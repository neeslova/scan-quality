"""Прогон VLM-судьи по страницам. Отвечает за ретрай, кеш и честный отказ.

Три правила, ради которых модуль существует отдельно от бэкенда:

**Неудача записывается как неудача.** Ни один сбой не превращается в оценки по
умолчанию. Страница со `status="failed"` не попадает ни в метрики, ни в вердикт —
она видна в отчёте как непосчитанная. Дефолтные нули на месте несостоявшегося
вызова отправили бы весь брак в `good`, и заметить это по сводным цифрам нельзя.

**Ретрай ровно один.** Модели свойственно испортить JSON случайно, и второй
заход это чинит. Дальше повторять смысла нет: устойчиво негодный ответ означает,
что дело в промпте или в самой странице, и прятать это за пятью попытками —
значит платить за одну и ту же ошибку пять раз.

**Кеш по sha256.** Судья стоит денег, и повторный запуск не должен пересчитывать
готовое. Ключ — хеш файла и номер страницы, тот же, что в остальных прогонах.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.config import Config, load_config
from src.judge.backends import BackendError, JudgeBackend, encode_image, extract_json, get_backend
from src.judge.prompt import SYSTEM, build_prompt
from src.judge.schema import JudgeAnswer, JudgeRecord

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # первый заход плюс один ретрай


def decide_verdict(scores: dict[str, float], config: Config) -> str:
    """Тот же вердикт, что и в пайплайне, по оценкам судьи в шкале 0..1.

    Правило намеренно продублировано из `src.pipeline`, а не импортировано:
    судья обязан подчиняться тем же порогам, и расхождение между тем, что
    записано в промпте, и тем, что считает код, должно ловиться тестом.
    """
    from src.pipeline import decide_verdict as pipeline_verdict

    return pipeline_verdict(scores, config.verdict)


def ask_once(
    backend: JudgeBackend,
    image_path: Path,
    prompt: str,
    config: Config,
) -> tuple[JudgeAnswer, str]:
    """Один запрос: отправить, разобрать, проверить. Любая беда — исключение."""
    image, media_type = encode_image(image_path)
    raw = backend.ask(image, media_type, prompt, SYSTEM)
    answer = JudgeAnswer.model_validate_json(extract_json(raw))
    return answer.validate_axes(config.labels), raw


def judge_page(
    backend: JudgeBackend,
    image_path: Path,
    prompt: str,
    config: Config,
    model_name: str,
    page: int = 0,
    sha256: str = "",
    relative: Optional[str] = None,
) -> JudgeRecord:
    """Судит одну страницу. Возвращает запись — успешную или с честным отказом."""
    started = time.perf_counter()
    last_error = ""
    last_raw = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer, raw = ask_once(backend, image_path, prompt, config)
        except (BackendError, ValidationError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            last_raw = last_raw or getattr(error, "raw", "")
            if attempt < MAX_ATTEMPTS:
                logger.info("%s: попытка %d не удалась (%s)", image_path.name, attempt, last_error)
                continue
            break
        else:
            return JudgeRecord(
                image=relative or image_path.name,
                page=page,
                sha256=sha256,
                status="ok",
                answer=answer,
                verdict=decide_verdict(answer.as_unit_scores(), config),
                model=model_name,
                elapsed_s=round(time.perf_counter() - started, 2),
            )

    logger.warning("%s: судья не ответил (%s)", image_path.name, last_error)
    return JudgeRecord(
        image=relative or image_path.name,
        page=page,
        sha256=sha256,
        status="failed",
        error=last_error,
        model=model_name,
        elapsed_s=round(time.perf_counter() - started, 2),
        raw=last_raw[:2000],
    )


def load_done(path: Path) -> set[str]:
    """Ключи уже осуждённых страниц. Провалы повторяем, успехи — нет."""
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                done.add(f"{record.get('sha256', '')}#{record.get('page', 0)}")
    return done


def select_grey_zone(reports_path: Path, config: Config) -> set[str]:
    """Имена страниц, попавших в серую зону по дешёвым этапам.

    Это и есть каскад: судья вызывается там, где система сомневается, а не по
    всему корпусу. На Tobacco серая зона — примерно 28% страниц, то есть три
    четверти вызовов не делаются вовсе.
    """
    grey: set[str] = set()
    with reports_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            report = json.loads(line)
            if report.get("verdict") == "acceptable":
                grey.add(report["image"])
    logger.info("серая зона: %d страниц", len(grey))
    return grey


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--out", type=Path, required=True, help="куда дописывать jsonl")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--only-grey-zone",
        type=Path,
        default=None,
        help="отчёты пайплайна: судить только страницы с вердиктом acceptable",
    )
    parser.add_argument("--limit", type=int, default=None, help="взять только N страниц")
    parser.add_argument("--few-shot", type=Path, default=None, help="файл с примерами для промпта")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, args.corpus)

    if not config.judge.enabled:
        raise SystemExit("judge.enabled = false — включите судью в конфиге корпуса")

    from src.data.golden import file_sha256
    from src.ocr.deepseek import plan_jobs

    few_shot = args.few_shot.read_text(encoding="utf-8") if args.few_shot else None
    prompt = build_prompt(config, few_shot)
    backend = get_backend(
        config.judge.backend, config.judge.model, config.judge.max_tokens, config.judge.timeout_s
    )

    # Перемешанный порядок, а не алфавитный: корпус разложен по папкам классов,
    # и `--limit` по алфавиту отдал бы одни только `Good`. Судья платный, и
    # выборка из одного класса — это выброшенные деньги: метрику по ней не
    # посчитать вовсе.
    jobs = plan_jobs(args.data)
    done = load_done(args.out)
    pending = [job for job in jobs if job.key not in done]

    if args.only_grey_zone is not None:
        grey = select_grey_zone(args.only_grey_zone, config)
        before = len(pending)
        pending = [job for job in pending if Path(job.relative).name in grey]
        logger.info("каскад отсеял %d из %d страниц", before - len(pending), before)

    if args.limit is not None:
        pending = pending[: args.limit]

    logger.info("к обработке: %d страниц", len(pending))
    if not pending:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for index, job in enumerate(pending, start=1):
            record = judge_page(
                backend,
                job.path,
                prompt,
                config,
                model_name=config.judge.model,
                page=job.page,
                sha256=job.sha256 or file_sha256(job.path),
                relative=job.relative,
            )
            fh.write(record.model_dump_json() + "\n")
            fh.flush()
            failed += record.status == "failed"
            if index % 20 == 0 or index == len(pending):
                logger.info("%d/%d, не ответил на %d", index, len(pending), failed)

    logger.info("готово: %d страниц, из них не осуждено %d", len(pending), failed)


if __name__ == "__main__":
    main()
