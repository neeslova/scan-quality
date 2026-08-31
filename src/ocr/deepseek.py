"""Прогон DeepSeek-OCR по корпусу. Запускается на GPU-хосте, не локально.

**Это офлайн-экстрактор, а не часть пайплайна.** Модель требует CUDA, Python
3.12 и torch 2.6, тогда как рабочее окружение проекта — Windows, CPU, Python 3.9.
Поэтому модуль устроен как разовый прогон: он читает корпус, пишет `.jsonl` с
сырыми прочтениями, и на этом его роль заканчивается. Локальный код потом читает
этот файл и считает по нему сигналы — без GPU и без единой зависимости отсюда.
Продакшн модель не требует, и это тот же принцип, по которому в проекте живёт
ONNX-сеть: дорогое считается один раз.

**Сохраняется сырой текст, а не готовые метрики.** Прогон стоит часов GPU и
повторить его дёшево не выйдет, а формулы сигналов ещё будут меняться. Всё, что
можно пересчитать локально, здесь не считается принципиально.

**Каждая страница читается дважды — в низком и высоком разрешении.** Это и есть
self-consistency: расхождение двух прочтений показывает, что модель не столько
читала, сколько додумывала, и никакой разметки для этого не нужно. Из-за этого
прогон стоит вдвое дороже, и это осознанная плата за главный сигнал этапа.

Устойчивость к обрыву обязательна: бесплатная сессия Colab отключается по
таймауту в середине корпуса. Готовое адресуется по sha256 файла, при повторном
запуске уже посчитанное пропускается, файл открывается на дозапись.

Запуск (в Colab, после установки зависимостей):
    python -m src.ocr.deepseek --data /content/data --out /content/deepseek_tg.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_ID = "deepseek-ai/DeepSeek-OCR"

# Режимы разрешения из карточки модели. Имя -> (base_size, image_size, crop_mode).
RESOLUTION_MODES = {
    "tiny": (512, 512, False),
    "small": (640, 640, False),
    "base": (1024, 1024, False),
    "large": (1280, 1280, False),
    "gundam": (1024, 640, True),
    # Промежуточные режимы, которых в карточке нет. Заведены потому, что на T4
    # `base` и `gundam` не влезают в видеопамять (замерено: пик 14.3 и 13.7 ГБ
    # при потолке 14.56), а пара `tiny` + `small` — это 512 против 640, слишком
    # близко: расхождение прочтений на такой паре будет слабым и на плохом
    # скане тоже. Ищем самый широкий разрыв, который помещается в карту.
    "mid": (768, 768, False),
    # С нарезкой: страница видится тайлами, а не одним уменьшенным кадром.
    # Это другой взгляд, а не просто другое разрешение, поэтому расхождение с
    # `tiny` должно быть содержательнее, чем у пары близких масштабов.
    "tiles": (768, 640, True),
}

# Пара для self-consistency. Берём максимально разнесённые режимы из дешёвых:
# чем сильнее отличается вход, тем честнее проверка на «додумывание».
DEFAULT_MODES = ("tiny", "base")

# Просим текст без разметки макета: сигналы считаются по словам, а теги
# `<|ref|>`/`<|det|>` из режима grounding только зашумили бы их.
PROMPT = "<image>\nFree OCR. "

# Потолок длины генерации. Remote-код модели зашивает 8192 токена, и на здоровой
# странице это без разницы — она укладывается в тысячу с небольшим. Но модель,
# залипшая в повторе, идёт до упора: замер на живом корпусе дал медиану 38 с при
# максимуме 670 с на страницу, разброс в 94 раза. Дороже всего обходятся ровно
# те страницы, ради которых этап и затеян.
#
# 2048 токенов — это примерно 5-7 тысяч символов, вдвое больше плотной А4.
# Обрезанная страница не теряется как наблюдение: упёршийся в потолок текст сам
# по себе означает, что модель ушла в повтор, и детект зацикливания это увидит.
MAX_NEW_TOKENS = 2048

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"})
PDF_SUFFIXES = frozenset({".pdf"})


@dataclass
class PageJob:
    """Одна страница, подлежащая прочтению."""

    path: Path
    page: int
    sha256: str
    relative: str

    @property
    def key(self) -> str:
        return f"{self.sha256}#{self.page}"


@dataclass
class PageResult:
    """Прочтения одной страницы во всех режимах плюс исход."""

    image: str
    page: int
    sha256: str
    texts: dict[str, str] = field(default_factory=dict)
    elapsed_s: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "image": self.image,
                "page": self.page,
                "sha256": self.sha256,
                "texts": self.texts,
                "elapsed_s": self.elapsed_s,
                "status": self.status,
                "error": self.error,
            },
            ensure_ascii=False,
        )


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> int:
    import fitz

    try:
        with fitz.open(str(path)) as document:
            return int(document.page_count)
    except Exception as error:  # noqa: BLE001 — битый файл не повод бросать корпус
        logger.warning("%s: не читается (%s)", path.name, error)
        return 0


def collect_jobs(root: Path) -> list[PageJob]:
    """Все страницы корпуса. PDF разворачивается, порядок детерминирован."""
    jobs: list[PageJob] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES and suffix not in PDF_SUFFIXES:
            continue
        digest = file_sha256(path)
        relative = path.relative_to(root).as_posix()
        pages = _pdf_page_count(path) if suffix in PDF_SUFFIXES else 1
        for index in range(pages):
            jobs.append(PageJob(path=path, page=index, sha256=digest, relative=relative))
    return jobs


def load_done(path: Path) -> set[str]:
    """Ключи страниц, уже посчитанных в прошлых запусках.

    Битые строки игнорируются молча: обрыв сессии в момент записи оставляет
    неполную последнюю строку, и падать из-за неё при следующем запуске нельзя.
    """
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
                done.add(f"{record['sha256']}#{record['page']}")
    return done


def attention_implementation() -> str:
    """`flash_attention_2` или `eager` — по поколению карты.

    FlashAttention-2 требует Ampere и новее. На бесплатном Colab выдают T4
    (Turing, SM 7.5), где он не собирается вовсе, и единственный рабочий путь —
    штатное внимание. Проверяем железо, а не гадаем.

    Третьего варианта нет: `sdpa`, которое считало бы внимание без
    материализации матрицы и сильно сэкономило бы память, эта модель не
    поддерживает. В `ATTENTION_CLASSES` её remote-кода только `eager` и
    `flash_attention_2` (и их mla/mha-варианты), `_supports_sdpa` не выставлен.
    Отсюда и потолок по разрешению на Turing — см. `RESOLUTION_MODES`.
    """
    import torch

    if not torch.cuda.is_available():
        return "eager"
    major, _ = torch.cuda.get_device_capability()
    return "flash_attention_2" if major >= 8 else "eager"


def model_dtype():
    """Всегда bfloat16 — тип диктует remote-код модели, а не карта.

    Соблазн взять float16 на Turing выглядит разумно: bfloat16 там не поддержан
    аппаратно и считается через эмуляцию. Но `modeling_deepseekocr.py` зашивает
    bfloat16 жёстко — приводит к нему тензоры изображений
    (`image_transform(...).to(torch.bfloat16)`) и оборачивает генерацию в
    `torch.autocast("cuda", dtype=torch.bfloat16)`. С весами в float16 половины
    модели расходятся: текстовые эмбеддинги выходят Half, выход визуального
    проектора — Float, и `masked_scatter_` в `forward` падает на разнице типов.

    Выбирать тип по железу здесь нельзя: он часть контракта модели. Плата —
    эмуляция bfloat16 на T4, то есть медленнее; это видно в замере скорости.
    """
    import torch

    return torch.bfloat16


def _capped_generate(generate, limit: int = MAX_NEW_TOKENS):
    """Оборачивает `generate`, срезая запрошенную длину до `limit`.

    Иначе никак: `infer` из remote-кода передаёт `max_new_tokens=8192` явным
    аргументом, а явный аргумент бьёт любой `generation_config`. Параметра,
    которым это регулируется снаружи, у `infer` нет.
    """

    def wrapped(*args, **kwargs):
        kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", limit), limit)
        return generate(*args, **kwargs)

    return wrapped


class DeepSeekOCR:
    """Обёртка над моделью. Грузится один раз, читает страницу в заданном режиме."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        attention = attention_implementation()
        dtype = model_dtype()
        logger.info("Загрузка %s (attention=%s, dtype=%s)", self._model_id, attention, dtype)

        # ОЗУ здесь узкое место, а не видеопамять: у бесплатного Colab её 12.7 ГБ.
        # `torch_dtype` не даёт материализовать веса в float32 (было бы около
        # 12 ГБ на трёхмиллиардной модели). Но и этого мало: чекпойнт лежит
        # одним шардом на 6.7 ГБ, поэтому `low_cpu_mem_usage` со своей загрузкой
        # «по шарду за раз» не выигрывает ничего — шард ровно один. Копия
        # тензоров плюс mmap того же файла дают около 13 ГБ, и сеанс умирает.
        #
        # `device_map` снимает это: accelerate раскладывает веса по карте
        # потензорно, прямо из mmap, и полной копии в ОЗУ не возникает вовсе.
        # Указан явный `{"": 0}`, а не `"auto"`: последний требует от модели
        # объявленного `_no_split_modules`, которого у remote-кода может не быть,
        # а размазывать по устройствам нам и не нужно — карта одна.
        placement = {"device_map": {"": 0}} if torch.cuda.is_available() else {}

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            use_safetensors=True,
            _attn_implementation=attention,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **placement,
        )
        # Модель уже на карте и в нужном типе: `.cuda()` и `.to(dtype)` здесь не
        # только избыточны, но и опасны — на модели, разложенной accelerate,
        # ручной перенос ломает расставленные хуки.
        self._model = self._model.eval()
        self._model.generate = _capped_generate(self._model.generate)

    def read(self, image_path: Path, mode: str, output_dir: Path) -> str:
        """Текст страницы в заданном режиме разрешения.

        Возвращаемое значение `infer` в документации модели не описано: он может
        отдать строку, а может только записать файл в `output_path`. Поддержаны
        оба поведения, потому что выяснять это в середине многочасового прогона —
        не лучшее место.
        """
        if self._model is None:
            self.load()

        base_size, image_size, crop_mode = RESOLUTION_MODES[mode]
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = self._model.infer(
                self._tokenizer,
                prompt=PROMPT,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=False,
            )
        finally:
            # Освобождаем кэш аллокатора между прочтениями. Утечки здесь нет
            # (`infer` работает под `no_grad`), но каждая страница просит блоки
            # своего размера, и на длинном прогоне сегменты фрагментируются.
            # На T4 запас по видеопамяти измеряется сотнями мегабайт, так что
            # эта уборка — разница между прогоном и падением на середине.
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if isinstance(result, str) and result.strip():
            return result
        return _read_result_files(output_dir)


def _read_result_files(output_dir: Path) -> str:
    """Достаёт текст из того, что модель записала на диск."""
    candidates = sorted(
        (p for p in output_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".mmd"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    return ""


def _page_image(job: PageJob, workdir: Path) -> Path:
    """Путь к картинке страницы. PDF рендерится, изображение отдаётся как есть.

    Изображение намеренно не преобразуется: модель сама приводит вход к своему
    разрешению, и любой наш ресайз только исказил бы то, что она видит.
    """
    if job.path.suffix.lower() not in PDF_SUFFIXES:
        return job.path

    import fitz

    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / f"{job.sha256}_{job.page}.png"
    if not target.is_file():
        with fitz.open(str(job.path)) as document:
            document[job.page].get_pixmap(dpi=200).save(str(target))
    return target


def run(
    root: Path,
    out_path: Path,
    modes: tuple[str, ...] = DEFAULT_MODES,
    limit: Optional[int] = None,
    workdir: Optional[Path] = None,
) -> int:
    """Читает корпус и дописывает результаты. Возвращает число обработанных страниц."""
    for mode in modes:
        if mode not in RESOLUTION_MODES:
            raise SystemExit(f"Неизвестный режим {mode}; есть: {', '.join(RESOLUTION_MODES)}")

    workdir = workdir or out_path.parent / "_deepseek_work"
    jobs = collect_jobs(root)
    done = load_done(out_path)
    pending = [job for job in jobs if job.key not in done]
    if limit is not None:
        pending = pending[:limit]

    logger.info(
        "страниц в корпусе %d, уже готово %d, к обработке %d",
        len(jobs),
        len(done),
        len(pending),
    )
    if not pending:
        return 0

    engine = DeepSeekOCR()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    started_all = time.perf_counter()

    # Дозапись, а не перезапись: сессия Colab может оборваться в любой момент,
    # и всё посчитанное до обрыва должно пережить его.
    with out_path.open("a", encoding="utf-8") as fh:
        for index, job in enumerate(pending, start=1):
            result = PageResult(image=job.relative, page=job.page, sha256=job.sha256)
            try:
                image_path = _page_image(job, workdir)
                for mode in modes:
                    started = time.perf_counter()
                    result.texts[mode] = engine.read(image_path, mode, workdir / "out")
                    result.elapsed_s[mode] = round(time.perf_counter() - started, 2)
            except Exception as error:  # noqa: BLE001 — одна страница не роняет прогон
                result.status = "failed"
                result.error = f"{type(error).__name__}: {error}"
                logger.warning("%s#%d: %s", job.relative, job.page, result.error)

            fh.write(result.to_json() + "\n")
            fh.flush()
            processed += 1

            if index % 10 == 0 or index == len(pending):
                speed = (time.perf_counter() - started_all) / index
                left = (len(pending) - index) * speed
                logger.info(
                    "%d/%d, %.1f с/страница, осталось ~%.0f мин",
                    index,
                    len(pending),
                    speed,
                    left / 60,
                )

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--out", type=Path, required=True, help="куда дописывать jsonl")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="режимы разрешения через запятую: " + ", ".join(RESOLUTION_MODES),
    )
    parser.add_argument("--limit", type=int, default=None, help="взять только N страниц (замер)")
    parser.add_argument("--workdir", type=Path, default=None, help="временные файлы")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.data.is_dir():
        raise SystemExit(f"Не папка: {args.data}")

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    processed = run(args.data, args.out, modes=modes, limit=args.limit, workdir=args.workdir)
    logger.info("обработано страниц: %d -> %s", processed, args.out)


if __name__ == "__main__":
    main()
