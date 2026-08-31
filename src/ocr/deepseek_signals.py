"""Сигналы по выгрузке DeepSeek-OCR и проверка их против эталона. Считается локально.

Прогон модели остаётся в Colab, а всё, что можно посчитать по тексту, живёт
здесь: формулы сигналов будут меняться, и каждый раз занимать GPU ради этого
незачем. Вход — `.jsonl`, который написал `src.ocr.deepseek`, выход — таблица
с разделяющей способностью каждого сигнала.

**Главное, что печатает отчёт, — AUC на сигнал.** Он отвечает на единственный
вопрос этапа: видит ли DeepSeek-OCR разницу между хорошим и плохим сканом там,
где её не увидели CV-метрики. Сигнал с AUC около 0.5 бесполезен, каким бы
осмысленным он ни казался, и лучше узнать это до того, как он попадёт в вердикт.

Направление у сигналов разное: расхождение прогонов и зацикливание растут на
плохих страницах, а объём распознанного текста на них падает. Поэтому рядом с
AUC печатается направление — по нему видно, ведёт ли себя сигнал так, как
задумано, или он поймал что-то постороннее.

Запуск:
    python -m src.ocr.deepseek_signals --texts data/labeled/deepseek_tg.jsonl \\
        --golden data/labeled/golden_tg.jsonl --out reports/deepseek_tg.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.data.golden import read_golden
from src.ocr.signals import (
    foreign_char_ratio,
    max_repeat_run,
    ngram_repetition,
    normalized_distance,
    oov_ratio,
    tokenize,
)

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ("ru", "en")
# Знаки из `ocr.extra_chars` базового конфига плюс разметка markdown.
#
# Дописаны `|` и обратная кавычка не для красоты: DeepSeek-OCR возвращает
# markdown даже на промпт `Free OCR`, и на странице с таблицей вертикальных черт
# набирается несколько десятков. По категории Unicode `|` — это `Sm`,
# математический символ, а не пунктуация, поэтому `foreign_char_ratio` считал бы
# его чужим алфавитом. Тогда сигнал зависел бы от того, решила ли модель
# нарисовать таблицу, а это свойство режима: на одной и той же странице `tiny`
# сложил таблицу криво, `base` — верно. Мерить надо качество скана, а не стиль
# разметки.
#
# Список намеренно отдельный от `ocr.extra_chars`: тот откалиброван под EasyOCR,
# и трогать его — значит сдвинуть уже посчитанные якоря CV-метрик.
DEFAULT_EXTRA_CHARS = "0123456789.,;:!?()[]{}-–—«»\"'/\\%№@#&*+=<>_$|`~^×−"


@dataclass(frozen=True)
class PageSignals:
    """Сигналы по одной странице. Ключ — тот же, что в выгрузке."""

    image: str
    page: int
    sha256: str
    values: dict[str, float]
    status: str


def load_texts(paths: Union[Path, Sequence[Path]]) -> list[dict]:
    """Читает выгрузки прогона. Битые строки пропускаются с предупреждением.

    Файлов можно передать несколько, и тогда они сливаются по ключу
    `sha256#page`. Это нужно, чтобы режимы можно было добирать по одному:
    прогон дорогой, бесплатная сессия Colab короткая, и сначала имеет смысл
    прочитать весь корпус дешёвым режимом — три сигнала из четырёх считаются
    по одному прочтению, — а второй режим добрать отдельным заходом, когда
    станет ясно, что первые цифры того стоят.

    Текст режима берётся из первого файла, где он непустой: страница, упавшая
    в одном заходе и прочитанная в другом, не должна остаться пустой. Страница
    считается прочитанной, если у неё есть хоть один текст.
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]

    merged: dict[str, dict] = {}
    order: list[str] = []

    for path in paths:
        with Path(path).open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("%s, строка %d повреждена, пропущена", Path(path).name, number)
                    continue

                key = f"{row.get('sha256', '')}#{row.get('page', 0)}"
                target = merged.get(key)
                if target is None:
                    target = dict(row)
                    target["texts"] = dict(row.get("texts") or {})
                    target["elapsed_s"] = dict(row.get("elapsed_s") or {})
                    merged[key] = target
                    order.append(key)
                    continue

                for mode, text in (row.get("texts") or {}).items():
                    if text and not target["texts"].get(mode):
                        target["texts"][mode] = text
                        target["elapsed_s"][mode] = (row.get("elapsed_s") or {}).get(mode)
                if not row.get("error"):
                    target["error"] = target.get("error", "")

    for row in merged.values():
        row["status"] = "ok" if any(row["texts"].values()) else row.get("status", "failed")
        row["elapsed_s"] = {k: v for k, v in row["elapsed_s"].items() if v is not None}

    return [merged[key] for key in order]


def load_vocabulary(path: Optional[Path]) -> Optional[set[str]]:
    """Словарь для `oov_ratio`: по слову на строку.

    Без файла сигнал не считается вовсе. Подставлять сюда словарь, собранный по
    самому корпусу, нельзя: слова из плохих страниц попали бы в него наравне с
    хорошими, и метрика перестала бы что-либо различать.
    """
    if path is None:
        return None
    words = {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    logger.info("словарь: %d слов", len(words))
    return words


def page_signals(
    row: dict,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    extra_chars: str = DEFAULT_EXTRA_CHARS,
    vocabulary: Optional[set[str]] = None,
) -> PageSignals:
    """Считает все сигналы по одной записи выгрузки.

    Режимы не зашиты: их имена берутся из самой записи. Прогон мог идти с другой
    парой разрешений, и падать из-за этого при разборе результатов не следует.
    """
    texts: dict[str, str] = row.get("texts") or {}
    modes = sorted(texts)
    values: dict[str, float] = {}

    for mode in modes:
        text = texts[mode]
        values[f"repetition_{mode}"] = ngram_repetition(text)
        values[f"repeat_run_{mode}"] = float(max_repeat_run(text))
        values[f"foreign_{mode}"] = foreign_char_ratio(text, languages, extra_chars)
        values[f"words_{mode}"] = float(len(tokenize(text)))
        if vocabulary is not None:
            values[f"oov_{mode}"] = oov_ratio(text, vocabulary)

    # Главный сигнал: два разрешения прочли страницу по-разному.
    if len(modes) >= 2:
        values["divergence"] = normalized_distance(texts[modes[0]], texts[modes[1]])

    return PageSignals(
        image=row.get("image", ""),
        page=int(row.get("page", 0)),
        sha256=row.get("sha256", ""),
        values=values,
        status=row.get("status", "ok"),
    )


def collect(
    rows: list[dict],
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    extra_chars: str = DEFAULT_EXTRA_CHARS,
    vocabulary: Optional[set[str]] = None,
) -> list[PageSignals]:
    return [page_signals(row, languages, extra_chars, vocabulary) for row in rows]


def join_with_golden(signals: list[PageSignals], golden_path: Path) -> tuple[list[dict], list[int]]:
    """Сопоставляет сигналы с эталоном по sha256 и номеру страницы.

    Сверка идёт по хешу, а не по имени: имена файлов в корпусе повторяются, а
    хеш однозначен. Ровно поэтому он и сохраняется на обоих концах.
    """
    truth_by_key = {f"{r.sha256}#{r.page}": r.label for r in read_golden(golden_path)}

    joined: list[dict] = []
    labels: list[int] = []
    for item in signals:
        if item.status != "ok":
            continue
        label = truth_by_key.get(f"{item.sha256}#{item.page}")
        if label is None:
            continue
        joined.append(item.values)
        labels.append(1 if label == "bad" else 0)
    return joined, labels


def rank_signals(rows: list[dict], labels: list[int]) -> list[tuple[str, float, int]]:
    """AUC каждого сигнала. Возвращает (имя, AUC, число страниц), лучшие сверху.

    AUC оставляем как есть, не разворачивая к «больше — лучше»: значение ниже
    0.5 означает, что сигнал работает в обратную сторону, и это надо видеть, а
    не прятать за модулем разности.
    """
    from sklearn.metrics import roc_auc_score

    truth = np.array(labels)
    if truth.size == 0 or len(set(labels)) < 2:
        return []

    names = sorted({name for row in rows for name in row})
    result: list[tuple[str, float, int]] = []
    for name in names:
        values = np.array([row.get(name, np.nan) for row in rows], dtype=float)
        usable = ~np.isnan(values)
        if usable.sum() < 20 or len(set(truth[usable].tolist())) < 2:
            continue
        if np.nanstd(values[usable]) == 0:
            continue
        result.append(
            (name, float(roc_auc_score(truth[usable], values[usable])), int(usable.sum()))
        )

    result.sort(key=lambda item: abs(item[1] - 0.5), reverse=True)
    return result


def format_report(
    signals: list[PageSignals],
    rows: list[dict],
    labels: list[int],
) -> str:
    lines: list[str] = []
    failed = sum(1 for s in signals if s.status != "ok")
    lines.append(f"Страниц в выгрузке: {len(signals)} (не прочитано: {failed})")
    lines.append(f"Сопоставлено с эталоном: {len(rows)} (bad: {sum(labels)})")

    empty = sum(1 for s in signals if s.status == "ok" and s.values.get("words_base", 1) == 0)
    if empty:
        lines.append(f"Пустых прочтений: {empty}")

    ranked = rank_signals(rows, labels)
    if not ranked:
        lines.append("")
        lines.append("Сигналы не с чем сравнить: в эталоне один класс или мало страниц.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Разделяющая способность сигналов (bad — положительный класс):")
    lines.append("")
    header = "{:22s}{:>8s}{:>10s}{:>7s}".format("сигнал", "AUC", "направл.", "n")
    lines.append(header)
    lines.append("-" * len(header))
    for name, auc, count in ranked:
        direction = "растёт" if auc > 0.5 else "падает"
        lines.append(f"{name:22s}{auc:8.3f}{direction:>10s}{count:7d}")

    lines.append("")
    lines.append("Направление читается так: «растёт» — сигнал выше на плохих страницах.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--texts",
        type=Path,
        required=True,
        nargs="+",
        help="выгрузки src.ocr.deepseek; несколько файлов сливаются по sha256#page",
    )
    parser.add_argument("--golden", type=Path, required=True, help="эталон good/bad")
    parser.add_argument("--vocabulary", type=Path, default=None, help="словарь для oov, по слову")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--out", type=Path, default=None, help="куда записать отчёт")
    parser.add_argument("--dump", type=Path, default=None, help="куда сложить сигналы, jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    languages = tuple(code.strip() for code in args.languages.split(",") if code.strip())
    signals = collect(
        load_texts(args.texts),
        languages=languages,
        vocabulary=load_vocabulary(args.vocabulary),
    )
    if not signals:
        raise SystemExit(f"Выгрузка пуста: {args.texts}")

    rows, labels = join_with_golden(signals, args.golden)
    text = format_report(signals, rows, labels)
    print(text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("отчёт записан в %s", args.out)

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8") as fh:
            for item in signals:
                fh.write(
                    json.dumps(
                        {
                            "image": item.image,
                            "page": item.page,
                            "sha256": item.sha256,
                            "status": item.status,
                            **item.values,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("сигналы записаны в %s", args.dump)


if __name__ == "__main__":
    main()
