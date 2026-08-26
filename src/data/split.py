"""Сплит размеченного набора по ДОКУМЕНТАМ и правило группировки страниц.

Почему по документам. У страниц одного дела одна бумага, один прогон сканера и
один и тот же перекос. Если перемешать страницы случайно, соседние страницы одного
документа попадут и в train, и в test — модель узнает их по фактуре бумаги, а не по
дефекту, и метрики завышаются на 10-15%. Это риск №1 из PLAN.md.

Запуск:
    python -m src.data.split --labels data/labeled/labels.jsonl --out data/splits
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Union

from src.config import Config, load_config
from src.schema import LabelRecord

logger = logging.getLogger(__name__)

# Длина числового префикса Bates-номера, по которому страницы считаются одним делом.
BATES_PREFIX_LEN = 7
SPLIT_NAMES = ("train", "val", "test")


def document_id(path: Union[str, Path], strategy: str = "bates7") -> str:
    """Из пути к странице — id документа, по которому идёт сплит.

    `bates7` рассчитан на Tobacco3482: имена файлов там — Bates-номера
    (`0000002770`), и подряд идущие номера означают соседние страницы одного дела.
    Группируем по первым семи цифрам; в корпусе 119 таких групп размером до пяти.
    """
    p = Path(path)
    if strategy == "stem":
        return p.stem
    if strategy == "parent":
        return p.parent.name
    if strategy == "bates7":
        digits = re.sub(r"\D", "", p.stem)
        # Без цифр в имени группировать не по чему — считаем страницу отдельным делом.
        return digits[:BATES_PREFIX_LEN] if digits else p.stem
    raise ValueError(f"Неизвестная стратегия document_id: {strategy}")


def group_by_document(records: list[LabelRecord]) -> dict[str, list[LabelRecord]]:
    groups: dict[str, list[LabelRecord]] = defaultdict(list)
    for record in records:
        groups[record.document].append(record)
    return dict(groups)


def split_documents(
    groups: dict[str, list[LabelRecord]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Раскладывает документы по частям, выравнивая доли по СТРАНИЦАМ.

    Документы разной длины, поэтому делить их поровну по количеству — не то же
    самое, что поровну по страницам. Идём по перемешанному списку и каждый
    документ отдаём той части, которая сильнее всех отстаёт от своей цели.
    """
    if not groups:
        return {name: [] for name in SPLIT_NAMES}

    order = sorted(groups)  # стабильная база до перемешивания
    random.Random(seed).shuffle(order)
    # Длинные документы раскладываем первыми: под конец останутся мелкие,
    # которыми удобно доводить доли до целевых.
    order.sort(key=lambda doc: len(groups[doc]), reverse=True)

    total_pages = sum(len(pages) for pages in groups.values())
    targets = {name: ratios[name] * total_pages for name in SPLIT_NAMES}
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    filled = dict.fromkeys(SPLIT_NAMES, 0)

    for doc in order:
        name = max(SPLIT_NAMES, key=lambda s: targets[s] - filled[s])
        assigned[name].append(doc)
        filled[name] += len(groups[doc])

    for name in SPLIT_NAMES:
        assigned[name].sort()
    return assigned


def build_split(records: list[LabelRecord], config: Config) -> dict[str, dict]:
    """Готовый сплит: списки документов и страниц по каждой части."""
    groups = group_by_document(records)
    documents = split_documents(groups, config.split.ratios, config.split.seed)

    result: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        images = sorted(r.image for doc in documents[name] for r in groups[doc])
        result[name] = {
            "documents": documents[name],
            "images": images,
            "n_documents": len(documents[name]),
            "n_images": len(images),
        }
    return result


def label_frequencies(records: list[LabelRecord], labels: list[str]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for label, present in record.labels.items():
            if present:
                counts[label] += 1
    return {label: counts.get(label, 0) for label in labels}


def read_labels(path: Union[str, Path], dedupe: bool = True) -> list[LabelRecord]:
    """Читает labels.jsonl. Битые строки пропускает с предупреждением, а не падает.

    Файл только дозаписывается: переразмеченная страница добавляет строку, а не
    заменяет старую. По умолчанию применяем то же правило, что и разметчик, —
    побеждает последняя запись. Без этого переразмеченные страницы считались бы
    дважды и завышали объём набора. `dedupe=False` отдаёт всю историю правок:
    по ней видно, где предразметка врёт систематически.
    """
    latest: dict[str, LabelRecord] = {}
    history: list[LabelRecord] = []

    # utf-8-sig: инструменты Windows дописывают BOM, и запись не разбирается.
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = LabelRecord.model_validate_json(line)
            except ValueError as exc:
                logger.warning("%s:%d — строка пропущена: %s", path, number, exc)
                continue
            history.append(record)
            latest[record.image] = record

    if not dedupe:
        return history
    if len(history) != len(latest):
        logger.info(
            "%s: %d строк -> %d страниц (переразмечено %d)",
            path,
            len(history),
            len(latest),
            len(history) - len(latest),
        )
    return list(latest.values())


def write_split(split: dict[str, dict], out_dir: Union[str, Path]) -> None:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        target = directory / f"{name}.json"
        with target.open("w", encoding="utf-8") as fh:
            json.dump(split[name], fh, ensure_ascii=False, indent=2)
        logger.info(
            "%s: %d документов, %d страниц",
            target.name,
            split[name]["n_documents"],
            split[name]["n_images"],
        )


def write_frequency_report(
    records: list[LabelRecord],
    split: dict[str, dict],
    config: Config,
    path: Union[str, Path],
) -> None:
    """Отчёт по частотам меток для записки: сколько чего и не пропала ли редкая метка.

    Контрольная цифра из PLAN.md §7.4 — редкая метка должна встречаться не реже
    ~15% выборки, иначе `pos_weight` её не вытянет.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    by_split = {
        name: label_frequencies(
            [r for r in records if r.image in set(split[name]["images"])], config.labels
        )
        for name in SPLIT_NAMES
    }
    total = len(records)

    lines = [
        "# Частоты меток после ручной разметки",
        "",
        f"Всего размечено страниц: **{total}**, документов: "
        f"**{len(group_by_document(records))}**.",
        "",
        "| Метка | train | val | test | всего | доля | статус |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label in config.labels:
        counts = [by_split[name][label] for name in SPLIT_NAMES]
        overall = sum(counts)
        share = overall / total if total else 0.0
        if label in config.labeling.auto_labels:
            status = "ставится автоматом (С3)"
        elif overall == 0:
            status = "**нет примеров**"
        elif counts[1] == 0 or counts[2] == 0:
            status = "**нет в val или test**"
        elif share < 0.15:
            status = "редкая, нужно ≥15%"
        else:
            status = "ок"
        row = "".join(f" {c} |" for c in counts)
        lines.append(f"| `{label}` |{row} {overall} | {share:.1%} | {status} |")

    lines += [
        "",
        "## Разбиение",
        "",
        "| Часть | документов | страниц | доля |",
        "|---|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        part = split[name]
        share = part["n_images"] / total if total else 0.0
        lines.append(f"| {name} | {part['n_documents']} | {part['n_images']} | {share:.1%} |")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Отчёт по частотам: %s", target)


def check_leakage(split: dict[str, dict]) -> list[str]:
    """Один документ не должен встречаться в двух частях. Возвращает нарушителей."""
    seen: dict[str, str] = {}
    leaked: list[str] = []
    for name in SPLIT_NAMES:
        for doc in split[name]["documents"]:
            if doc in seen and seen[doc] != name:
                leaked.append(f"{doc}: {seen[doc]} и {name}")
            seen[doc] = name
    return leaked


def main() -> None:
    parser = argparse.ArgumentParser(description="Сплит размеченного набора по документам")
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="папка для train/val/test.json")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/label_frequencies.md"),
        help="куда положить отчёт по частотам меток",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    records = read_labels(args.labels)
    if not records:
        raise SystemExit(f"В {args.labels} нет ни одной записи")

    split = build_split(records, config)
    write_split(split, args.out)

    leaked = check_leakage(split)
    if leaked:
        raise SystemExit("Утечка документов между частями:\n  " + "\n  ".join(leaked))

    write_frequency_report(records, split, config, args.report)

    print(f"\nвсего: {len(records)} страниц, {len(group_by_document(records))} документов")
    print(f"{'часть':8s}{'документов':>12s}{'страниц':>10s}{'доля':>8s}")
    for name in SPLIT_NAMES:
        part = split[name]
        share = part["n_images"] / len(records)
        print(f"{name:8s}{part['n_documents']:12d}{part['n_images']:10d}{share:8.1%}")

    by_split = {
        name: label_frequencies(
            [r for r in records if r.image in set(split[name]["images"])], config.labels
        )
        for name in SPLIT_NAMES
    }
    print(f"\n{'метка':16s}" + "".join(f"{n:>8s}" for n in SPLIT_NAMES) + f"{'всего':>8s}")
    for label in config.labels:
        counts = [by_split[name][label] for name in SPLIT_NAMES]
        flag = "  <- нет в val/test" if counts[1] == 0 or counts[2] == 0 else ""
        print(f"{label:16s}" + "".join(f"{c:8d}" for c in counts) + f"{sum(counts):8d}{flag}")


if __name__ == "__main__":
    main()
