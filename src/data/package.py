"""Упаковка датасета для Drive: рецепт вместо пикселей.

В архив идут только те страницы, которые реально нужны обучению:
  * эталоны, из которых собирается синтетика (их перечисляет манифест);
  * размеченные вручную страницы train, val и test.

Готовые картинки синтетики не кладём. Восемь тысяч отрисованных страниц весят
9.4 ГБ, а манифест — три мегабайта: эталон, набор меток, силы дефектов и зерно
полностью задают страницу, и в Colab она собирается за минуты (`--from-manifest`).

Запуск:
    python -m src.data.package --data data/raw/tobacco3482/data \\
        --manifest data/synthetic/manifest.jsonl --labels data/labeled/labels.jsonl \\
        --splits data/splits --out data/scanq_dataset.tar
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

from src.data.generate import read_manifest
from src.data.split import read_labels

logger = logging.getLogger(__name__)


def collect_pages(manifest: Path, labels: Path) -> tuple[list[str], dict[str, int]]:
    """Уникальные относительные пути страниц, которые надо увезти."""
    references = {record.reference for record in read_manifest(manifest)}
    labelled = {record.image for record in read_labels(labels)}
    counts = {
        "эталоны": len(references),
        "размеченные": len(labelled),
        "пересечение": len(references & labelled),
    }
    return sorted(references | labelled), counts


def build(
    data_root: Path,
    manifest: Path,
    labels: Path,
    splits: Path,
    out: Path,
) -> tuple[int, int]:
    pages, counts = collect_pages(manifest, labels)
    out.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    total_bytes = 0
    with tarfile.open(out, "w") as tar:
        for relative in pages:
            source = data_root / relative
            if not source.is_file():
                logger.warning("нет файла: %s", relative)
                missing += 1
                continue
            total_bytes += source.stat().st_size
            tar.add(source, arcname=f"pages/{relative}")

        tar.add(manifest, arcname="manifest.jsonl")
        tar.add(labels, arcname="labels.jsonl")
        for name in ("train", "val", "test"):
            part = splits / f"{name}.json"
            if not part.is_file():
                raise SystemExit(f"Нет файла сплита: {part}")
            tar.add(part, arcname=f"splits/{name}.json")

    for name, value in counts.items():
        print(f"{name:14s}{value:6d}")
    if missing:
        print(f"{'пропущено':14s}{missing:6d}")
    return len(pages) - missing, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Упаковка датасета для Drive")
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="куда писать tar")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    packed, raw_bytes = build(args.data, args.manifest, args.labels, args.splits, args.out)

    size = args.out.stat().st_size
    print(f"\nстраниц в архиве: {packed}")
    print(f"архив: {args.out}  ({size / 1024**2:.0f} МБ)")
    print("для сравнения, отрисованная синтетика заняла бы около 9.4 ГБ")


if __name__ == "__main__":
    main()
