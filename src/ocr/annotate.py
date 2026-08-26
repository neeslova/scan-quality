"""Проставляет метку `unreadable` по всему размеченному набору.

Читает labels.jsonl, прогоняет OCR по каждой странице и дописывает обновлённую
запись: ручные метки сохраняются как есть, добавляется только `unreadable`.
Файл append-only, побеждает последняя запись — значит правки человека, сделанные
позже, перекроют автомат, и наоборот. Порядок запусков имеет значение.

Сам скор кладётся в `derived`: обучение бинарное и градацию теряет, а для
подбора порога и для корреляционного анализа она нужна.

Запуск:
    python -m src.ocr.annotate --labels data/labeled/labels.jsonl \\
        --data data/raw/tobacco3482/data
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.data.split import read_labels
from src.io.loader import load_page
from src.labeling.app import append_label
from src.ocr.engine import read_page, shared_engine
from src.ocr.readability import analyze_words, unreadable_score
from src.schema import LabelRecord

logger = logging.getLogger(__name__)

_STATE: dict[str, object] = {}


def _init_worker(config_path: Optional[str], overlays: Optional[list[str]]) -> None:
    import cv2

    cv2.setNumThreads(1)
    config = load_config(config_path, overlays)
    _STATE["config"] = config
    # Модель грузится один раз на процесс: инициализация EasyOCR стоит секунды.
    _STATE["engine"] = shared_engine(config.ocr.engine, config.ocr.languages)


def _annotate_one(args: tuple[str, str]) -> Optional[tuple[str, float, int, float, float]]:
    """(image, unreadable_score, n_boxes, confidence, nonword) или None."""
    image, root = args
    config: Config = _STATE["config"]  # type: ignore[assignment]
    engine = _STATE["engine"]

    try:
        page = load_page(
            Path(root) / image,
            target_dpi=config.data.target_dpi,
            dpi_fallback=config.data.dpi_fallback,
            allow_upscale=config.data.allow_upscale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s пропущен: %s", image, exc)
        return None

    words = read_page(page.gray, engine, config.ocr.work_side)
    result = analyze_words(words, config, page.width * page.height, engine.name)
    score = unreadable_score(result, config)
    if score is None:
        return None
    return (image, score, result.n_boxes, result.mean_confidence, result.nonword_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Автоматическая метка unreadable по OCR")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="каждый процесс держит свою копию модели OCR, память растёт линейно",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    records = read_labels(args.labels)
    if args.limit:
        records = records[: args.limit]
    todo = [r for r in records if "unreadable" not in r.labels]
    print(
        f"размечено страниц: {len(records)}, без unreadable: {len(todo)}",
        file=sys.stderr,
    )
    if not todo:
        return

    by_image = {r.image: r for r in records}
    payload = [(r.image, str(args.data)) for r in todo]
    started = time.perf_counter()
    threshold = config.ocr.unreadable_threshold
    written = 0
    skipped = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.config) if args.config else None,
            [str(p) for p in args.corpus] if args.corpus else None,
        ),
    ) as pool:
        for index, item in enumerate(pool.map(_annotate_one, payload, chunksize=2), 1):
            if item is None:
                skipped += 1
                continue
            image, score, n_boxes, confidence, nonword = item
            base = by_image[image]
            append_label(
                args.labels,
                LabelRecord(
                    image=base.image,
                    document=base.document,
                    corpus=base.corpus,
                    labels={**base.labels, "unreadable": score >= threshold},
                    prelabel=base.prelabel,
                    derived={
                        **base.derived,
                        "unreadable": score,
                        "ocr_confidence": confidence,
                        "ocr_nonword": nonword,
                        "ocr_boxes": float(n_boxes),
                    },
                    annotator=f"ocr:{config.ocr.engine}",
                    timestamp=base.timestamp,
                    duration_s=base.duration_s,
                    notes=base.notes,
                ),
            )
            written += 1
            if index % 20 == 0:
                rate = index / (time.perf_counter() - started)
                left = (len(todo) - index) / rate if rate else 0
                print(
                    f"  {index}/{len(todo)}  {rate:.2f} стр/с  осталось ~{left/60:.0f} мин",
                    file=sys.stderr,
                    flush=True,
                )

    print(f"\nпроставлено: {written}, пропущено (нечего распознавать): {skipped}")


if __name__ == "__main__":
    main()
