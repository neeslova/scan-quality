"""Пакетная обработка папки: сканы -> CSV и JSONL.

DoD спринта С7: папка из 50 сканов обрабатывается одной командой и без сети.
Сети здесь нет ни в каком виде — ни модель, ни OCR наружу не ходят, вся работа
идёт на локальных файлах. Это требование раздела 2 плана, а не удобство:
автоматическая проверка потока сканов не может зависеть от доступности сервиса.

Многостраничный PDF даёт по строке на страницу: вердикт ставится странице,
а не файлу. Вопрос про вердикт документу целиком открыт (раздел 14) и решается
не здесь.

Пустая ячейка в CSV означает «не измерено», а не ноль. Метка, которую источник
на этой странице не выдал, обязана отличаться от метки со скором 0.0 — иначе
непроверенный скан выглядит как чистый.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.io.loader import IMAGE_SUFFIXES
from src.schema import QualityReport

logger = logging.getLogger(__name__)

SUFFIXES = {*IMAGE_SUFFIXES, ".pdf"}

_WORKER_STATE: dict[str, object] = {}


def find_scans(root: Path) -> list[Path]:
    """Все поддерживаемые файлы под папкой. Одиночный файл тоже годится."""
    if root.is_file():
        return [root] if root.suffix.lower() in SUFFIXES else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)


def _init_worker(config_path: Optional[str], overlays: Optional[list[str]]) -> None:
    import cv2

    # OpenCV и onnxruntime многопоточны сами по себе; вместе с пулом процессов
    # это даёт переподписку ядер и замедляет всё в разы. Переменную ставим до
    # первого создания сессии: предсказатель поднимается лениво.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    _WORKER_STATE["config"] = load_config(config_path, overlays)


def _analyze_one(args: tuple[str, bool]) -> list[str]:
    """Один файл -> строки JSON по странице. Ошибка не роняет весь прогон."""
    path_str, with_ocr = args
    config: Config = _WORKER_STATE["config"]  # type: ignore[assignment]

    from src.pipeline import analyze_all_pages

    try:
        reports = analyze_all_pages(path_str, config, with_ocr=with_ocr)
    except Exception as error:  # noqa: BLE001 - битый файл не повод бросать папку
        logger.warning("%s: пропущен (%s)", path_str, error)
        return []
    return [report.to_json(indent=None) for report in reports]


def csv_header(config: Config) -> list[str]:
    return [
        "image",
        "verdict",
        "quality_score",
        *config.labels,
        "not_applicable",
        "pipeline_version",
        "width",
        "height",
        "elapsed_ms",
    ]


def csv_row(report: QualityReport, config: Config) -> list[object]:
    scores = report.scores()
    return [
        report.image,
        report.verdict,
        report.quality_score,
        # Пусто, а не ноль: метку на этой странице не измерили.
        *[scores.get(label, "") for label in config.labels],
        ";".join(report.not_applicable),
        report.pipeline_version,
        report.width,
        report.height,
        report.elapsed_ms,
    ]


def process(
    paths: list[Path],
    config: Config,
    workers: int,
    with_ocr: bool,
    config_path: Optional[Path],
    overlays: Optional[list[Path]],
):
    """Отчёты по всем файлам. Пул процессов, потому что работа упирается в CPU."""
    payload = [(str(path), with_ocr) for path in paths]

    if workers <= 1:
        _init_worker(str(config_path) if config_path else None, None)
        for index, item in enumerate(payload, 1):
            for line in _analyze_one(item):
                yield QualityReport.model_validate_json(line)
            if index % 10 == 0:
                logger.info("%d/%d", index, len(payload))
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            str(config_path) if config_path else None,
            [str(p) for p in overlays] if overlays else None,
        ),
    ) as pool:
        for index, lines in enumerate(pool.map(_analyze_one, payload, chunksize=2), 1):
            for line in lines:
                yield QualityReport.model_validate_json(line)
            if index % 10 == 0:
                logger.info("%d/%d", index, len(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетная оценка качества сканов")
    parser.add_argument("--input", type=Path, required=True, help="папка со сканами или файл")
    parser.add_argument("--csv", type=Path, default=None, help="таблица по странице")
    parser.add_argument("--jsonl", type=Path, default=None, help="полные отчёты, по строке")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--workers", type=int, default=1, help="процессов; 1 — без пула")
    parser.add_argument("--with-ocr", action="store_true", help="считать unreadable (медленно)")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N файлов")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.csv is None and args.jsonl is None:
        raise SystemExit("Нужен хотя бы один выход: --csv или --jsonl")

    config = load_config(args.config, args.corpus)
    paths = find_scans(args.input)[: args.limit]
    if not paths:
        raise SystemExit(f"В {args.input} не нашлось файлов {sorted(SUFFIXES)}")

    logger.info("файлов: %d, воркеров: %d", len(paths), args.workers)
    started = time.perf_counter()

    csv_file = args.csv.open("w", newline="", encoding="utf-8-sig") if args.csv else None
    jsonl_file = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    writer = None
    if csv_file is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(csv_header(config))

    counts: dict[str, int] = {"good": 0, "acceptable": 0, "bad": 0}
    pages = 0
    try:
        for report in process(paths, config, args.workers, args.with_ocr, args.config, args.corpus):
            pages += 1
            counts[report.verdict] = counts.get(report.verdict, 0) + 1
            if writer is not None:
                writer.writerow(csv_row(report, config))
            if jsonl_file is not None:
                jsonl_file.write(report.to_json(indent=None) + "\n")
    finally:
        for handle in (csv_file, jsonl_file):
            if handle is not None:
                handle.close()

    elapsed = time.perf_counter() - started
    print(f"\nстраниц: {pages} из {len(paths)} файлов, {elapsed:.1f} с", file=sys.stderr)
    print(
        "  " + ", ".join(f"{name} {counts.get(name, 0)}" for name in ("good", "acceptable", "bad")),
        file=sys.stderr,
    )
    for path in (args.csv, args.jsonl):
        if path is not None:
            print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
