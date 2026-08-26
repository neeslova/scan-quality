"""Черновая разметка по CV-метрикам и отбор очереди на ручную разметку.

Две отдельные задачи, потому что у них разная цена:

1. Прогон CV по корпусу — около секунды на страницу, тысячи страниц. Пишется в
   JSONL по мере обработки и умеет продолжаться с места обрыва.
2. Отбор очереди — мгновенный, читает готовые записи.

Почему очередь смешанная. Чистая случайная выборка даёт правдивые частоты меток,
но на 800 страниц придётся два-три примера редкого дефекта — модель его не выучит.
Отбор только по подозрительности даёт примеры, но учит модель ровно тем дефектам,
которые CV и так умеет ловить, и завышает оценку пользы CNN. Берём половину
случайно, половину — верхушкой по КАЖДОЙ метке отдельно, чтобы редкие не потерялись.

Запуск:
    python -m src.labeling.prelabel --data data/raw/tobacco3482/data \\
        --corpus-name tobacco3482 --out data/labeled/prelabels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

from src.config import Config, load_config
from src.data.split import document_id
from src.io.loader import IMAGE_SUFFIXES
from src.pipeline import analyze
from src.schema import PrelabelRecord

logger = logging.getLogger(__name__)

_WORKER_STATE: dict[str, object] = {}


def list_images(root: Union[str, Path]) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _init_worker(config_path: Optional[str], overlays: Optional[list[str]]) -> None:
    import cv2

    # OpenCV сам многопоточен; вместе с пулом процессов это даёт переподписку
    # ядер и замедляет всё в разы.
    cv2.setNumThreads(1)
    _WORKER_STATE["config"] = load_config(config_path, overlays)


def _score_one(args: tuple[str, str, str]) -> Optional[str]:
    """Одна страница -> строка JSONL. None, если файл не читается."""
    path_str, root_str, corpus = args
    config: Config = _WORKER_STATE["config"]  # type: ignore[assignment]
    path = Path(path_str)
    try:
        report = analyze(path, config)
    except Exception as exc:  # noqa: BLE001 — битый файл не должен ронять прогон
        logger.warning("%s пропущен: %s", path.name, exc)
        return None

    scores = report.scores()
    threshold = config.labeling.suggest_threshold
    manual = [label for label in config.manual_labels if label in scores]

    record = PrelabelRecord(
        image=str(path.relative_to(root_str)).replace("\\", "/"),
        document=document_id(path, config.split.document_id),
        corpus=corpus,
        scores={label: scores[label] for label in manual},
        suggested={label: scores[label] >= threshold for label in manual},
        not_applicable=report.not_applicable,
        verdict=report.verdict,
        quality_score=report.quality_score,
        width=report.width,
        height=report.height,
    )
    return record.model_dump_json()


def read_prelabels(path: Union[str, Path]) -> list[PrelabelRecord]:
    records: list[PrelabelRecord] = []
    target = Path(path)
    if not target.is_file():
        return records
    with target.open("r", encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(PrelabelRecord.model_validate_json(line))
            except ValueError as exc:
                logger.warning("%s:%d — строка пропущена: %s", target, number, exc)
    return records


def build_prelabels(
    root: Path,
    out_path: Path,
    corpus: str,
    config_path: Optional[Path],
    overlays: Optional[list[Path]],
    limit: Optional[int],
    seed: int,
    workers: int,
) -> int:
    """Считает CV-метрики по корпусу. Продолжает с места обрыва."""
    files = list_images(root)
    if not files:
        raise SystemExit(f"В {root} не найдено изображений")

    if limit is not None and limit < len(files):
        random.Random(seed).shuffle(files)
        files = sorted(files[:limit])

    done = {r.image for r in read_prelabels(out_path)}
    todo = [f for f in files if str(f.relative_to(root)).replace("\\", "/") not in done]
    print(
        f"корпус: {len(files)} страниц, готово {len(done)}, осталось {len(todo)}", file=sys.stderr
    )
    if not todo:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [(str(f), str(root), corpus) for f in todo]
    written = 0
    started = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(config_path) if config_path else None,
                [str(p) for p in overlays] if overlays else None,
            ),
        ) as pool:
            for index, line in enumerate(pool.map(_score_one, payload, chunksize=4), 1):
                if line is None:
                    continue
                fh.write(line + "\n")
                written += 1
                # Пишем сразу: прогон длинный, обрыв не должен стоить всей работы.
                if index % 25 == 0:
                    fh.flush()
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed
                    left = (len(todo) - index) / rate if rate else 0
                    print(
                        f"  {index}/{len(todo)}  {rate:.1f} стр/с  осталось ~{left/60:.0f} мин",
                        file=sys.stderr,
                        flush=True,
                    )
    return written


def select_queue(records: list[PrelabelRecord], config: Config) -> list[PrelabelRecord]:
    """Очередь на ручную разметку: половина случайно, половина по подозрительности.

    Вторая половина набирается по каждой метке отдельно, а не общей верхушкой:
    иначе туда попали бы только страницы с самым частым дефектом, а редкие метки
    (`streaks`, `cropped`) не набрали бы примеров вовсе.
    """
    if not records:
        return []

    total = min(config.labeling.sample_total, len(records))
    n_random = int(total * config.labeling.random_share)

    rng = random.Random(config.labeling.sample_seed)
    pool = sorted(records, key=lambda r: r.image)
    rng.shuffle(pool)

    chosen: dict[str, PrelabelRecord] = {}
    for record in pool[:n_random]:
        chosen[record.image] = record

    rest = [r for r in pool if r.image not in chosen]
    labels = [label for label in config.manual_labels if label in config.cv.scores]
    per_label = {
        label: sorted(rest, key=lambda r: r.scores.get(label, 0.0), reverse=True)
        for label in labels
    }
    cursor = dict.fromkeys(labels, 0)

    # По кругу берём лучшего кандидата каждой метки — так доля меток выравнивается.
    while len(chosen) < total and labels:
        progressed = False
        for label in labels:
            if len(chosen) >= total:
                break
            queue = per_label[label]
            while cursor[label] < len(queue):
                candidate = queue[cursor[label]]
                cursor[label] += 1
                if candidate.image not in chosen:
                    chosen[candidate.image] = candidate
                    progressed = True
                    break
        if not progressed:
            break

    # Разметчику показываем в перемешанном виде: подряд идущие «все плохие»
    # сбивают калибровку глаза.
    queue = list(chosen.values())
    rng.shuffle(queue)
    return queue


def write_queue(queue: list[PrelabelRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([r.model_dump(mode="json") for r in queue], fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Черновая разметка по CV-метрикам")
    parser.add_argument("--data", type=Path, required=True, help="папка корпуса")
    parser.add_argument("--corpus-name", default=None, help="имя корпуса в записях")
    parser.add_argument("--out", type=Path, required=True, help="prelabels.jsonl")
    parser.add_argument("--queue", type=Path, default=None, help="куда записать очередь разметки")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--corpus", type=Path, action="append", default=None, help="оверлей конфига"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="сколько страниц корпуса обсчитать (по умолчанию вдвое больше цели разметки)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--queue-only", action="store_true", help="не считать, только отобрать")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)
    corpus = args.corpus_name or args.data.parent.name

    limit = args.limit if args.limit is not None else config.labeling.sample_total * 2
    if not args.queue_only:
        written = build_prelabels(
            root=args.data,
            out_path=args.out,
            corpus=corpus,
            config_path=args.config,
            overlays=args.corpus,
            limit=limit,
            seed=config.labeling.sample_seed,
            workers=args.workers,
        )
        print(f"записано новых: {written}", file=sys.stderr)

    records = read_prelabels(args.out)
    queue = select_queue(records, config)
    if args.queue is not None:
        write_queue(queue, args.queue)
        print(f"очередь: {len(queue)} страниц -> {args.queue}", file=sys.stderr)

    print(f"\nвсего в prelabels: {len(records)}, в очереди: {len(queue)}")
    print(f"{'метка':16s}{'в очереди':>12s}{'во всём наборе':>16s}")
    for label in config.manual_labels:
        if label not in config.cv.scores:
            continue
        threshold = config.labeling.suggest_threshold
        in_queue = sum(1 for r in queue if r.scores.get(label, 0.0) >= threshold)
        in_all = sum(1 for r in records if r.scores.get(label, 0.0) >= threshold)
        share = in_queue / len(queue) if queue else 0.0
        print(f"{label:16s}{in_queue:8d} ({share:4.0%}){in_all:16d}")


if __name__ == "__main__":
    main()
