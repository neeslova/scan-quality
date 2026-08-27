"""Сколько стоит страница на CPU. Пункт DoD С6 и метрика защиты в С8.

Меряется медиана, а не среднее: первый прогон включает подъём сессии
onnxruntime и прогрев кэшей, и одно такое измерение утягивает среднее заметно
сильнее, чем медиану.

Слои разделены намеренно. «Полторы секунды на страницу» ничего не говорит, если
неизвестно, что в них CV-метрики занимают десятую часть, а всё остальное — сеть:
именно из такого разделения видно, за что платим и что имеет смысл ускорять.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import Config, load_config
from src.io.loader import load_page

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Timing:
    """Время одного слоя в миллисекундах."""

    name: str
    samples: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def low(self) -> float:
        return min(self.samples)

    @property
    def high(self) -> float:
        return max(self.samples)


def time_each(name: str, call: Callable[[object], object], items: list) -> Timing:
    """Время на КАЖДОМ элементе отдельно, а не на пачке целиком.

    Иначе медиана, минимум и максимум — одно и то же число, а разброс между
    страницами как раз и есть то, что интересно: страницы разного размера дают
    разное число патчей, и сеть на них стоит по-разному.

    Первый вызов прогревочный и в выборку не идёт: в него попадает подъём сессии
    onnxruntime и прогрев кэшей.
    """
    call(items[0])
    samples = []
    for item in items:
        started = time.perf_counter()
        call(item)
        samples.append((time.perf_counter() - started) * 1000.0)
    return Timing(name=name, samples=samples)


def format_table(timings: list[Timing], pages: int) -> str:
    lines = [f"{'слой':22s}{'медиана':>10s}{'мин':>9s}{'макс':>9s}", "-" * 50]
    for timing in timings:
        lines.append(f"{timing.name:22s}{timing.median:9.0f}м{timing.low:8.0f}м{timing.high:8.0f}м")
    lines.append("-" * 50)
    lines.append(f"страниц в замере: {pages}")
    return "\n".join(lines)


def benchmark(paths: list[Path], config: Config, with_ocr: bool = False) -> list[Timing]:
    from src.metrics.baseline import analyze_page
    from src.models.infer import shared_predictor
    from src.pipeline import build_report

    predictor = shared_predictor(config)
    if predictor is None:
        logger.warning("Модели нет — время сети не мерится")

    pages = [
        load_page(
            path,
            target_dpi=config.data.target_dpi,
            dpi_fallback=config.data.dpi_fallback,
            allow_upscale=config.data.allow_upscale,
        )
        for path in paths
    ]

    def load(path):
        return load_page(
            path,
            target_dpi=config.data.target_dpi,
            dpi_fallback=config.data.dpi_fallback,
            allow_upscale=config.data.allow_upscale,
        )

    timings = [
        time_each("загрузка страницы", load, paths),
        time_each("CV-метрики", lambda page: analyze_page(page, config), pages),
    ]
    if predictor is not None:
        timings.append(
            time_each("сеть (onnxruntime)", lambda page: predictor.predict(page.gray), pages)
        )
    timings.append(
        time_each(
            "весь отчёт",
            lambda page: build_report(page, config, time.perf_counter(), with_ocr, predictor),
            pages,
        )
    )
    return timings


def main() -> None:
    from src.data.dataset import collect_real, load_split

    parser = argparse.ArgumentParser(description="Время обработки страницы на CPU")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--part", default="val", choices=("train", "val"))
    parser.add_argument("--pages", type=int, default=12, help="сколько страниц взять")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--with-ocr", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    _, images = load_split(args.splits, args.part)
    samples = collect_real(args.labels, args.data, images)[: args.pages]
    if not samples:
        raise SystemExit("Страниц не нашлось")

    timings = benchmark([s.path for s in samples], config, args.with_ocr)
    print(f"\nCPU, {config.model.backbone}, патч {config.data.patch_size}, на страницу:")
    print(format_table(timings, len(samples)))


if __name__ == "__main__":
    main()
