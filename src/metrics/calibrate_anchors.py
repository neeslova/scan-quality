"""Подбор якорей CV-метрик по распределению реального корпуса.

Зачем это нужно. Якоря в `configs/base.yaml` задают шкалу «сырая метрика -> скор
0..1», и абсолютные значения у них не универсальны: Tenengrad зависит от гарнитуры,
кегля и dpi, а `min_margin_frac` — от того, как свёрстан документ. Якоря, подобранные
на одном корпусе, на другом дают вырожденные скоры (все единицы или все нули).

Что делает этот модуль. Считает сырые метрики по выборке корпуса и ставит якоря по
перцентилям: `good` — там, где лежит типичная страница, `bad` — в хвосте. Направление
берётся из текущего конфига: если `good > bad`, метрика «чем меньше, тем хуже».

Чего он НЕ делает. Это не калибровка по разметке. Полученный скор — ранг страницы
внутри корпуса, а не вероятность дефекта: инструмент исходит из того, что типичная
страница скорее хорошая, и проверить это без меток нельзя. Нужен он ровно для того,
чтобы предразметка в С2 показывала разметчику осмысленный порядок подозрительности.
Настоящие пороги считаются по PR-кривым в С6, когда метки уже есть.

Запуск:
    python -m src.metrics.calibrate_anchors --data data/raw/tobacco3482/data --limit 250
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.config import Config, load_config
from src.io.loader import IMAGE_SUFFIXES, load_page
from src.metrics.baseline import compute_raw_metrics, inapplicable_labels

logger = logging.getLogger(__name__)

# Перцентиль «типичной» страницы и перцентиль хвоста.
DEFAULT_GOOD_PCT = 60.0
DEFAULT_BAD_PCT = 5.0


def collect_metrics(
    files: list[Path],
    config: Config,
    on_progress: Optional[callable] = None,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Сырые метрики по выборке. Неприменимые метки исключаются пострранично."""
    values: dict[str, list[float]] = {}
    skipped: dict[str, int] = {}

    for index, path in enumerate(files, 1):
        try:
            page = load_page(
                path,
                target_dpi=config.data.target_dpi,
                dpi_fallback=config.data.dpi_fallback,
                allow_upscale=config.data.allow_upscale,
            )
            raw = compute_raw_metrics(page, config)
        except Exception as exc:  # noqa: BLE001 — один битый файл не должен ронять прогон
            logger.warning("%s пропущен: %s", path.name, exc)
            continue

        # Метрику, которая на этой странице неизмерима, в распределение не берём:
        # иначе битональные сканы утянут шкалу контраста в вырожденный ноль.
        blocked = {
            config.cv.scores[label].metric
            for label in inapplicable_labels(raw, config)
            if label in config.cv.scores
        }
        for key, value in raw.items():
            if key in blocked:
                skipped[key] = skipped.get(key, 0) + 1
                continue
            values.setdefault(key, []).append(value)

        if on_progress is not None:
            on_progress(index, len(files))

    return values, skipped


def suggest_anchors(
    values: dict[str, list[float]],
    config: Config,
    good_pct: float = DEFAULT_GOOD_PCT,
    bad_pct: float = DEFAULT_BAD_PCT,
) -> dict[str, dict[str, float]]:
    """Якоря по перцентилям, направление — из текущего конфига."""
    suggestions: dict[str, dict[str, float]] = {}

    for label, mapping in config.cv.scores.items():
        samples = values.get(mapping.metric)
        if not samples:
            logger.warning("Метрика %s (%s) не собрана — пропуск", mapping.metric, label)
            continue

        array = np.asarray(samples, dtype=np.float64)
        lower_is_worse = mapping.good > mapping.bad
        if lower_is_worse:
            good = float(np.percentile(array, good_pct))
            bad = float(np.percentile(array, bad_pct))
        else:
            good = float(np.percentile(array, 100.0 - good_pct))
            bad = float(np.percentile(array, 100.0 - bad_pct))

        if good == bad:
            logger.warning(
                "Метрика %s (%s) вырождена: перцентили совпали (%.4f) — якоря не меняем",
                mapping.metric,
                label,
                good,
            )
            continue

        suggestions[label] = {
            "metric": mapping.metric,
            "good": round(good, 4),
            "bad": round(bad, 4),
        }
    return suggestions


def _round_for_yaml(value: float) -> float:
    """Мелкие доли нужны точнее, крупные величины — грубее."""
    magnitude = abs(value)
    if magnitude >= 100:
        return round(value, 1)
    if magnitude >= 1:
        return round(value, 3)
    return round(value, 5)


def to_yaml(suggestions: dict[str, dict[str, float]]) -> str:
    block = {
        label: {
            "metric": data["metric"],
            "good": _round_for_yaml(data["good"]),
            "bad": _round_for_yaml(data["bad"]),
        }
        for label, data in suggestions.items()
    }
    return yaml.safe_dump({"scores": block}, allow_unicode=True, sort_keys=False, width=100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подбор якорей CV-метрик по корпусу")
    parser.add_argument("--data", type=Path, required=True, help="папка со сканами")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--corpus", type=Path, action="append", default=None, help="оверлей конфига под корпус"
    )
    parser.add_argument("--limit", type=int, default=250, help="размер случайной выборки")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--good-pct", type=float, default=DEFAULT_GOOD_PCT)
    parser.add_argument("--bad-pct", type=float, default=DEFAULT_BAD_PCT)
    parser.add_argument("--out", type=Path, default=None, help="куда записать YAML-блок")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    files = [p for p in sorted(args.data.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise SystemExit(f"В {args.data} не найдено изображений")

    random.seed(args.seed)
    sample = random.sample(files, min(args.limit, len(files)))
    print(f"корпус: {len(files)} файлов, выборка: {len(sample)}", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  обработано {done}/{total}", file=sys.stderr, flush=True)

    values, skipped = collect_metrics(sample, config, progress)
    suggestions = suggest_anchors(values, config, args.good_pct, args.bad_pct)

    print(f"\n{'метка':16s}{'метрика':24s}{'было':>20s}{'стало':>20s}", file=sys.stderr)
    for label, data in suggestions.items():
        old = config.cv.scores[label]
        was = f"{old.good:g} -> {old.bad:g}"
        now = f"{data['good']:g} -> {data['bad']:g}"
        print(f"{label:16s}{data['metric']:24s}{was:>20s}{now:>20s}", file=sys.stderr)
    for metric, count in sorted(skipped.items()):
        print(f"\n{metric}: {count} страниц исключено как неизмеримые", file=sys.stderr)

    block = to_yaml(suggestions)
    if args.out is not None:
        args.out.write_text(block, encoding="utf-8")
        print(f"\nзаписано в {args.out}", file=sys.stderr)
    print(block)


if __name__ == "__main__":
    main()
