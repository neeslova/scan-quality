"""Калибровка порогов по PR-кривым.

Зачем это нужно именно так. Скоры приходят из трёх источников с несовместимыми
шкалами: CV-метрика нормирована по перцентилям корпуса, сеть выдаёт вероятность
после сигмоиды, OCR — свою производную величину. Один глобальный порог 0.5 на всех
даёт то, что и дал на замере: у `blur` точность 1.000 при полноте 0.080 — сеть
ранжирует верно, но её средние по патчам не дотягивают до 0.5 никогда.

Поэтому калибруется не порог, а **шкала**. Каждой метке подбираются два якоря,
и `score_from_anchors` отображает её скор в общие 0..1 — ровно та же механика,
которой CV-слой приводит свои метрики к скорам (решение №22), только уровнем выше.
Якоря выбираются так, чтобы глобальные `tau_low` и `tau_high` из раздела 4 плана
попали точно в рабочие точки метки:

    normalized(t_recall) = tau_low     — точка, где полнота достигает целевой
    normalized(t_prec)   = tau_high    — точка, где точность достигает целевой

Отсюда два уравнения на два якоря, решение — в `anchors_from_operating_points`.
Правило вердикта при этом не меняется ни на строку: меняется то, что в него
подаётся.

Асимметрия из раздела 4 живёт в целевой полноте: пропустить плохой скан дороже,
чем отклонить хороший, поэтому по `blur` и `unreadable` цель 0.95, по остальным
ниже. Метка, у которой в выборке слишком мало положительных примеров, не
калибруется вовсе — подгонка по трём страницам это не калибровка.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, load_config
from src.metrics.baseline import score_from_anchors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Calibration:
    """Якоря метки и то, чем за них заплачено."""

    label: str
    good: float
    bad: float
    t_low: float
    t_high: float
    recall_low: float
    precision_low: float
    recall_high: float
    precision_high: float
    support: int


def sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точность и полнота во всех точках разреза. Пороги — сами значения скоров.

    Своя реализация вместо sklearn: scikit-learn стоит только в extra `train`,
    а калибровка обязана считаться там же, где работает приложение.
    """
    order = np.argsort(y_score)[::-1]
    scores = y_score[order]
    truth = y_true[order]

    positives = float(truth.sum())
    hits = np.cumsum(truth)
    seen = np.arange(1, len(truth) + 1)

    precision = hits / seen
    recall = hits / positives if positives else np.zeros_like(hits, dtype=float)
    return scores, precision, recall


def _threshold_for_recall(recall: np.ndarray, target: float) -> Optional[int]:
    """Индекс самого ВЫСОКОГО разреза, на котором полнота ещё не ниже целевой.

    Из всех порогов, ловящих нужную долю дефектов, берём самый придирчивый:
    лишняя мягкость не добавит пойманных дефектов, только ложные срабатывания.
    """
    reached = np.flatnonzero(recall >= target)
    return int(reached[0]) if len(reached) else None  # первый по убыванию скора


def operating_points(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float, confident_recall: float
) -> Optional[tuple[int, int]]:
    """Индексы двух рабочих точек: (бесспорно плохо, подозрительно).

    Обе задаются ПОЛНОТОЙ, и это не косметика. Полнота монотонно не возрастает
    с ростом порога, поэтому порядок двух точек гарантирован по построению.
    Первая версия брала вторую точку по целевой ТОЧНОСТИ — и у хорошо разделимой
    метки та держалась выше цели до самого низа ранжирования: «порог точности»
    оказывался НИЖЕ «порога полноты», срабатывала аварийная ветка, и шкала
    схлопывалась в ступеньку шириной 0.003. Так вышло у `shadow`, `skew`
    и `cropped` на первом же прогоне.

    Смысл прежний: `tau_low` ловит почти все дефекты, `tau_high` — только те,
    в которых источник уверен. Достигнутая точность в обеих точках попадает
    в отчёт: по ней видно, чем за полноту заплачено.
    """
    scores, _, recall = sweep(y_true, y_score)
    if not len(scores) or not y_true.any():
        return None

    low = _threshold_for_recall(recall, target_recall)
    high = _threshold_for_recall(recall, confident_recall)
    if low is None or high is None:
        return None
    return high, low


def anchors_from_operating_points(
    t_low: float, t_high: float, tau_low: float, tau_high: float
) -> tuple[float, float]:
    """Два якоря из двух рабочих точек.

    Решение системы:  (t_low  - good) / (bad - good) = tau_low
                      (t_high - good) / (bad - good) = tau_high

    Точки могут совпасть: у метки, где целевая и «бесспорная» полнота достигаются
    одним и тем же разрезом, шкала не определена. Тогда раздвигаем их на
    минимальный зазор — отображение останется возрастающим, а не поделится на
    ноль. Это честная ступенька, и она видна в отчёте по совпавшим t_low и t_high.
    """
    span = t_high - t_low
    if span <= 0.0:
        span = 1e-3
    scale = span / (tau_high - tau_low)
    good = t_low - tau_low * scale
    return float(good), float(good + scale)


def calibrate_label(
    label: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    config: Config,
) -> Optional[Calibration]:
    """Якоря одной метки или None, если калибровать не на чем."""
    cfg = config.calibrate
    support = int(y_true.sum())
    if support < cfg.min_support:
        logger.warning(
            "%s: положительных всего %d при пороге %d — не калибруем, "
            "подгонка по такой выборке хуже её отсутствия",
            label,
            support,
            cfg.min_support,
        )
        return None

    target_recall = cfg.recall.get(label, cfg.target_recall)
    points = operating_points(y_true, y_score, target_recall, cfg.confident_recall)
    if points is None:
        logger.warning("%s: целевая полнота %.2f недостижима — не калибруем", label, target_recall)
        return None

    scores, precision, recall = sweep(y_true, y_score)
    high, low = points
    # Опорные точки шкалы, а НЕ пороги вердикта. Иначе получается круг: якоря
    # строились бы относительно порога, а порог подбирается по якорям (№45).
    good, bad = anchors_from_operating_points(
        float(scores[low]),
        float(scores[high]),
        cfg.anchor_low,
        cfg.anchor_high,
    )
    return Calibration(
        label=label,
        good=round(good, 4),
        bad=round(bad, 4),
        t_low=round(float(scores[low]), 4),
        t_high=round(float(scores[high]), 4),
        recall_low=round(float(recall[low]), 4),
        precision_low=round(float(precision[low]), 4),
        recall_high=round(float(recall[high]), 4),
        precision_high=round(float(precision[high]), 4),
        support=support,
    )


def format_table(results: list[Calibration], config: Config) -> str:
    lines = [
        " " * 34 + "-- подозрительно --   --- бесспорно ---",
        f"{'метка':16s}{'источник':>9s}{'good':>9s}{'bad':>9s}"
        f"{'t_low':>8s}{'R':>7s}{'P':>7s}"
        f"{'t_high':>9s}{'R':>7s}{'P':>7s}{'n+':>6s}",
    ]
    for item in results:
        lines.append(
            f"{item.label:16s}{config.sources.of(item.label):>9s}"
            f"{item.good:9.3f}{item.bad:9.3f}"
            f"{item.t_low:8.3f}{item.recall_low:7.3f}{item.precision_low:7.3f}"
            f"{item.t_high:9.3f}{item.recall_high:7.3f}{item.precision_high:7.3f}"
            f"{item.support:6d}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class VerdictTradeoff:
    """Во что обходится пара порогов вердикта на странице целиком."""

    tau_low: float
    tau_high: float
    clean_good: int
    clean_bad: int
    clean_total: int
    severe_good: int
    severe_bad: int
    severe_total: int


def verdict_tradeoff(
    page_scores: list[float], defects: list[int], tau_low: float, tau_high: float, severe: int = 3
) -> VerdictTradeoff:
    """Что даёт пара порогов на уровне СТРАНИЦЫ, а не метки.

    Калибровка меток настраивает каждую по отдельности, но вердикт — объединение
    по девяти: `good` требует, чтобы НИ ОДНА не превысила `tau_low`. Даже при
    хорошей точности каждой метки объединение накрывает почти всё, и `good`
    становится редким механически. Это отмечено ещё в С1 и калибровкой меток
    не лечится — здесь считается цена уже по странице.

    Смотреть надо на две колонки сразу. Смягчение порогов спасает чистые
    страницы, но начинает пропускать тяжело битые: на val переход с 0.30/0.60 на
    0.45/0.70 снизил долю отвергнутых чистых с 37% до 3% ценой падения отлова
    страниц с тремя дефектами с 89% до 59%, и одна такая страница получила
    `good`. Раздел 4 говорит, что дороже, поэтому выбор не автоматический.
    """
    tops = np.asarray(page_scores, dtype=float)
    counts = np.asarray(defects, dtype=int)

    clean = tops[counts == 0]
    heavy = tops[counts >= severe]
    return VerdictTradeoff(
        tau_low=tau_low,
        tau_high=tau_high,
        clean_good=int((clean <= tau_low).sum()),
        clean_bad=int((clean > tau_high).sum()),
        clean_total=len(clean),
        severe_good=int((heavy <= tau_low).sum()),
        severe_bad=int((heavy > tau_high).sum()),
        severe_total=len(heavy),
    )


def format_tradeoff(rows: list[VerdictTradeoff]) -> str:
    lines = [
        f"{'tau_low':>8s}{'tau_high':>9s} | "
        f"{'чистые good':>12s}{'чистые bad':>12s} | {'3+ good':>9s}{'3+ bad':>9s}",
        "-" * 66,
    ]
    for row in rows:
        lines.append(
            f"{row.tau_low:8.2f}{row.tau_high:9.2f} | "
            f"{row.clean_good:6d}/{row.clean_total:<5d}{row.clean_bad:6d}/{row.clean_total:<5d} | "
            f"{row.severe_good:4d}/{row.severe_total:<4d}{row.severe_bad:4d}/{row.severe_total:<4d}"
        )
    lines.append("")
    lines.append("`3+ good` обязан оставаться нулём: пропустить тяжело битую страницу —")
    lines.append("самая дорогая ошибка системы (раздел 4 плана).")
    return "\n".join(lines)


def to_overlay(results: list[Calibration]) -> dict:
    """Оверлей поверх base.yaml: только то, что отличается (решение №24)."""
    return {
        "verdict": {
            "anchors": {item.label: {"good": item.good, "bad": item.bad} for item in results}
        }
    }


@dataclass(frozen=True)
class PageScores:
    """Одна страница: что выдал пайплайн и что стоит в разметке."""

    scores: dict[str, float]
    labels: dict[str, bool]

    def defect_count(self, exclude: Sequence[str] = ()) -> int:
        """Сколько дефектов у страницы по разметке. `unreadable` обычно исключают:
        она выводится из OCR и в правиле вердикта имеет свой порог."""
        skip = set(exclude)
        return sum(1 for label, on in self.labels.items() if on and label not in skip)

    def top(self) -> float:
        """Худшая метка страницы: правило вердикта смотрит именно на неё."""
        return max(self.scores.values()) if self.scores else 0.0


def collect_scores(samples, config: Config, with_ocr: bool) -> list[PageScores]:
    """Прогон страниц целиком через пайплайн.

    Именно через пайплайн, а не мимо него: калибруются те самые числа, которые
    потом попадут в отчёт, со всеми источниками и агрегацией.
    """
    from src.io.loader import load_page
    from src.models.infer import shared_predictor
    from src.pipeline import build_report

    predictor = shared_predictor(config)
    pages: list[PageScores] = []

    for index, sample in enumerate(samples, 1):
        page = load_page(
            sample.path,
            target_dpi=config.data.target_dpi,
            dpi_fallback=config.data.dpi_fallback,
            allow_upscale=config.data.allow_upscale,
        )
        report = build_report(page, config, time.perf_counter(), with_ocr, predictor)
        pages.append(
            PageScores(
                scores=report.scores(),
                labels={label: bool(sample.labels.get(label)) for label in config.labels},
            )
        )
        if index % 20 == 0:
            logger.info("%d/%d", index, len(samples))

    return pages


def by_label(pages: list[PageScores], labels: Sequence[str]) -> tuple[dict, dict]:
    """Развернуть постранично собранное по меткам: (истина, скоры).

    Метка попадает в выборку только со страниц, где источник её ВЫДАЛ. Длины
    поэтому разные, и это правильно: метку нельзя калибровать по страницам,
    на которых её не измеряли (битональный скан, строки не найдены).
    """
    truth: dict[str, list[float]] = {label: [] for label in labels}
    scored: dict[str, list[float]] = {label: [] for label in labels}
    for page in pages:
        for label in labels:
            if label not in page.scores:
                continue
            truth[label].append(1.0 if page.labels.get(label) else 0.0)
            scored[label].append(page.scores[label])
    return truth, scored


def main() -> None:
    import yaml

    from src.data.dataset import collect_real, load_split

    parser = argparse.ArgumentParser(description="Калибровка порогов по PR-кривым")
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--part", default="val", choices=("train", "val"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--out", type=Path, default=Path("configs/thresholds.yaml"))
    parser.add_argument("--with-ocr", action="store_true", help="считать и unreadable")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    _, images = load_split(args.splits, args.part)
    samples = collect_real(args.labels, args.data, images)
    pages = collect_scores(samples, config, args.with_ocr)
    truth, scored = by_label(pages, config.labels)

    results = []
    for label in config.labels:
        if not scored[label]:
            logger.warning("%s: ни одной страницы со скором — источник не работал", label)
            continue
        calibration = calibrate_label(
            label, np.asarray(truth[label]), np.asarray(scored[label]), config
        )
        if calibration is not None:
            results.append(calibration)

    print(f"\n{args.part}: {len(samples)} страниц")
    print(format_table(results, config))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(to_overlay(results), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nякоря записаны в {args.out} — накладывать флагом --corpus")

    # Второй уровень: пороги самого вердикта. Калибровка меток настраивает
    # каждую по отдельности, а `good` требует, чтобы НИ ОДНА не превысила порог.
    # Цену этого объединения видно только на странице целиком.
    anchors = to_overlay(results)["verdict"]["anchors"]
    tops, counts = [], []
    for page in pages:
        fixed = [
            (
                score_from_anchors(value, anchors[label]["good"], anchors[label]["bad"])
                if label in anchors
                else value
            )
            for label, value in page.scores.items()
        ]
        tops.append(max(fixed) if fixed else 0.0)
        counts.append(page.defect_count(exclude=config.ocr_derived))

    grid = [(0.30, 0.60), (0.40, 0.65), (0.45, 0.70), (0.50, 0.75), (0.60, 0.80)]
    print(
        f"\nпороги вердикта на странице целиком (сейчас в конфиге "
        f"{config.verdict.tau_low:.2f}/{config.verdict.tau_high:.2f}):"
    )
    print(format_tradeoff([verdict_tradeoff(tops, counts, lo, hi) for lo, hi in grid]))


if __name__ == "__main__":
    main()
