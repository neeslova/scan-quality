"""Оркестратор: изображение -> QualityReport.

Каждая метка берётся у того источника, который меряет её лучше — это результат
замера на реальных страницах, а не замысел (решения №39-40): CV-метрики дали
macro-AP 0.665 против 0.526 у сети, но ошибаются они на непересекающихся метках.
Распределение живёт в конфиге, в секции `sources`:

  [A] CV-метрики (С1) — blur, glare, shadow, skew, cropped, low_resolution, streaks
  [B] CNN через onnxruntime (С5/С6) — low_contrast и noise: единственные две,
      где CV-слой не выдаёт значения вовсе, потому что корпус битональный
  [C] OCR-слой (С3) — unreadable, метка по построению выводится из распознавания

Ни один источник не обязателен: нет модели или OCR — соответствующие метки
уходят в `not_applicable`, а отчёт собирается по остальным.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

from src.config import Config, VerdictConfig, load_config
from src.io.loader import LoadedPage, load_page, load_pages
from src.metrics.baseline import analyze_page, score_from_anchors
from src.models.infer import PatchPredictor, shared_predictor
from src.ocr.engine import read_page, shared_engine
from src.ocr.readability import analyze_words, unreadable_score
from src.schema import DefectScore, QualityReport, Verdict

logger = logging.getLogger(__name__)


def decide_verdict(scores: Mapping[str, float], cfg: VerdictConfig) -> Verdict:
    """Правило вердикта из PLAN.md §4.

    Асимметрия намеренная: пропустить плохой скан дороже, чем отклонить хороший,
    поэтому `unreadable` имеет собственный, более низкий порог.
    """
    if scores.get("unreadable", 0.0) > cfg.tau_unreadable:
        return "bad"
    if any(score > cfg.tau_high for score in scores.values()):
        return "bad"
    if any(score > cfg.tau_low for score in scores.values()):
        return "acceptable"
    return "good"


def apply_anchors(scores: Mapping[str, float], config: Config) -> dict[str, float]:
    """Скоры источников -> общая шкала.

    Три источника меряют в трёх разных единицах, и одно правило вердикта на них
    работать не может: на замере у `blur` при пороге 0.5 выходила точность 1.000
    при полноте 0.080 — сеть ранжирует верно, но её средние по патчам до 0.5 не
    доходят. Якоря (С6, `calibrate.py`) растягивают шкалу каждой метки так, чтобы
    общие `tau_low` и `tau_high` попали в её рабочие точки.

    Метка без якорей идёт как есть: это честнее, чем притворяться откалиброванной.
    """
    anchors = config.verdict.anchors
    return {
        label: (
            round(score_from_anchors(value, anchors[label].good, anchors[label].bad), 4)
            if label in anchors
            else value
        )
        for label, value in scores.items()
    }


def quality_score(scores: Mapping[str, float]) -> float:
    """Сводный балл 0..1: 1 — идеальный скан. Пока просто «один минус худший дефект»."""
    if not scores:
        return 1.0
    return round(1.0 - max(scores.values()), 4)


def _run_ocr(page: LoadedPage, config: Config):
    """OCR-слой: (результат, скор unreadable или None). Ошибки не роняют отчёт."""
    engine = shared_engine(config.ocr.engine, config.ocr.languages)
    words = read_page(page.gray, engine, config.ocr.work_side)
    result = analyze_words(words, config, page.width * page.height, engine.name)
    return result, unreadable_score(result, config)


def build_report(
    page: LoadedPage,
    config: Config,
    started: float,
    with_ocr: bool = False,
    predictor: Optional[PatchPredictor] = None,
) -> QualityReport:
    """Собирает отчёт по уже загруженной странице.

    `started` — отметка `time.perf_counter()`, взятая ДО загрузки страницы.
    Раньше сюда передавали уже посчитанную длительность, и она успевала учесть
    только загрузку: отчёт по странице с сетью показывал 42 мс при 1235 мс на
    одном инференсе. В С6 время инференса — отдельная метрика, и мерить его
    мимо самого анализа бессмысленно.
    """
    raw, cv_scores, cv_skipped = analyze_page(page, config)

    # CV-слой умеет считать больше меток, чем ему назначено: `low_contrast`
    # и `noise` он тоже выдаёт, но проигрывает по ним сети, поэтому берём только
    # своё. Сырые метрики при этом остаются в отчёте все — они для разбора.
    raw_scores: dict[str, float] = {}
    origin: dict[str, str] = {}
    hot: dict[str, list[int]] = {}
    for label, score in cv_scores.items():
        if config.sources.of(label) != "cv":
            continue
        raw_scores[label] = score
        origin[label] = "cv"

    # Неприменимость чужой метки нас не касается: за неё отвечает другой источник.
    missing = {label for label in cv_skipped if config.sources.of(label) == "cv"}

    if config.sources.cnn:
        if predictor is None:
            missing.update(config.sources.cnn)
        else:
            # Один прогон на страницу: и скоры, и горячие патчи — из него.
            prediction = predictor.predict(page.gray)
            predicted = prediction.scores()
            for label in config.sources.cnn:
                raw_scores[label] = predicted[label]
                origin[label] = "cnn"
                hot[label] = prediction.top_patches(label)

    ocr_result = None
    if with_ocr:
        ocr_result, unreadable = _run_ocr(page, config)
        if unreadable is None:
            # Пустая страница не нечитаема: распознавать нечего, а не плохо.
            missing.update(config.sources.ocr)
        else:
            raw_scores["unreadable"] = unreadable
            origin["unreadable"] = "ocr"
    else:
        missing.update(config.sources.ocr)

    # Вердикт считается по приведённой шкале, а не по сырым числам источников.
    scores = apply_anchors(raw_scores, config)
    defects = [
        DefectScore(
            label=label,
            score=value,
            raw=raw_scores[label],
            source=origin[label],
            top_patches=hot.get(label, []),
        )
        for label, value in scores.items()
    ]

    not_applicable = sorted(missing)
    defects.sort(key=lambda d: d.score, reverse=True)

    name = page.source.name if page.page == 0 else f"{page.source.name}#{page.page + 1}"
    return QualityReport(
        pipeline_version="+".join(
            part
            for part, on in (
                ("cv", True),
                ("cnn", predictor is not None and bool(config.sources.cnn)),
                ("ocr", with_ocr),
            )
            if on
        ),
        image=name,
        width=page.width,
        height=page.height,
        verdict=decide_verdict(scores, config.verdict),
        quality_score=quality_score(scores),
        defects=defects,
        cv_metrics={key: round(value, 4) for key, value in raw.items()},
        not_applicable=not_applicable,
        ocr=ocr_result,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def analyze(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
    page: int = 0,
    with_ocr: bool = False,
) -> QualityReport:
    """Одна страница файла -> отчёт. OCR по флагу: он в разы медленнее метрик."""
    started = time.perf_counter()
    cfg = config or load_config()
    loaded = load_page(
        image_path,
        target_dpi=cfg.data.target_dpi,
        dpi_fallback=cfg.data.dpi_fallback,
        page=page,
        allow_upscale=cfg.data.allow_upscale,
    )
    report = build_report(loaded, cfg, started, with_ocr, shared_predictor(cfg))
    logger.info(
        "%s: %s (score %.2f, %.0f мс)",
        report.image,
        report.verdict,
        report.quality_score,
        report.elapsed_ms,
    )
    return report


def analyze_all_pages(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
) -> list[QualityReport]:
    """Все страницы файла. Для картинки — один отчёт, для PDF — по одному на страницу."""
    cfg = config or load_config()
    predictor = shared_predictor(cfg)
    reports: list[QualityReport] = []
    for loaded in load_pages(
        image_path,
        target_dpi=cfg.data.target_dpi,
        dpi_fallback=cfg.data.dpi_fallback,
        allow_upscale=cfg.data.allow_upscale,
    ):
        started = time.perf_counter()
        reports.append(build_report(loaded, cfg, started, predictor=predictor))
    return reports
