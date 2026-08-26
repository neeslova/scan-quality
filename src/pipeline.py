"""Оркестратор: изображение -> QualityReport.

Готово:
  [A] CV-метрики (С1) — детерминированный baseline, обучения не требует.
Впереди:
  [B] CNN multi-label через onnxruntime (С5/С6), [C] OCR-слой и метка unreadable (С3).
Пока CNN нет, вердикт считается по CV-скорам — это и есть точка отсчёта для записки.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

from src.config import Config, VerdictConfig, load_config
from src.io.loader import LoadedPage, load_page, load_pages
from src.metrics.baseline import analyze_page
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
    elapsed_ms: float,
    with_ocr: bool = False,
) -> QualityReport:
    """Собирает отчёт по уже загруженной странице."""
    raw, scores, not_applicable = analyze_page(page, config)

    defects = [
        DefectScore(label=label, score=score, source="cv") for label, score in scores.items()
    ]

    ocr_result = None
    if with_ocr:
        ocr_result, unreadable = _run_ocr(page, config)
        if unreadable is None:
            # Пустая страница не нечитаема: распознавать нечего, а не плохо.
            not_applicable = sorted({*not_applicable, "unreadable"})
        else:
            scores["unreadable"] = unreadable
            defects.append(DefectScore(label="unreadable", score=unreadable, source="ocr"))

    defects.sort(key=lambda d: d.score, reverse=True)

    name = page.source.name if page.page == 0 else f"{page.source.name}#{page.page + 1}"
    return QualityReport(
        pipeline_version="cv-baseline+ocr" if with_ocr else "cv-baseline",
        image=name,
        width=page.width,
        height=page.height,
        verdict=decide_verdict(scores, config.verdict),
        quality_score=quality_score(scores),
        defects=defects,
        cv_metrics={key: round(value, 4) for key, value in raw.items()},
        not_applicable=not_applicable,
        ocr=ocr_result,
        elapsed_ms=round(elapsed_ms, 2),
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
    report = build_report(loaded, cfg, (time.perf_counter() - started) * 1000, with_ocr)
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
    reports: list[QualityReport] = []
    for loaded in load_pages(
        image_path,
        target_dpi=cfg.data.target_dpi,
        dpi_fallback=cfg.data.dpi_fallback,
        allow_upscale=cfg.data.allow_upscale,
    ):
        started = time.perf_counter()
        reports.append(build_report(loaded, cfg, (time.perf_counter() - started) * 1000))
    return reports
