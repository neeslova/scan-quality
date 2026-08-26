"""CV-baseline: сборка сырых метрик по странице и перевод их в скоры по меткам.

Это опорная точка для записки — с ней сравнивается CNN. Обучения здесь нет:
только детерминированные измерения и линейные пороги из конфига.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from src.config import Config
from src.io.loader import LoadedPage
from src.io.patches import grid, select_informative
from src.metrics import blur, contrast, geometry, glare, noise, shadow, streaks, text_scale

logger = logging.getLogger(__name__)


def score_from_anchors(value: float, good: float, bad: float) -> float:
    """0 в точке good, 1 в точке bad, линейно между, зажато в [0, 1].

    Направление выводится из якорей: для резкости good > bad (чем меньше, тем хуже),
    для шума good < bad. Одна функция на все метрики — меньше мест для ошибки знака.
    """
    span = bad - good
    if span == 0.0:
        raise ValueError("good и bad не должны совпадать")
    return float(min(1.0, max(0.0, (value - good) / span)))


def _source_scale(page: LoadedPage) -> float:
    """Во сколько раз исходный файл был мельче того, что мы анализируем."""
    original_width = page.original_size[0]
    if original_width <= 0 or page.width <= 0:
        return 1.0
    return original_width / page.width


def compute_raw_metrics(page: LoadedPage, config: Config) -> dict[str, float]:
    """Все сырые CV-метрики страницы. Ключи совпадают с cv.scores[*].metric."""
    cfg = config.cv
    gray = page.gray

    # Резкость и шум меряем по патчам с текстом: на пустом поле бумаги краёв нет,
    # и любая метрика резкости покажет «размыто».
    boxes = grid(
        page.width,
        page.height,
        config.data.patch_size,
        config.data.grid.rows,
        config.data.grid.cols,
    )
    informative = select_informative(gray, boxes, cfg.min_ink_frac)
    crops = [box.crop(gray) for box in informative]

    tenengrad_values = [blur.tenengrad(c) for c in crops]
    laplacian_values = [blur.laplacian_variance(c) for c in crops]
    noise_values = [noise.noise_sigma(c) for c in crops]

    # Перекос ищем один раз и им же выравниваем страницу под замер высоты строки:
    # на завале строки сливаются и высота уезжает вверх. Метрики резкости считаем
    # ДО поворота — интерполяция сама по себе замыливает края.
    skew_deg = geometry.estimate_skew(
        gray,
        max_angle=cfg.skew_max_angle,
        coarse_step=cfg.skew_coarse_step,
        fine_step=cfg.skew_fine_step,
        work_height=cfg.skew_work_height,
        block_frac=cfg.ink_block_frac,
        offset=cfg.ink_offset,
    )
    deskewed = geometry.rotate(gray, skew_deg) if abs(skew_deg) >= cfg.skew_coarse_step else gray

    glare_stats = glare.glare_stats(
        gray,
        threshold=cfg.glare_threshold,
        min_cluster_frac=cfg.glare_min_cluster_frac,
        flat_window_frac=cfg.glare_flat_window_frac,
        flat_std=cfg.glare_flat_std,
        min_excess=cfg.glare_min_excess,
    )
    shadow_stats = shadow.shadow_stats(
        gray,
        rel_threshold=cfg.shadow_rel_threshold,
        window_frac=cfg.shadow_background_frac,
    )
    streak_stats = streaks.streak_stats(gray, smooth_frac=cfg.streak_smooth_frac)
    scale = text_scale.estimate_text_scale(
        deskewed,
        min_row_ink_frac=cfg.line_min_row_ink_frac,
        block_frac=cfg.ink_block_frac,
        offset=cfg.ink_offset,
    )
    margins = geometry.margin_fractions(gray, block_frac=cfg.ink_block_frac, offset=cfg.ink_offset)

    tenengrad_mean = float(np.mean(tenengrad_values)) if tenengrad_values else 0.0
    gap = contrast.ink_paper_gap(gray)

    return {
        # резкость
        "tenengrad": tenengrad_mean,
        # Нормировка на квадрат контраста: градиент пропорционален перепаду яркости,
        # поэтому без деления блёклая печать неотличима от расфокуса.
        "tenengrad_norm": tenengrad_mean / max(gap, 1.0) ** 2,
        "laplacian_var": float(np.mean(laplacian_values)) if laplacian_values else 0.0,
        # шум
        "noise_sigma": float(np.mean(noise_values)) if noise_values else 0.0,
        "high_freq_energy": noise.high_freq_energy(crops[0]) if crops else 0.0,
        # контраст
        "rms_contrast": contrast.rms_contrast(gray),
        "dynamic_range": contrast.dynamic_range(gray),
        "ink_paper_gap": gap,
        # пересвет
        "glare_bright_frac": glare_stats.bright_frac,
        "glare_cluster_frac": glare_stats.cluster_frac,
        "glare_clusters": float(glare_stats.n_clusters),
        # тень
        "shadow_frac": shadow_stats.dark_frac,
        "illumination_range": shadow_stats.illumination_range,
        # геометрия
        "skew_deg": skew_deg,
        "skew_abs_deg": abs(skew_deg),
        "margin_left": margins.left,
        "margin_right": margins.right,
        "margin_top": margins.top,
        "margin_bottom": margins.bottom,
        "min_margin_frac": margins.minimum,
        # масштаб текста
        "line_height_px": scale.line_height,
        # То же, но в пикселях исходного файла: загрузчик привёл страницу к 300 dpi,
        # и после апскейла скан 150 dpi по высоте строки уже не отличить от нормального.
        "source_line_height_px": scale.line_height * _source_scale(page),
        "n_lines": float(scale.n_lines),
        "text_density": scale.text_density,
        # полосы
        "streak_energy": streak_stats.energy,
        "streak_column_energy": streak_stats.column_energy,
        "streak_row_energy": streak_stats.row_energy,
        # служебное
        "dpi": page.dpi,
        "n_patches": float(len(boxes)),
        "n_informative_patches": float(len(informative)),
    }


def scores_from_metrics(raw: Mapping[str, float], config: Config) -> dict[str, float]:
    """Сырые метрики -> вероятности дефектов 0..1 по якорям из конфига."""
    scores: dict[str, float] = {}
    for label, mapping in config.cv.scores.items():
        if mapping.metric not in raw:
            logger.warning("Метрика %s для метки %s не посчитана — пропуск", mapping.metric, label)
            continue
        scores[label] = round(score_from_anchors(raw[mapping.metric], mapping.good, mapping.bad), 4)
    return scores


def analyze_page(page: LoadedPage, config: Config) -> tuple[dict[str, float], dict[str, float]]:
    """Удобная обёртка: (сырые метрики, скоры по меткам)."""
    raw = compute_raw_metrics(page, config)
    return raw, scores_from_metrics(raw, config)
