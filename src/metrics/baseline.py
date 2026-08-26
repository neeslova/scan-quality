"""CV-baseline: сборка сырых метрик по странице и перевод их в скоры по меткам.

Это опорная точка для записки — с ней сравнивается CNN. Обучения здесь нет:
только детерминированные измерения и линейные пороги из конфига.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Optional

import numpy as np

from src.config import Config
from src.imaging import binarize_ink, mid_tone_fraction
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

    # Бинаризация страницы стоит около 200 мс на крупном скане, а нужна четырём
    # метрикам сразу — считаем один раз. Для выровненной копии её приходится
    # пересчитывать: поворот меняет пиксели.
    page_ink = binarize_ink(gray, cfg.ink_block_frac, cfg.ink_offset)
    page_clean = geometry.denoise_ink(page_ink, cfg.ink_denoise_frac)
    deskewed_ink = page_ink if deskewed is gray else None

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
        binary=deskewed_ink,
        max_line_height_frac=cfg.line_max_height_frac,
        frame_span_frac=cfg.crop_frame_span_frac,
        smooth_frac=cfg.line_profile_smooth_frac,
    )
    margins = geometry.margin_fractions(
        gray,
        block_frac=cfg.ink_block_frac,
        offset=cfg.ink_offset,
        denoise_frac=cfg.ink_denoise_frac,
        binary=page_clean,
    )
    border = geometry.border_ink(
        gray,
        band_frac=cfg.crop_band_frac,
        frame_span_frac=cfg.crop_frame_span_frac,
        block_frac=cfg.ink_block_frac,
        offset=cfg.ink_offset,
        denoise_frac=cfg.ink_denoise_frac,
        binary=page_clean,
    )

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
        # Доля средних тонов: ниже cv.bitonal_max_mid_frac скан битональный,
        # и метрики по градациям серого на нём неприменимы.
        "mid_tone_frac": mid_tone_fraction(gray, cfg.bitonal_mid_low, cfg.bitonal_mid_high),
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
        # Поля — справочная величина, а не признак обреза: скан, подрезанный ровно
        # по листу, полей не имеет при полностью сохранном документе.
        "min_margin_frac": margins.minimum,
        # Обрез: доля периметра, где штрихи рассечены границей кадра.
        "border_ink_frac": border.coverage,
        "border_ink_left": border.left,
        "border_ink_right": border.right,
        "border_ink_top": border.top,
        "border_ink_bottom": border.bottom,
        "has_scan_frame": float(border.has_frame),
        # масштаб текста
        "line_height_px": scale.line_height,
        # То же, но в пикселях исходного файла. Загрузчик мог уменьшить страницу
        # (скан 600 dpi -> 300), и тогда высота строки в рабочем разрешении занижена
        # относительно того, что реально было в файле.
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


def inapplicable_labels(raw: Mapping[str, float], config: Config) -> set[str]:
    """Метки, которые на этой странице измерять нечем.

    На битональном скане (факс, микрофильм, режим Ч/Б) разрыв бумага-чернила всегда
    максимален, а шум обнулён самой бинаризацией. Формально метрики посчитаются и
    выдадут уверенный ноль — но это ноль от отсутствия шкалы, а не от отсутствия
    дефекта. Такую метку честнее не выдавать вовсе.

    То же с высотой строки: если строк не нашлось (пустая страница, разреженная
    рукопись), нулевая высота означала бы «текст нечитаемо мелкий», то есть
    максимальный дефект, — хотя на деле измерить просто не удалось.
    """
    skip: set[str] = set()
    if raw.get("mid_tone_frac", 1.0) < config.cv.bitonal_max_mid_frac:
        skip |= {"low_contrast", "noise"}
    if raw.get("n_lines", 0.0) <= 0.0:
        skip.add("low_resolution")
    return skip


def scores_from_metrics(
    raw: Mapping[str, float],
    config: Config,
    skip: Optional[Collection[str]] = None,
) -> dict[str, float]:
    """Сырые метрики -> вероятности дефектов 0..1 по якорям из конфига."""
    skipped = set(skip or ())
    scores: dict[str, float] = {}
    for label, mapping in config.cv.scores.items():
        if label in skipped:
            continue
        if mapping.metric not in raw:
            logger.warning("Метрика %s для метки %s не посчитана — пропуск", mapping.metric, label)
            continue
        scores[label] = round(score_from_anchors(raw[mapping.metric], mapping.good, mapping.bad), 4)
    return scores


def analyze_page(
    page: LoadedPage, config: Config
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Удобная обёртка: (сырые метрики, скоры, неприменимые метки)."""
    raw = compute_raw_metrics(page, config)
    skip = inapplicable_labels(raw, config)
    return raw, scores_from_metrics(raw, config, skip), sorted(skip)
