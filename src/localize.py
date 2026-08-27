"""Где на странице дефект. Карта поверх скана для приложения (С7).

**Grad-CAM из плана здесь не используется, и это осознанно.** Он объясняет
решение СЕТИ, а после замера (решения №39-40) сеть отвечает лишь за две метки
из десяти: `low_contrast` и `noise`, обе глобальные — «где именно» у них
не спрашивают. Остальные семь дают CV-метрики, и у них уже есть точное
доказательство: маска кластеров пересвета, карта освещённости, аномальные
колонки, приграничные штрихи. Рисовать поверх них Grad-CAM значило бы объяснять
не тот источник, который принял решение. Второе, независимое основание: Grad-CAM
требует градиентов, то есть torch, а он стоит только в extra `train` —
приложение обязано работать на базовых зависимостях.

Поэтому локализация идёт от источника метки:

  CV, локальные (`glare`, `shadow`, `streaks`, `cropped`) — ровно та маска,
      по которой посчитан скор. Не похожая, а та же самая: иначе показанное
      не объясняет выставленное.
  CNN (`low_contrast`, `noise`) — карта по патчам: сеть и предсказывает по патчам.
  Глобальные CV (`blur`, `skew`, `low_resolution`) и `unreadable` — карты нет.
      Расфокус и перекос относятся ко всей странице целиком, и подсветить
      «здесь размыто» на равномерно размытом скане невозможно честно.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.config import Config
from src.metrics.geometry import border_ink_mask
from src.metrics.glare import glare_mask
from src.metrics.shadow import shadow_mask
from src.metrics.streaks import streak_mask

logger = logging.getLogger(__name__)

# Метки, у которых локализация осмысленна, и чем она считается.
CV_MASKS = {
    "glare": glare_mask,
    "shadow": shadow_mask,
    "streaks": streak_mask,
    "cropped": border_ink_mask,
}


def cv_mask(label: str, gray: np.ndarray, config: Config) -> Optional[np.ndarray]:
    """Маска дефекта по CV-метрике или None, если у метки её нет.

    Параметры берутся из конфига, чтобы маска считалась ровно теми же числами,
    что и скор: разойдись они, картинка перестала бы объяснять вердикт.
    """
    if label not in CV_MASKS or gray.size == 0:
        return None

    cv = config.cv
    if label == "glare":
        return glare_mask(
            gray,
            threshold=cv.glare_threshold,
            min_cluster_frac=cv.glare_min_cluster_frac,
            flat_window_frac=cv.glare_flat_window_frac,
            flat_std=cv.glare_flat_std,
            min_excess=cv.glare_min_excess,
        )
    if label == "shadow":
        return shadow_mask(gray, cv.shadow_rel_threshold, cv.shadow_background_frac)
    if label == "streaks":
        return streak_mask(gray, cv.streak_smooth_frac)
    return border_ink_mask(gray, band_frac=cv.crop_band_frac)


def patch_heat(prediction, label: str, shape: tuple[int, int]) -> np.ndarray:
    """Карта по патчам: вероятность метки размазана по площади своего патча.

    Патчи сетки перекрываются, поэтому берём максимум, а не сумму: перекрытие
    иначе давало бы яркое пятно там, где просто плотнее сетка, а не дефект.
    """
    heat = np.zeros(shape, dtype=np.float32)
    position = list(prediction.labels).index(label)
    for patch, probability in zip(prediction.patches, prediction.probabilities[:, position]):
        window = heat[patch.y0 : patch.y1, patch.x0 : patch.x1]
        np.maximum(window, float(probability), out=window)
    return heat


def heatmap(
    label: str,
    gray: np.ndarray,
    config: Config,
    prediction=None,
) -> Optional[np.ndarray]:
    """Карта 0..1 по размеру страницы или None, если метку не локализовать."""
    source = config.sources.of(label)
    if source == "cv":
        mask = cv_mask(label, gray, config)
        return None if mask is None else mask.astype(np.float32)
    if source == "cnn" and prediction is not None:
        return patch_heat(prediction, label, gray.shape[:2])
    return None


def overlay(gray: np.ndarray, heat: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Скан с подсветкой дефекта. RGB, чтобы Gradio показал как есть.

    Подсветка красная и полупрозрачная: под ней должен остаться виден сам скан,
    иначе по картинке нельзя проверить, что система права.
    """
    base = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32)
    if heat.shape != gray.shape[:2]:
        heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    weight = np.clip(heat, 0.0, 1.0)[:, :, None] * strength
    red = np.zeros_like(base)
    red[:, :, 0] = 255.0
    return np.clip(base * (1.0 - weight) + red * weight, 0, 255).astype(np.uint8)


def localizable(config: Config) -> list[str]:
    """Метки, для которых карта вообще существует."""
    return [
        label
        for label in config.labels
        if (config.sources.of(label) == "cv" and label in CV_MASKS)
        or config.sources.of(label) == "cnn"
    ]
