"""Полосы: следы валиков, грязь на стекле, артефакты копира.

Признак — устойчивая аномалия одной колонки (или строки) фона по всей странице.
Считаем профиль фона, вычитаем сглаженную версию и смотрим на остаток: у чистого
скана он около нуля, у полосатого — регулярные выбросы.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StreakStats:
    column_energy: float
    row_energy: float

    @property
    def energy(self) -> float:
        """Полоса может быть и вертикальной, и горизонтальной — берём худшее."""
        return max(self.column_energy, self.row_energy)


_MAD_TO_SIGMA = 1.4826
# Во сколько робастных σ должно уложиться «нормальное» дрожание профиля.
_OUTLIER_SIGMAS = 3.0


def _profile_energy(profile: np.ndarray, smooth: int) -> float:
    """Энергия аномальных колонок после вычитания плавного тренда.

    Обычное СКО остатка здесь не работает: равномерный шум даёт по всему профилю
    примерно ту же энергию, что и несколько настоящих полос, и метрика перестаёт
    их различать. Полоса — редкий сильный выброс, поэтому считаем только ту часть
    остатка, что вылезает за робастный разброс.
    """
    if profile.size < 5:
        return 0.0
    kernel = max(3, smooth | 1)
    trend = cv2.blur(profile.reshape(1, -1).astype(np.float64), (kernel, 1))[0]
    residual = profile - trend

    level = float(np.median(profile))
    if level <= 0.0:
        return 0.0

    sigma = float(np.median(np.abs(residual))) * _MAD_TO_SIGMA
    excess = np.maximum(np.abs(residual) - _OUTLIER_SIGMAS * sigma, 0.0)
    return float(np.sqrt(np.mean(excess**2)) / level)


def _background_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """Профиль фона вдоль оси: берём светлые перцентили, чтобы текст не мешал."""
    return np.percentile(gray.astype(np.float64), 75.0, axis=axis)


def streak_stats(gray: np.ndarray, smooth_frac: float = 0.02) -> StreakStats:
    if gray.size == 0:
        return StreakStats(0.0, 0.0)

    height, width = gray.shape
    return StreakStats(
        column_energy=_profile_energy(_background_profile(gray, axis=0), int(width * smooth_frac)),
        row_energy=_profile_energy(_background_profile(gray, axis=1), int(height * smooth_frac)),
    )
