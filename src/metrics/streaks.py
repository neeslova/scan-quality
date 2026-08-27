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


def _profile_excess(profile: np.ndarray, smooth: int) -> tuple[np.ndarray, float]:
    """Превышение над робастным разбросом по каждой колонке и уровень фона.

    Обычное СКО остатка здесь не работает: равномерный шум даёт по всему профилю
    примерно ту же энергию, что и несколько настоящих полос, и метрика перестаёт
    их различать. Полоса — редкий сильный выброс, поэтому считаем только ту часть
    остатка, что вылезает за робастный разброс.

    Возвращается сам вектор превышений, а не только его энергия: по нему слой
    локализации (С7) показывает, КАКИЕ колонки признаны полосами. Считать это
    вторым способом значило бы показывать не то, по чему выставлен скор.
    """
    if profile.size < 5:
        return np.zeros_like(profile, dtype=np.float64), 0.0
    kernel = max(3, smooth | 1)
    trend = cv2.blur(profile.reshape(1, -1).astype(np.float64), (kernel, 1))[0]
    residual = profile - trend

    level = float(np.median(profile))
    if level <= 0.0:
        return np.zeros_like(profile, dtype=np.float64), 0.0

    sigma = float(np.median(np.abs(residual))) * _MAD_TO_SIGMA
    return np.maximum(np.abs(residual) - _OUTLIER_SIGMAS * sigma, 0.0), level


def _profile_energy(profile: np.ndarray, smooth: int) -> float:
    """Энергия аномальных колонок после вычитания плавного тренда."""
    excess, level = _profile_excess(profile, smooth)
    if level <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean(excess**2)) / level)


def streak_mask(gray: np.ndarray, smooth_frac: float = 0.02) -> np.ndarray:
    """Колонки и строки, признанные полосами. Та же арифметика, что у скора."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)

    height, width = gray.shape
    columns, _ = _profile_excess(_background_profile(gray, axis=0), int(width * smooth_frac))
    rows, _ = _profile_excess(_background_profile(gray, axis=1), int(height * smooth_frac))

    mask = np.zeros((height, width), dtype=bool)
    mask[:, columns > 0.0] = True
    mask[rows > 0.0, :] = True
    return mask


def _without_rules(gray: np.ndarray) -> np.ndarray:
    """Страница без структурных линий: рамок, линеек таблиц, подчёркиваний.

    Без этого таблица гарантированно объявляется полосами. Профиль фона берёт
    перцентиль по колонке, и вертикальная линейка таблицы — тёмная колонка во всю
    высоту — для профиля неотличима от следа валика: тот же одиночный сильный
    выброс. Скан с таблицей получал `streaks` и уходил в `bad`.

    Разница между ними структурная, а не яркостная: линейка — это ровная линия
    из чернил, а след валика — размазанное изменение фона, у которого чёткой
    линии нет вовсе. Поэтому линии выделяются морфологическим открытием длинным
    тонким ядром и заменяются уровнем бумаги: колонка после этого выглядит так,
    как выглядела бы без таблицы.

    Связные компоненты здесь не годятся, и это проверено: линейки таблицы сшиты
    в одну решётку, её habitus — большой бокс по обеим осям, и правило «тонкая
    и длинная» её не опознаёт. Открытие смотрит на форму локально и решётку
    разбирает на составляющие линии.
    """
    from src.imaging import binarize_ink

    ink = binarize_ink(gray) > 0
    height, width = gray.shape
    # Ядро длиной в пятую часть стороны: короче — начнёт цеплять строки текста,
    # длиннее — пропустит линейки внутри таблицы.
    horizontal = cv2.morphologyEx(
        ink.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 5), 1)),
    )
    vertical = cv2.morphologyEx(
        ink.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 5))),
    )
    rules = (horizontal | vertical) > 0
    if not rules.any():
        return gray.astype(np.float64)

    # Расширяем на пару пикселей: у линии есть сглаженная кромка, и без запаса
    # она остаётся в профиле полутоном.
    thick = cv2.dilate(rules.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    cleaned = gray.astype(np.float64).copy()
    cleaned[thick] = float(np.median(gray[~thick])) if (~thick).any() else float(np.median(gray))
    return cleaned


def _background_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """Профиль фона вдоль оси: берём светлые перцентили, чтобы текст не мешал."""
    return np.percentile(_without_rules(gray), 75.0, axis=axis)


def streak_stats(gray: np.ndarray, smooth_frac: float = 0.02) -> StreakStats:
    if gray.size == 0:
        return StreakStats(0.0, 0.0)

    height, width = gray.shape
    return StreakStats(
        column_energy=_profile_energy(_background_profile(gray, axis=0), int(width * smooth_frac)),
        row_energy=_profile_energy(_background_profile(gray, axis=1), int(height * smooth_frac)),
    )
