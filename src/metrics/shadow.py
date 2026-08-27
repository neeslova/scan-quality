"""Тень от сгиба, переплёта или края крышки сканера.

Тень — это свойство фона, а не текста: сильно размываем страницу, получаем карту
освещённости и ищем области, которые заметно темнее её же медианы.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ShadowStats:
    dark_frac: float  # доля площади в тени
    illumination_range: float  # (p95 - p5) фона в долях медианы
    darkest_ratio: float  # самая тёмная точка фона к медиане


WORK_SIDE = 256


def background(gray: np.ndarray, window_frac: float = 0.06) -> np.ndarray:
    """Карта освещённости.

    Размытия недостаточно: строки текста сами по себе тянут среднее вниз, и ровная
    страница выглядит как наполовину затенённая. Поэтому сначала морфологическое
    закрытие ядром заведомо крупнее строки — оно затирает все тёмные структуры и
    оставляет только бумагу, — и лишь потом сглаживание.
    """
    if gray.size == 0:
        return gray.astype(np.float64)

    scale = min(1.0, WORK_SIDE / max(gray.shape))
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    ksize = max(3, int(min(small.shape) * window_frac) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    paper = cv2.GaussianBlur(paper, (ksize, ksize), 0)
    return cv2.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def _background_and_mask(
    gray: np.ndarray, rel_threshold: float, window_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Карта освещённости и маска затенённого. Одна точка правды для обеих."""
    bg = background(gray, window_frac).astype(np.float64)
    median = float(np.median(bg))
    if median <= 0.0:
        return bg, np.ones(gray.shape, dtype=bool)
    return bg, bg < rel_threshold * median


def shadow_mask(
    gray: np.ndarray,
    rel_threshold: float = 0.82,
    window_frac: float = 0.06,
) -> np.ndarray:
    """Где именно тень. Ровно та маска, по которой считается `dark_frac`.

    Нужна слою локализации (С7): рисовать пользователю вторую, отдельно
    написанную тень значило бы показывать не то, по чему выставлен скор.
    """
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    return _background_and_mask(gray, rel_threshold, window_frac)[1]


def shadow_stats(
    gray: np.ndarray,
    rel_threshold: float = 0.82,
    window_frac: float = 0.06,
) -> ShadowStats:
    """Доля площади, где фон темнее rel_threshold от медианы фона."""
    if gray.size == 0:
        return ShadowStats(0.0, 0.0, 1.0)

    bg, mask = _background_and_mask(gray, rel_threshold, window_frac)
    median = float(np.median(bg))
    if median <= 0.0:
        return ShadowStats(1.0, 0.0, 0.0)

    dark_frac = float(mask.mean())
    p5, p95 = np.percentile(bg, [5.0, 95.0])
    return ShadowStats(
        dark_frac=dark_frac,
        illumination_range=float((p95 - p5) / median),
        darkest_ratio=float(np.min(bg) / median),
    )
