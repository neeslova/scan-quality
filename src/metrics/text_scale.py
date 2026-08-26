"""Оценка высоты строки — прокси для разрешения текста.

Ниже ~15 px на строку кириллический печатный текст перестаёт читаться, поэтому
эта метрика напрямую отвечает за метку low_resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.imaging import (
    DEFAULT_INK_BLOCK_FRAC,
    DEFAULT_INK_OFFSET,
    binarize_ink,
    frame_component_mask,
)

# Участок профиля выше этой доли высоты страницы — не строка текста.
DEFAULT_MAX_LINE_HEIGHT_FRAC = 0.25
DEFAULT_FRAME_SPAN_FRAC = 0.8
# Сглаживание горизонтального профиля, доля высоты страницы.
DEFAULT_PROFILE_SMOOTH_FRAC = 0.002


@dataclass(frozen=True)
class TextScale:
    line_height: float  # медианная высота полосы текста, px
    n_lines: int
    text_density: float  # доля чернил на странице


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные участки True как список (start, stop)."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def estimate_text_scale(
    gray: np.ndarray,
    min_row_ink_frac: float = 0.02,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
    binary: Optional[np.ndarray] = None,
    max_line_height_frac: float = DEFAULT_MAX_LINE_HEIGHT_FRAC,
    frame_span_frac: float = DEFAULT_FRAME_SPAN_FRAC,
    smooth_frac: float = DEFAULT_PROFILE_SMOOTH_FRAC,
) -> TextScale:
    """Горизонтальный профиль чернил -> полосы строк -> медианная высота.

    Страницу желательно предварительно выровнять: при перекосе строки сливаются
    в один сплошной участок и высота уезжает вверх.

    Перед профилированием убираются компоненты, растянутые почти на всю сторону.
    Без этого тёмный край скана даёт чернила в каждой строке профиля, и вся
    страница читается как одна строка высотой в лист — на архивном корпусе так
    ломалось около 40% страниц. Участки выше `max_line_height_frac` от высоты
    страницы строками не считаются: это остатки рамок и слитые блоки.
    """
    if gray.size == 0:
        return TextScale(0.0, 0, 0.0)

    if binary is None:
        binary = binarize_ink(gray, block_frac, offset)

    height, width = binary.shape
    density = float(np.count_nonzero(binary) / binary.size)

    text_only = binary.copy()
    text_only[frame_component_mask(binary, frame_span_frac)] = 0

    profile = text_only.sum(axis=1, dtype=np.float64) / (255.0 * width)
    # Сглаживание обязательно: на фактурной бумаге профиль дрожит вокруг порога и
    # одна строка разваливается на несколько коротких участков — медианная высота
    # уезжает вниз в разы. Ядро берём от высоты страницы, а не от высоты строки:
    # её мы как раз и ищем.
    smooth = max(1, int(height * smooth_frac))
    if smooth > 1:
        profile = np.convolve(profile, np.ones(smooth) / smooth, mode="same")

    runs = _runs(profile >= min_row_ink_frac)
    if not runs:
        return TextScale(0.0, 0, density)

    heights = np.array([stop - start for start, stop in runs], dtype=np.float64)
    # Один пиксель — линейка таблицы, не строка; слишком высокий участок — не текст.
    heights = heights[(heights >= 2.0) & (heights <= max_line_height_frac * height)]
    if heights.size == 0:
        return TextScale(0.0, 0, density)

    return TextScale(
        line_height=float(np.median(heights)),
        n_lines=int(heights.size),
        text_density=density,
    )
