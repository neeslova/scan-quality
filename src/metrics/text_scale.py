"""Оценка высоты строки — прокси для разрешения текста.

Ниже ~15 px на строку кириллический печатный текст перестаёт читаться, поэтому
эта метрика напрямую отвечает за метку low_resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.imaging import DEFAULT_INK_BLOCK_FRAC, DEFAULT_INK_OFFSET, binarize_ink


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
) -> TextScale:
    """Горизонтальный профиль чернил -> полосы строк -> медианная высота.

    Страницу желательно предварительно выровнять: при перекосе строки сливаются
    в один сплошной участок и высота уезжает вверх.
    """
    if gray.size == 0:
        return TextScale(0.0, 0, 0.0)

    binary = binarize_ink(gray, block_frac, offset)
    height, width = binary.shape
    density = float(np.count_nonzero(binary) / binary.size)

    profile = binary.sum(axis=1, dtype=np.float64) / (255.0 * width)
    runs = _runs(profile >= min_row_ink_frac)
    if not runs:
        return TextScale(0.0, 0, density)

    heights = np.array([stop - start for start, stop in runs], dtype=np.float64)
    # Отбрасываем «строки» высотой в один пиксель — это линейки таблиц, не текст.
    heights = heights[heights >= 2.0]
    if heights.size == 0:
        return TextScale(0.0, 0, density)

    return TextScale(
        line_height=float(np.median(heights)),
        n_lines=int(heights.size),
        text_density=density,
    )
