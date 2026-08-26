"""Геометрия страницы: перекос строк и поля документа.

Перекос ищем перебором угла по профилю проекции: у ровного текста строки дают
резкий пилообразный профиль, у перекошенного он размазывается. Это устойчивее
Хафа на сканах с печатями и таблицами, где линий много и они не про текст.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.imaging import DEFAULT_INK_BLOCK_FRAC, DEFAULT_INK_OFFSET, binarize_ink


@dataclass(frozen=True)
class Margins:
    left: float
    right: float
    top: float
    bottom: float

    @property
    def minimum(self) -> float:
        """Самое узкое поле — по нему судим об обрезе."""
        return min(self.left, self.right, self.top, self.bottom)


def rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворот вокруг центра с добиванием фона до цвета бумаги."""
    if abs(angle_deg) < 1e-3:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    border = int(np.percentile(gray, 90))  # бумага, а не чёрная рамка
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _profile_sharpness(binary: np.ndarray, angle_deg: float) -> float:
    """Насколько «пилообразен» горизонтальный профиль при данном повороте."""
    rotated = binary if abs(angle_deg) < 1e-3 else rotate(binary, angle_deg)
    profile = rotated.sum(axis=1, dtype=np.float64)
    if profile.size < 2:
        return 0.0
    return float(np.sum(np.diff(profile) ** 2))


def estimate_skew(
    gray: np.ndarray,
    max_angle: float = 15.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
    work_height: int = 800,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
) -> float:
    """Угол перекоса в градусах (>0 — текст завален против часовой стрелки).

    Двухпроходный перебор: грубый по всему диапазону, затем точный вокруг найденного
    угла. Полное разрешение здесь не нужно и стоит секунд, поэтому уменьшаем страницу.
    """
    if gray.size == 0:
        return 0.0

    scale = min(1.0, work_height / gray.shape[0])
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    binary = binarize_ink(small, block_frac, offset)
    if np.count_nonzero(binary) == 0:
        return 0.0

    coarse = np.arange(-max_angle, max_angle + coarse_step, coarse_step)
    best = max(coarse, key=lambda a: _profile_sharpness(binary, float(a)))

    fine = np.arange(best - coarse_step, best + coarse_step + fine_step, fine_step)
    best = max(fine, key=lambda a: _profile_sharpness(binary, float(a)))
    return round(float(best), 2)


def text_bbox(
    gray: np.ndarray,
    min_ink_frac: float = 0.005,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
) -> tuple[int, int, int, int]:
    """Прямоугольник текста: (x0, y0, x1, y1). Пустая страница -> вся страница."""
    if gray.size == 0:
        return (0, 0, 0, 0)

    binary = binarize_ink(gray, block_frac, offset)
    # Мелкий мусор (пыль, точки) не должен растягивать бокс до краёв.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    height, width = binary.shape
    rows = binary.sum(axis=1, dtype=np.float64) / (255.0 * width)
    cols = binary.sum(axis=0, dtype=np.float64) / (255.0 * height)

    row_idx = np.flatnonzero(rows >= min_ink_frac)
    col_idx = np.flatnonzero(cols >= min_ink_frac)
    if row_idx.size == 0 or col_idx.size == 0:
        return (0, 0, width, height)
    return (int(col_idx[0]), int(row_idx[0]), int(col_idx[-1]) + 1, int(row_idx[-1]) + 1)


def margin_fractions(
    gray: np.ndarray,
    min_ink_frac: float = 0.005,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
) -> Margins:
    """Поля вокруг текста в долях соответствующей стороны страницы."""
    if gray.size == 0:
        return Margins(0.0, 0.0, 0.0, 0.0)

    height, width = gray.shape
    x0, y0, x1, y1 = text_bbox(gray, min_ink_frac, block_frac, offset)
    return Margins(
        left=x0 / width,
        right=(width - x1) / width,
        top=y0 / height,
        bottom=(height - y1) / height,
    )
