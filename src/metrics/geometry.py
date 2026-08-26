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

# Ядро открытия для подавления зерна бумаги, доля меньшей стороны страницы.
DEFAULT_DENOISE_FRAC = 0.0015
# Полоса вдоль края, в которой ищем рассечённые штрихи.
DEFAULT_BAND_FRAC = 0.005
# Компонента, растянутая вдоль стороны больше этой доли, — рамка, а не текст.
DEFAULT_FRAME_SPAN_FRAC = 0.8


@dataclass(frozen=True)
class Margins:
    left: float
    right: float
    top: float
    bottom: float

    @property
    def minimum(self) -> float:
        return min(self.left, self.right, self.top, self.bottom)


@dataclass(frozen=True)
class BorderInk:
    """Сколько текста рассечено границей кадра."""

    coverage: float  # доля периметра, занятая срезанными штрихами
    left: float
    right: float
    top: float
    bottom: float
    has_frame: bool  # найдена сплошная рамка: край сканера, тень переплёта


def _denoise(binary: np.ndarray, denoise_frac: float) -> np.ndarray:
    """Убирает зерно бумаги. Ядро от размера страницы, а не фиксированные 3×3.

    На фактурной архивной бумаге пятна и потемневшие края проходят порог локальной
    бинаризации, и без масштабируемого открытия бокс текста растягивается на весь кадр.
    """
    size = max(3, int(min(binary.shape) * denoise_frac) | 1)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


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
    denoise_frac: float = DEFAULT_DENOISE_FRAC,
) -> tuple[int, int, int, int]:
    """Прямоугольник текста: (x0, y0, x1, y1). Пустая страница -> вся страница."""
    if gray.size == 0:
        return (0, 0, 0, 0)

    binary = _denoise(binarize_ink(gray, block_frac, offset), denoise_frac)
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
    denoise_frac: float = DEFAULT_DENOISE_FRAC,
) -> Margins:
    """Поля вокруг текста в долях соответствующей стороны страницы.

    Внимание: это НЕ признак обреза. Поля зависят от того, как обрезали скан, а не
    от того, потерян ли текст: архивный скан, подрезанный ровно по листу, даёт
    нулевые поля при полностью сохранном документе. Для обреза см. `border_ink`.
    """
    if gray.size == 0:
        return Margins(0.0, 0.0, 0.0, 0.0)

    height, width = gray.shape
    x0, y0, x1, y1 = text_bbox(gray, min_ink_frac, block_frac, offset, denoise_frac)
    return Margins(
        left=x0 / width,
        right=(width - x1) / width,
        top=y0 / height,
        bottom=(height - y1) / height,
    )


def border_ink(
    gray: np.ndarray,
    band_frac: float = DEFAULT_BAND_FRAC,
    frame_span_frac: float = DEFAULT_FRAME_SPAN_FRAC,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
    denoise_frac: float = DEFAULT_DENOISE_FRAC,
) -> BorderInk:
    """Доля периметра, где штрихи рассечены границей кадра.

    Это и есть признак обреза. «Нет полей» им не является: скан, подрезанный ровно
    по листу, полей не имеет, но документ на нём сохранён целиком. Обрез виден иначе —
    строки текста упираются в границу и обрываются на ней.

    Сплошные компоненты, растянутые почти на всю сторону, из подсчёта исключаются:
    это край сканера, тень переплёта или чёрная рамка — свой дефект, но не обрез.
    Настоящий срезанный текст даёт прерывистое покрытие: штрихи с промежутками.
    """
    if gray.size == 0:
        return BorderInk(0.0, 0.0, 0.0, 0.0, 0.0, False)

    binary = _denoise(binarize_ink(gray, block_frac, offset), denoise_frac)
    height, width = binary.shape
    band = max(1, int(min(height, width) * band_frac))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return BorderInk(0.0, 0.0, 0.0, 0.0, 0.0, False)

    spans_width = stats[:, cv2.CC_STAT_WIDTH] >= frame_span_frac * width
    spans_height = stats[:, cv2.CC_STAT_HEIGHT] >= frame_span_frac * height
    is_frame = spans_width | spans_height
    is_frame[0] = True  # фон

    frame_lookup = is_frame[labels]
    text = (labels > 0) & ~frame_lookup

    edges = {
        "top": text[:band, :].any(axis=0),
        "bottom": text[-band:, :].any(axis=0),
        "left": text[:, :band].any(axis=1),
        "right": text[:, -band:].any(axis=1),
    }
    coverage = {name: float(mask.mean()) for name, mask in edges.items()}

    return BorderInk(
        coverage=max(coverage.values()),
        left=coverage["left"],
        right=coverage["right"],
        top=coverage["top"],
        bottom=coverage["bottom"],
        has_frame=bool(is_frame[1:].any()),
    )
