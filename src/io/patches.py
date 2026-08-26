"""Нарезка страницы на сетку патчей и агрегация предсказаний обратно в скор страницы.

Патчи нужны потому, что A4@300dpi — это 2480×3508: при ресайзе в 224 px текст
превращается в серый шум и blur/low_resolution становятся недетектируемыми.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.imaging import binarize_ink

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Patch:
    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]


def _positions(total: int, size: int, count: int) -> list[int]:
    """Начала окон: равномерно от 0 до total-size. Перекрытие допустимо, дыры — нет."""
    if total <= size or count <= 1:
        return [0]
    span = total - size
    return [round(i * span / (count - 1)) for i in range(count)]


def grid(width: int, height: int, patch_size: int, rows: int, cols: int) -> list[Patch]:
    """Регулярная сетка патчей по странице. Возвращает боксы, не пиксели."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Некорректный размер страницы: {width}x{height}")

    box_w = min(patch_size, width)
    box_h = min(patch_size, height)
    xs = _positions(width, box_w, cols)
    ys = _positions(height, box_h, rows)

    patches: list[Patch] = []
    index = 0
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            patches.append(
                Patch(index=index, row=row, col=col, x0=x0, y0=y0, x1=x0 + box_w, y1=y0 + box_h)
            )
            index += 1
    return patches


def ink_fraction(patch_gray: np.ndarray) -> float:
    """Доля «чернил» на патче. Пустое поле бумаги даёт ~0.

    Порог локальный: у Оцу на однородном патче нет двух классов, и он честно
    делит пополам шум — пустое поле выглядело бы наполовину исписанным.
    """
    if patch_gray.size == 0:
        return 0.0
    binary = binarize_ink(patch_gray)
    return float(np.count_nonzero(binary) / binary.size)


def select_informative(
    gray: np.ndarray, patches: Sequence[Patch], min_ink_frac: float
) -> list[Patch]:
    """Оставляет патчи с текстом.

    Резкость и шум на пустом поле бумаги измерять бессмысленно: там нет краёв,
    и любая метрика скажет «размыто». Если текста не нашлось нигде (пустая страница),
    возвращаем всё — пусть решает вызывающий, а не пустой список.
    """
    informative = [p for p in patches if ink_fraction(p.crop(gray)) >= min_ink_frac]
    if not informative:
        logger.debug("Информативных патчей нет (порог %.3f) — берём все", min_ink_frac)
        return list(patches)
    return informative


def aggregate(
    per_patch: Mapping[str, Sequence[float]],
    local_labels: Sequence[str],
) -> dict[str, float]:
    """Схлопывает скоры по патчам в скор страницы.

    Локальные дефекты (блик, тень, полосы, обрез) — max: достаточно одного плохого места.
    Глобальные (размытие, шум, контраст) — mean: это свойство всего прогона сканера.
    """
    local = set(local_labels)
    result: dict[str, float] = {}
    for label, values in per_patch.items():
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        result[label] = float(array.max() if label in local else array.mean())
    return result
