"""Резкость: дисперсия лапласиана и Tenengrad.

Обе метрики измеряют энергию краёв. Считать их нужно на патче с текстом и при
нормализованном dpi — иначе числа несравнимы между сканами.
"""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Дисперсия лапласиана. Классика: у размытого изображения края «съедены»."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    """Средний квадрат градиента Собеля.

    Устойчивее лапласиана к шуму: шум даёт высокочастотный отклик, но малую
    амплитуду градиента, а лапласиан на нём заметно раздувается.
    """
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx * gx + gy * gy))
