"""Контраст: RMS, динамический диапазон и разрыв «бумага — чернила».

Для документа важнее всего последняя метрика: блёклая печать — это когда текст
и фон сблизились по яркости, даже если гистограмма в целом широкая.
"""

from __future__ import annotations

import cv2
import numpy as np


def rms_contrast(gray: np.ndarray) -> float:
    """Среднеквадратичный контраст (σ яркости, 0..127)."""
    if gray.size == 0:
        return 0.0
    return float(np.std(gray.astype(np.float64)))


def dynamic_range(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> float:
    """Разброс яркости по перцентилям — устойчив к одиночным выбросам."""
    if gray.size == 0:
        return 0.0
    low, high = np.percentile(gray, [low_pct, high_pct])
    return float(high - low)


def ink_paper_gap(gray: np.ndarray) -> float:
    """Разрыв медиан бумаги и чернил после порога Оцу (0..255).

    Прямое выражение «блёклой печати»: чем меньше разрыв, тем ближе серый текст
    к серому фону и тем хуже он читается.
    """
    if gray.size == 0:
        return 0.0
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray[gray <= threshold]
    paper = gray[gray > threshold]
    if ink.size == 0 or paper.size == 0:
        return 0.0
    return float(np.median(paper) - np.median(ink))
