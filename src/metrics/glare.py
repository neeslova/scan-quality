"""Пересвет: участок выбит в белое.

Одного порога яркости мало по двум причинам. Во-первых, белая бумага сама по себе
ярче 245, и любой чистый скан получил бы максимальный балл. Во-вторых, блик — это
потеря информации: внутри него нет фактуры. Поэтому требуем три условия сразу:
пиксель ярче абсолютного порога, заметно ярче собственного уровня бумаги страницы,
и вокруг него нет никакой структуры.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.imaging import local_std


@dataclass(frozen=True)
class GlareStats:
    bright_frac: float  # доля пикселей ярче абсолютного порога
    cluster_frac: float  # доля площади в блик-кластерах достаточного размера
    n_clusters: int
    largest_cluster_frac: float
    paper_level: float  # уровень бумаги, от которого считался пересвет


def glare_stats(
    gray: np.ndarray,
    threshold: int = 245,
    min_cluster_frac: float = 0.0005,
    flat_window_frac: float = 0.02,
    flat_std: float = 3.0,
    min_excess: int = 8,
) -> GlareStats:
    """Кластеры «яркое, ярче бумаги и плоское». Мелкие отбрасываем как шум."""
    if gray.size == 0:
        return GlareStats(0.0, 0.0, 0, 0.0, 0.0)

    area = float(gray.size)
    # Медиана страницы — это бумага: текст всегда в меньшинстве.
    paper_level = float(np.median(gray))

    bright = gray >= max(threshold, paper_level + min_excess)
    bright_frac = float(np.count_nonzero(gray >= threshold) / area)

    window = max(3, int(min(gray.shape) * flat_window_frac) | 1)
    flat = local_std(gray, window) < flat_std
    mask = (bright & flat).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_cluster_frac * area
    kept = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not kept:
        return GlareStats(bright_frac, 0.0, 0, 0.0, paper_level)

    return GlareStats(
        bright_frac=bright_frac,
        cluster_frac=float(sum(kept) / area),
        n_clusters=len(kept),
        largest_cluster_frac=float(max(kept) / area),
        paper_level=paper_level,
    )
