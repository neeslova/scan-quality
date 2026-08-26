"""Синтетические «сканы» и деградации для тестов метрик.

Настоящих сканов в репозитории нет (и не будет — `data/` в .gitignore), поэтому
проверяем метрики на искусственной странице: важно не абсолютное значение, а то,
что деградация сдвигает метрику в ожидаемую сторону.
"""

from __future__ import annotations

import cv2
import numpy as np

PAPER = 238
INK = 45

GLYPH_WIDTH = 4
GLYPH_GAP = 3
WORD_GAP = 14


def text_page(
    width: int = 1200,
    height: int = 1600,
    line_height: int = 24,
    line_gap: int = 18,
    margin: int = 90,
    paper: int = PAPER,
    ink: int = INK,
    seed: int = 0,
) -> np.ndarray:
    """Страница «печатного текста»: строки из слов, слова из вертикальных штрихов."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), paper, dtype=np.uint8)

    y = margin
    while y + line_height < height - margin:
        x = margin
        limit = width - margin
        while x < limit - GLYPH_WIDTH:
            word_width = int(rng.integers(30, 110))
            if x + word_width > limit:
                break
            for gx in range(x, x + word_width, GLYPH_WIDTH + GLYPH_GAP):
                cv2.rectangle(page, (gx, y), (gx + GLYPH_WIDTH, y + line_height), int(ink), -1)
            x += word_width + WORD_GAP
        y += line_height + line_gap
    return page


def blurred(page: np.ndarray, ksize: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(page, (ksize | 1, ksize | 1), 0)


def noisy(page: np.ndarray, sigma: float = 12.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, page.shape)
    return np.clip(page.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def low_contrast(page: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Сжимает диапазон к среднему: печать становится блёклой."""
    mean = float(np.mean(page))
    return np.clip((page.astype(np.float64) - mean) * factor + mean, 0, 255).astype(np.uint8)


def with_glare(page: np.ndarray, radius_frac: float = 0.18) -> np.ndarray:
    """Эллиптический участок, выбитый в белое: ни текста, ни фактуры."""
    out = page.copy()
    height, width = page.shape
    center = (int(width * 0.35), int(height * 0.4))
    axes = (int(width * radius_frac), int(height * radius_frac * 0.8))
    cv2.ellipse(out, center, axes, 0, 0, 360, 255, thickness=-1)
    return out


def with_shadow(page: np.ndarray, depth: float = 0.45, width_frac: float = 0.25) -> np.ndarray:
    """Затемнение у левого края — как от переплёта."""
    height, width = page.shape
    band = max(1, int(width * width_frac))
    ramp = np.ones(width, dtype=np.float64)
    ramp[:band] = np.linspace(1.0 - depth, 1.0, band)
    return np.clip(page.astype(np.float64) * ramp[None, :], 0, 255).astype(np.uint8)


def with_streaks(page: np.ndarray, count: int = 6, depth: int = 35, seed: int = 2) -> np.ndarray:
    """Вертикальные полосы валиков: несколько колонок стабильно темнее."""
    rng = np.random.default_rng(seed)
    out = page.astype(np.int16)
    width = page.shape[1]
    for column in rng.choice(width - 4, size=count, replace=False):
        out[:, column : column + 2] -= depth
    return np.clip(out, 0, 255).astype(np.uint8)


def rotated(page: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = page.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        page,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(np.percentile(page, 90)),
    )


def downscaled(page: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Потеря разрешения: уменьшили и вернули обратно — символы уже не разделить."""
    small = cv2.resize(page, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_CUBIC)


def cropped(page: np.ndarray, cut_frac: float = 0.12) -> np.ndarray:
    """Срезали левый край вместе с полем — текст упирается в границу кадра."""
    cut = int(page.shape[1] * cut_frac)
    return page[:, cut:].copy()


def save(page: np.ndarray, path, dpi: int = 300) -> None:
    """Сохраняет с явным dpi, чтобы загрузчик не пересчитывал масштаб."""
    from PIL import Image

    Image.fromarray(page).save(str(path), dpi=(dpi, dpi))
