"""Общие пиксельные операции, нужные сразу нескольким модулям.

Отдельный модуль, чтобы `io` не импортировал `metrics` и наоборот.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_INK_BLOCK_FRAC = 0.03
DEFAULT_INK_OFFSET = 10

# Ядро Immerkær: гасит линейную составляющую и оставляет высокие частоты.
LAPLACE_MASK = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
MASK_NORM = 6.0  # L2-норма ядра: для белого шума std(ответа) = 6σ
_MAD_TO_SIGMA = 1.4826
# На сколько σ шума порог чернил должен отступать от локального среднего,
# чтобы шум не считался текстом. 3σ — примерно одна ложная точка на тысячу.
_INK_NOISE_SIGMAS = 3.0
_MAX_INK_OFFSET = 60
# Ниже этого уровня шума медианный фильтр только зря мылит текст.
_DENOISE_SIGMA = 2.0


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """σ шума по медиане высокочастотного отклика.

    Медиана, а не среднее: края букв дают тот же высокочастотный отклик, что и шум,
    и по среднему (классический Immerkær) оценка на документе завышается в разы.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    response = cv2.filter2D(
        gray.astype(np.float64), ddepth=-1, kernel=LAPLACE_MASK, borderType=cv2.BORDER_REFLECT
    )[1:-1, 1:-1]
    if response.size == 0:
        return 0.0
    return float(np.median(np.abs(response))) * _MAD_TO_SIGMA / MASK_NORM


def binarize_ink(
    gray: np.ndarray,
    block_frac: float = DEFAULT_INK_BLOCK_FRAC,
    offset: int = DEFAULT_INK_OFFSET,
) -> np.ndarray:
    """Бинаризация «чернила = 255» по локальному порогу.

    Оцу здесь не годится: тень от переплёта или неравномерная подсветка сдвигают
    глобальный порог, и затемнённая половина страницы целиком попадает в «текст» —
    после чего поля документа схлопываются и скан ложно помечается как обрезанный.
    Локальный порог сравнивает пиксель с его окрестностью и к наклону фона нечувствителен.

    Отступ от локального среднего растёт вместе с шумом. С фиксированным отступом
    зернистый скан бинаризуется целиком в «текст»: строки сливаются, и высота строки
    уезжает до высоты страницы.
    """
    if gray.size == 0:
        return gray

    sigma = estimate_noise_sigma(gray)
    source = cv2.medianBlur(gray, 3) if sigma > _DENOISE_SIGMA else gray
    effective_offset = int(min(_MAX_INK_OFFSET, max(offset, round(_INK_NOISE_SIGMAS * sigma))))

    block = max(3, int(min(gray.shape) * block_frac) | 1)
    return cv2.adaptiveThreshold(
        source, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, effective_offset
    )


def mid_tone_fraction(gray: np.ndarray, low: int = 40, high: int = 215) -> float:
    """Доля пикселей в средних тонах.

    Отличает полутоновый скан от битонального. У битонального (факс, микрофильм,
    режим «чёрно-белый» на МФУ) почти всё лежит в двух пиках у 0 и 255, средних
    тонов почти нет — и метрики, которые опираются на градации серого, там
    измерять нечего.
    """
    if gray.size == 0:
        return 0.0
    return float(((gray > low) & (gray < high)).mean())


def local_std(gray: np.ndarray, window: int) -> np.ndarray:
    """σ в скользящем окне через E[x²] - E[x]²."""
    values = gray.astype(np.float64)
    size = max(3, window | 1)
    mean = cv2.blur(values, (size, size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(values * values, (size, size), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
