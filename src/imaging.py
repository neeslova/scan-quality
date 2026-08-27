"""Общие пиксельные операции, нужные сразу нескольким модулям.

Отдельный модуль, чтобы `io` не импортировал `metrics` и наоборот.
"""

from __future__ import annotations

import cv2
import numpy as np

# ImageNet-статистика: backbone предобучен на ней, и своя нормировка сбила бы её.
# Живёт здесь, а не рядом с обучением: ровно так же патч нормируют экспорт
# и инференс, а тянуть ради двух чисел модуль обучающего датасета им незачем.
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226

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


def _spanning(
    binary: np.ndarray,
    frame_span_frac: float,
    thin_span_frac: float = 0.2,
    thin_thickness_frac: float = 0.005,
) -> np.ndarray:
    """Компоненты, похожие на линию: рамка, край листа, длинная линейка.

    Два признака, и второй важнее первого. «Бокс почти на всю сторону» ловит
    только полноразмерную рамку: замером край сканера тянулся на 75.8% высоты
    при пороге 80% и не опознавался, хотя это одна сплошная компонента толщиной
    в двенадцать пикселей.

    Поэтому линия распознаётся по форме: тонкая и длинная. Буква, рассечённая
    границей кадра, — компонента в десяток-другой пикселей по обеим осям, у неё
    таких пропорций не бывает.

    Порог толщины намеренно жёсткий — доля стороны, равная ширине приграничной
    полосы. При вдвое большем значении правило начинало съедать сами строки
    текста: на странице 2544 px «тонким» становилось всё до 25 px, а это ровно
    высота строки, и оценка числа строк падала вдвое.
    """
    height, width = binary.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros(binary.shape, dtype=bool)

    box_w = stats[:, cv2.CC_STAT_WIDTH]
    box_h = stats[:, cv2.CC_STAT_HEIGHT]
    spans = (box_w >= frame_span_frac * width) | (box_h >= frame_span_frac * height)

    limit = max(2.0, thin_thickness_frac * min(height, width))
    spans |= (box_h <= limit) & (box_w >= thin_span_frac * width)
    spans |= (box_w <= limit) & (box_h >= thin_span_frac * height)

    spans[0] = False  # фон
    return spans[labels]


def frame_component_mask(binary: np.ndarray, frame_span_frac: float = 0.8) -> np.ndarray:
    """Маска рамки сканера и края листа: всё, что тянется почти на всю сторону.

    Такая компонента — не текст, и оставлять её в анализе нельзя. Тёмная полоса
    вдоль края даёт чернила в каждой строке горизонтального профиля: порог «в строке
    есть текст» превышается везде, страница схлопывается в один сплошной участок,
    и оценка высоты строки возвращает высоту всего листа.

    Проверки «бокс почти на всю сторону» недостаточно, и это выяснилось замером.
    Край сканера на реальных страницах тянулся на 75.8% высоты при пороге 80% —
    одна сплошная компонента толщиной в двенадцать пикселей, которая не
    опознавалась и целиком засчитывалась как текст, рассечённый границей кадра.
    Отсюда `border_ink_frac` 0.63-0.84 на страницах без всякого обреза; это была
    главная причина ложных вердиктов `bad` на чистых сканах.

    Признак линии в `_spanning` устроен так, чтобы такой край ловился по форме,
    а не по длине. Сшивать разрывы морфологией пробовали — не годится: замыкание
    ядром в пару процентов стороны склеивает соседние буквы в одну длинную
    строку, и она опознаётся как рамка. Число найденных строк текста падало вдвое.
    """
    if binary.size == 0:
        return np.zeros_like(binary, dtype=bool)

    return _spanning(binary, frame_span_frac)


def local_std(gray: np.ndarray, window: int) -> np.ndarray:
    """σ в скользящем окне через E[x²] - E[x]²."""
    values = gray.astype(np.float64)
    size = max(3, window | 1)
    mean = cv2.blur(values, (size, size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(values * values, (size, size), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
