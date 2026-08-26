"""Загрузка сканов: jpg/png/tiff/pdf -> grayscale ndarray, нормализация к target_dpi.

Нормализация dpi обязательна: почти все метрики (резкость, высота строки, шум)
зависят от масштаба, и без приведения к общему dpi их пороги несравнимы между сканами.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
PDF_SUFFIXES = frozenset({".pdf"})

# Длинная сторона A4 в дюймах — из неё выводим dpi, если метаданных нет.
A4_LONG_SIDE_INCHES = 297.0 / 25.4


@dataclass(frozen=True)
class LoadedPage:
    """Одна страница, приведённая к рабочему dpi."""

    gray: np.ndarray  # uint8, HxW
    dpi: float
    source: Path
    page: int = 0
    original_size: tuple[int, int] = (0, 0)  # (width, height) до масштабирования

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])


def _imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread не открывает пути с кириллицей на Windows — читаем через numpy."""
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"Пустой файл: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return image


def _read_dpi(path: Path) -> Optional[float]:
    """dpi из метаданных файла; None — если их нет или они бессмысленны."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None

    try:
        with Image.open(path) as img:
            dpi = img.info.get("dpi")
    except Exception as exc:  # noqa: BLE001 — битые метаданные не повод падать
        logger.debug("dpi не прочитан из %s: %s", path, exc)
        return None

    if not dpi:
        return None
    value = float(dpi[0])
    # Некоторые сканеры пишут 1 или 72 «по умолчанию» — это не настоящий dpi.
    return value if value > 72.0 else None


def _guess_dpi(height: int, width: int, fallback: str) -> Optional[float]:
    if fallback != "a4":
        return None
    return max(height, width) / A4_LONG_SIDE_INCHES


def _rescale(gray: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.02:  # разница меньше 2% — не трогаем
        return gray
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    new_size = (max(1, round(gray.shape[1] * factor)), max(1, round(gray.shape[0] * factor)))
    return cv2.resize(gray, new_size, interpolation=interpolation)


def _normalize_dpi(
    gray: np.ndarray,
    source_dpi: Optional[float],
    target_dpi: Optional[int],
    allow_upscale: bool = False,
) -> tuple[np.ndarray, float]:
    """Приводит страницу к целевому dpi. По умолчанию — только вниз.

    Апскейл запрещён не из экономии: растягивание скана 200 dpi до 300 кубической
    интерполяцией сглаживает ступенчатые края букв, и метрики резкости честно
    показывают размытие, которого в исходнике не было. Дефект создавался бы самим
    загрузчиком. Скан ниже целевого dpi остаётся в родном разрешении, а его
    настоящий dpi возвращается наружу — метрики масштаба опираются на него.
    """
    if target_dpi is None or source_dpi is None:
        return gray, float(source_dpi or 0.0)

    factor = target_dpi / source_dpi
    if factor > 1.0 and not allow_upscale:
        logger.debug(
            "dpi %.0f ниже целевого %d — оставляем родное разрешение", source_dpi, target_dpi
        )
        return gray, float(source_dpi)
    return _rescale(gray, factor), float(target_dpi)


def _load_pdf_pages(
    path: Path, target_dpi: int, page: Optional[int]
) -> list[tuple[np.ndarray, float, int, tuple[int, int]]]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Для чтения PDF нужен PyMuPDF: pip install pymupdf") from exc

    pages: list[tuple[np.ndarray, float, int, tuple[int, int]]] = []
    with fitz.open(str(path)) as doc:
        indices = range(doc.page_count) if page is None else [page]
        for index in indices:
            # PDF векторный — рендерим сразу в нужном dpi, пересэмплировать нечего.
            pixmap = doc[index].get_pixmap(dpi=target_dpi, colorspace=fitz.csGRAY)
            gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width
            )
            pages.append((gray.copy(), float(target_dpi), index, (pixmap.width, pixmap.height)))
    return pages


def load_pages(
    path: Union[str, Path],
    target_dpi: Optional[int] = 300,
    dpi_fallback: str = "a4",
    allow_upscale: bool = False,
) -> list[LoadedPage]:
    """Читает файл целиком: картинка -> одна страница, PDF -> все страницы."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Файл не найден: {src}")

    suffix = src.suffix.lower()
    if suffix in PDF_SUFFIXES:
        if target_dpi is None:
            raise ValueError("Для PDF нужен target_dpi: рендер идёт сразу в нужном разрешении")
        return [
            LoadedPage(gray=gray, dpi=dpi, source=src, page=index, original_size=size)
            for gray, dpi, index, size in _load_pdf_pages(src, target_dpi, page=None)
        ]

    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"Неподдерживаемый формат: {suffix} ({src.name})")

    gray = _imread_unicode(src)
    original_size = (gray.shape[1], gray.shape[0])
    source_dpi = _read_dpi(src) or _guess_dpi(gray.shape[0], gray.shape[1], dpi_fallback)
    gray, dpi = _normalize_dpi(gray, source_dpi, target_dpi, allow_upscale)

    logger.debug(
        "%s: %dx%d -> %dx%d при %.0f dpi",
        src.name,
        original_size[0],
        original_size[1],
        gray.shape[1],
        gray.shape[0],
        dpi,
    )
    return [LoadedPage(gray=gray, dpi=dpi, source=src, page=0, original_size=original_size)]


def load_page(
    path: Union[str, Path],
    target_dpi: Optional[int] = 300,
    dpi_fallback: str = "a4",
    page: int = 0,
    allow_upscale: bool = False,
) -> LoadedPage:
    """Одна страница. Для многостраничных PDF — по индексу."""
    src = Path(path)
    if src.suffix.lower() in PDF_SUFFIXES:
        if target_dpi is None:
            raise ValueError("Для PDF нужен target_dpi")
        rendered = _load_pdf_pages(src, target_dpi, page=page)
        gray, dpi, index, size = rendered[0]
        return LoadedPage(gray=gray, dpi=dpi, source=src, page=index, original_size=size)

    pages = load_pages(
        src, target_dpi=target_dpi, dpi_fallback=dpi_fallback, allow_upscale=allow_upscale
    )
    return pages[page]
