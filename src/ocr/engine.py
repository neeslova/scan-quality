"""Обёртка над OCR: EasyOCR как основной движок, Tesseract как запасной.

Наружу оба отдают один и тот же список слов с уверенностью, чтобы `readability`
не знал, кто именно распознавал. Это важно не для красоты: корпуса разноязычные,
и движок будет меняться вместе с корпусом.

Модель грузится лениво и один раз на процесс — инициализация EasyOCR занимает
несколько секунд, и делать её на каждой странице нельзя.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)


class OCREngine(Protocol):
    name: str

    def read(self, gray: np.ndarray) -> list[OCRWord]:
        """Распознаёт страницу и возвращает слова с уверенностью."""


def downscale(gray: np.ndarray, work_side: int) -> np.ndarray:
    """Уменьшает страницу до рабочего размера.

    На 300 dpi OCR не точнее, чем на ~150, а времени ест втрое больше: движки
    внутри всё равно приводят строки к своей высоте.
    """
    longest = max(gray.shape)
    if longest <= work_side:
        return gray
    scale = work_side / longest
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class EasyOCREngine:
    """EasyOCR. Тянет torch (CPU-версия), модель скачивается при первом запуске."""

    name = "easyocr"

    def __init__(self, languages: Sequence[str], gpu: bool = False) -> None:
        self._languages = list(languages)
        self._gpu = gpu
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover — зависит от окружения
                raise SystemExit(
                    "EasyOCR не установлен: py -m pip install easyocr\n"
                    "Либо переключите ocr.engine на tesseract."
                ) from exc
            logger.info("Загрузка EasyOCR (%s)...", ", ".join(self._languages))
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
        return self._reader

    def read(self, gray: np.ndarray) -> list[OCRWord]:
        reader = self._ensure_reader()
        words: list[OCRWord] = []
        for box, text, confidence in reader.readtext(gray):
            xs = [int(point[0]) for point in box]
            ys = [int(point[1]) for point in box]
            words.append(
                OCRWord(
                    text=str(text),
                    confidence=float(confidence),
                    box=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return words


class TesseractEngine:
    """Tesseract через pytesseract. Требует системной установки самого Tesseract."""

    name = "tesseract"

    # Коды языков у Tesseract свои.
    _LANG_MAP = {"en": "eng", "ru": "rus"}

    def __init__(self, languages: Sequence[str]) -> None:
        self._lang = "+".join(self._LANG_MAP.get(code, code) for code in languages)

    def read(self, gray: np.ndarray) -> list[OCRWord]:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pytesseract не установлен: py -m pip install pytesseract") from exc

        data = pytesseract.image_to_data(gray, lang=self._lang, output_type=pytesseract.Output.DICT)
        words: list[OCRWord] = []
        for index, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            # Tesseract отдаёт уверенность в процентах, -1 для служебных строк.
            confidence = float(data["conf"][index])
            if confidence < 0:
                continue
            x, y = int(data["left"][index]), int(data["top"][index])
            w, h = int(data["width"][index]), int(data["height"][index])
            words.append(
                OCRWord(text=text, confidence=confidence / 100.0, box=(x, y, x + w, y + h))
            )
        return words


def get_engine(name: str, languages: Sequence[str], gpu: bool = False) -> OCREngine:
    if name == "easyocr":
        return EasyOCREngine(languages, gpu=gpu)
    if name == "tesseract":
        return TesseractEngine(languages)
    raise ValueError(f"Неизвестный движок OCR: {name}")


_CACHE: dict[tuple[str, tuple[str, ...], bool], OCREngine] = {}


def shared_engine(name: str, languages: Sequence[str], gpu: bool = False) -> OCREngine:
    """Один экземпляр движка на процесс: инициализация стоит секунды."""
    key = (name, tuple(languages), gpu)
    if key not in _CACHE:
        _CACHE[key] = get_engine(name, languages, gpu)
    return _CACHE[key]


def read_page(
    gray: np.ndarray,
    engine: OCREngine,
    work_side: Optional[int] = None,
) -> list[OCRWord]:
    """Распознаёт страницу, при необходимости уменьшив её."""
    if gray.size == 0:
        return []
    prepared = downscale(gray, work_side) if work_side else gray
    try:
        return engine.read(prepared)
    except Exception as exc:  # noqa: BLE001 — одна страница не должна ронять прогон
        logger.warning("OCR не справился: %s", exc)
        return []
