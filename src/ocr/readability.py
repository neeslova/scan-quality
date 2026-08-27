"""Читаемость страницы по результату OCR и вывод метки `unreadable`.

`unreadable` — не отдельный дефект, а следствие: скан плох ровно настолько,
насколько плохо с него снимается текст. Поэтому метка выводится, а не размечается
руками — так у неё объективный ground truth вместо субъективной оценки.

Три сигнала, потому что поодиночке каждый обманывается:
  - **средняя уверенность** падает на плохом скане, но движок бывает уверенно
    неправ: на грязи он «узнаёт» символы и ставит им высокий балл;
  - **доля мусорных символов** ловит именно этот случай — распознанное не похоже
    на язык. Но у EasyOCR распознаватель имеет фиксированный алфавит и физически
    не может выдать символ вне языка: на пробном прогоне по Tobacco эта доля была
    ровно 0.00 на всех страницах, то есть сигнал мёртв. С Tesseract он работает;
  - **доля неправдоподобных слов** — тот же смысл, но выживает при закрытом
    алфавите: «rn1lI» состоит из легальных букв и всё равно не слово.
Берём худший из трёх.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Sequence
from typing import Optional

import numpy as np

from src.config import Config
from src.metrics.baseline import score_from_anchors
from src.ocr.engine import OCRWord
from src.schema import OCRResult

logger = logging.getLogger(__name__)

# Какие юникод-категории считаем буквами языка.
_LETTER_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo"})
# Диапазоны, которые для наших корпусов являются осмысленными буквами.
_SCRIPT_RANGES = {
    "en": ((0x0041, 0x005A), (0x0061, 0x007A)),
    "ru": ((0x0410, 0x044F), (0x0401, 0x0401), (0x0451, 0x0451)),
}


def _expected_letters(languages: Sequence[str]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for code in languages:
        ranges.extend(_SCRIPT_RANGES.get(code, ()))
    return tuple(ranges)


def garbage_ratio(text: str, languages: Sequence[str], extra_chars: str) -> float:
    """Доля символов, которых в этом языке быть не должно.

    Считаем только по «значащим» символам: пробелы и переводы строк не в счёт,
    иначе доля мусора зависела бы от того, как движок расставил пробелы.
    """
    ranges = _expected_letters(languages)
    allowed = set(extra_chars)

    significant = 0
    garbage = 0
    for char in text:
        if char.isspace():
            continue
        significant += 1
        if char in allowed:
            continue
        code = ord(char)
        if any(low <= code <= high for low, high in ranges):
            continue
        # Буква чужого алфавита — тоже мусор, но помечаем отдельно от знаков:
        # на смешанных документах это ожидаемо, а вот «℘» или «□» — нет.
        if unicodedata.category(char) in _LETTER_CATEGORIES and not ranges:
            continue
        garbage += 1

    return garbage / significant if significant else 0.0


_VOWELS = {"en": "aeiouy", "ru": "аеёиоуыэюя"}
_PUNCT_ONLY_MAX = 2


def _vowels_for(languages: Sequence[str]) -> set[str]:
    return {ch for code in languages for ch in _VOWELS.get(code, "")}


def _is_plausible_word(token: str, vowels: set[str]) -> bool:
    if token.isdigit():
        return True

    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        # Чистая пунктуация правдоподобна только короткая: «—», «».», но не «;;;/..»
        return len(token) <= _PUNCT_ONLY_MAX
    if len(letters) >= 3 and vowels and not any(ch.lower() in vowels for ch in letters):
        return False
    # Смесь букв и цифр внутри одного слова — типичный вид распознанной грязи.
    if len(letters) >= 2 and any(ch.isdigit() for ch in token):
        return False
    return True


def nonword_ratio(words: Sequence[OCRWord], languages: Sequence[str]) -> float:
    """Доля токенов, не похожих на слова языка.

    Нужна потому, что доля мусорных СИМВОЛОВ бесполезна с движком, у которого
    закрытый алфавит: он не может выдать ничего вне языка, и метрика всегда ноль.
    Правдоподобность слова этим не ограничена — «rn1lI» собран из легальных букв.
    """
    tokens = [
        token.strip(".,;:!?()[]{}«»\"'-–—/\\") for word in words for token in word.text.split()
    ]
    tokens = [token for token in tokens if token]
    if not tokens:
        return 0.0

    vowels = _vowels_for(languages)
    bad = sum(1 for token in tokens if not _is_plausible_word(token, vowels))
    return bad / len(tokens)


def readable_share(words: Sequence[OCRWord], config: Config) -> float:
    """Доля текста страницы, прочитанная уверенно и осмысленно.

    Отвечает на вопрос «сколько текста я могу прочесть», а не «есть ли на
    странице испорченное место», и в этом вся разница. Печать поверх текста,
    подпись, штамп, вклейка — локальные помехи: они делают нечитаемым свой
    участок, а остальная страница читается как обычно. Прежние сигналы считали
    долю плохих токенов по всей странице, и печать, испортившая треть слов,
    объявляла нечитаемым весь скан.

    Здесь доля считается по ПЛОЩАДИ распознанных блоков, а не по их числу:
    подпись из пяти неразборчивых росчерков не должна весить столько же,
    сколько пять абзацев ровного текста.
    """
    if not words:
        return 0.0

    vowels = _vowels_for(config.ocr.languages)
    total = sum(max(w.area, 1) for w in words)
    good = sum(
        max(w.area, 1)
        for w in words
        if w.confidence >= config.ocr.min_confidence
        and any(
            _is_plausible_word(t.strip(".,;:!?()[]{}«»\"'-–—/\\"), vowels) for t in w.text.split()
        )
    )
    return good / total if total else 0.0


def analyze_words(
    words: Sequence[OCRWord],
    config: Config,
    page_area: int,
    engine_name: str,
) -> OCRResult:
    """Сводка по распознанному тексту страницы."""
    cfg = config.ocr
    confident = [w for w in words if w.confidence >= cfg.min_confidence]

    # Средняя уверенность — по уверенно распознанным словам: обрывки из пустых
    # полей иначе тянут её вниз на любом, даже отличном скане.
    mean_confidence = float(np.mean([w.confidence for w in confident])) if confident else 0.0
    text = " ".join(w.text for w in words)
    density = sum(w.area for w in words) / page_area if page_area else 0.0

    return OCRResult(
        engine=engine_name,
        mean_confidence=round(min(1.0, max(0.0, mean_confidence)), 4),
        garbage_ratio=round(garbage_ratio(text, cfg.languages, cfg.extra_chars), 4),
        nonword_ratio=round(nonword_ratio(words, cfg.languages), 4),
        readable_share=round(readable_share(words, config), 4),
        text_density=round(min(1.0, density), 4),
        n_boxes=len(words),
    )


def unreadable_score(result: OCRResult, config: Config) -> Optional[float]:
    """Метка `unreadable` из трёх сигналов, берётся худший. None — если считать не по чему.

    Пустая страница не является нечитаемой: если распознанных блоков почти нет,
    честнее не выдавать метку вовсе, чем объявить чистый лист худшим дефектом.
    """
    if result.n_boxes < config.ocr.min_boxes:
        return None

    anchors = config.ocr.unreadable
    signals = {
        "mean_confidence": result.mean_confidence,
        "garbage_ratio": result.garbage_ratio,
        "nonword_ratio": result.nonword_ratio,
        "readable_share": result.readable_share,
    }
    scores = [
        score_from_anchors(signals[name], pair.good, pair.bad)
        for name, pair in anchors.items()
        if name in signals
    ]
    return round(max(scores), 4)
