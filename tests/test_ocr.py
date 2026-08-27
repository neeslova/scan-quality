"""OCR-слой: доля мусора, сводка по словам, вывод метки unreadable.

Сам движок здесь не запускается — он медленный и тянет модель. Проверяется логика
поверх его результата, а она и есть содержательная часть метки `unreadable`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config
from src.ocr.engine import OCRWord, TesseractEngine, downscale, get_engine
from src.ocr.readability import analyze_words, garbage_ratio, nonword_ratio, unreadable_score
from src.schema import OCRResult


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def word(text: str, confidence: float = 0.9, size: int = 100) -> OCRWord:
    return OCRWord(text=text, confidence=confidence, box=(0, 0, size, size))


# --- доля мусора ------------------------------------------------------------


def test_clean_english_has_no_garbage(config: Config) -> None:
    text = "The Council for Tobacco Research, U.S.A., Inc. (1996)"
    assert garbage_ratio(text, ["en"], config.ocr.extra_chars) == 0.0


def test_broken_symbols_are_garbage(config: Config) -> None:
    ratio = garbage_ratio("Th□ C℘un¤il ▪▪ ﬗ Ⴟ", ["en"], config.ocr.extra_chars)
    assert ratio > 0.3


def test_cyrillic_is_garbage_for_english_but_not_for_russian(config: Config) -> None:
    text = "Совет по исследованиям"
    assert garbage_ratio(text, ["en"], config.ocr.extra_chars) > 0.9
    assert garbage_ratio(text, ["ru"], config.ocr.extra_chars) == 0.0


def test_whitespace_does_not_affect_ratio(config: Config) -> None:
    """Иначе доля мусора зависела бы от того, как движок расставил пробелы."""
    tight = garbage_ratio("abc□", ["en"], config.ocr.extra_chars)
    loose = garbage_ratio("a b c\n\n□   ", ["en"], config.ocr.extra_chars)
    assert tight == pytest.approx(loose)


def test_empty_text(config: Config) -> None:
    assert garbage_ratio("", ["en"], config.ocr.extra_chars) == 0.0
    assert garbage_ratio("   \n ", ["en"], config.ocr.extra_chars) == 0.0


# --- сводка по словам -------------------------------------------------------


def test_low_confidence_words_excluded_from_mean(config: Config) -> None:
    """Обрывки из пустых полей не должны топить среднюю уверенность."""
    words = [word("Dear", 0.95), word("Sir", 0.92), word("~", 0.05), word("''", 0.02)]
    result = analyze_words(words, config, page_area=1_000_000, engine_name="test")

    assert result.mean_confidence == pytest.approx(0.935, abs=0.01)
    assert result.n_boxes == 4  # в счётчик попадают все


def test_no_confident_words_gives_zero(config: Config) -> None:
    result = analyze_words([word("~", 0.01)], config, page_area=1000, engine_name="test")
    assert result.mean_confidence == 0.0


def test_text_density_is_bounded(config: Config) -> None:
    words = [word("x", 0.9, size=2000) for _ in range(10)]
    result = analyze_words(words, config, page_area=1000, engine_name="test")
    assert result.text_density == 1.0


def test_empty_page(config: Config) -> None:
    result = analyze_words([], config, page_area=1000, engine_name="test")
    assert result.n_boxes == 0
    assert result.mean_confidence == 0.0
    assert result.garbage_ratio == 0.0


# --- метка unreadable -------------------------------------------------------


def make_result(
    confidence: float,
    garbage: float,
    boxes: int = 50,
    nonword: float = 0.0,
    readable: float = 1.0,
) -> OCRResult:
    return OCRResult(
        engine="test",
        mean_confidence=confidence,
        garbage_ratio=garbage,
        nonword_ratio=nonword,
        readable_share=readable,
        text_density=0.2,
        n_boxes=boxes,
    )


# --- доля неправдоподобных слов --------------------------------------------


def test_real_words_are_plausible() -> None:
    words = [word(t) for t in "Dear Sir we have reviewed your application 1996".split()]
    assert nonword_ratio(words, ["en"]) == 0.0


def test_confident_nonsense_without_foreign_symbols() -> None:
    """Ключевой случай: EasyOCR не может выдать символ вне алфавита.

    «rn1lI» состоит из легальных латинских букв, доля мусорных СИМВОЛОВ на нём
    равна нулю — ловит только правдоподобность слова.
    """
    words = [word(t) for t in ["rn1lI", "hjkl", "vvvv", "t3st", "wrtz"]]
    assert nonword_ratio(words, ["en"]) > 0.7


def test_digits_and_short_punctuation_are_fine() -> None:
    words = [word(t) for t in ["1996", "42", "-", "—"]]
    assert nonword_ratio(words, ["en"]) == 0.0


def test_nonword_ratio_on_empty() -> None:
    assert nonword_ratio([], ["en"]) == 0.0
    assert nonword_ratio([word("   ")], ["en"]) == 0.0


def test_nonword_signal_reaches_unreadable(config: Config) -> None:
    """Без этого сигнала уверенная белиберда с EasyOCR прошла бы как хороший скан."""
    clean = unreadable_score(make_result(0.90, 0.0, nonword=0.05), config)
    nonsense = unreadable_score(make_result(0.90, 0.0, nonword=0.60), config)
    total = unreadable_score(make_result(0.90, 0.0, nonword=0.95), config)

    assert clean == 0.0
    assert nonsense > clean
    assert total == 1.0


def test_local_damage_does_not_make_the_page_unreadable(config: Config) -> None:
    """Печать поверх текста, подпись, штамп — локальные помехи.

    Они делают нечитаемым свой участок, а не страницу. До появления
    `readable_share` печать, испортившая треть слов, уводила скан в `bad`:
    доля плохих токенов считалась по всей странице и объявлялась её свойством.
    """
    stamped = unreadable_score(make_result(0.88, 0.0, nonword=0.30, readable=0.75), config)
    assert stamped < config.verdict.tau_unreadable


def test_page_that_truly_cannot_be_read_is_flagged(config: Config) -> None:
    """Обратная проверка: смягчение не должно обезвредить саму метку."""
    ruined = unreadable_score(make_result(0.40, 0.0, nonword=0.70, readable=0.12), config)
    assert ruined > config.verdict.tau_unreadable


def test_good_scan_is_readable(config: Config) -> None:
    assert unreadable_score(make_result(0.92, 0.01), config) == 0.0


def test_low_confidence_makes_unreadable(config: Config) -> None:
    assert unreadable_score(make_result(0.30, 0.01), config) == 1.0


def test_confident_nonsense_is_caught_by_garbage(config: Config) -> None:
    """Ключевой случай: движок уверенно распознаёт кракозябры из грязи.

    По одной уверенности такой скан выглядел бы отличным — ловит доля мусора.
    """
    score = unreadable_score(make_result(0.90, 0.50), config)
    assert score == 1.0


def test_worst_of_two_signals_wins(config: Config) -> None:
    both_mild = unreadable_score(make_result(0.60, 0.22), config)
    assert 0.0 < both_mild < 1.0


def test_empty_page_is_not_unreadable(config: Config) -> None:
    """Чистый лист не нечитаем: распознавать нечего, а не плохо."""
    assert unreadable_score(make_result(0.0, 0.0, boxes=0), config) is None
    assert unreadable_score(make_result(0.0, 0.0, boxes=2), config) is None


# --- движок -----------------------------------------------------------------


def test_downscale_only_shrinks() -> None:
    big = np.zeros((3000, 2000), dtype=np.uint8)
    small = np.zeros((400, 300), dtype=np.uint8)

    assert max(downscale(big, 1600).shape) == 1600
    assert downscale(small, 1600).shape == small.shape


def test_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестный движок"):
        get_engine("magic", ["en"])


def test_tesseract_language_codes() -> None:
    """У Tesseract свои коды языков — обёртка обязана переводить."""
    assert TesseractEngine(["ru"])._lang == "rus"
    assert TesseractEngine(["en", "ru"])._lang == "eng+rus"


def test_word_area() -> None:
    assert OCRWord(text="a", confidence=1.0, box=(10, 20, 30, 50)).area == 20 * 30
    assert OCRWord(text="a", confidence=1.0, box=(30, 50, 10, 20)).area == 0
