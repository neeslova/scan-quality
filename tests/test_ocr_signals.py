"""Сигналы здоровья текста: зацикливание, чужой алфавит, словарь, расхождение прогонов."""

from __future__ import annotations

import pytest

from src.ocr.signals import (
    foreign_char_ratio,
    max_repeat_run,
    ngram_repetition,
    normalized_distance,
    oov_ratio,
    tokenize,
)

CLEAN = (
    "Настоящий договор заключён между сторонами в целях выполнения работ "
    "по ремонту помещения согласно утверждённой смете и календарному плану "
    "работ на текущий календарный год с учётом всех дополнительных условий"
)

# Так выглядит выход генеративной модели, потерявшей опору в изображении.
LOOPED = CLEAN + " " + " ".join(["в целях выполнения работ по ремонту"] * 12)


def test_tokenize_drops_punctuation_and_case() -> None:
    assert tokenize("Договор, ДОГОВОР — договор!") == ["договор", "договор", "договор"]


def test_repetition_is_near_zero_on_normal_text() -> None:
    assert ngram_repetition(CLEAN) < 0.1


def test_repetition_rises_on_looped_output() -> None:
    """Главный признак вранья VLM: она ушла в повтор одной фразы."""
    assert ngram_repetition(LOOPED) > 0.5
    assert ngram_repetition(LOOPED) > ngram_repetition(CLEAN)


def test_repetition_counts_occurrences_not_unique_grams() -> None:
    """Фраза, повторённая много раз, должна дать высокую долю, а не одну N-ю."""
    text = " ".join(["альфа бета гамма дельта эпсилон"] * 20)
    assert ngram_repetition(text) > 0.9


def test_short_text_has_no_repetition() -> None:
    assert ngram_repetition("две строки") == 0.0


def test_max_repeat_run_finds_stuck_tail() -> None:
    assert max_repeat_run(LOOPED) > 20
    assert max_repeat_run(CLEAN) < max_repeat_run(LOOPED)


def test_max_repeat_run_separates_scattered_from_contiguous() -> None:
    """Рассыпанные дубли и залипший хвост — разные болезни, и меряются порознь.

    В обоих текстах доля повторяющихся n-грамм высока, но непрерывный повтор
    есть только во втором.
    """
    phrase = "один два три четыре пять"
    scattered = f"{phrase} шесть семь восемь девять десять {phrase}"
    contiguous = " ".join([phrase] * 6)

    assert max_repeat_run(contiguous) > max_repeat_run(scattered)


def test_foreign_chars_absent_in_expected_scripts() -> None:
    assert foreign_char_ratio(CLEAN, ["ru"]) == 0.0
    assert foreign_char_ratio("Order number 12345, dated 01.02.2024.", ["en"]) == 0.0


def test_foreign_chars_detected_in_wrong_script() -> None:
    """Кириллица под английским алфавитом — мусор, и наоборот."""
    assert foreign_char_ratio("Договор подписан", ["en"]) > 0.9
    assert foreign_char_ratio("значки ℘ □ ▩ ", ["ru"]) > 0.0


def test_digits_and_punctuation_are_not_foreign() -> None:
    """Иначе любая таблица выглядела бы мусором."""
    assert foreign_char_ratio("12/45-А (2024) — 87,5%", ["ru"]) == 0.0


def test_typographic_signs_are_allowed_only_explicitly() -> None:
    """`№` и `□` — одна юникод-категория, а смысл противоположный.

    Разрешить категорию целиком нельзя: тогда вместе с номером знака прошли бы
    и `□` с `℘`, которыми как раз сыплет сломавшийся OCR. Поэтому законные знаки
    перечисляются явно — тем же списком `ocr.extra_chars`, что и в конфиге.
    """
    assert foreign_char_ratio("№ 5", ["ru"]) > 0.0
    assert foreign_char_ratio("№ 5", ["ru"], extra_chars="№") == 0.0
    assert foreign_char_ratio("□ 5", ["ru"], extra_chars="№") > 0.0


def test_oov_ratio_uses_supplied_vocabulary() -> None:
    vocabulary = {"договор", "подписан", "сторонами"}
    assert oov_ratio("Договор подписан сторонами", vocabulary) == 0.0
    assert oov_ratio("Дгвр пдпсн стрнми", vocabulary) == 1.0


def test_oov_ratio_requires_vocabulary() -> None:
    """Молча подставленный чужой словарь дал бы правдоподобную чушь."""
    with pytest.raises(ValueError, match="словарь"):
        oov_ratio("любой текст")


def test_identical_readings_do_not_diverge() -> None:
    assert normalized_distance(CLEAN, CLEAN) == 0.0


def test_divergence_grows_with_disagreement() -> None:
    """Основной сигнал: два разрешения прочли страницу по-разному."""
    half = " ".join(CLEAN.split()[:12]) + " совершенно другие слова внезапно возникли тут"
    assert 0.0 < normalized_distance(CLEAN, half) < 1.0
    assert normalized_distance(CLEAN, "полностью иной набор слов") > 0.8


def test_empty_against_text_is_total_divergence() -> None:
    """В одном разрешении пусто, в другом страница текста — худший из признаков."""
    assert normalized_distance("", CLEAN) == 1.0
    assert normalized_distance(CLEAN, "") == 1.0


def test_both_empty_readings_agree() -> None:
    """Пустая страница, прочитанная одинаково пусто, — это согласие, а не провал."""
    assert normalized_distance("", "") == 0.0


def test_whitespace_differences_do_not_count() -> None:
    """Разрешения всегда расходятся в переносах — расхождение должно быть по словам."""
    assert normalized_distance(CLEAN, CLEAN.replace(" ", "\n")) == 0.0
