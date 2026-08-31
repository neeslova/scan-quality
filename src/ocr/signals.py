"""Сигналы «здоровья» распознанного текста — то, чем меряется VLM-OCR.

Обычный OCR (EasyOCR, Tesseract) отдаёт уверенность на каждое слово, и по ней
видно, где он поплыл. Генеративный OCR — DeepSeek-OCR и подобные — устроен иначе:
это языковая модель, и на нечитаемом входе она не понижает уверенность, а
**додумывает правдоподобный текст**. Уверенность декодера отражает предсказуемость
языка, а не разборчивость картинки: на размытом «Министерство финансов Российской»
модель уверенно допишет продолжение, потому что знает русский язык, а не потому
что что-то разглядела.

Поэтому здесь считаются признаки, которые ловят именно машинное враньё:

  * **зацикливание** — самый надёжный. Уйдя в повтор одной фразы, VLM показывает,
    что опоры в изображении не осталось и она генерирует из языковой модели.
    Считаем и долю повторяющихся n-грамм, и длину самого длинного повтора: первое
    ловит рассыпанные по тексту дубли, второе — классический «залипший» хвост;
  * **чужой алфавит** — доля символов вне ожидаемых письменностей;
  * **доля слов вне словаря** — распознанное не похоже на язык;
  * **расхождение двух прогонов** (self-consistency) — главный сигнал. Один и тот
    же скан читается в двух разрешениях, и выходы сравниваются. На хорошем скане
    они почти совпадают; на плохом модель каждый раз выдумывает своё, и
    расхождение велико. Ground truth для этого не нужен вовсе — именно поэтому
    сигнал применим ко всему корпусу, а не к размеченной его части.

Функции здесь чистые: на вход строки, на выход числа. Ничего не грузят и не
знают, какая модель дала текст, — это позволяет проверять их синтетикой и
переиспользовать с любым движком.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Optional

logger = logging.getLogger(__name__)

# Длина n-граммы для детекта зацикливания. Пять слов — компромисс: на трёх
# срабатывают обычные обороты («в соответствии с настоящим договором»), на семи
# теряются короткие циклы, которыми как раз залипает VLM.
DEFAULT_NGRAM = 5

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Ожидаемые письменности. Совпадают с теми, что различает `readability`.
_SCRIPT_RANGES = {
    "ru": ((0x0410, 0x044F), (0x0401, 0x0401), (0x0451, 0x0451)),
    "en": ((0x0041, 0x005A), (0x0061, 0x007A)),
}


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре. Пунктуация и разметка выбрасываются.

    Регистр снимается намеренно: повтор «Договор Договор договор» — это то же
    зацикливание, и различать его по заглавной букве смысла нет.
    """
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def ngram_repetition(text: str, n: int = DEFAULT_NGRAM) -> float:
    """Доля n-грамм, встретившихся в тексте больше одного раза.

    Считается по вхождениям, а не по уникальным n-граммам: фраза, повторённая
    сорок раз, должна дать долю около единицы, а не одну сороковую.
    """
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0.0

    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)


MAX_REPEAT_PERIOD = 60


def max_repeat_run(text: str, n: int = DEFAULT_NGRAM, max_period: int = MAX_REPEAT_PERIOD) -> int:
    """Длина самого длинного непрерывного повтора, в словах.

    Ищет «залипший» хвост: одна и та же фраза, идущая подряд много раз. Доля
    повторяющихся n-грамм такой хвост тоже увидит, но не отличит его от текста,
    где дубли рассыпаны по всей странице, — а это разные болезни.

    Период перебирается не до половины текста, а до `max_period`. Полный перебор
    квадратичен по длине, и на странице в пару тысяч слов это миллионы сравнений
    на пустом месте: залипания генеративных моделей — это циклы в несколько слов
    или одну фразу, а не в половину страницы.
    """
    tokens = tokenize(text)
    if len(tokens) < 2 * n:
        return 0

    best = 0
    for period in range(1, min(max_period, len(tokens) // 2) + 1):
        run = 0
        for index in range(len(tokens) - period):
            if tokens[index] == tokens[index + period]:
                run += 1
                best = max(best, run + period)
            else:
                run = 0
    return min(best, len(tokens))


def foreign_char_ratio(text: str, languages: Sequence[str], extra_chars: str = "") -> float:
    """Доля значащих символов вне ожидаемых письменностей.

    Пробелы не в счёт: иначе доля зависела бы от того, как модель расставила
    переносы. Цифры и пунктуация задаются через `extra_chars` — они законны в
    любом языке, и без них счёт мусора был бы завышен на любой таблице.
    """
    ranges: list[tuple[int, int]] = []
    for code in languages:
        ranges.extend(_SCRIPT_RANGES.get(code, ()))
    allowed = set(extra_chars)

    significant = 0
    foreign = 0
    for char in text:
        if char.isspace():
            continue
        significant += 1
        if char in allowed or unicodedata.category(char).startswith(("N", "P")):
            continue
        code = ord(char)
        if any(low <= code <= high for low, high in ranges):
            continue
        foreign += 1

    return foreign / significant if significant else 0.0


def oov_ratio(text: str, vocabulary: Optional[set[str]] = None, min_length: int = 3) -> float:
    """Доля слов вне словаря.

    Словарь обязателен и передаётся снаружи. Молча подставленный словарь не того
    языка дал бы правдоподобную, но бессмысленную цифру, поэтому его отсутствие —
    ошибка, а не повод вернуть ноль.

    Короткие слова пропускаются: предлоги и обрывки в словаре есть не всегда,
    а веса они не несут.
    """
    if vocabulary is None:
        raise ValueError("oov_ratio требует словарь: без него цифра не имеет смысла")

    tokens = [t for t in tokenize(text) if len(t) >= min_length and not t.isdigit()]
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token not in vocabulary) / len(tokens)


def _levenshtein(left: Sequence, right: Sequence) -> int:
    """Расстояние Левенштейна на двух строках матрицы.

    Своя реализация, а не библиотека: считать её нужно один раз на страницу по
    коротким последовательностям, и тянуть ради этого зависимость в базовую
    установку не стоит. Если `rapidfuzz` в окружении есть, используется он —
    на длинных текстах он на порядок быстрее.
    """
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # удаление
                    current[j - 1] + 1,  # вставка
                    previous[j - 1] + (left_item != right_item),  # замена
                )
            )
        previous = current
    return previous[-1]


def normalized_distance(left: str, right: str, by_words: bool = True) -> float:
    """Расхождение двух прочтений одной страницы, 0..1.

    Ноль — тексты совпали, единица — не совпало ничего. Нормируется на длину
    большей последовательности, поэтому величина сравнима между страницами
    разного объёма.

    По умолчанию считается **по словам, а не по символам**. Причина не в
    скорости: посимвольное расстояние штрафует за расхождения в пробелах,
    переносах и раскладке таблиц, которых у двух разрешений всегда полно, и
    ровный скан из-за них получал бы заметное расхождение на пустом месте.
    Нас же интересует, совпал ли ПРОЧИТАННЫЙ ТЕКСТ.

    Две пустые строки считаются совпавшими (0.0), а пустая с непустой —
    полным расхождением: если в одном разрешении модель не увидела ничего,
    а в другом увидела страницу текста, это максимально плохой признак.
    """
    left_seq: Sequence = tokenize(left) if by_words else left
    right_seq: Sequence = tokenize(right) if by_words else right

    if not left_seq and not right_seq:
        return 0.0
    longest = max(len(left_seq), len(right_seq))
    if not left_seq or not right_seq:
        return 1.0

    try:
        from rapidfuzz.distance import Levenshtein

        distance = Levenshtein.distance(left_seq, right_seq)
    except ImportError:
        distance = _levenshtein(left_seq, right_seq)

    return min(1.0, distance / longest)
