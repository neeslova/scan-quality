"""Сигналы по выгрузке DeepSeek-OCR: расчёт, сверка с эталоном, ранжирование."""

from __future__ import annotations

import json

from src.data.golden import write_golden
from src.ocr.deepseek_signals import (
    collect,
    format_report,
    join_with_golden,
    load_texts,
    page_signals,
    rank_signals,
)
from src.schema import GoldenRecord

CLEAN = (
    "Настоящий договор заключён между сторонами в целях выполнения работ "
    "по ремонту помещения согласно утверждённой смете и календарному плану"
)
LOOPED = "в целях выполнения работ " * 15


def _row(sha: str, tiny: str, base: str, page: int = 0, status: str = "ok") -> dict:
    return {
        "image": f"{sha}.png",
        "page": page,
        "sha256": sha,
        "texts": {"tiny": tiny, "base": base},
        "elapsed_s": {"tiny": 1.0, "base": 3.0},
        "status": status,
        "error": "",
    }


def test_signals_computed_for_each_mode() -> None:
    result = page_signals(_row("a", CLEAN, CLEAN))

    assert result.values["repetition_tiny"] == result.values["repetition_base"]
    assert result.values["words_base"] > 0
    # Прочтения совпали — расхождения нет.
    assert result.values["divergence"] == 0.0


def test_divergence_flags_unstable_page() -> None:
    """Разные прочтения одной страницы — признак того, что модель додумывала."""
    stable = page_signals(_row("a", CLEAN, CLEAN))
    unstable = page_signals(_row("b", CLEAN, LOOPED))

    assert unstable.values["divergence"] > stable.values["divergence"]
    assert unstable.values["repetition_base"] > unstable.values["repetition_tiny"]


def test_modes_are_read_from_the_record() -> None:
    """Прогон мог идти другой парой разрешений — разбор не должен падать."""
    row = {"image": "a.png", "page": 0, "sha256": "a", "texts": {"small": CLEAN}, "status": "ok"}
    values = page_signals(row).values

    assert "repetition_small" in values
    # Один режим — сравнивать не с чем, расхождение не считается.
    assert "divergence" not in values


def test_oov_counted_only_with_vocabulary() -> None:
    row = _row("a", CLEAN, CLEAN)
    assert "oov_tiny" not in page_signals(row).values
    assert "oov_tiny" in page_signals(row, vocabulary={"договор"}).values


def test_join_matches_by_hash_and_skips_failures(tmp_path) -> None:
    """Сверка идёт по хешу: имена файлов в корпусе повторяются, хеш — нет."""
    golden_path = tmp_path / "golden.jsonl"
    write_golden(
        [
            GoldenRecord(
                image="Good/a.png", page=0, document="a", corpus="tg", label="good", sha256="a"
            ),
            GoldenRecord(
                image="bad/b.png", page=0, document="b", corpus="tg", label="bad", sha256="b"
            ),
        ],
        golden_path,
    )

    signals = collect(
        [
            _row("a", CLEAN, CLEAN),
            _row("b", CLEAN, LOOPED),
            _row("c", CLEAN, CLEAN),  # нет в эталоне
            _row("d", "", "", status="failed"),  # не прочиталась
        ]
    )
    rows, labels = join_with_golden(signals, golden_path)

    assert len(rows) == 2
    assert labels == [0, 1]


def test_ranking_finds_the_separating_signal() -> None:
    """Сигнал, различающий классы, должен оказаться выше шумового.

    Проверка ровно на том, ради чего этап делается: расхождение прогонов растёт
    на плохих страницах, а посторонний сигнал не растёт.
    """
    rows = [{"divergence": 0.05, "noise": 0.5} for _ in range(15)]
    rows += [{"divergence": 0.80, "noise": 0.5} for _ in range(15)]
    labels = [0] * 15 + [1] * 15

    ranked = rank_signals(rows, labels)

    assert ranked[0][0] == "divergence"
    assert ranked[0][1] > 0.9
    # Постоянный сигнал не ранжируется вовсе: разделять им нечего.
    assert all(name != "noise" for name, _, _ in ranked)


def test_report_survives_single_class_golden() -> None:
    signals = collect([_row("a", CLEAN, CLEAN)])
    text = format_report(signals, [{"divergence": 0.1}], [1])

    assert "один класс" in text


def test_load_texts_skips_broken_lines(tmp_path) -> None:
    path = tmp_path / "texts.jsonl"
    path.write_text(
        json.dumps(_row("a", CLEAN, CLEAN), ensure_ascii=False) + "\n{ broken\n",
        encoding="utf-8",
    )

    assert len(load_texts(path)) == 1
