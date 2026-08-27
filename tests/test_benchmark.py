"""Замер времени. Проверяется не скорость, а честность самого замера."""

from __future__ import annotations

import pytest

from src.models.benchmark import Timing, format_table, time_each


def test_every_item_gives_its_own_sample() -> None:
    """По образцу на элемент, а не один на всю пачку.

    Пачкой медиана, минимум и максимум выродились бы в одно число, а разброс
    между страницами — это ровно то, что интересно: страницы разного размера
    дают разное число патчей.
    """
    seen = []
    timing = time_each("шаг", seen.append, [1, 2, 3])

    assert len(timing.samples) == 3


def test_warm_up_call_is_not_counted() -> None:
    """В первый вызов попадает подъём сессии onnxruntime — он не про страницу."""
    calls = []
    timing = time_each("шаг", calls.append, ["a", "b"])

    assert calls == ["a", "a", "b"]  # прогрев на первом, потом замеры
    assert len(timing.samples) == 2


def test_summary_values_come_from_the_samples() -> None:
    timing = Timing(name="слой", samples=[10.0, 30.0, 20.0])

    assert timing.median == pytest.approx(20.0)
    assert timing.low == pytest.approx(10.0)
    assert timing.high == pytest.approx(30.0)


def test_table_shows_every_layer() -> None:
    """Одно число «2.4 с на страницу» не говорит, за что заплачено."""
    table = format_table(
        [Timing("CV-метрики", [1137.0]), Timing("сеть (onnxruntime)", [1410.0])], pages=12
    )

    assert "CV-метрики" in table
    assert "сеть (onnxruntime)" in table
    assert "12" in table
