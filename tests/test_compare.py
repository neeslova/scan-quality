"""Сравнение трёх систем. Главное — таблица не льстит тому, кто не отвечает."""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.models.compare import PageRow, format_ap_table, metrics_for, worst_errors


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def page(image: str, truth: dict, cv: dict, cnn: dict, verdict: str = "bad") -> PageRow:
    """Гибрид собирается по источникам, как в конфиге: cv своё, cnn своё."""
    hybrid = {**{k: v for k, v in cv.items() if k in ("blur", "skew")}, **cnn}
    return PageRow(
        image=image,
        truth=truth,
        cv=cv,
        cnn=cnn,
        hybrid=hybrid,
        hybrid_raw=hybrid,
        verdict=verdict,
    )


# --- какие страницы попадают в выборку метки --------------------------------


def test_label_is_scored_only_where_the_source_answered(config: Config) -> None:
    """На битональном скане CV не измеряет шум вовсе (решение №21).

    Подставить туда ноль значило бы засчитать «не мерил» за «дефекта нет» —
    и метрика вышла бы лучше, чем система заслуживает.
    """
    rows = [
        page("a.jpg", {"noise": True}, {"blur": 0.2}, {"noise": 0.9}),
        page("b.jpg", {"noise": True}, {"blur": 0.3}, {}),  # сеть промолчала
    ]

    item = metrics_for(rows, "cnn", "noise", config)

    assert item is not None
    assert item.support == 1  # вторая страница в выборку не попала


def test_label_nobody_answered_is_none(config: Config) -> None:
    rows = [page("a.jpg", {"noise": True}, {"blur": 0.2}, {})]
    assert metrics_for(rows, "cnn", "noise", config) is None


# --- два макро-средних ------------------------------------------------------


def test_declining_a_hard_label_does_not_inflate_the_macro(config: Config) -> None:
    """Ловушка, ради которой в таблице два средних.

    Система, которая отказывается от трудной метки, получила бы завышенное
    среднее по остальным. «macro, все» засчитывает отказ на уровне случайного
    угадывания — то есть ровно тем, чего он стоит.
    """
    rows = []
    for index in range(10):
        has_noise = index < 5
        rows.append(
            page(
                f"{index}.jpg",
                {"blur": index < 5, "noise": has_noise},
                {"blur": 0.9 if index < 5 else 0.1},  # CV отвечает уверенно
                {"blur": 0.5, "noise": 0.9 if has_noise else 0.1},
            )
        )

    table = format_ap_table(rows, config)

    assert "macro, общие" in table
    assert "macro, все" in table
    # CV не выдал noise -> в строке метки у него прочерк.
    noise_row = next(line for line in table.splitlines() if line.startswith("noise"))
    assert "—" in noise_row


# --- худшие ошибки ----------------------------------------------------------


def test_a_miss_outranks_a_false_alarm(config: Config) -> None:
    """Пропустить плохой скан дороже, чем отклонить хороший (раздел 4).

    Разбор ошибок обязан ставить пропуски выше ложных тревог той же величины,
    иначе двадцать «худших» окажутся списком безобидных перестраховок.
    """
    missed = page("missed.jpg", {"blur": True}, {"blur": 0.05}, {}, verdict="good")
    false_alarm = page("alarm.jpg", {"blur": False}, {"blur": 0.95}, {}, verdict="bad")

    ranked = worst_errors([false_alarm, missed], config, count=2)

    assert ranked[0][0].image == "missed.jpg"
    assert "пропуск" in ranked[0][1]
    assert "ложная тревога" in ranked[1][1]


def test_correct_pages_are_not_listed(config: Config) -> None:
    """Список ошибок должен состоять из ошибок."""
    good = page("clean.jpg", {"blur": False}, {"blur": 0.02}, {}, verdict="good")
    assert worst_errors([good], config) == []
