"""Пакетная обработка папки. Главное — таблица не врёт про неизмеренное."""

from __future__ import annotations

import pytest

from src.cli import csv_header, csv_row, find_scans
from src.config import Config, load_config
from src.schema import DefectScore, QualityReport
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def test_finds_images_and_pdfs_recursively(tmp_path) -> None:
    fx.save(fx.text_page(width=60, height=80), tmp_path / "a.png", dpi=300)
    (tmp_path / "вложенная").mkdir()
    fx.save(fx.text_page(width=60, height=80), tmp_path / "вложенная" / "b.jpg", dpi=300)
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "заметки.txt").write_text("не скан", encoding="utf-8")

    found = {path.name for path in find_scans(tmp_path)}

    assert found == {"a.png", "b.jpg", "c.pdf"}


def test_single_file_is_accepted(tmp_path) -> None:
    path = tmp_path / "one.png"
    fx.save(fx.text_page(width=60, height=80), path, dpi=300)

    assert find_scans(path) == [path]
    assert find_scans(tmp_path / "one.txt") == []


def test_unmeasured_label_is_empty_not_zero(config: Config) -> None:
    """Ноль означает «дефекта нет», пусто — «не измерено». Путать их опасно.

    На битональном скане контраст и шум неизмеримы, и если бы таблица ставила
    там 0.0, непроверенный скан выглядел бы чистым — ровно та ошибка, ради
    которой в отчёте есть `not_applicable` (решение №21).
    """
    report = QualityReport(
        image="fax.png",
        width=1200,
        height=1600,
        verdict="acceptable",
        quality_score=1.0,
        defects=[DefectScore(label="blur", score=0.0, source="cv")],
        not_applicable=["low_contrast", "noise"],
    )

    row = dict(zip(csv_header(config), csv_row(report, config)))

    assert row["blur"] == 0.0  # измерено, дефекта нет
    assert row["noise"] == ""  # не измерено
    assert row["low_contrast"] == ""
    assert row["not_applicable"] == "low_contrast;noise"


def test_header_covers_every_label(config: Config) -> None:
    header = csv_header(config)

    assert set(config.labels) <= set(header)
    assert header[:3] == ["image", "verdict", "quality_score"]
