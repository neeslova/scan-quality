"""Загрузчик: нормализация dpi, кириллица в пути, отказ на чужих форматах."""

from __future__ import annotations

import pytest

from src.io.loader import A4_LONG_SIDE_INCHES, load_page, load_pages
from tests import factories as fx


def test_dpi_from_metadata_is_respected(tmp_path) -> None:
    page = fx.text_page(width=600, height=800)
    path = tmp_path / "scan.png"
    fx.save(page, path, dpi=300)

    loaded = load_page(path, target_dpi=300)
    assert (loaded.width, loaded.height) == (600, 800)
    # PIL хранит dpi рациональной дробью, поэтому 300 возвращается как 299.98
    assert loaded.dpi == pytest.approx(300.0, rel=1e-3)


def test_low_dpi_scan_is_not_upscaled(tmp_path) -> None:
    """Скан ниже целевого dpi остаётся в родном разрешении.

    Апскейл кубической интерполяцией сгладил бы края букв и создал размытие,
    которого в исходнике нет: загрузчик сам произвёл бы дефект.
    """
    page = fx.text_page(width=600, height=800)
    path = tmp_path / "scan_150.png"
    fx.save(page, path, dpi=150)

    loaded = load_page(path, target_dpi=300)
    assert (loaded.width, loaded.height) == (600, 800)
    assert loaded.dpi == pytest.approx(150.0, rel=1e-3)

    # С явным разрешением апскейл всё-таки делается — поведение управляемое.
    stretched = load_page(path, target_dpi=300, allow_upscale=True)
    assert (stretched.width, stretched.height) == (1200, 1600)
    assert stretched.dpi == pytest.approx(300.0)


def test_high_dpi_scan_is_downscaled(tmp_path) -> None:
    page = fx.text_page(width=1200, height=1600)
    path = tmp_path / "scan_600.png"
    fx.save(page, path, dpi=600)

    loaded = load_page(path, target_dpi=300)
    assert (loaded.width, loaded.height) == (600, 800)
    assert loaded.dpi == pytest.approx(300.0)


def test_dpi_guessed_from_a4_when_metadata_missing(tmp_path) -> None:
    import cv2
    import numpy as np

    page = fx.text_page(width=1240, height=1754)
    path = tmp_path / "no_meta.png"
    cv2.imencode(".png", page)[1].tofile(str(path))

    # Длинная сторона 1754 px при A4 -> ~150 dpi, то есть ниже целевого:
    # растягивать нельзя, работаем в родном разрешении и честно сообщаем dpi.
    loaded = load_page(path, target_dpi=300, dpi_fallback="a4")
    assert loaded.dpi == pytest.approx(1754 / A4_LONG_SIDE_INCHES, rel=0.01)
    assert (loaded.width, loaded.height) == (1240, 1754)
    assert np.asarray(loaded.gray).dtype == np.uint8


def test_cyrillic_path(tmp_path) -> None:
    """cv2.imread не открывает такие пути на Windows — загрузчик обязан уметь."""
    folder = tmp_path / "Сканы договоров"
    folder.mkdir()
    path = folder / "договор_01.png"
    fx.save(fx.text_page(width=400, height=500), path, dpi=300)

    assert load_page(path, target_dpi=300).width == 400


def test_unknown_format_rejected(tmp_path) -> None:
    path = tmp_path / "scan.docx"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        load_pages(path)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pages(tmp_path / "nope.png")
