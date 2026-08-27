"""Локализация дефекта. Главное — карта показывает то, по чему выставлен скор."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config
from src.data.degrade import DEGRADATIONS
from src.localize import heatmap, localizable, overlay, patch_heat
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def page() -> np.ndarray:
    return fx.text_page(width=900, height=1200)


def rng() -> np.random.Generator:
    return np.random.default_rng(7)


# --- какие метки вообще локализуемы -----------------------------------------


def test_global_defects_have_no_map(config: Config, page: np.ndarray) -> None:
    """Подсветить «здесь размыто» на равномерно размытом скане честно нельзя.

    Расфокус, перекос и потеря разрешения — свойства всего прогона сканера,
    у них нет места на странице. Врать красивой картинкой хуже, чем её не дать.
    """
    for label in ("blur", "skew", "low_resolution"):
        assert heatmap(label, page, config) is None


def test_ocr_label_has_no_map(config: Config, page: np.ndarray) -> None:
    """`unreadable` приходит от OCR по всей странице разом (решение №7)."""
    assert heatmap("unreadable", page, config) is None


def test_localizable_lists_only_what_can_be_shown(config: Config) -> None:
    labels = set(localizable(config))

    assert {"glare", "shadow", "streaks", "cropped"} <= labels
    assert set(config.sources.cnn) <= labels  # сеть предсказывает по патчам
    assert not {"blur", "skew", "low_resolution", "unreadable"} & labels


# --- карта совпадает с дефектом ---------------------------------------------


def test_glare_map_lands_on_the_synthesised_glare(config: Config, page: np.ndarray) -> None:
    """Карта строится той же функцией, что и скор, — значит обязана совпасть с маской
    деградации, которой блик и был нарисован."""
    result = DEGRADATIONS["glare"](page, 0.8, config, rng())
    heat = heatmap("glare", result.image, config)

    assert heat is not None
    truth = result.mask > 0
    assert truth.any()
    # Подсвеченное лежит внутри нарисованного, а не где-то ещё.
    hit = (heat > 0.5) & truth
    assert hit.sum() > 0.5 * float((heat > 0.5).sum())


def test_clean_page_gets_an_empty_glare_map(config: Config, page: np.ndarray) -> None:
    """Чистый скан не должен подсвечиваться: иначе картинка обесценивается."""
    heat = heatmap("glare", page, config)

    assert heat is not None
    assert float(heat.mean()) < 0.01


# --- карта по патчам --------------------------------------------------------


class _Prediction:
    """Заглушка предсказания: важна только арифметика перекрытия."""

    def __init__(self, patches, probabilities, labels) -> None:
        self.patches = patches
        self.probabilities = probabilities
        self.labels = labels


def test_overlapping_patches_take_the_max_not_the_sum() -> None:
    """Патчи сетки перекрываются. Сумма давала бы яркое пятно там, где просто
    гуще сетка, а не там, где дефект."""
    from src.io.patches import Patch

    patches = [
        Patch(index=0, row=0, col=0, x0=0, y0=0, x1=10, y1=10),
        Patch(index=1, row=0, col=1, x0=5, y0=0, x1=15, y1=10),
    ]
    prediction = _Prediction(patches, np.array([[0.4], [0.6]], dtype=np.float32), ["noise"])

    heat = patch_heat(prediction, "noise", (10, 15))

    assert heat[0, 0] == pytest.approx(0.4)
    assert heat[0, 7] == pytest.approx(0.6)  # перекрытие: максимум, не 1.0
    assert heat.max() <= 1.0


# --- отрисовка --------------------------------------------------------------


def test_overlay_keeps_the_scan_visible(page: np.ndarray) -> None:
    """Под подсветкой должен остаться виден сам скан — иначе по картинке нельзя
    проверить, что система права."""
    heat = np.ones(page.shape, dtype=np.float32)
    image = overlay(page, heat)

    assert image.shape == (*page.shape, 3)
    assert image.dtype == np.uint8
    # Полностью красным не заливаем даже при карте из единиц.
    assert image[:, :, 1].max() > 0


def test_overlay_resizes_a_smaller_map(page: np.ndarray) -> None:
    """Карта сети считается по патчам и может прийти в другом разрешении."""
    heat = np.zeros((page.shape[0] // 4, page.shape[1] // 4), dtype=np.float32)
    heat[:, :] = 1.0

    image = overlay(page, heat)

    assert image.shape == (*page.shape, 3)
