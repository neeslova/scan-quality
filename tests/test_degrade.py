"""Деградации: каждая должна сдвигать «свою» метрику и отдавать корректную маску."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config
from src.data.degrade import DEGRADATIONS, LOCAL, apply
from src.metrics import blur as blur_metric
from src.metrics import contrast, geometry, glare, noise, shadow, streaks
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def page() -> np.ndarray:
    return fx.text_page(width=1200, height=1600)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_every_taxonomy_label_has_a_degradation(config: Config) -> None:
    """Кроме unreadable — она выводится из OCR, а не рисуется."""
    expected = {label for label in config.labels if label not in config.labeling.auto_labels}
    assert set(DEGRADATIONS) == expected


def test_local_set_matches_config(config: Config) -> None:
    """Маски есть ровно у тех дефектов, что агрегируются по max."""
    assert LOCAL == set(config.data.aggregation.local)


# --- каждый дефект двигает свою метрику -------------------------------------


def test_blur_reduces_sharpness(page: np.ndarray, config: Config) -> None:
    for seed in range(4):  # покрываем обе ветки: гаусс и смаз
        out = DEGRADATIONS["blur"](page, 0.8, config, rng(seed)).image
        assert blur_metric.tenengrad(out) < blur_metric.tenengrad(page) * 0.6


def test_noise_raises_sigma(page: np.ndarray, config: Config) -> None:
    out = DEGRADATIONS["noise"](page, 0.8, config, rng()).image
    assert noise.noise_sigma(out) > noise.noise_sigma(page) + 3.0


def test_low_contrast_closes_gap(page: np.ndarray, config: Config) -> None:
    out = DEGRADATIONS["low_contrast"](page, 0.9, config, rng()).image
    assert contrast.ink_paper_gap(out) < contrast.ink_paper_gap(page) * 0.5


def test_low_contrast_keeps_paper_light(page: np.ndarray, config: Config) -> None:
    """Выцветает краска, а не бумага: страница не должна потемнеть целиком."""
    out = DEGRADATIONS["low_contrast"](page, 0.9, config, rng()).image
    assert np.median(out) == pytest.approx(np.median(page), abs=6)


def test_low_resolution_shrinks_the_page(page: np.ndarray, config: Config) -> None:
    """Страница реально становится мельче — апскейл обратно запрещён (решение №20)."""
    out = DEGRADATIONS["low_resolution"](page, 0.9, config, rng()).image
    assert out.shape[0] < page.shape[0] * 0.5


def test_skew_is_detected(page: np.ndarray, config: Config) -> None:
    out = DEGRADATIONS["skew"](page, 0.9, config, rng()).image
    assert abs(geometry.estimate_skew(out)) > 2.0


def test_glare_is_detected_and_masked(page: np.ndarray, config: Config) -> None:
    result = DEGRADATIONS["glare"](page, 0.8, config, rng())
    assert glare.glare_stats(result.image).cluster_frac > 0.01
    assert result.mask is not None
    assert 0.0 < result.mask.mean() / 255 < 0.6


def test_shadow_is_detected_and_masked(page: np.ndarray, config: Config) -> None:
    result = DEGRADATIONS["shadow"](page, 0.9, config, rng())
    assert shadow.shadow_stats(result.image).dark_frac > 0.03
    assert result.mask is not None and result.mask.any()


def test_streaks_are_detected_and_masked(page: np.ndarray, config: Config) -> None:
    result = DEGRADATIONS["streaks"](page, 0.9, config, rng(2))
    assert streaks.streak_stats(result.image).energy > streaks.streak_stats(page).energy * 2
    assert result.mask is not None and result.mask.any()


def test_cropped_severs_text_and_masks_the_edge(page: np.ndarray, config: Config) -> None:
    result = DEGRADATIONS["cropped"](page, 0.9, config, rng())
    assert result.image.shape != page.shape
    assert result.mask is not None
    assert result.mask.shape == result.image.shape
    assert geometry.border_ink(result.image).coverage > 0.15


# --- маски и порядок применения ---------------------------------------------


def test_global_defects_have_no_mask(page: np.ndarray, config: Config) -> None:
    for label in set(DEGRADATIONS) - LOCAL:
        assert DEGRADATIONS[label](page, 0.6, config, rng()).mask is None, label


def test_masks_survive_later_geometry(page: np.ndarray, config: Config) -> None:
    """Маска блика должна подгоняться под обрез, иначе она уедет относительно кадра."""
    labels = ["glare", "cropped"]
    image, masks = apply(page, labels, dict.fromkeys(labels, 0.8), config, rng())

    assert masks["glare"] is not None
    assert masks["glare"].shape == image.shape
    assert masks["cropped"].shape == image.shape


def test_severity_is_monotone(page: np.ndarray, config: Config) -> None:
    weak = DEGRADATIONS["blur"](page, 0.3, config, rng(1)).image
    strong = DEGRADATIONS["blur"](page, 1.0, config, rng(1)).image
    assert blur_metric.tenengrad(strong) < blur_metric.tenengrad(weak)


def test_apply_is_deterministic(page: np.ndarray, config: Config) -> None:
    labels = ["blur", "noise", "shadow"]
    severities = dict.fromkeys(labels, 0.7)
    first, _ = apply(page, labels, severities, config, rng(5))
    second, _ = apply(page, labels, severities, config, rng(5))
    assert np.array_equal(first, second)


def test_apply_without_labels_returns_source(page: np.ndarray, config: Config) -> None:
    image, masks = apply(page, [], {}, config, rng())
    assert np.array_equal(image, page)
    assert masks == {}
