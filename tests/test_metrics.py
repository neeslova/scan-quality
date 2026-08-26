"""С1: каждая деградация должна сдвигать «свою» метрику в ожидаемую сторону.

Абсолютные значения здесь не проверяются — они зависят от шрифта и dpi и калибруются
в С2 по реальным сканам. Проверяется знак эффекта: именно он и есть содержание метрики.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, load_config
from src.io import patches
from src.metrics import blur, contrast, geometry, glare, noise, shadow, streaks, text_scale
from src.metrics.baseline import score_from_anchors
from tests import factories as fx


@pytest.fixture(scope="module")
def page() -> np.ndarray:
    return fx.text_page()


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


# --- резкость ---------------------------------------------------------------


def test_blur_drops_sharpness(page: np.ndarray) -> None:
    soft = fx.blurred(page, ksize=11)
    assert blur.tenengrad(soft) < blur.tenengrad(page) * 0.5
    assert blur.laplacian_variance(soft) < blur.laplacian_variance(page) * 0.5


def test_sharpness_of_blank_page_is_low(page: np.ndarray) -> None:
    blank = np.full_like(page, fx.PAPER)
    assert blur.tenengrad(blank) == pytest.approx(0.0, abs=1e-6)


# --- шум --------------------------------------------------------------------


def test_noise_raises_sigma(page: np.ndarray) -> None:
    clean_sigma = noise.noise_sigma(page)
    dirty_sigma = noise.noise_sigma(fx.noisy(page, sigma=12.0))
    assert dirty_sigma > clean_sigma + 3.0


def test_noise_sigma_tracks_true_sigma(page: np.ndarray) -> None:
    """На плоском поле оценка должна попадать в настоящую σ с точностью ~30%."""
    flat = np.full((400, 400), 200, dtype=np.uint8)
    estimated = noise.noise_sigma(fx.noisy(flat, sigma=6.0))
    assert 4.0 < estimated < 8.5


# --- контраст ---------------------------------------------------------------


def test_low_contrast_closes_ink_paper_gap(page: np.ndarray) -> None:
    faded = fx.low_contrast(page, factor=0.2)
    assert contrast.ink_paper_gap(faded) < contrast.ink_paper_gap(page) * 0.35
    assert contrast.rms_contrast(faded) < contrast.rms_contrast(page)


# --- пересвет ---------------------------------------------------------------


def test_glare_detected(page: np.ndarray) -> None:
    stats = glare.glare_stats(fx.with_glare(page))
    assert stats.cluster_frac > 0.02
    assert stats.n_clusters >= 1


def test_clean_page_has_no_glare(page: np.ndarray) -> None:
    """Ключевая проверка: белая бумага сама по себе бликом считаться не должна."""
    assert glare.glare_stats(page).cluster_frac == pytest.approx(0.0, abs=1e-9)


# --- тень -------------------------------------------------------------------


def test_shadow_detected(page: np.ndarray) -> None:
    assert shadow.shadow_stats(fx.with_shadow(page)).dark_frac > 0.05


def test_clean_page_has_no_shadow(page: np.ndarray) -> None:
    assert shadow.shadow_stats(page).dark_frac < 0.01


# --- полосы -----------------------------------------------------------------


def test_streaks_detected(page: np.ndarray) -> None:
    dirty = streaks.streak_stats(fx.with_streaks(page)).energy
    clean = streaks.streak_stats(page).energy
    assert dirty > clean * 2.0


# --- геометрия --------------------------------------------------------------


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.5, 6.0])
def test_skew_estimated(page: np.ndarray, angle: float) -> None:
    estimated = geometry.estimate_skew(fx.rotated(page, angle))
    # Оцениваем угол, которым страницу нужно вернуть обратно.
    assert estimated == pytest.approx(-angle, abs=0.6)


def test_straight_page_has_no_skew(page: np.ndarray) -> None:
    assert abs(geometry.estimate_skew(page)) < 0.6


def test_cropped_page_loses_margin(page: np.ndarray) -> None:
    full = geometry.margin_fractions(page)
    cut = geometry.margin_fractions(fx.cropped(page, cut_frac=0.12))
    assert full.minimum > 0.03
    assert cut.left < 0.01
    assert cut.minimum < full.minimum


# --- масштаб текста ---------------------------------------------------------


def test_line_height_matches_generator(page: np.ndarray) -> None:
    scale = text_scale.estimate_text_scale(page)
    assert scale.line_height == pytest.approx(24.0, abs=3.0)
    assert scale.n_lines > 20


def test_downscale_reduces_line_height(page: np.ndarray) -> None:
    small = fx.text_page(width=480, height=640, line_height=10, line_gap=7, margin=36)
    assert text_scale.estimate_text_scale(small).line_height < 14.0


# --- патчи ------------------------------------------------------------------


def test_grid_covers_page() -> None:
    boxes = patches.grid(1200, 1600, 384, rows=9, cols=6)
    assert len(boxes) == 54
    assert boxes[0].x0 == 0 and boxes[0].y0 == 0
    assert boxes[-1].x1 == 1200 and boxes[-1].y1 == 1600
    assert all(b.width == 384 and b.height == 384 for b in boxes)


def test_grid_handles_small_page() -> None:
    boxes = patches.grid(200, 150, 384, rows=9, cols=6)
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].y1) == (200, 150)


def test_select_informative_skips_blank(page: np.ndarray) -> None:
    boxes = patches.grid(page.shape[1], page.shape[0], 384, rows=9, cols=6)
    blank = np.full_like(page, fx.PAPER)
    # На пустой странице информативных нет — возвращаются все, а не пустой список.
    assert len(patches.select_informative(blank, boxes, min_ink_frac=0.01)) == len(boxes)
    assert len(patches.select_informative(page, boxes, min_ink_frac=0.01)) > 0


def test_aggregate_local_vs_global() -> None:
    result = patches.aggregate(
        {"glare": [0.1, 0.9, 0.2], "blur": [0.1, 0.9, 0.2]}, local_labels=["glare"]
    )
    assert result["glare"] == pytest.approx(0.9)
    assert result["blur"] == pytest.approx(0.4)


# --- перевод метрики в скор -------------------------------------------------


def test_score_from_anchors_both_directions() -> None:
    # «чем меньше, тем хуже» (резкость)
    assert score_from_anchors(900.0, good=900.0, bad=120.0) == 0.0
    assert score_from_anchors(120.0, good=900.0, bad=120.0) == 1.0
    assert score_from_anchors(510.0, good=900.0, bad=120.0) == pytest.approx(0.5)
    # «чем больше, тем хуже» (шум)
    assert score_from_anchors(1.5, good=1.5, bad=8.0) == 0.0
    assert score_from_anchors(20.0, good=1.5, bad=8.0) == 1.0


def test_score_from_anchors_rejects_equal_anchors() -> None:
    with pytest.raises(ValueError):
        score_from_anchors(1.0, good=2.0, bad=2.0)
