"""Подбор якорей по перцентилям корпуса: направление, вырожденные случаи, YAML."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.config import Config, load_config
from src.metrics.baseline import score_from_anchors
from src.metrics.calibrate_anchors import suggest_anchors, to_yaml


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def test_direction_is_taken_from_config(config: Config) -> None:
    """У «чем меньше, тем хуже» good обязан остаться больше bad, и наоборот."""
    rng = np.random.default_rng(0)
    values = {
        mapping.metric: list(rng.uniform(1.0, 100.0, 500)) for mapping in config.cv.scores.values()
    }

    suggestions = suggest_anchors(values, config)

    assert set(suggestions) == set(config.cv.scores)
    for label, data in suggestions.items():
        old = config.cv.scores[label]
        assert (data["good"] > data["bad"]) == (old.good > old.bad), label


def test_anchors_land_on_requested_percentiles(config: Config) -> None:
    samples = list(range(1000))  # равномерное распределение 0..999
    values = {mapping.metric: samples for mapping in config.cv.scores.values()}

    suggestions = suggest_anchors(values, config, good_pct=60.0, bad_pct=5.0)

    # blur: чем меньше tenengrad_norm, тем хуже -> good = p60, bad = p5
    blur = suggestions["blur"]
    assert blur["good"] == pytest.approx(599.4, abs=1.0)
    assert blur["bad"] == pytest.approx(49.95, abs=1.0)

    # noise: чем больше, тем хуже -> good = p40, bad = p95
    noise = suggestions["noise"]
    assert noise["good"] == pytest.approx(399.6, abs=1.0)
    assert noise["bad"] == pytest.approx(949.05, abs=1.0)


def test_typical_page_scores_near_zero(config: Config) -> None:
    """Смысл калибровки: типичная страница корпуса должна получать низкий скор."""
    rng = np.random.default_rng(7)
    samples = list(rng.normal(50.0, 10.0, 2000))
    values = {mapping.metric: samples for mapping in config.cv.scores.values()}

    suggestions = suggest_anchors(values, config)
    median = float(np.median(samples))

    for label, data in suggestions.items():
        score = score_from_anchors(median, data["good"], data["bad"])
        assert score < 0.35, f"{label}: медиана корпуса получила скор {score:.2f}"


def test_degenerate_metric_is_skipped(config: Config) -> None:
    """Если все значения одинаковы, перцентили совпадают — якоря не подменяем."""
    values = {mapping.metric: [3.0] * 100 for mapping in config.cv.scores.values()}
    assert suggest_anchors(values, config) == {}


def test_missing_metric_is_skipped(config: Config) -> None:
    assert suggest_anchors({}, config) == {}


def test_yaml_block_is_valid_and_complete(config: Config) -> None:
    values = {mapping.metric: list(range(100)) for mapping in config.cv.scores.values()}
    parsed = yaml.safe_load(to_yaml(suggest_anchors(values, config)))

    assert set(parsed["scores"]) == set(config.cv.scores)
    for label, block in parsed["scores"].items():
        assert set(block) == {"metric", "good", "bad"}
        assert block["metric"] == config.cv.scores[label].metric
