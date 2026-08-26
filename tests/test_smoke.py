"""Smoke-тест спринта С0: конфиг грузится, схема валидна, путь файл -> JSON работает.

Специально не тянет numpy/torch/gradio — должен проходить на голом окружении.
"""

from __future__ import annotations

import json

import pytest

from src.config import Config, load_config
from src.pipeline import analyze, decide_verdict, quality_score
from src.schema import QualityReport


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def test_config_loads(config: Config) -> None:
    assert config.n_labels == 10
    assert "unreadable" in config.labels
    assert config.data.patch_size == 384
    assert config.data.grid.n_patches == 54


def test_aggregation_covers_all_labels(config: Config) -> None:
    local = set(config.data.aggregation.local)
    global_ = set(config.data.aggregation.global_)
    assert local | global_ == set(config.labels)
    assert not local & global_
    assert config.is_local("glare")
    assert not config.is_local("blur")


def test_config_rejects_unknown_key(config: Config) -> None:
    raw = config.model_dump(by_alias=True)
    raw["oops"] = 1
    with pytest.raises(ValueError):
        Config.model_validate(raw)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"blur": 0.1, "unreadable": 0.05}, "good"),
        ({"blur": 0.45, "unreadable": 0.05}, "acceptable"),
        ({"blur": 0.85, "unreadable": 0.05}, "bad"),
        # unreadable бьёт по собственному, более низкому порогу
        ({"blur": 0.01, "unreadable": 0.55}, "bad"),
        ({}, "good"),
    ],
)
def test_decide_verdict(config: Config, scores: dict, expected: str) -> None:
    assert decide_verdict(scores, config.verdict) == expected


def test_quality_score() -> None:
    assert quality_score({}) == 1.0
    assert quality_score({"blur": 0.25, "noise": 0.1}) == 0.75


def test_analyze_returns_valid_report(config: Config, tmp_path) -> None:
    image = tmp_path / "scan_001.jpg"
    image.write_bytes(b"not-a-real-image-but-enough-for-the-stub")

    report = analyze(image, config)

    assert isinstance(report, QualityReport)
    assert report.image == "scan_001.jpg"
    assert report.verdict in {"good", "acceptable", "bad"}
    assert len(report.defects) == config.n_labels
    assert set(report.scores()) == set(config.labels)
    # отчёт отсортирован по убыванию вероятности
    assert report.defects == sorted(report.defects, key=lambda d: d.score, reverse=True)

    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["verdict"] == report.verdict


def test_analyze_is_deterministic(config: Config, tmp_path) -> None:
    image = tmp_path / "scan_002.jpg"
    image.write_bytes(b"same-bytes-same-report")
    assert analyze(image, config).scores() == analyze(image, config).scores()


def test_analyze_missing_file(config: Config, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "nope.jpg", config)
