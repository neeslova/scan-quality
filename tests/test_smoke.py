"""Сквозной тест: конфиг грузится, схема валидна, путь файл -> QualityReport -> JSON."""

from __future__ import annotations

import json

import pytest

from src.config import Config, load_config
from src.pipeline import analyze, decide_verdict, quality_score
from src.schema import QualityReport
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def scan(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("scans") / "scan_001.png"
    fx.save(fx.text_page(width=1200, height=1600), path, dpi=300)
    return str(path)


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


def test_cv_scores_are_known_labels(config: Config) -> None:
    assert set(config.cv.scores) <= set(config.labels)
    # unreadable приходит из OCR (С3), CV-baseline его не считает
    assert "unreadable" not in config.cv.scores


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


def test_analyze_returns_valid_report(config: Config, scan: str) -> None:
    report = analyze(scan, config)

    assert isinstance(report, QualityReport)
    assert report.image == "scan_001.png"
    assert (report.width, report.height) == (1200, 1600)
    assert report.pipeline_version == "cv-baseline"
    assert report.verdict in {"good", "acceptable", "bad"}
    assert set(report.scores()) == set(config.cv.scores)
    assert all(d.source == "cv" for d in report.defects)
    # отчёт отсортирован по убыванию вероятности
    assert report.defects == sorted(report.defects, key=lambda d: d.score, reverse=True)
    assert report.cv_metrics["n_informative_patches"] > 0

    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    assert payload["verdict"] == report.verdict


def test_clean_page_is_not_rejected(config: Config, scan: str) -> None:
    """DoD С1: на чистом скане baseline не должен кричать о дефектах."""
    report = analyze(scan, config)
    assert report.verdict in {"good", "acceptable"}
    assert report.scores()["glare"] == pytest.approx(0.0, abs=1e-6)
    assert report.scores()["shadow"] < 0.2
    assert report.scores()["skew"] < 0.3


def test_low_resolution_survives_dpi_normalization(config: Config, tmp_path) -> None:
    """Скан 150 dpi загрузчик растянет до 300 — метка low_resolution обязана уцелеть.

    После апскейла высота строки в пикселях возвращается к норме, поэтому метрика
    считается в пикселях ИСХОДНОГО файла. Иначе низкое разрешение просто исчезает.
    """
    path = tmp_path / "lowres.png"
    fx.save(fx.text_page(width=620, height=800, line_height=12, line_gap=9, margin=45), path, 150)

    report = analyze(path, config)
    assert report.width == 1240  # растянут до 300 dpi
    assert report.cv_metrics["source_line_height_px"] < 16
    assert report.scores()["low_resolution"] > 0.7


def test_analyze_is_deterministic(config: Config, scan: str) -> None:
    assert analyze(scan, config).scores() == analyze(scan, config).scores()


def test_analyze_missing_file(config: Config, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "nope.jpg", config)


def test_app_handler(config: Config, scan: str) -> None:
    """Хендлер Gradio отдаёт четыре выхода и не требует самой Gradio."""
    from src.app import _run

    verdict, defects, metrics, payload = _run(None, config)
    assert (defects, metrics, payload) == ({}, [], {})

    verdict, defects, metrics, payload = _run(scan, config)
    assert payload["verdict"] in {"good", "acceptable", "bad"}
    assert set(defects) == set(config.cv.scores)
    assert len(metrics) > 5
