"""Объяснение отчёта. Главное — оно не имеет права ни на что повлиять."""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.explain import explain, report_digest, with_explanation
from src.schema import DefectScore, QualityReport


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture
def report() -> QualityReport:
    return QualityReport(
        image="scan_042.jpg",
        width=2544,
        height=3269,
        verdict="bad",
        quality_score=0.12,
        defects=[
            DefectScore(label="blur", score=0.88, raw=0.71, source="cv"),
            DefectScore(label="noise", score=0.41, raw=0.13, source="cnn"),
        ],
        not_applicable=["unreadable"],
    )


def test_disabled_by_default_returns_nothing(config: Config, report: QualityReport) -> None:
    """Система обязана работать без сети, поэтому выключено по умолчанию."""
    assert config.explain.enabled is False
    assert explain(report, config) is None


def test_missing_package_is_not_an_error(config: Config, report: QualityReport) -> None:
    """`anthropic` живёт в extra `explain` и на обычной установке отсутствует.

    Это рабочее состояние, а не поломка: отчёт полон и без пояснения.
    """
    enabled = config.model_copy(
        update={"explain": config.explain.model_copy(update={"enabled": True})}
    )
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError:
        assert explain(report, enabled) is None
    else:
        pytest.skip("anthropic установлен — путь отсутствия пакета здесь не проверить")


def test_report_survives_a_failed_explanation(config: Config, report: QualityReport) -> None:
    """Ключевое свойство: внешний API — декоратор, а не источник вердикта (№2)."""
    enabled = config.model_copy(
        update={"explain": config.explain.model_copy(update={"enabled": True})}
    )
    result = with_explanation(report, enabled)

    assert result.verdict == report.verdict
    assert result.quality_score == report.quality_score
    assert result.scores() == report.scores()


# --- что уходит наружу ------------------------------------------------------


def test_digest_carries_numbers_and_sources(config: Config, report: QualityReport) -> None:
    text = report_digest(report, config)

    assert "bad" in text
    assert "blur" in text and "cv" in text
    assert "noise" in text and "cnn" in text
    # Неизмеренное названо отдельно: иначе модель опишет его как «дефекта нет».
    assert "unreadable" in text
    assert "Не измерено" in text


def test_digest_never_contains_the_scan_itself(config: Config, report: QualityReport) -> None:
    """Наружу уходят только числа и метки.

    Tobacco3482 — документы табачных процессов с настоящими именами и адресами.
    Отправлять сам скан в сторонний сервис нельзя, и выжимка это гарантирует
    по построению: в ней нет ни пикселей, ни распознанного текста.
    """
    text = report_digest(report, config)

    assert "base64" not in text.lower()
    assert len(text) < 2000  # выжимка, а не документ
    # Размеры страницы — не содержимое, но и их незачем: их тут нет.
    assert "2544" not in text
