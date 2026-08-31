"""Обученное сведение в пайплайне: подключается конфигом, ошибку загрузки не глушит.

Правило `1 - max(дефект)` задано руками и не знает, какие метки на корпусе
информативны: одной вырожденной хватает, чтобы утопить страницу. Модель
подбирает веса по эталону — на `Data iz tg` это 0.770 против 0.505 честной
5-fold. Здесь проверяется не качество модели (оно меряется в `from_metrics`), а
то, что подключение работает и молча не отваливается.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.pipeline import learned_verdict, shared_verdict_model


class _Stub:
    """Модель, чей риск равен метрике `risk`. Пиклится: класс уровня модуля."""

    def predict_proba(self, rows):
        risk = np.asarray(rows, dtype=float)[:, 0]
        return np.stack([1.0 - risk, risk], axis=1)


def _bundle(tmp_path, tau_low: float = 0.3, tau_high: float = 0.7):
    import joblib

    path = tmp_path / "verdict.joblib"
    joblib.dump(
        {
            "pipeline": _Stub(),
            "names": ["risk"],
            "tau_low": tau_low,
            "tau_high": tau_high,
            "pages": 10,
        },
        path,
    )
    shared_verdict_model.cache_clear()
    return path


def _config(model: str | None):
    config = load_config(None, None)
    return config.model_copy(update={"verdict": config.verdict.model_copy(update={"model": model})})


def test_without_a_model_the_old_rule_stays() -> None:
    """Пустое поле — не ошибка, а режим по умолчанию."""
    assert learned_verdict({"risk": 0.9}, _config(None)) is None


@pytest.mark.parametrize(
    ("risk", "expected"),
    [(0.10, "good"), (0.50, "acceptable"), (0.90, "bad")],
)
def test_thresholds_split_the_risk_into_three_verdicts(tmp_path, risk, expected) -> None:
    config = _config(str(_bundle(tmp_path)))

    verdict, quality = learned_verdict({"risk": risk}, config)

    assert verdict == expected
    assert quality == pytest.approx(1.0 - risk)


def test_missing_metric_does_not_crash(tmp_path) -> None:
    """Метрики может не быть на странице: заполнит импьютер, обученный с моделью."""
    config = _config(str(_bundle(tmp_path)))

    verdict, _ = learned_verdict({}, config)

    assert verdict in {"good", "acceptable", "bad"}


def test_broken_model_file_is_not_swallowed(tmp_path) -> None:
    """Молчаливый откат означал бы демонстрацию старых вердиктов под видом новых."""
    import joblib

    path = tmp_path / "broken.joblib"
    joblib.dump({"names": ["risk"]}, path)
    shared_verdict_model.cache_clear()

    with pytest.raises(ValueError):
        learned_verdict({"risk": 0.5}, _config(str(path)))
