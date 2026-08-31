"""Классификатор поверх CV-метрик: сборка выборки, честность оценки, рабочая точка."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.golden import write_golden
from src.models.from_metrics import (
    build_dataset,
    out_of_fold_risk,
    threshold_for_recall,
)
from src.schema import GoldenRecord


def _corpus(tmp_path: Path, pages: int = 60) -> tuple[Path, Path]:
    """Синтетический корпус: одна метрика разделяет классы, две — шум.

    Пайплайну при этом подсовывается вырожденный `quality_score`: он объявляет
    браком почти всё, как и на живом корпусе. Так проверяется не абстрактная
    обучаемость, а именно то, ради чего модуль написан.
    """
    rng = np.random.default_rng(0)
    reports: list[dict] = []
    golden: list[GoldenRecord] = []

    for index in range(pages):
        bad = index % 2 == 0
        name = f"{index:03d}.png"
        reports.append(
            {
                "image": name,
                "verdict": "bad" if index % 10 else "good",
                "quality_score": 0.0 if index % 10 else 1.0,
                "cv_metrics": {
                    "signal": float(rng.normal(3.0 if bad else 0.0, 0.6)),
                    "noise_a": float(rng.normal(0.0, 1.0)),
                    "noise_b": float(rng.normal(0.0, 1.0)),
                },
            }
        )
        golden.append(
            GoldenRecord(
                image=("bad/" if bad else "Good/") + name,
                page=0,
                document=name,
                corpus="synthetic",
                label="bad" if bad else "good",
                sha256=f"{index:064d}",
            )
        )

    reports_path = tmp_path / "reports.jsonl"
    reports_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in reports), encoding="utf-8"
    )
    golden_path = tmp_path / "golden.jsonl"
    write_golden(golden, golden_path)
    return reports_path, golden_path


def test_dataset_joins_metrics_with_labels(tmp_path) -> None:
    reports_path, golden_path = _corpus(tmp_path, pages=20)

    data = build_dataset(golden_path, reports_path)

    assert data.features.shape == (20, 3)
    assert data.names == ["noise_a", "noise_b", "signal"]
    assert set(data.labels.tolist()) == {0, 1}


def test_learned_risk_beats_a_saturated_pipeline_score(tmp_path) -> None:
    """Ради этого модуль и написан: правило `max` насыщается, метрики — нет."""
    from src.models.against_golden import roc_auc

    reports_path, golden_path = _corpus(tmp_path)
    data = build_dataset(golden_path, reports_path)

    risk, per_fold = out_of_fold_risk(data)

    from sklearn.metrics import roc_auc_score

    learned = roc_auc_score(data.labels, risk)
    assert learned > 0.9
    assert learned > roc_auc(data.baseline)
    assert len(per_fold) == 5


def test_every_page_gets_an_out_of_fold_risk(tmp_path) -> None:
    """Ни одна страница не остаётся без предсказания, иначе AUC считается не по всем."""
    reports_path, golden_path = _corpus(tmp_path)
    data = build_dataset(golden_path, reports_path)

    risk, _ = out_of_fold_risk(data)

    assert not np.isnan(risk).any()


def test_threshold_takes_the_highest_that_still_meets_recall() -> None:
    """Полнота падает с ростом порога, поэтому любой порог ниже — лишние тревоги."""
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    risk = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    assert threshold_for_recall(labels, risk, 0.5) == 0.8
    assert threshold_for_recall(labels, risk, 1.0) == 0.6


def test_threshold_is_undefined_without_positives() -> None:
    labels = np.array([0, 0, 0])
    risk = np.array([0.1, 0.2, 0.3])

    assert np.isnan(threshold_for_recall(labels, risk, 0.85))
