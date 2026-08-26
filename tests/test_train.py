"""Обучение: отбор лучшей эпохи, лог метрик, сводные метрики."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.models.evaluate import LabelMetrics, macro, per_label, summary
from src.models.train import CSV_FIELDS, append_csv, update_best

# --- отбор лучшей эпохи -----------------------------------------------------


def test_best_updates_only_on_improvement() -> None:
    best, improved = update_best(0.5, 0.6)
    assert (best, improved) == (0.6, True)

    best, improved = update_best(0.6, 0.55)
    assert (best, improved) == (0.6, False)

    best, improved = update_best(0.6, 0.6)
    assert (best, improved) == (0.6, False)


def test_nan_never_becomes_best() -> None:
    """Метрика бывает не определена — она не должна побеждать."""
    assert update_best(0.4, float("nan")) == (0.4, False)
    assert update_best(float("-inf"), float("nan")) == (float("-inf"), False)


def test_first_epoch_always_improves() -> None:
    best, improved = update_best(float("-inf"), 0.01)
    assert improved and best == 0.01


def test_best_survives_a_resume() -> None:
    """Ключевая проверка: после обрыва сессии лучшая модель не должна затираться.

    В чекпоинт пишется УЖЕ обновлённое значение, иначе продолжение стартует
    с best = -inf и первая же эпоха перезаписывает best.ckpt худшей моделью.
    """
    best = float("-inf")
    for value in (0.30, 0.45, 0.40):
        best, _ = update_best(best, value)
    saved = best  # что попало бы в last.ckpt

    restored = saved
    restored, improved = update_best(restored, 0.42)
    assert not improved
    assert restored == pytest.approx(0.45)


# --- потолок pos_weight -----------------------------------------------------


def test_pos_weight_is_capped(tmp_path) -> None:
    """Метка с тремя примерами на восемь тысяч получала вес 2686.

    Её слагаемое съедало всю функцию потерь: на реальном прогоне train-потеря
    была 0.52, а val-потеря 23.7 — почти целиком из этого веса.
    """
    from src.config import load_config
    from src.data.dataset import Sample
    from src.models.train import pos_weight_tensor

    config = load_config()
    # 3 положительных примера unreadable на 8000 страниц — как в реальном прогоне.
    samples = [
        Sample(path=tmp_path / "x.png", labels={"unreadable": i < 3}, masks={}, source="real")
        for i in range(8000)
    ]

    weights = pos_weight_tensor(config, samples, "cpu").numpy()
    position = config.labels.index("unreadable")

    assert weights[position] == pytest.approx(config.train.pos_weight_max)
    assert weights.max() <= config.train.pos_weight_max


def test_explicit_pos_weight_is_also_capped(tmp_path) -> None:
    from src.config import load_config
    from src.models.train import pos_weight_tensor

    config = load_config()
    explicit = [1000.0] * config.n_labels
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"pos_weight": explicit})}
    )

    weights = pos_weight_tensor(config, [], "cpu").numpy()
    assert weights.max() <= config.train.pos_weight_max


# --- лог метрик -------------------------------------------------------------


def test_csv_appends_with_one_header(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    for epoch in range(3):
        append_csv(path, {"epoch": epoch, "macro_ap": 0.1 * epoch})

    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    assert path.read_text(encoding="utf-8").count("epoch,train_loss") == 1


def test_csv_tolerates_missing_fields(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 0})
    with path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert set(row) == set(CSV_FIELDS)


# --- сводные метрики --------------------------------------------------------


def test_macro_skips_labels_without_examples() -> None:
    """Метка, которой нет в выборке, не должна штрафовать модель нулём."""
    metrics = [
        LabelMetrics("blur", 0.8, 0.8, 0.8, 0.9, support=10),
        LabelMetrics("glare", 0.0, 0.0, 0.0, float("nan"), support=0),
    ]
    assert macro(metrics, "f1") == pytest.approx(0.8)
    assert macro(metrics, "average_precision") == pytest.approx(0.9)


def test_macro_on_nothing_is_nan() -> None:
    metrics = [LabelMetrics("blur", 0.0, 0.0, 0.0, float("nan"), support=0)]
    assert np.isnan(macro(metrics, "f1"))


def test_per_label_counts_support_and_scores() -> None:
    labels = ["a", "b"]
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.7], [0.1, 0.3]])

    metrics = per_label(y_true, y_score, labels, threshold=0.5)

    by_name = {m.label: m for m in metrics}
    assert by_name["a"].support == 2
    assert by_name["a"].precision == pytest.approx(1.0)
    assert by_name["a"].recall == pytest.approx(1.0)
    assert by_name["b"].support == 1


def test_summary_has_all_four_macros() -> None:
    metrics = [LabelMetrics("blur", 0.5, 0.6, 0.55, 0.7, support=4)]
    assert set(summary(metrics)) == {
        "macro_f1",
        "macro_ap",
        "macro_precision",
        "macro_recall",
    }
