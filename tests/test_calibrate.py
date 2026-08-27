"""Калибровка порогов. Главное свойство — якоря попадают в рабочие точки метки."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AnchorPair, Config, load_config
from src.metrics.baseline import score_from_anchors
from src.models.calibrate import (
    anchors_from_operating_points,
    calibrate_label,
    operating_points,
    sweep,
    to_overlay,
)
from src.pipeline import apply_anchors


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


# --- PR-кривая --------------------------------------------------------------


def test_sweep_counts_precision_and_recall_at_every_cut() -> None:
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1])

    scores, precision, recall = sweep(y_true, y_score)

    assert list(scores) == [0.9, 0.8, 0.7, 0.1]
    # После первого разреза: один из одного верен, поймана половина дефектов.
    assert precision[0] == pytest.approx(1.0)
    assert recall[0] == pytest.approx(0.5)
    # После третьего: двое из трёх верны, пойманы оба дефекта.
    assert precision[2] == pytest.approx(2 / 3)
    assert recall[2] == pytest.approx(1.0)


def test_sweep_survives_a_label_without_positives() -> None:
    """Делить на ноль тут нельзя, а метка без примеров в выборке бывает."""
    _, _, recall = sweep(np.zeros(3), np.array([0.9, 0.5, 0.1]))
    assert list(recall) == [0.0, 0.0, 0.0]


# --- рабочие точки ----------------------------------------------------------


def test_recall_point_is_the_strictest_that_still_catches_enough() -> None:
    """Из всех порогов с нужной полнотой берётся самый высокий: он придирчивее."""
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_score = np.array([0.9, 0.6, 0.5, 0.1])
    scores, _, recall = sweep(y_true, y_score)

    high, low = operating_points(y_true, y_score, target_recall=1.0, confident_recall=0.5)

    assert scores[low] == pytest.approx(0.6)  # 0.9 ловит лишь половину
    assert recall[low] == pytest.approx(1.0)
    assert scores[high] == pytest.approx(0.9)  # половины хватает уже здесь


def test_confident_point_is_never_below_the_suspicion_point() -> None:
    """Регрессия: через целевую ТОЧНОСТЬ вторая точка уезжала ниже первой.

    У хорошо разделимой метки точность держится выше цели до самого низа
    ранжирования, и «порог точности» оказывался меньше «порога полноты».
    Срабатывала аварийная ветка, и шкала схлопывалась в ступеньку шириной
    0.003 — так вышло у shadow, skew и cropped на первом же прогоне.
    Полнота монотонна по порогу, поэтому теперь порядок гарантирован.
    """
    # Идеально разделимая метка: все дефекты выше всех чистых страниц.
    y_true = np.array([1.0] * 20 + [0.0] * 70)
    y_score = np.concatenate([np.linspace(1.0, 0.8, 20), np.linspace(0.2, 0.0, 70)])
    scores, _, _ = sweep(y_true, y_score)

    high, low = operating_points(y_true, y_score, target_recall=0.85, confident_recall=0.5)

    assert scores[high] >= scores[low]


def test_unreachable_recall_is_reported_as_such() -> None:
    """Полноты 0.95 может не быть ни при каком пороге — это не повод врать."""
    assert operating_points(np.zeros(3), np.array([0.9, 0.5, 0.1]), 0.95, 0.5) is None


# --- якоря ------------------------------------------------------------------


def test_anchors_put_the_global_thresholds_on_the_operating_points(config: Config) -> None:
    """Ключевое свойство: правило вердикта из раздела 4 не меняется ни на строку.

    Меняется шкала, на которой оно применяется. Порог полноты обязан попасть
    ровно в `tau_low`, порог точности — ровно в `tau_high`, иначе калибровка
    сдвигает рабочую точку метки и цифры перестают значить обещанное.
    """
    tau_low, tau_high = config.verdict.tau_low, config.verdict.tau_high
    t_recall, t_precision = 0.18, 0.42

    good, bad = anchors_from_operating_points(t_recall, t_precision, tau_low, tau_high)

    assert score_from_anchors(t_recall, good, bad) == pytest.approx(tau_low, abs=1e-9)
    assert score_from_anchors(t_precision, good, bad) == pytest.approx(tau_high, abs=1e-9)


def test_anchors_stay_monotone_when_the_points_collapse(config: Config) -> None:
    """Плохо разделимая метка может дать порог точности НИЖЕ порога полноты.

    Делить на ноль или переворачивать шкалу нельзя: отображение должно остаться
    возрастающим, иначе высокий скор начнёт означать «дефекта нет».
    """
    good, bad = anchors_from_operating_points(
        0.5, 0.5, config.verdict.tau_low, config.verdict.tau_high
    )
    assert bad > good
    assert score_from_anchors(0.9, good, bad) > score_from_anchors(0.1, good, bad)


# --- отказ от калибровки ----------------------------------------------------


def test_label_with_too_few_examples_is_not_calibrated(config: Config) -> None:
    """Подгонка по трём страницам — не калибровка, а её видимость."""
    y_true = np.zeros(90)
    y_true[:3] = 1.0
    y_score = np.linspace(1.0, 0.0, 90)

    assert calibrate_label("unreadable", y_true, y_score, config) is None


def test_label_with_enough_examples_is_calibrated(config: Config) -> None:
    rng = np.random.default_rng(0)
    y_true = np.zeros(90)
    y_true[:20] = 1.0
    # Разделимая метка: у дефектных страниц скор выше.
    y_score = np.where(y_true > 0, rng.uniform(0.5, 1.0, 90), rng.uniform(0.0, 0.5, 90))

    result = calibrate_label("blur", y_true, y_score, config)

    assert result is not None
    assert result.support == 20
    assert result.recall_low >= config.calibrate.recall["blur"]
    # Точка «бесспорно плохо» строже точки «подозрительно» и потому точнее.
    assert result.t_high >= result.t_low
    assert result.precision_high >= result.precision_low
    assert result.bad > result.good


# --- применение в пайплайне -------------------------------------------------


def test_score_without_anchors_passes_through_untouched(config: Config) -> None:
    """Притворяться откалиброванной метка не должна: без якорей скор идёт как есть."""
    assert apply_anchors({"blur": 0.42}, config) == {"blur": 0.42}


def test_anchors_move_the_score_onto_the_common_scale(config: Config) -> None:
    """Сеть на замере не доводила `blur` до 0.5 никогда: точность 1.000, полнота 0.080.

    После приведения та же величина обязана попадать в диапазон, где правило
    вердикта её вообще видит.
    """
    calibrated = config.model_copy(
        update={
            "verdict": config.verdict.model_copy(
                update={"anchors": {"blur": AnchorPair(good=0.10, bad=0.30)}}
            )
        }
    )

    assert apply_anchors({"blur": 0.10}, calibrated)["blur"] == pytest.approx(0.0)
    assert apply_anchors({"blur": 0.30}, calibrated)["blur"] == pytest.approx(1.0)
    assert apply_anchors({"blur": 0.20}, calibrated)["blur"] == pytest.approx(0.5)
    # Метка вне якорей в том же вызове не трогается.
    assert apply_anchors({"blur": 0.2, "noise": 0.7}, calibrated)["noise"] == 0.7


def test_overlay_loads_as_a_real_config(config: Config, tmp_path) -> None:
    """Оверлей задаёт лишь отличия (решение №24) и обязан проходить ту же валидацию.

    Проверяем настоящей загрузкой, а не подменой поля: файл, который пишет
    калибровка, должен накладываться на base.yaml как любой другой оверлей.
    """
    import yaml

    result = calibrate_label(
        "blur",
        np.array([1.0] * 20 + [0.0] * 70),
        np.concatenate([np.linspace(1.0, 0.6, 20), np.linspace(0.4, 0.0, 70)]),
        config,
    )
    overlay = to_overlay([result])
    assert set(overlay) == {"verdict"}
    assert set(overlay["verdict"]["anchors"]) == {"blur"}

    path = tmp_path / "thresholds.yaml"
    path.write_text(yaml.safe_dump(overlay, allow_unicode=True), encoding="utf-8")

    loaded = load_config(overlays=path)
    assert loaded.verdict.anchors["blur"].good == pytest.approx(result.good)
    assert loaded.verdict.anchors["blur"].bad == pytest.approx(result.bad)
    # Остальное наследуется от базового конфига без изменений.
    assert loaded.verdict.tau_low == config.verdict.tau_low
