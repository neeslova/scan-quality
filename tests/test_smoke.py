"""Сквозной тест: конфиг грузится, схема валидна, путь файл -> QualityReport -> JSON."""

from __future__ import annotations

import json

import pytest

from src.config import Config, load_config
from src.pipeline import analyze, decide_verdict, quality_score
from src.schema import QualityReport
from tests import factories as fx


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> Config:
    """Конфиг без модели: путь к quality.onnx намеренно указывает в пустоту.

    Иначе тест зависел бы от того, лежит ли на этой машине обученная сеть, —
    а в репозиторий она не попадает. Гибрид с сетью проверяется отдельно,
    в `test_cnn_labels_come_from_the_network`.
    """
    config = load_config()
    absent = tmp_path_factory.mktemp("nomodel") / "quality.onnx"
    return config.model_copy(
        update={"paths": config.paths.model_copy(update={"onnx_model": absent})}
    )


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
    ("where", "expected"),
    [
        ("ниже tau_low", "good"),
        ("между порогами", "acceptable"),
        ("выше tau_high", "bad"),
        ("пусто", "good"),
    ],
)
def test_decide_verdict(config: Config, where: str, expected: str) -> None:
    """Проверяем ПРАВИЛО, а не конкретные числа порогов.

    Пороги калибруются и меняются (0.30/0.60 -> 0.40/0.65 -> 0.45/0.70), и тест,
    приколоченный к их значениям, ломается на каждой калибровке, ничего при этом
    не проверяя по существу.
    """
    low, high = config.verdict.tau_low, config.verdict.tau_high
    scores = {
        "ниже tau_low": {"blur": low - 0.05},
        "между порогами": {"blur": (low + high) / 2},
        "выше tau_high": {"blur": high + 0.05},
        "пусто": {},
    }[where]

    assert decide_verdict(scores, config.verdict) == expected


def test_unreadable_has_its_own_lower_threshold(config: Config) -> None:
    """Метка нечитаемости бьёт по своему порогу: текст, который нельзя прочесть,
    обесценивает скан сильнее любого другого дефекта (раздел 4)."""
    scores = {"blur": 0.01, "unreadable": config.verdict.tau_unreadable + 0.05}
    assert decide_verdict(scores, config.verdict) == "bad"


def test_quality_score() -> None:
    assert quality_score({}) == 1.0
    assert quality_score({"blur": 0.25, "noise": 0.1}) == 0.75


def test_analyze_returns_valid_report(config: Config, scan: str) -> None:
    report = analyze(scan, config)

    assert isinstance(report, QualityReport)
    assert report.image == "scan_001.png"
    assert (report.width, report.height) == (1200, 1600)
    # Модели нет — работают CV и только он, и версия это говорит.
    assert report.pipeline_version == "cv"
    assert report.verdict in {"good", "acceptable", "bad"}
    assert set(report.scores()) == set(config.sources.cv)
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


def test_low_resolution_detected(config: Config, tmp_path) -> None:
    """Скан 150 dpi остаётся в родном разрешении, и low_resolution срабатывает."""
    path = tmp_path / "lowres.png"
    fx.save(fx.text_page(width=620, height=800, line_height=12, line_gap=9, margin=45), path, 150)

    report = analyze(path, config)
    assert (report.width, report.height) == (620, 800)  # не растянут
    assert report.cv_metrics["dpi"] == pytest.approx(150.0, rel=1e-3)
    assert report.cv_metrics["line_height_px"] < 16
    # Порог, а не конкретное число: якоря пересчитываются под корпус,
    # а требование — чтобы метка поднялась выше tau_low и попала в вердикт.
    assert report.scores()["low_resolution"] > config.verdict.tau_low
    assert report.verdict != "good"


def test_bitonal_scan_without_a_model_reports_nothing_measured(config: Config, tmp_path) -> None:
    """Битональный скан: CV не измеряет контраст и шум, а сети нет — значит нечем.

    Разрыв бумага/чернила там всегда максимален, а шум обнулён самой бинаризацией:
    формально метрики дадут уверенный ноль, но это ноль от отсутствия шкалы.
    Отсутствие метки в defects обязано читаться как «не измерено», а не «дефекта
    нет», — иначе автоматическая проверка примет непроверенный скан за хороший.
    Ровно поэтому по замеру эти две метки отданы сети (решение №40).
    """
    page = fx.text_page(width=1200, height=1600)
    bitonal = ((page > 128) * 255).astype(page.dtype)
    path = tmp_path / "fax.png"
    fx.save(bitonal, path, 300)

    report = analyze(path, config)
    assert report.cv_metrics["mid_tone_frac"] < config.cv.bitonal_max_mid_frac
    assert "low_contrast" in report.not_applicable
    assert "noise" in report.not_applicable
    assert "low_contrast" not in report.scores()
    assert "noise" not in report.scores()

    # Метки чужих источников тоже помечены: OCR не запускался, сети нет.
    assert set(report.not_applicable) == {"low_contrast", "noise", "unreadable"}
    # На полутоновой странице CV свои метки измеряет, чужие остаются чужими.
    halftone = analyze(scan_path(tmp_path), config)
    assert set(halftone.not_applicable) == {"low_contrast", "noise", "unreadable"}
    assert set(halftone.scores()) == set(config.sources.cv)


def test_cnn_labels_come_from_the_network(config: Config, tmp_path) -> None:
    """С моделью две метки приходят от сети, и CV по ним больше не спрашивают.

    CV-слой умеет считать контраст и шум, но проигрывает по ним сети (AP 0.425
    и 0.543 против неизмеримого на битональном корпусе), поэтому берётся сеть.
    """
    from src.models.export_onnx import page_like_batch  # noqa: F401  (тот же модуль)
    from src.models.model import build_model, export_onnx

    model_path = tmp_path / "tiny.onnx"
    export_onnx(
        build_model("mobilenetv3_small_100", config.n_labels, pretrained=False),
        model_path,
        config.data.patch_size,
    )
    with_model = config.model_copy(
        update={"paths": config.paths.model_copy(update={"onnx_model": model_path})}
    )

    report = analyze(scan_path(tmp_path), with_model)

    assert report.pipeline_version == "cv+cnn"
    by_source = {d.label: d.source for d in report.defects}
    assert by_source["low_contrast"] == "cnn"
    assert by_source["noise"] == "cnn"
    assert by_source["blur"] == "cv"
    # Неизмеримость у CV этих меток больше не наша забота: за них отвечает сеть.
    assert report.not_applicable == ["unreadable"]


def scan_path(tmp_path) -> str:
    path = tmp_path / "halftone.png"
    fx.save(fx.text_page(width=1200, height=1600), path, 300)
    return str(path)


def test_analyze_is_deterministic(config: Config, scan: str) -> None:
    assert analyze(scan, config).scores() == analyze(scan, config).scores()


def test_analyze_missing_file(config: Config, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path / "nope.jpg", config)


def test_app_handler(config: Config, scan: str) -> None:
    """Хендлер Gradio отдаёт свои выходы и не требует самой Gradio."""
    from src.app import _run

    verdict, defects, metrics, payload, page, shown = _run(None, False, config)
    assert (defects, metrics, payload, page) == ({}, [], {}, None)

    verdict, defects, metrics, payload, page, shown = _run(scan, False, config)
    assert payload["verdict"] in {"good", "acceptable", "bad"}
    # Модели в этом конфиге нет: приходят ровно метки CV-слоя, не все, что он
    # умеет считать. `low_contrast` и `noise` отданы сети (решение №40).
    assert set(defects) == set(config.sources.cv)
    assert len(metrics) > 5
    # Страница возвращается для карты дефекта, список — что можно показать.
    assert page is not None and page.ndim == 2
    assert set(shown) <= set(config.sources.cv)


# --- приложение: источники, карта, батч -------------------------------------


def test_defect_table_names_the_source(config: Config, scan: str) -> None:
    """Семь меток из десяти — детерминированные CV-метрики, две — сеть, одна — OCR.

    Без подписи пользователь читает десять чисел как одинаковые по природе.
    """
    from src.app import defect_rows
    from src.pipeline import analyze

    report = analyze(scan, config)
    rows = defect_rows(report, config)

    by_label = {row[0]: row for row in rows}
    assert by_label["blur"][2] == "CV-метрика"
    # Неизмеренное показано строкой, а не молчанием: пустое место читается
    # как «дефекта нет» (решение №21).
    assert by_label["noise"][1] == "не измерено"
    assert by_label["unreadable"][1] == "не измерено"


def test_map_is_refused_for_a_global_defect(config: Config, scan: str) -> None:
    """Подсветить «здесь размыто» на равномерно размытом скане честно нельзя."""
    import numpy as np

    from src.app import _heatmap

    gray = np.asarray(fx.text_page(width=600, height=800))
    assert _heatmap(gray, "blur", config) is None
    assert _heatmap(None, "glare", config) is None
    assert _heatmap(gray, None, config) is None


def test_map_is_drawn_for_a_local_defect(config: Config) -> None:
    import numpy as np

    from src.app import _heatmap

    gray = np.asarray(fx.text_page(width=600, height=800))
    image = _heatmap(gray, "glare", config)

    assert image is not None
    assert image.shape == (*gray.shape, 3)


def test_batch_survives_a_broken_file(config: Config, tmp_path, scan: str) -> None:
    """Один битый файл не должен ронять весь батч: это пакетный режим."""
    from src.app import batch_rows

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    rows = batch_rows([scan, str(broken)], False, config)

    assert len(rows) == 2
    assert rows[0][1] in {"good", "acceptable", "bad"}
    assert rows[1][1] == "ошибка"
