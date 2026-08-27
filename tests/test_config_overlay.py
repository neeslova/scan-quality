"""Покорпусные оверлеи конфига: слияние, валидация, изоляция от базового файла."""

from __future__ import annotations

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, deep_merge, load_config


def test_deep_merge_replaces_scalars_and_merges_dicts() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "keep": "same"}
    result = deep_merge(base, {"a": 2, "nested": {"y": 20, "z": 30}})

    assert result == {"a": 2, "nested": {"x": 1, "y": 20, "z": 30}, "keep": "same"}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}, "keep": "same"}  # исходник не тронут


def test_deep_merge_replaces_lists_wholesale() -> None:
    """Список заменяется целиком: иначе оверлей мог бы только расширять набор меток."""
    assert deep_merge({"labels": ["a", "b", "c"]}, {"labels": ["a"]}) == {"labels": ["a"]}


def test_overlay_changes_only_what_it_names(tmp_path) -> None:
    base = load_config()
    overlay = tmp_path / "corpus.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {"cv": {"scores": {"cropped": {"metric": "border_ink_frac", "good": 0.5, "bad": 0.9}}}}
        ),
        encoding="utf-8",
    )

    merged = load_config(overlays=overlay)

    assert merged.cv.scores["cropped"].good == 0.5
    assert merged.cv.scores["cropped"].bad == 0.9
    # всё остальное осталось от базового конфига
    assert merged.cv.scores["blur"] == base.cv.scores["blur"]
    assert merged.labels == base.labels
    assert merged.verdict == base.verdict


def test_overlays_apply_in_order(tmp_path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump({"verdict": {"tau_low": 0.1}}), encoding="utf-8")
    second.write_text(yaml.safe_dump({"verdict": {"tau_low": 0.2}}), encoding="utf-8")

    assert load_config(overlays=[first, second]).verdict.tau_low == 0.2


def test_overlay_is_validated(tmp_path) -> None:
    """Оверлей проходит ту же валидацию: сломать конфиг мимо схемы нельзя."""
    overlay = tmp_path / "broken.yaml"
    overlay.write_text(yaml.safe_dump({"verdict": {"tau_low": 0.9}}), encoding="utf-8")

    with pytest.raises(ValueError, match="tau_low"):
        load_config(overlays=overlay)


def test_missing_overlay_reports_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(overlays=tmp_path / "nope.yaml")


def test_shipped_corpus_overlays_load() -> None:
    """Все оверлеи в configs/corpora должны накладываться на базовый без ошибок."""
    corpora = DEFAULT_CONFIG_PATH.parent / "corpora"
    if not corpora.is_dir():
        pytest.skip("оверлеи ещё не добавлены")

    files = sorted(corpora.glob("*.yaml"))
    assert files, "папка corpora есть, но пуста"
    for path in files:
        config = load_config(overlays=path)
        assert set(config.cv.scores) <= set(config.labels), path.name


# --- источники меток --------------------------------------------------------


def test_sources_cover_every_label_exactly_once() -> None:
    """Пропуск означал бы метку, которую никто не считает, дубль — тихий арбитраж."""
    config = load_config()
    assigned = config.sources.all_labels()

    assert sorted(assigned) == sorted(config.labels)
    assert len(assigned) == len(set(assigned))


def test_label_without_a_source_is_rejected(tmp_path) -> None:
    overlay = tmp_path / "gap.yaml"
    overlay.write_text(
        yaml.safe_dump({"sources": {"cv": ["blur"], "cnn": [], "ocr": ["unreadable"]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="без источника"):
        load_config(overlays=overlay)


def test_label_from_two_sources_is_rejected(tmp_path) -> None:
    config = load_config()
    both = {
        "cv": list(config.sources.cv),
        "cnn": [*config.sources.cnn, "blur"],
        "ocr": list(config.sources.ocr),
    }
    overlay = tmp_path / "clash.yaml"
    overlay.write_text(yaml.safe_dump({"sources": both}), encoding="utf-8")

    with pytest.raises(ValueError, match="у двух источников"):
        load_config(overlays=overlay)


def test_ocr_derived_follows_sources() -> None:
    """`ocr_derived` больше не отдельное поле: обучение спрашивает у `sources`."""
    config = load_config()
    assert config.ocr_derived == config.sources.ocr
    assert config.sources.of("unreadable") == "ocr"
