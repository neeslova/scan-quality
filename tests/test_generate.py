"""Отбор эталонов и контроль частот: главное — не пустить val/test в синтетику."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from src.config import Config, load_config
from src.data.degrade import DEGRADATIONS
from src.data.generate import (
    downscale_mask,
    held_out_documents,
    pick_labels,
    select_references,
)
from src.labeling.app import append_label
from src.schema import LabelRecord, PrelabelRecord
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def make_prelabels(config: Config, n: int = 200) -> list[PrelabelRecord]:
    labels = list(config.cv.scores)
    pool = []
    for i in range(n):
        # Каждая третья страница чистая, остальные с заметным дефектом.
        clean = i % 3 == 0
        scores = dict.fromkeys(labels, 0.02 if clean else 0.75)
        pool.append(
            PrelabelRecord(
                image=f"{i:07d}.jpg",
                document=f"{i // 4:07d}",
                corpus="test",
                scores=scores,
            )
        )
    return pool


def write_splits(tmp_path, documents: dict[str, list[str]]):
    directory = tmp_path / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    for name, docs in documents.items():
        payload = {"documents": docs, "images": [], "n_documents": len(docs), "n_images": 0}
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


# --- отложенные документы ---------------------------------------------------


def test_held_out_covers_val_and_test(tmp_path) -> None:
    directory = write_splits(tmp_path, {"train": ["a"], "val": ["b", "c"], "test": ["d"]})
    assert held_out_documents(directory) == {"b", "c", "d"}


def test_missing_split_file_is_fatal(tmp_path) -> None:
    directory = write_splits(tmp_path, {"train": ["a"], "val": ["b"]})
    with pytest.raises(SystemExit, match="test.json"):
        held_out_documents(directory)


# --- отбор эталонов ---------------------------------------------------------


def test_references_never_come_from_val_or_test(config: Config) -> None:
    """Риск утечки №1 через синтетику: деградированная копия тестовой страницы."""
    prelabels = make_prelabels(config)
    held_out = {r.document for r in prelabels[:120]}

    references = select_references(prelabels, None, held_out, config)

    assert references
    assert not {r.document for r in references} & held_out


def test_only_clean_pages_become_references(config: Config) -> None:
    prelabels = make_prelabels(config)
    references = select_references(prelabels, None, set(), config)

    limit = config.synth.reference.max_defect_score
    for record in references:
        assert all(score <= limit for score in record.scores.values())


def test_human_confirmed_clean_goes_first(config: Config, tmp_path) -> None:
    """Ручная отметка «чисто» — самое надёжное свидетельство, какое у нас есть."""
    prelabels = make_prelabels(config)
    dirty_by_cv = prelabels[1]  # у него все скоры 0.75, по CV не эталон

    labels_path = tmp_path / "labels.jsonl"
    append_label(
        labels_path,
        LabelRecord(
            image=dirty_by_cv.image,
            document=dirty_by_cv.document,
            corpus="test",
            labels=dict.fromkeys(config.manual_labels, False),
        ),
    )

    references = select_references(prelabels, labels_path, set(), config)
    assert references[0].image == dirty_by_cv.image


def test_no_references_is_fatal_downstream(config: Config) -> None:
    prelabels = make_prelabels(config)
    held_out = {r.document for r in prelabels}
    assert select_references(prelabels, None, held_out, config) == []


# --- уменьшение масок -------------------------------------------------------


def test_thin_stripes_survive_downscale() -> None:
    """Полосы шириной в пиксель обязаны пережить уменьшение маски.

    При усреднении такая полоса даёт значение около 25 и при пороге 127
    пропадает — метка локализации теряется молча.
    """
    mask = np.zeros((2300, 1700), dtype=np.uint8)
    for column in (200, 640, 1100, 1500):
        mask[:, column : column + 1] = 255

    small = downscale_mask(mask)

    assert max(small.shape) == 256
    assert small.max() == 255
    assert (small > 127).any(axis=0).sum() >= 4  # все четыре полосы на месте


def test_downscale_keeps_aspect_and_area() -> None:
    mask = np.zeros((2000, 1000), dtype=np.uint8)
    mask[500:1500, 250:750] = 255  # четверть площади

    small = downscale_mask(mask)

    assert small.shape[0] / small.shape[1] == pytest.approx(2.0, abs=0.05)
    assert float((small > 127).mean()) == pytest.approx(0.25, abs=0.05)


def test_small_mask_is_left_alone() -> None:
    mask = np.zeros((120, 90), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert np.array_equal(downscale_mask(mask), mask)


# --- контроль частот --------------------------------------------------------


def test_never_exceeds_max_defects(config: Config) -> None:
    rng = np.random.default_rng(0)
    counts: Counter = Counter()
    for index in range(200):
        picked = pick_labels(rng, config, counts, index, 200)
        assert 1 <= len(picked) <= config.synth.max_defects_per_image
        assert len(set(picked)) == len(picked)


def test_always_at_least_one_defect(config: Config) -> None:
    """Чистых страниц в синтетике не делаем — чистые у нас и так реальные."""
    rng = np.random.default_rng(1)
    counts: Counter = Counter()
    for index in range(100):
        assert pick_labels(rng, config, counts, index, 100)


def test_rare_labels_reach_the_quota(config: Config) -> None:
    """Смысл всего механизма: к концу прогона отстающие метки догоняют квоту."""
    rng = np.random.default_rng(7)
    counts: Counter = Counter()
    total = 1200
    for index in range(total):
        counts.update(pick_labels(rng, config, counts, index, total))

    quota = config.synth.min_label_share
    for label in DEGRADATIONS:
        share = counts[label] / total
        assert share >= quota * 0.9, f"{label}: {share:.1%} против квоты {quota:.0%}"


def test_manifest_survives_a_bom(tmp_path) -> None:
    """Инструменты Windows дописывают BOM, и парсер JSON падает невнятно."""
    from src.data.generate import read_manifest
    from src.schema import SyntheticRecord

    record = SyntheticRecord(
        image="images/a.jpg", reference="a.jpg", document="a", corpus="t", width=10, height=10
    )
    path = tmp_path / "manifest.jsonl"
    path.write_text("﻿" + record.model_dump_json() + "\n", encoding="utf-8")

    assert read_manifest(path)[0].image == "images/a.jpg"


def test_recipe_reproduces_the_same_page(config: Config, tmp_path) -> None:
    """Ради этого в записи хранится seed: рецепт восстанавливает страницу побитово.

    Иначе синтетику пришлось бы возить в Colab гигабайтами вместо мегабайтов.
    """
    from src.data.degrade import apply as apply_degradations

    page = fx.text_page(width=800, height=1000)
    labels = ["blur", "shadow", "streaks"]
    severities = dict.fromkeys(labels, 0.7)
    seed = 12345

    first, first_masks = apply_degradations(
        page, labels, severities, config, np.random.default_rng(seed)
    )
    second, second_masks = apply_degradations(
        page, labels, severities, config, np.random.default_rng(seed)
    )

    assert np.array_equal(first, second)
    for label in ("shadow", "streaks"):
        assert np.array_equal(first_masks[label], second_masks[label])


def test_deficit_raises_probability(config: Config) -> None:
    """Метка, отставшая под конец, должна выпадать почти всегда."""
    rng = np.random.default_rng(3)
    total = 100
    behind: Counter = Counter(dict.fromkeys(DEGRADATIONS, total))
    behind["streaks"] = 0

    picks = [pick_labels(rng, config, behind, 95, total) for _ in range(40)]
    assert sum("streaks" in picked for picked in picks) > 30
