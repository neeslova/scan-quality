"""Отбор эталонов и контроль частот: главное — не пустить val/test в синтетику."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from src.config import Config, load_config
from src.data.degrade import DEGRADATIONS
from src.data.generate import held_out_documents, pick_labels, select_references
from src.labeling.app import append_label
from src.schema import LabelRecord, PrelabelRecord


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


def test_deficit_raises_probability(config: Config) -> None:
    """Метка, отставшая под конец, должна выпадать почти всегда."""
    rng = np.random.default_rng(3)
    total = 100
    behind: Counter = Counter(dict.fromkeys(DEGRADATIONS, total))
    behind["streaks"] = 0

    picks = [pick_labels(rng, config, behind, 95, total) for _ in range(40)]
    assert sum("streaks" in picked for picked in picks) > 30
