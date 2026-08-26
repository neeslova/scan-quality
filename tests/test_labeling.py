"""Предразметка, отбор очереди и запись меток."""

from __future__ import annotations

import random

import pytest

from src.config import Config, load_config
from src.labeling.app import append_label, first_unlabeled, load_done
from src.labeling.prelabel import read_prelabels, select_queue
from src.schema import LabelRecord, PrelabelRecord


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def make_pool(config: Config, n: int = 400, seed: int = 3) -> list[PrelabelRecord]:
    """Корпус, где blur встречается часто, а streaks — редко."""
    rng = random.Random(seed)
    labels = [label for label in config.manual_labels if label in config.cv.scores]
    pool = []
    for i in range(n):
        scores = {label: round(rng.betavariate(1.2, 8.0), 3) for label in labels}
        if i % 3 == 0:
            scores["blur"] = round(rng.uniform(0.5, 1.0), 3)
        if i % 97 == 0:  # редкий дефект: ~4 страницы на 400
            scores["streaks"] = round(rng.uniform(0.7, 1.0), 3)
        pool.append(
            PrelabelRecord(
                image=f"{i:07d}.jpg",
                document=f"{i // 3:07d}",
                corpus="test",
                scores=scores,
                suggested={k: v >= config.labeling.suggest_threshold for k, v in scores.items()},
            )
        )
    return pool


def test_queue_size_and_uniqueness(config: Config) -> None:
    pool = make_pool(config, n=400)
    queue = select_queue(pool, config)

    assert len(queue) == min(config.labeling.sample_total, len(pool))
    assert len({r.image for r in queue}) == len(queue)


def test_queue_never_exceeds_pool(config: Config) -> None:
    pool = make_pool(config, n=25)
    assert len(select_queue(pool, config)) == 25


def test_queue_is_deterministic(config: Config) -> None:
    pool = make_pool(config, n=300)
    first = [r.image for r in select_queue(pool, config)]
    second = [r.image for r in select_queue(pool, config)]
    assert first == second


def test_rare_defect_gets_into_queue(config: Config) -> None:
    """Смысл отбора: редкая метка не должна потеряться в случайной выборке.

    Берём заведомо маленькую цель, при которой чистый рандом почти наверняка
    пропустил бы четыре страницы с `streaks` из четырёхсот.
    """
    pool = make_pool(config, n=400)
    small = config.model_copy(
        update={"labeling": config.labeling.model_copy(update={"sample_total": 40})}
    )
    queue = select_queue(pool, small)

    rare = [r for r in queue if r.scores.get("streaks", 0.0) > 0.6]
    assert rare, "редкий дефект не попал в очередь"


def test_random_share_is_respected(config: Config) -> None:
    """При random_share = 0 очередь целиком набирается по подозрительности."""
    pool = make_pool(config, n=400)
    only_suspicious = config.model_copy(
        update={
            "labeling": config.labeling.model_copy(update={"sample_total": 60, "random_share": 0.0})
        }
    )
    queue = select_queue(pool, only_suspicious)
    suspicion = [r.suspicion for r in queue]

    assert len(queue) == 60
    assert sum(s > 0.5 for s in suspicion) > len(queue) * 0.7


def test_empty_pool(config: Config) -> None:
    assert select_queue([], config) == []


def test_suspicion_is_worst_label() -> None:
    record = PrelabelRecord(
        image="a.jpg", document="a", corpus="t", scores={"blur": 0.2, "glare": 0.8}
    )
    assert record.suspicion == pytest.approx(0.8)
    assert PrelabelRecord(image="b.jpg", document="b", corpus="t").suspicion == 0.0


def test_prelabels_roundtrip(tmp_path, config: Config) -> None:
    path = tmp_path / "prelabels.jsonl"
    pool = make_pool(config, n=5)
    path.write_text("\n".join(r.model_dump_json() for r in pool), encoding="utf-8")

    restored = read_prelabels(path)
    assert [r.image for r in restored] == [r.image for r in pool]


def test_missing_prelabels_file_is_empty(tmp_path) -> None:
    assert read_prelabels(tmp_path / "nope.jsonl") == []


# --- запись меток -----------------------------------------------------------


def test_append_and_last_record_wins(tmp_path) -> None:
    """Повторная разметка дописывает строку; читается последняя, история цела."""
    path = tmp_path / "labels.jsonl"
    append_label(path, LabelRecord(image="a.jpg", document="a", corpus="t", labels={"blur": True}))
    append_label(path, LabelRecord(image="a.jpg", document="a", corpus="t", labels={"blur": False}))

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    done = load_done(path)
    assert len(done) == 1
    assert done["a.jpg"].labels["blur"] is False


def test_load_done_missing_file(tmp_path) -> None:
    assert load_done(tmp_path / "nope.jsonl") == {}


def test_first_unlabeled_resumes(config: Config, tmp_path) -> None:
    queue = make_pool(config, n=5)
    path = tmp_path / "labels.jsonl"
    for record in queue[:3]:
        append_label(
            path,
            LabelRecord(image=record.image, document=record.document, corpus="t", labels={}),
        )

    assert first_unlabeled(queue, load_done(path)) == 3


def test_first_unlabeled_on_finished_queue(config: Config) -> None:
    queue = make_pool(config, n=3)
    done = {
        r.image: LabelRecord(image=r.image, document=r.document, corpus="t", labels={})
        for r in queue
    }
    assert first_unlabeled(queue, done) == 0


def test_positive_labels_helper() -> None:
    record = LabelRecord(
        image="a.jpg",
        document="a",
        corpus="t",
        labels={"blur": True, "glare": False, "skew": True},
    )
    assert record.positive == ["blur", "skew"]
