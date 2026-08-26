"""Сплит по документам: главное требование — страницы одного дела не разъезжаются."""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.data.split import (
    build_split,
    check_leakage,
    document_id,
    group_by_document,
    label_frequencies,
    read_labels,
    split_documents,
    write_frequency_report,
    write_split,
)
from src.schema import LabelRecord


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def make_records(n_docs: int = 40, pages: int = 3) -> list[LabelRecord]:
    return [
        LabelRecord(
            image=f"{doc:07d}{page}.jpg",
            document=f"{doc:07d}",
            corpus="test",
            labels={"blur": page == 0, "glare": doc % 7 == 0},
        )
        for doc in range(n_docs)
        for page in range(pages)
    ]


@pytest.mark.parametrize(
    ("name", "strategy", "expected"),
    [
        ("0000002770.jpg", "bates7", "0000002"),
        ("0000002771.jpg", "bates7", "0000002"),
        ("515575465+-5466.jpg", "bates7", "5155754"),
        ("scan_no_digits.jpg", "bates7", "scan_no_digits"),
        ("0000002770.jpg", "stem", "0000002770"),
    ],
)
def test_document_id(name: str, strategy: str, expected: str) -> None:
    assert document_id(f"data/raw/{name}", strategy) == expected


def test_document_id_parent() -> None:
    assert document_id("data/raw/dogovor_17/page_1.jpg", "parent") == "dogovor_17"


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        document_id("a.jpg", "magic")


def test_no_document_crosses_splits(config: Config) -> None:
    """Риск №1: страницы одного дела в train и test завышают метрики на 10-15%."""
    split = build_split(make_records(), config)
    assert check_leakage(split) == []

    owner: dict[str, str] = {}
    for name, part in split.items():
        for image in part["images"]:
            doc = image[:7]
            assert owner.setdefault(doc, name) == name, f"{doc} разъехался"


def test_page_shares_match_ratios(config: Config) -> None:
    records = make_records(n_docs=60, pages=4)
    split = build_split(records, config)

    total = len(records)
    for name, expected in config.split.ratios.items():
        share = split[name]["n_images"] / total
        assert share == pytest.approx(expected, abs=0.05), name


def test_uneven_documents_still_balance_pages(config: Config) -> None:
    """Документы разной длины: делим по страницам, а не по количеству документов."""
    records = [
        LabelRecord(image=f"{doc:07d}{page}.jpg", document=f"{doc:07d}", corpus="t", labels={})
        for doc, count in enumerate([20, 15, 10, 5, 3, 3, 2, 2, 1, 1, 1, 1])
        for page in range(count)
    ]
    split = build_split(records, config)

    # Проверяем against конфиг, а не against зашитые числа: доли сплита меняются
    # вместе со стратегией разметки. Допуск шире, чем в ровном случае: документ
    # в 20 страниц из 64 нельзя разрезать.
    for name, expected in config.split.ratios.items():
        share = split[name]["n_images"] / len(records)
        assert share == pytest.approx(expected, abs=0.15), name
        assert split[name]["n_images"] > 0, name


def test_split_is_deterministic(config: Config) -> None:
    records = make_records()
    assert build_split(records, config) == build_split(records, config)


def test_split_seed_changes_result(config: Config) -> None:
    records = make_records()
    other = config.model_copy(update={"split": config.split.model_copy(update={"seed": 777})})
    assert build_split(records, config) != build_split(records, other)


def test_empty_input(config: Config) -> None:
    split = build_split([], config)
    assert all(part["n_images"] == 0 for part in split.values())


def test_split_documents_covers_everything(config: Config) -> None:
    groups = group_by_document(make_records())
    assigned = split_documents(groups, config.split.ratios, config.split.seed)
    flat = [doc for docs in assigned.values() for doc in docs]
    assert sorted(flat) == sorted(groups)
    assert len(flat) == len(set(flat))


def test_label_frequencies(config: Config) -> None:
    counts = label_frequencies(make_records(n_docs=7, pages=3), config.labels)
    assert counts["blur"] == 7  # по одной первой странице на документ
    assert counts["unreadable"] == 0


def test_read_labels_skips_broken_lines(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    first = LabelRecord(image="a.jpg", document="a", corpus="t", labels={"blur": True})
    second = LabelRecord(image="b.jpg", document="b", corpus="t", labels={"blur": False})
    path.write_text(
        first.model_dump_json() + "\n{не json}\n\n" + second.model_dump_json() + "\n",
        encoding="utf-8",
    )
    assert len(read_labels(path)) == 2


def test_read_labels_applies_last_wins(tmp_path) -> None:
    """Переразмеченная страница не должна считаться дважды и раздувать набор."""
    path = tmp_path / "labels.jsonl"
    old = LabelRecord(image="a.jpg", document="a", corpus="t", labels={"blur": True})
    new = LabelRecord(image="a.jpg", document="a", corpus="t", labels={"blur": False})
    path.write_text(old.model_dump_json() + "\n" + new.model_dump_json() + "\n", encoding="utf-8")

    records = read_labels(path)
    assert len(records) == 1
    assert records[0].labels["blur"] is False

    # История правок доступна отдельно: по ней видно, где предразметка врёт
    assert len(read_labels(path, dedupe=False)) == 2


def test_write_split_creates_three_files(config: Config, tmp_path) -> None:
    split = build_split(make_records(), config)
    write_split(split, tmp_path)

    for name in ("train", "val", "test"):
        assert (tmp_path / f"{name}.json").is_file()


def test_frequency_report_flags_problems(config: Config, tmp_path) -> None:
    records = make_records(n_docs=20, pages=3)
    split = build_split(records, config)
    path = tmp_path / "reports" / "label_frequencies.md"

    write_frequency_report(records, split, config, path)
    text = path.read_text(encoding="utf-8")

    assert "Частоты меток" in text
    for label in config.labels:
        assert f"`{label}`" in text
    # streaks в этих записях не встречается ни разу — это должно быть видно
    assert "нет примеров" in text
    # unreadable ставится автоматом в С3, а не руками
    assert "ставится автоматом" in text


def test_check_leakage_detects_planted_duplicate(config: Config) -> None:
    split = build_split(make_records(), config)
    split["val"]["documents"].append(split["train"]["documents"][0])
    assert check_leakage(split)
