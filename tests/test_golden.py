"""Сборка эталонного набора из папок good/bad: метки, PDF, дубликаты."""

from __future__ import annotations

from pathlib import Path

from src.data.golden import collect, read_golden, summarize, write_golden
from tests import factories as fx


def _dataset(root: Path) -> Path:
    """Датасет с разным регистром имён папок — как разложено на диске."""
    (root / "Good").mkdir(parents=True)
    (root / "bad").mkdir()
    fx.save(fx.text_page(width=400, height=500), root / "Good" / "clean.png")
    fx.save(fx.blurred(fx.text_page(width=400, height=500)), root / "bad" / "blurred.png")
    return root


def test_folder_names_become_labels(tmp_path) -> None:
    records = collect(_dataset(tmp_path / "corpus"), corpus="tg")

    assert {r.image: r.label for r in records} == {
        "Good/clean.png": "good",
        "bad/blurred.png": "bad",
    }
    assert {r.corpus for r in records} == {"tg"}
    assert all(r.source == "folder" and r.sha256 for r in records)


def test_unknown_folders_are_ignored(tmp_path) -> None:
    root = _dataset(tmp_path / "corpus")
    (root / "черновики").mkdir()
    fx.save(fx.text_page(width=200, height=200), root / "черновики" / "draft.png")

    assert all("черновики" not in r.image for r in collect(root, corpus="tg"))


def test_pdf_expands_to_pages_sharing_one_document(tmp_path) -> None:
    """Метка стоит на файле, но страницы внутри него оцениваются порознь.

    Общий `document` обязателен: разъедься страницы одного PDF по train и test,
    оценка была бы завышена за счёт утечки.
    """
    fitz = __import__("fitz")
    root = _dataset(tmp_path / "corpus")
    document = fitz.open()
    for _ in range(3):
        document.new_page()
    document.save(str(root / "bad" / "scan.pdf"))
    document.close()

    pages = [r for r in collect(root, corpus="tg") if r.image.endswith("scan.pdf")]

    assert [r.page for r in pages] == [0, 1, 2]
    assert all(r.label == "bad" for r in pages)
    assert len({r.document for r in pages}) == 1


def test_same_file_twice_in_one_class_is_counted_once(tmp_path) -> None:
    root = _dataset(tmp_path / "corpus")
    (root / "bad" / "копия.png").write_bytes((root / "bad" / "blurred.png").read_bytes())

    bad = [r for r in collect(root, corpus="tg") if r.label == "bad"]
    assert len(bad) == 1


def test_same_file_in_both_classes_is_dropped_entirely(tmp_path) -> None:
    """Один скан не может быть одновременно эталоном good и bad.

    Взять любую из двух меток означало бы подмешать в эталон заведомо неверную
    строку, поэтому страница выбывает целиком, а разметчик получает предупреждение.
    """
    root = _dataset(tmp_path / "corpus")
    (root / "bad" / "clean.png").write_bytes((root / "Good" / "clean.png").read_bytes())

    records = collect(root, corpus="tg")
    assert all("clean.png" not in r.image for r in records)
    assert [r.image for r in records] == ["bad/blurred.png"]


def test_roundtrip_through_jsonl(tmp_path) -> None:
    records = collect(_dataset(tmp_path / "corpus"), corpus="tg")
    path = tmp_path / "golden.jsonl"
    write_golden(records, path)

    assert read_golden(path) == records
    assert "good 1" in summarize(records)
