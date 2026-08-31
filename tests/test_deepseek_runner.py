"""Прогон DeepSeek-OCR: сбор заданий, докатка после обрыва, запись результата.

Сама модель здесь не участвует — она требует CUDA и в тестовом окружении её нет.
Проверяется обвязка, из-за отказов которой теряются часы GPU: пропуск готового,
переживание битой строки и то, что PDF разворачивается постранично.
"""

from __future__ import annotations

import json

from src.ocr.deepseek import (
    RESOLUTION_MODES,
    DeepSeekOCR,
    PageResult,
    collect_jobs,
    load_done,
)
from tests import factories as fx


def test_collect_expands_pdf_and_keeps_images(tmp_path) -> None:
    import fitz

    fx.save(fx.text_page(width=300, height=400), tmp_path / "scan.png")
    document = fitz.open()
    for _ in range(3):
        document.new_page()
    document.save(str(tmp_path / "doc.pdf"))
    document.close()

    jobs = collect_jobs(tmp_path)

    assert [(j.relative, j.page) for j in jobs] == [
        ("doc.pdf", 0),
        ("doc.pdf", 1),
        ("doc.pdf", 2),
        ("scan.png", 0),
    ]
    # Страницы одного PDF делят хеш файла, но различаются ключом.
    pdf_jobs = [j for j in jobs if j.relative == "doc.pdf"]
    assert len({j.sha256 for j in pdf_jobs}) == 1
    assert len({j.key for j in pdf_jobs}) == 3


def test_unsupported_files_are_skipped(tmp_path) -> None:
    fx.save(fx.text_page(width=200, height=200), tmp_path / "scan.png")
    (tmp_path / "notes.docx").write_bytes(b"nope")

    assert [j.relative for j in collect_jobs(tmp_path)] == ["scan.png"]


def test_done_keys_skip_finished_pages(tmp_path) -> None:
    """Прерванная сессия Colab не должна пересчитывать готовое: это часы GPU."""
    path = tmp_path / "out.jsonl"
    path.write_text(
        PageResult(image="a.png", page=0, sha256="abc", texts={"tiny": "текст"}).to_json() + "\n",
        encoding="utf-8",
    )

    assert load_done(path) == {"abc#0"}


def test_failed_pages_are_retried(tmp_path) -> None:
    """Страница с ошибкой не считается готовой — на следующем запуске повторяем."""
    path = tmp_path / "out.jsonl"
    failed = PageResult(image="a.png", page=0, sha256="abc", status="failed", error="OOM")
    path.write_text(failed.to_json() + "\n", encoding="utf-8")

    assert load_done(path) == set()


def test_truncated_last_line_survives(tmp_path) -> None:
    """Обрыв в момент записи оставляет неполную строку — падать из-за неё нельзя."""
    path = tmp_path / "out.jsonl"
    good = PageResult(image="a.png", page=0, sha256="abc").to_json()
    path.write_text(good + '\n{"image": "b.png", "sha', encoding="utf-8")

    assert load_done(path) == {"abc#0"}


def test_result_serialises_all_modes(tmp_path) -> None:
    result = PageResult(
        image="a.png",
        page=1,
        sha256="abc",
        texts={"tiny": "низкое", "base": "высокое"},
        elapsed_s={"tiny": 1.5, "base": 4.0},
    )
    payload = json.loads(result.to_json())

    assert payload["texts"] == {"tiny": "низкое", "base": "высокое"}
    assert payload["status"] == "ok"
    assert payload["page"] == 1


def test_self_consistency_modes_differ_in_resolution() -> None:
    """Пара режимов обязана реально отличаться входом, иначе сравнивать нечего."""
    from src.ocr.deepseek import DEFAULT_MODES

    sizes = {RESOLUTION_MODES[mode][0] for mode in DEFAULT_MODES}
    assert len(DEFAULT_MODES) == 2
    assert len(sizes) == 2


def test_load_asks_for_half_precision_and_sharded_weights(monkeypatch) -> None:
    """Без `torch_dtype` веса материализуются на CPU в float32 — и сеанс умирает.

    Три миллиарда параметров в float32 — почти 12 ГБ, а бесплатный Colab даёт
    12.7 ГБ ОЗУ: загрузка убивает сеанс, не дойдя до карты. Это уже стоило
    одного прогона. Проверяем на подставном `transformers`, потому что настоящая
    модель требует CUDA, которой в тестовом окружении нет.
    """
    import sys
    import types

    import torch

    captured: dict = {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(eval=lambda: "модель")

    fake = types.ModuleType("transformers")
    fake.AutoModel = _AutoModel
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: "токенизатор")
    monkeypatch.setitem(sys.modules, "transformers", fake)

    DeepSeekOCR().load()

    assert captured["torch_dtype"] is torch.bfloat16
    assert captured["low_cpu_mem_usage"] is True
