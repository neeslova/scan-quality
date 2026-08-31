"""Сверка с эталоном: сопоставление страниц, матрица ошибок, вырожденные случаи."""

from __future__ import annotations

import numpy as np

from src.models.against_golden import (
    Pair,
    cohen_kappa,
    confusion,
    index_golden,
    match,
    rates,
    report_key,
    roc_auc,
)
from src.schema import GoldenRecord


def _golden(image: str, label: str, page: int = 0, sha256: str = "") -> GoldenRecord:
    return GoldenRecord(
        image=image,
        page=page,
        document=image,
        corpus="tg",
        label=label,
        sha256=sha256 or image,
    )


def _report(image: str, verdict: str, quality: float) -> dict:
    return {"image": image, "verdict": verdict, "quality_score": quality}


def test_report_key_matches_pipeline_numbering() -> None:
    """Пайплайн нумерует страницы PDF с единицы, эталон хранит индекс с нуля."""
    assert report_key("scan.pdf", 0) == "scan.pdf"
    assert report_key("scan.pdf", 1) == "scan.pdf#2"


def test_same_name_in_both_classes_is_excluded() -> None:
    """Два разных файла с одним именем неразличимы в отчёте — оба выбывают.

    Иначе одна из двух страниц гарантированно засчиталась бы как ошибка: отчёт
    хранит только имя файла, и какой из двух он описывает, установить нечем.
    """
    records = [
        _golden("Good/scale.png", "good", sha256="aaa"),
        _golden("bad/scale.png", "bad", sha256="bbb"),
    ]
    index, ambiguous = index_golden(records)

    assert ambiguous == {"scale.png"}
    assert index == {}


def test_same_file_listed_twice_is_kept_once() -> None:
    """Одинаковый хеш — это один файл, а не конфликт: страница остаётся в оценке."""
    records = [_golden("Good/a.png", "good", sha256="x"), _golden("Good/a.png", "good", sha256="x")]
    index, ambiguous = index_golden(records)

    assert not ambiguous
    assert list(index) == ["a.png"]


def test_match_reports_pages_without_reports() -> None:
    golden = [_golden("Good/a.png", "good"), _golden("bad/b.png", "bad")]
    pairs, missing = match(golden, {"a.png": _report("a.png", "good", 0.9)})

    assert [p.key for p in pairs] == ["a.png"]
    assert missing == ["b.png"]


def test_confusion_treats_bad_as_positive_class() -> None:
    pairs = [
        Pair("1", truth="bad", verdict="bad", risk=0.9),
        Pair("2", truth="bad", verdict="good", risk=0.1),
        Pair("3", truth="good", verdict="bad", risk=0.8),
        Pair("4", truth="good", verdict="good", risk=0.2),
    ]
    assert confusion(pairs, ("good",)) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}

    values = rates(confusion(pairs, ("good",)))
    assert values["precision"] == 0.5
    assert values["recall"] == 0.5
    assert values["false_alarm"] == 0.5


def test_acceptable_moves_between_classes_with_binarization() -> None:
    """`acceptable` — это «посмотри глазами», и его отнесение меняет картину."""
    pairs = [Pair("1", truth="good", verdict="acceptable", risk=0.5)]

    assert confusion(pairs, ("good",))["fp"] == 1
    assert confusion(pairs, ("good", "acceptable"))["tn"] == 1


def test_kappa_is_undefined_when_system_answers_one_class() -> None:
    """Вырожденный ответ — не нулевое согласие, а отсутствие величины.

    Ровно этот случай дал прогон на реальном корпусе: система назвала браком
    почти всё, и accuracy при этом осталась похожей на правду.
    """
    pairs = [
        Pair("1", truth="good", verdict="bad", risk=0.9),
        Pair("2", truth="bad", verdict="bad", risk=0.9),
    ]
    assert np.isnan(cohen_kappa(pairs, ("good",)))


def test_roc_auc_is_blind_to_threshold() -> None:
    """Риск ранжирует классы верно, хотя вердикт у всех страниц одинаковый."""
    pairs = [
        Pair("1", truth="good", verdict="bad", risk=0.10),
        Pair("2", truth="good", verdict="bad", risk=0.20),
        Pair("3", truth="bad", verdict="bad", risk=0.80),
        Pair("4", truth="bad", verdict="bad", risk=0.90),
    ]
    assert roc_auc(pairs) == 1.0
    assert np.isnan(cohen_kappa(pairs, ("good",)))


def test_roc_auc_undefined_on_single_class() -> None:
    assert np.isnan(roc_auc([Pair("1", truth="bad", verdict="bad", risk=0.9)]))
