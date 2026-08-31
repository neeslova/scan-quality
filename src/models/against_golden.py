"""Сверка вердиктов системы с эталонным набором good/bad.

Отвечает на вопрос защиты «насколько система согласна с человеком» и делает это
двумя независимыми способами, потому что они ловят разное.

**Порогозависимые метрики** (матрица ошибок, precision/recall/F1, каппа Коэна)
меряют систему целиком — вместе с порогами вердикта. Они честно проседают, если
шкала корпуса смещена, даже когда сам детектор различает классы прекрасно.

**ROC-AUC считается по риску `1 - quality_score` и порога не знает вовсе.**
Именно он отделяет «детектор слепой» от «детектор видит, но порог не тот»:
в первом случае AUC около 0.5, во втором он высокий при никуда не годном F1.
Различать эти два случая обязательно — лечатся они противоположным.

Трёхуровневый вердикт сводится к двум классам двумя способами сразу, потому что
`acceptable` — это «посмотри глазами», и относить его к годным или к браку
зависит от того, есть ли у потока ручной досмотр. Обе строки печатаются рядом.

Запуск:
    python -m src.models.against_golden --golden data/labeled/golden_tg.jsonl \\
        --reports reports/tg_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.golden import read_golden
from src.schema import GoldenRecord

logger = logging.getLogger(__name__)

# Как трёхуровневый вердикт сводится к двум классам.
BINARIZATIONS = {
    "strict": ("good",),  # годным считается только `good`
    "lenient": ("good", "acceptable"),  # `acceptable` уходит в годные
}


@dataclass(frozen=True)
class Pair:
    """Одна сопоставленная страница: что сказал человек и что сказала система."""

    key: str
    truth: str  # good | bad
    verdict: str  # good | acceptable | bad
    risk: float  # 1 - quality_score, выше — хуже


def report_key(image: str, page: int) -> str:
    """Ключ страницы в том виде, в каком его пишет пайплайн.

    `build_report` кладёт в отчёт только имя файла, а страницы PDF помечает
    единицей больше: `scan.pdf#2` — это вторая страница, индекс 1.
    """
    return image if page == 0 else f"{image}#{page + 1}"


def index_golden(records: list[GoldenRecord]) -> tuple[dict[str, GoldenRecord], set[str]]:
    """Эталон по ключу отчёта плюс ключи, которые оказались неоднозначны.

    Отчёт не хранит папку класса, поэтому два разных файла с одинаковым именем
    в разных классах дают один и тот же ключ. Сопоставить их не с чем: из отчёта
    не видно, какой из двух это был. Такие ключи выбывают из оценки — иначе
    половина пары гарантированно засчиталась бы как ошибка на ровном месте.
    """
    index: dict[str, GoldenRecord] = {}
    ambiguous: set[str] = set()

    for record in records:
        key = report_key(Path(record.image).name, record.page)
        previous = index.get(key)
        if previous is not None and previous.sha256 != record.sha256:
            ambiguous.add(key)
            continue
        index[key] = record

    for key in ambiguous:
        index.pop(key, None)
    return index, ambiguous


def load_reports(path: Path) -> dict[str, dict]:
    """Отчёты по ключу. Побеждает последняя запись: прогон мог быть повторён."""
    reports: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                report = json.loads(line)
                reports[report["image"]] = report
    return reports


def match(golden: list[GoldenRecord], reports: dict[str, dict]) -> tuple[list[Pair], list[str]]:
    """Пары «эталон — отчёт» и список эталонных страниц, для которых отчёта нет."""
    index, ambiguous = index_golden(golden)
    if ambiguous:
        logger.warning(
            "%d страниц исключены: одинаковое имя файла в разных классах (%s)",
            len(ambiguous),
            ", ".join(sorted(ambiguous)[:3]),
        )

    pairs: list[Pair] = []
    missing: list[str] = []
    for key, record in sorted(index.items()):
        report = reports.get(key)
        if report is None:
            missing.append(key)
            continue
        pairs.append(
            Pair(
                key=key,
                truth=record.label,
                verdict=report["verdict"],
                risk=1.0 - float(report["quality_score"]),
            )
        )
    return pairs, missing


def confusion(pairs: list[Pair], good_verdicts: tuple[str, ...]) -> dict[str, int]:
    """Матрица ошибок. `bad` — положительный класс: пропустить брак дороже."""
    counts: Counter = Counter()
    for pair in pairs:
        predicted = "good" if pair.verdict in good_verdicts else "bad"
        counts[(pair.truth, predicted)] += 1
    return {
        "tp": counts[("bad", "bad")],  # брак пойман
        "fn": counts[("bad", "good")],  # брак пропущен — самая дорогая ошибка
        "fp": counts[("good", "bad")],  # годная страница отвергнута
        "tn": counts[("good", "good")],
    }


def rates(matrix: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = matrix["tp"], matrix["fn"], matrix["fp"], matrix["tn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    total = tp + fn + fp + tn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else float("nan"),
        # Доля годных страниц, ошибочно отвергнутых: цена внедрения для оператора.
        "false_alarm": fp / (fp + tn) if fp + tn else float("nan"),
    }


def cohen_kappa(pairs: list[Pair], good_verdicts: tuple[str, ...]) -> float:
    """Согласие с человеком поверх случайного. Без него accuracy обманывает.

    На вырожденном классификаторе, объявляющем брак почти всегда, accuracy равна
    доле брака в наборе и выглядит прилично. Каппа в этом случае около нуля,
    потому что такое согласие достигается и случайным угадыванием.
    """
    from sklearn.metrics import cohen_kappa_score

    if not pairs:
        return float("nan")
    truth = [p.truth for p in pairs]
    predicted = ["good" if p.verdict in good_verdicts else "bad" for p in pairs]
    if len(set(truth)) < 2 or len(set(predicted)) < 2:
        # Один класс с любой стороны — каппа не определена, а не равна нулю.
        return float("nan")
    return float(cohen_kappa_score(truth, predicted))


def roc_auc(pairs: list[Pair]) -> float:
    """AUC по риску. Порог не участвует — это свойство самой шкалы."""
    from sklearn.metrics import roc_auc_score

    truth = np.array([1 if p.truth == "bad" else 0 for p in pairs])
    if truth.size == 0 or len(set(truth.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(truth, np.array([p.risk for p in pairs])))


def format_report(pairs: list[Pair], missing: list[str]) -> str:
    lines: list[str] = []
    good = sum(1 for p in pairs if p.truth == "good")
    lines.append(
        f"Сопоставлено страниц: {len(pairs)} (эталон: good {good}, bad {len(pairs) - good})"
    )
    if missing:
        lines.append("Нет отчёта для {} страниц: {}".format(len(missing), ", ".join(missing[:5])))

    lines.append("")
    lines.append(f"ROC-AUC по риску (порога не знает): {roc_auc(pairs):.3f}")
    lines.append("")

    header = "{:10s}{:>7s}{:>7s}{:>7s}{:>7s}{:>13s}{:>8s}".format(
        "сведение", "P", "R", "F1", "acc", "ложн.тревог", "каппа"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, good_verdicts in BINARIZATIONS.items():
        values = rates(confusion(pairs, good_verdicts))
        kappa = cohen_kappa(pairs, good_verdicts)
        kappa_text = "—" if np.isnan(kappa) else f"{kappa:.3f}"
        lines.append(
            "{:10s}{:7.3f}{:7.3f}{:7.3f}{:7.3f}{:13.3f}{:>8s}".format(
                name,
                values["precision"],
                values["recall"],
                values["f1"],
                values["accuracy"],
                values["false_alarm"],
                kappa_text,
            )
        )

    lines.append("")
    for name, good_verdicts in BINARIZATIONS.items():
        matrix = confusion(pairs, good_verdicts)
        lines.append(
            "{}: брак пойман {}, брак пропущен {}, годных отвергнуто {}, годных принято {}".format(
                name, matrix["tp"], matrix["fn"], matrix["fp"], matrix["tn"]
            )
        )

    counts = Counter(p.verdict for p in pairs)
    lines.append("")
    lines.append(
        "вердикты системы: "
        + ", ".join(f"{name} {counts.get(name, 0)}" for name in ("good", "acceptable", "bad"))
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True, help="эталон, jsonl")
    parser.add_argument("--reports", type=Path, required=True, help="отчёты пайплайна, jsonl")
    parser.add_argument("--out", type=Path, default=None, help="куда записать текст отчёта")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pairs, missing = match(read_golden(args.golden), load_reports(args.reports))
    if not pairs:
        raise SystemExit("Ни одна страница эталона не сопоставилась с отчётами")

    text = format_report(pairs, missing)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("отчёт записан в %s", args.out)


if __name__ == "__main__":
    main()
