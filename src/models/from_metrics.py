"""Вердикт good/bad как классификатор поверх CV-метрик, а не максимум по дефектам.

**Зачем.** Пайплайн сводит девять оценок дефектов в одно число правилом
`quality_score = 1 - max(дефект)`. На Tobacco, по которому калибровались якоря,
это работало. На `Data iz tg` — нет: ROC-AUC 0.508, медиана риска ровно 1.000,
186 страниц из 199 объявлены `bad`. Максимум насыщается: достаточно одной
задранной метки, чтобы страница стала плохой, а с чужими якорями задрана хоть
одна почти у каждой.

При этом сами метрики классы различают. Значит дело не в измерениях, а в том,
как они сводятся: правило `max` задано руками и ничего не знает о том, какие
метки на этом корпусе информативны. Здесь сведение не назначается, а
**выучивается** по эталону.

**Почему логистическая регрессия.** Не потому что лучшая, а потому что честная
на такой выборке: 199 страниц и 37 признаков — это область, где деревья и
бустинги показывают красивые числа на обучении и разваливаются на контроле.
Линейная модель со стандартизацией даёт ещё и читаемые веса: видно, какие
метрики несут сигнал, а какие балласт. Это ответ на вопрос «почему», а не только
«насколько», и на защите он нужнее лишних процентов.

**Честность оценки.** Риск каждой страницы берётся из той складки, где страница
была в контроле (out-of-fold), поэтому AUC не мерит способность запомнить
выборку. Заполнение пропусков и стандартизация живут внутри складки: среднее,
посчитанное по всему набору, протекло бы в контроль.

Страницы одного документа не расходятся по складкам — `StratifiedGroupKFold`
получает `document` из эталона. На tg это почти ничего не меняет, PDF там
одностраничные, но на корпусе сканов одной книги утечка была бы грубой.

Запуск:
    python -m src.models.from_metrics --reports reports/tg_baseline.jsonl \\
        --golden data/labeled/golden_tg.jsonl --out reports/tg_learned.md
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.golden import read_golden
from src.models.against_golden import (
    Pair,
    cohen_kappa,
    confusion,
    index_golden,
    load_reports,
    match,
    rates,
    roc_auc,
)

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5

# Полнота в рабочей точке. Та же, что у якорей шкалы в `calibrate.target_recall`:
# пропустить брак дороже, чем отвергнуть годную страницу (раздел 4 плана).
DEFAULT_TARGET_RECALL = 0.85

# Полнота во второй точке — та же, что у `calibrate.confident_recall`. Она задаёт
# порог «бесспорно плохо». Обе точки заданы полнотой, а не точностью, ровно по той
# же причине, что и у шкал меток: полнота монотонна по порогу, и порядок двух
# точек тем самым гарантирован по построению.
DEFAULT_CONFIDENT_RECALL = 0.50

# Порог отсечки для вердикта из непрерывного риска. Только два класса: `acceptable`
# у обученной шкалы нет — серую зону вводят порогами уровнем выше, и смешивать
# это с обучением незачем.
GOOD_VERDICTS = ("good",)


@dataclass(frozen=True)
class Dataset:
    """Сопоставленные страницы: признаки, метки и то, что на них сказал пайплайн."""

    keys: list[str]
    features: np.ndarray  # (страниц, признаков)
    labels: np.ndarray  # 1 = bad
    groups: list[str]
    names: list[str]
    baseline: list[Pair]  # пары пайплайна на тех же страницах — для сравнения


def build_dataset(golden_path: Path, reports_path: Path) -> Dataset:
    """Собирает матрицу признаков из отчётов пайплайна и меток из эталона.

    Сопоставление отдано `against_golden.match`: там же живёт разбор случая,
    когда два файла с одинаковым именем лежат в разных классах — такие страницы
    выбывают, иначе половина пары гарантированно засчиталась бы как ошибка.
    """
    golden = read_golden(golden_path)
    reports = load_reports(reports_path)
    pairs, missing = match(golden, reports)
    if missing:
        logger.warning("нет отчёта для %d страниц эталона", len(missing))

    index, _ = index_golden(golden)

    # Порядок признаков фиксируем пересечением: метрика, посчитанная не на всех
    # страницах, портила бы матрицу молча.
    common: set[str] | None = None
    for pair in pairs:
        metrics = reports[pair.key].get("cv_metrics") or {}
        common = set(metrics) if common is None else (common & set(metrics))
    names = sorted(common or set())
    if not names:
        raise SystemExit("в отчётах нет общих cv_metrics — нечему учиться")

    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    keys: list[str] = []
    for pair in pairs:
        metrics = reports[pair.key]["cv_metrics"]
        rows.append([_as_float(metrics.get(name)) for name in names])
        labels.append(1 if pair.truth == "bad" else 0)
        record = index.get(pair.key)
        groups.append((record.document if record else "") or pair.key)
        keys.append(pair.key)

    logger.info("страниц %d, признаков %d", len(rows), len(names))
    return Dataset(
        keys=keys,
        features=np.asarray(rows, dtype=float),
        labels=np.asarray(labels, dtype=int),
        groups=groups,
        names=names,
        baseline=pairs,
    )


def _as_float(value: object) -> float:
    """Пропуски и нечисловое — NaN. Заполняются они внутри складки, не здесь."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _pipeline():
    """Заполнение пропусков, стандартизация, логрегрессия — одним объектом.

    Собрано пайплайном намеренно: только так медиана и масштаб считаются по
    обучающей части складки. Разложенные по шагам вручную, они почти всегда
    оказываются посчитанными по всему набору — это самая незаметная утечка в
    такой задаче.

    `class_weight="balanced"` — из той же асимметрии, что и целевая полнота:
    брак составляет меньше половины набора, и без веса модель охотнее
    ошибается в его сторону.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def _splitter(data: Dataset, folds: int):
    """Складки с учётом документа, если документов хватает на разбиение."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if len(set(data.groups)) >= folds:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    logger.warning("документов меньше складок — разбиваем без группировки")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)


def out_of_fold_risk(data: Dataset, folds: int = DEFAULT_FOLDS) -> tuple[np.ndarray, list[float]]:
    """Риск каждой страницы из складки, где она была в контроле, и AUC по складкам.

    Возвращаются и поскладочные AUC: одно среднее число скрывает, держится
    результат на всех складках или вытянут одной удачной.
    """
    from sklearn.metrics import roc_auc_score

    risk = np.full(len(data.labels), np.nan, dtype=float)
    per_fold: list[float] = []

    for train, test in _splitter(data, folds).split(data.features, data.labels, data.groups):
        pipeline = _pipeline()
        pipeline.fit(data.features[train], data.labels[train])
        predicted = pipeline.predict_proba(data.features[test])[:, 1]
        risk[test] = predicted
        fold_truth = data.labels[test]
        if len(set(fold_truth.tolist())) > 1:
            per_fold.append(float(roc_auc_score(fold_truth, predicted)))

    return risk, per_fold


def threshold_for_recall(labels: np.ndarray, risk: np.ndarray, target: float) -> float:
    """Наибольший порог, при котором полнота по браку ещё не ниже целевой.

    Наибольший, а не любой подходящий: полнота монотонно падает с ростом порога,
    и любой порог ниже даст ту же полноту ценой лишних ложных тревог.
    """
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")

    best = float("nan")
    for candidate in sorted(set(risk.tolist()), reverse=True):
        caught = int(((risk >= candidate) & (labels == 1)).sum())
        if caught / positives >= target:
            best = float(candidate)
            break
    return best


def coefficients(data: Dataset) -> list[tuple[str, float]]:
    """Веса модели, обученной на всём наборе. Только для интерпретации.

    Оценивать по ней нельзя — она видела все страницы. Но именно она отвечает на
    вопрос, какие метрики несут сигнал, и этот ответ устойчивее, чем веса любой
    отдельной складки.
    """
    pipeline = _pipeline()
    pipeline.fit(data.features, data.labels)
    weights = pipeline.named_steps["model"].coef_[0]
    return sorted(zip(data.names, weights.tolist()), key=lambda item: -abs(item[1]))


def _pairs_at(data: Dataset, risk: np.ndarray, threshold: float) -> list[Pair]:
    """Непрерывный риск превращается в вердикт одним порогом."""
    return [
        Pair(
            key=key,
            truth="bad" if label else "good",
            verdict="bad" if value >= threshold else "good",
            risk=float(value),
        )
        for key, label, value in zip(data.keys, data.labels.tolist(), risk.tolist())
    ]


def format_report(
    data: Dataset,
    risk: np.ndarray,
    per_fold: list[float],
    target_recall: float,
    folds: int,
) -> str:
    lines: list[str] = []
    bad = int(data.labels.sum())
    lines.append(
        f"Страниц {len(data.labels)} (good {len(data.labels) - bad}, bad {bad}), "
        f"признаков {len(data.names)}, складок {folds}"
    )
    lines.append("")

    learned = _pairs_at(data, risk, threshold_for_recall(data.labels, risk, target_recall))
    lines.append("ROC-AUC (out-of-fold, порога не знает)")
    lines.append(f"  выучено:  {roc_auc(learned):.3f}")
    lines.append(f"  пайплайн: {roc_auc(data.baseline):.3f}   (правило 1 - max по дефектам)")
    if per_fold:
        spread = ", ".join(f"{value:.3f}" for value in per_fold)
        lines.append(f"  по складкам: {spread}")
    lines.append("")

    threshold = threshold_for_recall(data.labels, risk, target_recall)
    lines.append(f"Рабочая точка: порог {threshold:.3f} (целевая полнота {target_recall:.2f})")
    lines.append("")
    lines.append("модель          P      R     F1    acc  ложн.тревог   каппа")
    lines.append("-" * 62)
    for name, pairs in (("выучено", learned), ("пайплайн", data.baseline)):
        matrix = confusion(pairs, GOOD_VERDICTS)
        rate = rates(matrix)
        lines.append(
            f"{name:<12}{rate['precision']:>7.3f}{rate['recall']:>7.3f}{rate['f1']:>7.3f}"
            f"{rate['accuracy']:>7.3f}{rate['false_alarm']:>13.3f}"
            f"{cohen_kappa(pairs, GOOD_VERDICTS):>8.3f}"
        )
    lines.append("")
    for name, pairs in (("выучено", learned), ("пайплайн", data.baseline)):
        matrix = confusion(pairs, GOOD_VERDICTS)
        lines.append(
            f"{name}: брак пойман {matrix['tp']}, брак пропущен {matrix['fn']}, "
            f"годных отвергнуто {matrix['fp']}, годных принято {matrix['tn']}"
        )

    lines.append("")
    lines.append("Веса модели, обученной на всём наборе (только для интерпретации):")
    for name, weight in coefficients(data)[:12]:
        lines.append(f"  {weight:+.3f}  {name}")

    return "\n".join(lines) + "\n"


def save_bundle(
    data: Dataset,
    path: Path,
    target_recall: float = DEFAULT_TARGET_RECALL,
    confident_recall: float = DEFAULT_CONFIDENT_RECALL,
    folds: int = DEFAULT_FOLDS,
) -> dict:
    """Обучает модель на всём наборе и кладёт её рядом с порогами и именами метрик.

    Порядок признаков сохраняется вместе с моделью намеренно: матрица собирается
    из словаря `cv_metrics`, а словарь порядка не гарантирует. Разошедшийся
    порядок не упал бы, а тихо перепутал бы признаки местами — и модель
    продолжила бы выдавать правдоподобные числа.

    Пороги считаются по out-of-fold риску, а не по обучающему: на обучающем они
    вышли бы оптимистично смещёнными, и рабочая точка на живых данных оказалась
    бы не той, что обещана.
    """
    risk, _ = out_of_fold_risk(data, folds)
    pipeline = _pipeline()
    pipeline.fit(data.features, data.labels)

    payload = {
        "pipeline": pipeline,
        "names": list(data.names),
        "tau_low": threshold_for_recall(data.labels, risk, target_recall),
        "tau_high": threshold_for_recall(data.labels, risk, confident_recall),
        "pages": int(len(data.labels)),
        "target_recall": float(target_recall),
        "confident_recall": float(confident_recall),
    }

    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    logger.info("модель сохранена в %s", path)
    return payload


def load_bundle(path: Path) -> dict:
    """Модель, имена признаков и пороги — тем же составом, что сохранены.

    Файла может не быть: `models/` не хранится в репозитории, как и ONNX-сеть.
    Сообщение поэтому говорит, чем его получить, — иначе на чистом клоне вылезал
    бы голый `FileNotFoundError` из середины пайплайна.
    """
    import joblib

    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path}: модели вердикта нет. Обучить: python -m src.models.from_metrics "
            f"--reports reports/<корпус>_baseline.jsonl --golden data/labeled/<эталон>.jsonl "
            f"--save {path}"
        )
    payload = joblib.load(path)
    missing = {"pipeline", "names", "tau_low", "tau_high"} - set(payload)
    if missing:
        raise ValueError(f"{path}: в модели нет полей {sorted(missing)}")
    return payload


def bundle_risk(metrics: dict, bundle: dict) -> float:
    """Риск страницы по её сырым метрикам. Порядок признаков берётся из модели.

    Метрика, которой на странице нет, идёт как пропуск: заполнит её тот же
    `SimpleImputer`, что обучался вместе с моделью. Подставить сюда ноль было бы
    хуже молчания — ноль у половины метрик означает «идеально».
    """
    row = [[_as_float(metrics.get(name)) for name in bundle["names"]]]
    return float(bundle["pipeline"].predict_proba(np.asarray(row, dtype=float))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="jsonl прогона пайплайна")
    parser.add_argument("--golden", type=Path, required=True, help="эталон good/bad")
    parser.add_argument("--out", type=Path, default=None, help="куда записать отчёт")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--confident-recall", type=float, default=DEFAULT_CONFIDENT_RECALL)
    parser.add_argument("--save", type=Path, default=None, help="куда сохранить модель")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = build_dataset(args.golden, args.reports)
    if len(set(data.labels.tolist())) < 2:
        raise SystemExit("в выборке один класс — считать нечего")

    risk, per_fold = out_of_fold_risk(data, args.folds)
    report = format_report(data, risk, per_fold, args.target_recall, args.folds)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        logger.info("отчёт записан в %s", args.out)

    if args.save:
        payload = save_bundle(
            data, args.save, args.target_recall, args.confident_recall, args.folds
        )
        print(
            f"модель: {args.save}, пороги "
            f"acceptable >= {payload['tau_low']:.3f}, bad >= {payload['tau_high']:.3f}"
        )


if __name__ == "__main__":
    main()
