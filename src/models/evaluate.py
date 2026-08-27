"""Метрики multi-label: per-label precision/recall/F1/AP и макро-сводки.

Отбор лучшей эпохи идёт по **macro-AP**, а не по F1. Причина: F1 считается при
каком-то пороге, а пороги у нас ещё не калиброваны — это работа С6, и делается
она по PR-кривым уже обученной модели. Выбирать модель порогом, который потом
изменится, значит выбирать не ту модель. AP порога не требует.

F1 при 0.5 всё равно логируется: по нему в записке идёт сравнение с CV-baseline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    average_precision: float
    support: int  # сколько положительных примеров


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP по метке. Без положительных примеров он не определён — возвращаем nan."""
    from sklearn.metrics import average_precision_score

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def per_label(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
) -> list[LabelMetrics]:
    from sklearn.metrics import precision_recall_fscore_support

    results = []
    for position, label in enumerate(labels):
        truth = y_true[:, position].astype(int)
        score = y_score[:, position]
        predicted = (score >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        results.append(
            LabelMetrics(
                label=label,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                average_precision=_safe_ap(truth, score),
                support=int(truth.sum()),
            )
        )
    return results


def macro(metrics: Sequence[LabelMetrics], field: str, exclude: Sequence[str] = ()) -> float:
    """Макро-среднее по меткам, у которых метрика определена.

    Метки без единого положительного примера в выборке пропускаются: усреднять
    с нулём значило бы штрафовать модель за то, чего в данных нет. `exclude` —
    для меток, которые в пайплайне приходят не от сети (`config.ocr_derived`):
    сеть их предсказывает общей головой, но лучшую эпоху по ним выбирать нельзя.
    """
    skip = set(exclude)
    values = [
        getattr(m, field)
        for m in metrics
        if m.label not in skip and m.support > 0 and not np.isnan(getattr(m, field))
    ]
    return float(np.mean(values)) if values else float("nan")


def summary(metrics: Sequence[LabelMetrics], exclude: Sequence[str] = ()) -> dict[str, float]:
    return {
        "macro_f1": macro(metrics, "f1", exclude),
        "macro_ap": macro(metrics, "average_precision", exclude),
        "macro_precision": macro(metrics, "precision", exclude),
        "macro_recall": macro(metrics, "recall", exclude),
    }


def format_table(metrics: Sequence[LabelMetrics], exclude: Sequence[str] = ()) -> str:
    skip = set(exclude)
    lines = [f"{'метка':16s}{'P':>8s}{'R':>8s}{'F1':>8s}{'AP':>8s}{'n+':>7s}"]
    for m in metrics:
        ap = "—" if np.isnan(m.average_precision) else f"{m.average_precision:.3f}"
        # Звёздочка — метка приходит не от сети: показана, но в макро не входит.
        name = f"{m.label}*" if m.label in skip else m.label
        lines.append(
            f"{name:16s}{m.precision:8.3f}{m.recall:8.3f}{m.f1:8.3f}{ap:>8s}{m.support:7d}"
        )
    totals = summary(metrics, exclude)
    lines.append(
        f"{'macro':16s}{totals['macro_precision']:8.3f}{totals['macro_recall']:8.3f}"
        f"{totals['macro_f1']:8.3f}{totals['macro_ap']:8.3f}"
    )
    if skip:
        lines.append(f"* в макро не входит: {', '.join(sorted(skip))}")
    return "\n".join(lines)


def predict(model, loader, config, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Прогон по загрузчику -> (истина, оценки) НА УРОВНЕ СТРАНИЦЫ.

    Патчи схлопываются ровно так, как это делает приложение: локальные метки —
    максимумом, глобальные — средним (`data.aggregation`). По патчам мерить
    нельзя: у реальной страницы масок нет, её метка раздаётся всем патчам, и
    патч из центра отвечал бы за блик в углу. Локальные метки от этого получали
    AP на уровне своей доли в выборке — то есть уровень случайного угадывания.

    Истину схлопываем максимумом: у реальной страницы она одинакова во всех
    патчах, и максимум возвращает исходную страничную разметку без изменений.
    """
    import torch

    local = np.array([label in set(config.data.aggregation.local) for label in config.labels])
    model.eval()
    truths: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for pages, page_targets in loader:
            n_pages, per_page = pages.shape[0], pages.shape[1]
            logits = model(pages.flatten(0, 1).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(n_pages, per_page, -1)
            truth = page_targets.numpy().reshape(n_pages, per_page, -1)
            scores.append(np.where(local, probs.max(axis=1), probs.mean(axis=1)))
            truths.append(truth.max(axis=1))
    return np.concatenate(truths), np.concatenate(scores)


def evaluate(
    model,
    loader,
    config,
    device: str = "cpu",
    threshold: float = 0.5,
) -> tuple[list[LabelMetrics], np.ndarray, np.ndarray]:
    y_true, y_score = predict(model, loader, config, device)
    return per_label(y_true, y_score, config.labels, threshold), y_true, y_score


def main() -> None:
    import argparse
    from pathlib import Path

    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.data.dataset import PatchDataset, collect_real, load_split
    from src.models.model import build_model, load_checkpoint

    parser = argparse.ArgumentParser(description="Оценка модели на реальных страницах")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--part", default="val", choices=("train", "val", "test"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    if args.part == "test":
        print("ВНИМАНИЕ: тест открывается один раз, в С8. Уверены?", flush=True)

    _, images = load_split(args.splits, args.part)
    samples = collect_real(args.labels, args.data, images)
    dataset = PatchDataset(samples, config, train=False)
    # Батч в СТРАНИЦАХ, а патчей со страницы восемь: батч в 32 страницы означал бы
    # 256 патчей за проход вместо тридцати двух.
    pages_per_batch = max(1, config.train.batch_size // config.dataset.val_patches_per_page)
    loader = DataLoader(dataset, batch_size=pages_per_batch, num_workers=0)

    model = build_model(
        config.model.backbone, config.n_labels, pretrained=False, dropout=config.model.dropout
    )
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    model.to(args.device)

    metrics, _, _ = evaluate(model, loader, config, args.device, args.threshold)
    print(f"\n{args.part}: {len(samples)} страниц, порог {args.threshold}")
    print(format_table(metrics, config.ocr_derived))


if __name__ == "__main__":
    main()
