"""Обучение multi-label CNN на патчах. Рассчитано на Colab: сессия отвалится.

Отсюда три обязательных свойства (PLAN.md §8):
  * чекпоинт **каждую эпоху** на Drive и `--resume auto`;
  * копия использованного конфига рядом с чекпоинтом — иначе через месяц не
    восстановить, на каких метках и порогах училась именно эта модель;
  * лог метрик в CSV на Drive, дописыванием.

Обучающая часть — синтетика плюс реальные страницы train, валидация — только
реальные. Смысл: синтетика нужна для объёма и для редких меток, но отбирать
модель по ней нельзя — она измеряла бы качество генератора деградаций.

Запуск:
    python -m src.models.train --config configs/base.yaml --data /content/data \\
        --out /content/drive/MyDrive/scanq/runs --resume auto
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, load_config
from src.data.dataset import PatchDataset, collect_real, collect_synthetic, load_split
from src.data.dataset import positive_weights
from src.models.evaluate import evaluate, format_table, summary
from src.models.model import build_model, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "macro_f1",
    "macro_ap",
    "macro_precision",
    "macro_recall",
    "lr",
    "seconds",
]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args, config: Config):
    from torch.utils.data import DataLoader

    _, train_images = load_split(args.splits, "train")
    val_docs, val_images = load_split(args.splits, "val")
    test_docs, _ = load_split(args.splits, "test")

    # Синтетику берём всю, кроме документов из val и test: её эталоны — страницы
    # всего корпуса, а сплит покрывает только размеченные три сотни.
    synthetic = collect_synthetic(args.manifest, args.synthetic, val_docs | test_docs)
    real_train = collect_real(args.labels, args.data, train_images)
    real_val = collect_real(args.labels, args.data, val_images)

    logger.info(
        "train: %d синтетики + %d реальных, val: %d реальных",
        len(synthetic),
        len(real_train),
        len(real_val),
    )
    if not real_val:
        raise SystemExit("val пуст — обучение без валидации бессмысленно")

    train_set = PatchDataset(synthetic + real_train, config, train=True, seed=config.train.seed)
    # Валидация детерминированная: те же патчи каждую эпоху, иначе кривая val
    # дрожит от смены патчей, а не от обучения.
    val_set = PatchDataset(real_val, config, train=False, seed=config.train.seed)

    loaders = (
        DataLoader(
            train_set,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.dataset.workers,
            drop_last=True,
            pin_memory=True,
        ),
        DataLoader(
            val_set,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.dataset.workers,
        ),
    )
    return loaders, synthetic + real_train


def make_optimizer(model, config: Config):
    import torch

    if config.train.optimizer != "adamw":
        raise SystemExit(f"Оптимизатор {config.train.optimizer} не поддержан")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    if config.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    elif config.train.scheduler == "none":
        scheduler = None
    else:
        raise SystemExit(f"Планировщик {config.train.scheduler} не поддержан")
    return optimizer, scheduler


def pos_weight_tensor(config: Config, samples, device: str):
    import torch

    if isinstance(config.train.pos_weight, list):
        weights = np.asarray(config.train.pos_weight, dtype=np.float32)
        if len(weights) != config.n_labels:
            raise SystemExit(f"train.pos_weight: нужно {config.n_labels} чисел")
    else:
        weights = positive_weights(samples, config.labels)
    logger.info(
        "pos_weight: %s",
        ", ".join(f"{label} {value:.1f}" for label, value in zip(config.labels, weights)),
    )
    return torch.tensor(weights, device=device)


def update_best(best: float, value: float) -> tuple[float, bool]:
    """Новое лучшее значение и признак улучшения.

    Вынесено отдельно, потому что порядок здесь легко перепутать: если сохранить
    `last.ckpt` со старым `best`, то после обрыва сессии Colab обучение продолжится
    с `best = -inf` и первая же эпоха перезапишет `best.ckpt` худшей моделью.
    """
    if np.isnan(value) or value <= best:
        return best, False
    return value, True


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def train_epoch(model, loader, criterion, optimizer, device: str) -> float:
    import torch

    model.train()
    total = 0.0
    seen = 0
    for batch, target in loader:
        batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), target)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * batch.size(0)
        seen += batch.size(0)
    return total / max(1, seen)


def validation_loss(model, loader, criterion, device: str) -> float:
    import torch

    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            loss = criterion(model(batch), target)
            total += float(loss.item()) * batch.size(0)
            seen += batch.size(0)
    return total / max(1, seen)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Обучение multi-label CNN на патчах")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса реальных сканов")
    parser.add_argument("--synthetic", type=Path, required=True, help="корень синтетики")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="куда писать чекпоинты и лог")
    parser.add_argument("--resume", default="auto", help="auto | путь к чекпоинту | none")
    parser.add_argument("--epochs", type=int, default=None, help="переопределить конфиг")
    parser.add_argument("--device", default=None, help="по умолчанию cuda, если доступна")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or config.train.epochs

    seed_everything(config.train.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # Конфиг рядом с чекпоинтом — обязательное требование раздела 8.
    (args.out / "config.json").write_text(
        json.dumps(config.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (train_loader, val_loader), train_samples = build_loaders(args, config)
    model = build_model(
        config.model.backbone,
        config.n_labels,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    optimizer, scheduler = make_optimizer(model, config)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor(config, train_samples, device)
    )

    start_epoch = 0
    best = float("-inf")
    last_path = args.out / "last.ckpt"
    resume_from: Optional[Path] = None
    if args.resume == "auto":
        resume_from = last_path if last_path.is_file() else None
    elif args.resume not in ("none", ""):
        resume_from = Path(args.resume)

    if resume_from is not None and resume_from.is_file():
        _, payload = load_checkpoint(resume_from, model, map_location=device)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best = float(payload["best"])
        logger.info("Продолжаем с эпохи %d, лучший macro-AP %.4f", start_epoch, best)

    config_dump = config.model_dump(mode="json", by_alias=True)
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validation_loss(model, val_loader, criterion, device)
        metrics, _, _ = evaluate(model, val_loader, config.labels, device)
        totals = summary(metrics)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - started
        append_csv(
            args.out / "metrics.csv",
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                **{key: round(value, 5) for key, value in totals.items()},
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(elapsed, 1),
            },
        )
        print(
            f"эпоха {epoch:3d}  loss {train_loss:.4f}/{val_loss:.4f}  "
            f"macro-AP {totals['macro_ap']:.4f}  macro-F1 {totals['macro_f1']:.4f}  "
            f"{elapsed:.0f} с",
            flush=True,
        )

        # Отбор по macro-AP: пороги ещё не калиброваны (это С6), и F1 при 0.5
        # выбирал бы модель под порог, который потом изменится.
        # Обновляем ДО сохранения last.ckpt — иначе после обрыва сессии обучение
        # продолжится с best = -inf и затрёт лучшую модель худшей.
        best, improved = update_best(best, totals["macro_ap"])

        # Чекпоинт КАЖДУЮ эпоху: сессия Colab отвалится, это вопрос времени.
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, config_dump)
        if improved:
            save_checkpoint(
                args.out / "best.ckpt", model, optimizer, scheduler, epoch, best, config_dump
            )
            print(f"  новый лучший macro-AP {best:.4f}", flush=True)

    metrics, _, _ = evaluate(model, val_loader, config.labels, device)
    print("\nитог на val:", file=sys.stderr)
    print(format_table(metrics))


if __name__ == "__main__":
    main()
