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
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, load_config
from src.data.dataset import (
    PatchDataset,
    collect_real,
    collect_synthetic,
    flatten_patches,
    load_split,
    page_positive_rates,
    patch_label_dilution,
)
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
    synthetic = []
    if args.source in ("mixed", "synthetic"):
        if args.manifest is None or args.synthetic is None:
            raise SystemExit("--manifest и --synthetic обязательны при source с синтетикой")
        synthetic = collect_synthetic(args.manifest, args.synthetic, val_docs | test_docs)

    real_train = []
    if args.source in ("mixed", "real"):
        real_train = collect_real(args.labels, args.data, train_images)

    real_val = collect_real(args.labels, args.data, val_images)

    logger.info(
        "train (%s): %d синтетики + %d реальных, val: %d реальных",
        args.source,
        len(synthetic),
        len(real_train),
        len(real_val),
    )
    if not synthetic and not real_train:
        raise SystemExit(f"train пуст при source={args.source}")
    if not real_val:
        raise SystemExit("val пуст — обучение без валидации бессмысленно")

    train_set = PatchDataset(synthetic + real_train, config, train=True, seed=config.train.seed)
    # Валидация детерминированная: те же патчи каждую эпоху, иначе кривая val
    # дрожит от смены патчей, а не от обучения.
    val_set = PatchDataset(real_val, config, train=False, seed=config.train.seed)

    # Батч теперь измеряется в СТРАНИЦАХ: датасет отдаёт все патчи страницы
    # сразу, чтобы не декодировать её по разу на каждый патч. Пересчитывать надо
    # и валидационный: патчей со страницы там вдвое больше, и батч в страницах
    # молча раздул бы прогон до 256 патчей за проход.
    pages_per_batch = max(1, config.train.batch_size // config.dataset.patches_per_page)
    val_pages_per_batch = max(1, config.train.batch_size // config.dataset.val_patches_per_page)
    # Воркеров не больше, чем ядер: в Colab их два, и лишние процессы только
    # мешают друг другу.
    workers = min(config.dataset.workers, os.cpu_count() or 1)
    if workers < config.dataset.workers:
        logger.info("Воркеров: %d вместо %d — столько ядер", workers, config.dataset.workers)

    loaders = (
        DataLoader(
            train_set,
            batch_size=pages_per_batch,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
        ),
        DataLoader(
            val_set,
            batch_size=val_pages_per_batch,
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
        ),
    )
    return loaders


def make_optimizer(model, config: Config, epochs: Optional[int] = None):
    """Оптимизатор и планировщик.

    `epochs` — сколько эпох реально запрошено: косинус должен укладываться именно
    в них. По `config.train.epochs` график при `--epochs 10` прошёл бы лишь треть
    цикла, и lr к концу обучения остался бы почти стартовым.
    """
    import torch

    if config.train.optimizer != "adamw":
        raise SystemExit(f"Оптимизатор {config.train.optimizer} не поддержан")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    if config.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs or config.train.epochs
        )
    elif config.train.scheduler == "none":
        scheduler = None
    else:
        raise SystemExit(f"Планировщик {config.train.scheduler} не поддержан")
    return optimizer, scheduler


def pos_weight_tensor(config: Config, dataset, device: str):
    import torch

    if isinstance(config.train.pos_weight, list):
        weights = np.asarray(config.train.pos_weight, dtype=np.float32)
        if len(weights) != config.n_labels:
            raise SystemExit(f"train.pos_weight: нужно {config.n_labels} чисел")
    else:
        # Дисбаланс считается по ПАТЧАМ, а не по страницам. Локальную метку патч
        # получает, только если реально накрывает дефект, поэтому страничная доля
        # умножается на измеренное разбавление: у `cropped` метку получают 7.5%
        # патчей со страниц с этой меткой, и без поправки вес выходил вчетверо
        # меньше нужного — ровно там, где метка и так редкая.
        page_rate = page_positive_rates(dataset.samples, config.labels)
        dilution = patch_label_dilution(dataset, config.train.pos_weight_sample, config.train.seed)
        rate = page_rate * dilution

        weights = np.ones(config.n_labels, dtype=np.float32)
        for position, label in enumerate(config.labels):
            if rate[position] <= 0.0:
                logger.warning("Метка %s без единого примера — вес 1.0", label)
                continue
            weights[position] = (1.0 - rate[position]) / rate[position]

        for label, share, diluted in zip(config.labels, page_rate, dilution):
            if diluted < 0.9 and share > 0.0:
                logger.info(
                    "%s: страниц с меткой %.0f%%, но метку получают лишь %.0f%% их патчей",
                    label,
                    100.0 * share,
                    100.0 * diluted,
                )

    # Потолок обязателен: метка с тремя примерами на восемь тысяч страниц
    # получает вес 2686, и её слагаемое съедает всю функцию потерь.
    capped = np.minimum(weights, config.train.pos_weight_max)
    for label, before, after in zip(config.labels, weights, capped):
        if after < before:
            logger.warning(
                "pos_weight[%s] урезан с %.0f до %.0f — примеров почти нет, метка так не выучится",
                label,
                before,
                after,
            )
    logger.info(
        "pos_weight: %s",
        ", ".join(f"{label} {value:.1f}" for label, value in zip(config.labels, capped)),
    )
    return torch.tensor(capped, device=device)


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

    model.train()
    total = 0.0
    seen = 0
    for pages, page_targets in loader:
        batch, target = flatten_patches(pages, page_targets)
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
        for pages, page_targets in loader:
            batch, target = flatten_patches(pages, page_targets)
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
    parser.add_argument("--synthetic", type=Path, default=None, help="корень синтетики")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--source",
        default="mixed",
        choices=("mixed", "real", "synthetic"),
        help="что подавать в train: синтетика с реальными, только реальные, только синтетика",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="куда писать чекпоинты и лог")
    parser.add_argument("--resume", default="auto", help="auto | путь к чекпоинту | none")
    parser.add_argument("--epochs", type=int, default=None, help="переопределить конфиг")
    parser.add_argument("--lr", type=float, default=None, help="переопределить конфиг")
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="взять ТОЛЬКО веса из чекпоинта: свежий оптимизатор, эпоха 0, best сброшен",
    )
    parser.add_argument("--device", default=None, help="по умолчанию cuda, если доступна")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)
    if args.lr is not None:
        config = config.model_copy(
            update={"train": config.train.model_copy(update={"lr": args.lr})}
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or config.train.epochs

    seed_everything(config.train.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # Конфиг рядом с чекпоинтом — обязательное требование раздела 8.
    (args.out / "config.json").write_text(
        json.dumps(config.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_loader, val_loader = build_loaders(args, config)
    model = build_model(
        config.model.backbone,
        config.n_labels,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    optimizer, scheduler = make_optimizer(model, config, epochs)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor(config, train_loader.dataset, device)
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
    elif args.init is not None:
        # Дообучение: признаки берём, историю обучения — нет. Оптимизатор со своими
        # моментами и `best` от прошлой задачи сюда не относятся, а `best` вдобавок
        # мерился по патчам и с новой, страничной величиной несопоставим.
        load_checkpoint(args.init, model, map_location=device)
        logger.info("Веса из %s; оптимизатор, счётчик эпох и best — с нуля", args.init)

    config_dump = config.model_dump(mode="json", by_alias=True)
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validation_loss(model, val_loader, criterion, device)
        metrics, _, _ = evaluate(model, val_loader, config, device)
        totals = summary(metrics, config.ocr_derived)
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

    metrics, _, _ = evaluate(model, val_loader, config, device)
    print("\nитог на val:", file=sys.stderr)
    print(format_table(metrics, config.ocr_derived))


if __name__ == "__main__":
    main()
