"""Модель: backbone из timm плюс multi-label голова на 10 сигмоид.

Сигмоиды, а не softmax: дефекты сочетаются. Размытый скан с тенью — это две
метки одновременно, а не выбор одной из десяти.

Backbone не тяжелее EfficientNet-B0 по прямому требованию из ограничений:
инференс идёт на локальном CPU через ONNX Runtime, и всё, что медленнее,
не влезает в бюджет времени на страницу.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_model(
    backbone: str,
    n_labels: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    in_channels: int = 3,
):
    """timm-backbone с головой на `n_labels` выходов (логиты, без сигмоиды).

    Логиты, а не вероятности: `BCEWithLogitsLoss` устойчивее численно, чем
    отдельная сигмоида с последующим `BCELoss`. Сигмоида применяется на инференсе.
    """
    import timm

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=n_labels,
        drop_rate=dropout,
        in_chans=in_channels,
    )
    logger.info(
        "%s: %.1f млн параметров, выходов %d",
        backbone,
        sum(p.numel() for p in model.parameters()) / 1e6,
        n_labels,
    )
    return model


def load_checkpoint(path, model=None, map_location: str = "cpu"):
    """Читает чекпоинт. Возвращает (модель, состояние обучения)."""
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    return model, payload


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, best: float, config_dump: dict):
    """Сохраняет всё, что нужно для продолжения, и копию конфига рядом.

    Конфиг обязателен: без него через месяц не восстановить, на каких порогах
    и с какими метками училась именно эта модель.
    """
    import torch

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best": best,
        "config": config_dump,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    # Пишем через временный файл: обрыв сессии Colab посреди сохранения
    # оставил бы битый чекпоинт вместо предыдущего рабочего.
    import os

    os.replace(tmp, path)


def export_onnx(model, path, patch_size: int, opset: int = 17, in_channels: int = 3) -> None:
    """Экспорт в ONNX для локального инференса без torch."""
    import torch

    model.eval()
    dummy = torch.randn(1, in_channels, patch_size, patch_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["patch"],
        output_names=["logits"],
        dynamic_axes={"patch": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )


def head_name(model) -> Optional[str]:
    """Имя классификационной головы — нужно, чтобы задать ей отдельный learning rate."""
    for name in ("classifier", "head", "fc"):
        if hasattr(model, name):
            return name
    return None
