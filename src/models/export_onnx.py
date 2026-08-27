"""Экспорт обученной сети в ONNX и сверка с torch.

Инференс обязан работать без torch: он стоит только в extra `train`, а приложение
живёт на `onnxruntime` из базовых зависимостей. Поэтому экспорт — не удобство,
а условие работоспособности пайплайна на машине пользователя.

Сверка обязательна и делается здесь же, но **по вероятностям, а не по логитам**.
Логит — величина без своего масштаба: на входе из белого шума сеть выдаёт
логиты в диапазоне -211..76, и расхождение в 0.13 там означает 0.4%
относительных и 1e-4 после сигмоиды. Пороги вердикта калибруются по
вероятностям, значит и допуск относится к ним.

По той же причине сверяемся не на шуме, а на входе, похожем на страницу: белый
фон с тёмными штрихами. Белый шум гонит каждый фильтр в насыщение и меряет
режим, в котором сеть никогда не работает.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import Config, load_config
from src.imaging import IMAGENET_MEAN, IMAGENET_STD
from src.models.model import build_model, export_onnx, load_checkpoint

logger = logging.getLogger(__name__)

TOLERANCE = 1e-3


def page_like_batch(patch_size: int, batch: int = 4, seed: int = 0) -> np.ndarray:
    """Вход, похожий на скан: белая бумага со строками тёмных штрихов."""
    rng = np.random.default_rng(seed)
    pages = np.full((batch, patch_size, patch_size), 235.0, dtype=np.float32)
    for index in range(batch):
        for row in range(12, patch_size - 12, 28):
            for column in range(10, patch_size - 20, 14):
                if rng.random() < 0.75:
                    width = int(rng.integers(4, 10))
                    pages[index, row : row + 11, column : column + width] = rng.integers(20, 90)
        pages[index] += rng.normal(0.0, 3.0, (patch_size, patch_size))

    values = (np.clip(pages, 0, 255) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.repeat(values[:, None, :, :], 3, axis=1).astype(np.float32)


def real_patch_batch(path: Path, config: Config, batch: int = 4) -> np.ndarray:
    """Патчи настоящей страницы — сверка строже, чем на нарисованной."""
    from src.data.dataset import PatchDataset, Sample

    sample = Sample(path=path, labels={}, masks={}, source="real")
    tensor, _ = PatchDataset([sample], config, train=False)[0]
    return tensor.numpy()[:batch]


def max_divergence(onnx_path: Path, model, sample: np.ndarray) -> float:
    """Наибольшее расхождение ВЕРОЯТНОСТЕЙ torch и ONNX на одном и том же входе."""
    import onnxruntime as ort
    import torch

    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(sample))).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"patch": sample})[0]
    got = 1.0 / (1.0 + np.exp(-logits))

    return float(np.abs(expected - got).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт модели в ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/quality.onnx"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--sample", type=Path, default=None, help="страница для сверки вместо нарисованной"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    model = build_model(
        config.model.backbone, config.n_labels, pretrained=False, dropout=config.model.dropout
    )
    load_checkpoint(args.checkpoint, model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.out, config.data.patch_size, opset=args.opset)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"{args.out}: {size_mb:.1f} МБ, вход {config.data.patch_size}x{config.data.patch_size}")

    if args.sample is not None:
        sample = real_patch_batch(args.sample, config)
        origin = f"патчи страницы {args.sample.name}"
    else:
        sample = page_like_batch(config.data.patch_size)
        origin = "нарисованная страница"

    divergence = max_divergence(args.out, model, sample)
    print(f"расхождение вероятностей torch vs onnx ({origin}): {divergence:.2e}")
    if divergence > TOLERANCE:
        raise SystemExit(
            f"Расхождение {divergence:.2e} больше {TOLERANCE:.0e}: пороги, "
            "посчитанные на torch, к этой модели уже не относятся"
        )
    print(f"сверка пройдена, допуск {TOLERANCE:.0e}", file=sys.stderr)


if __name__ == "__main__":
    main()
