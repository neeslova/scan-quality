"""Экспорт в ONNX. Главное здесь — что именно сверяется и на чём."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.export_onnx import TOLERANCE, max_divergence, page_like_batch


def test_sample_looks_like_a_page_not_like_noise() -> None:
    """Белый шум гонит фильтры в насыщение: логиты уходят в -211..76.

    Это режим, в котором сеть никогда не работает, и мерить расхождение в нём
    бессмысленно. Вход должен быть бумагой со штрихами: светлым в основном.
    """
    batch = page_like_batch(patch_size=128, batch=2)

    assert batch.shape == (2, 3, 128, 128)
    # Три канала — это повторённый серый, а не три разных.
    assert np.array_equal(batch[:, 0], batch[:, 2])

    brightness = batch * IMAGENET_STD + IMAGENET_MEAN
    assert brightness.mean() > 0.75  # бумага, а не серая каша
    assert brightness.min() < 0.45  # но штрихи на ней есть


def test_divergence_is_measured_on_probabilities(tmp_path) -> None:
    """Допуск 1e-3 относится к вероятностям: пороги вердикта применяются к ним.

    На логитах та же модель давала расхождение 1.3e-1 — на два порядка выше
    допуска, — хотя после сигмоиды это 1e-4. Сверка по логитам забраковала бы
    исправный экспорт.
    """
    from src.models.model import build_model, export_onnx

    config = load_config()
    patch = 96
    model = build_model("mobilenetv3_small_100", config.n_labels, pretrained=False)
    path = tmp_path / "tiny.onnx"
    export_onnx(model, path, patch)

    divergence = max_divergence(path, model, page_like_batch(patch, batch=2))
    assert divergence < TOLERANCE


def test_export_rejects_a_mismatched_input_size(tmp_path) -> None:
    """Батч динамический, размер патча — нет: модель обучена на своём разрешении."""
    import onnxruntime as ort

    from src.models.model import build_model, export_onnx

    config = load_config()
    model = build_model("mobilenetv3_small_100", config.n_labels, pretrained=False)
    path = tmp_path / "tiny.onnx"
    export_onnx(model, path, patch_size=96)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Другое число патчей проходит.
    session.run(["logits"], {"patch": page_like_batch(96, batch=3)})
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
        session.run(["logits"], {"patch": page_like_batch(64, batch=2)})
