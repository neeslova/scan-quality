"""Датасет патчей. Главное — правило метки: локальный дефект не наследуется вслепую."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import Config, load_config
from src.data.dataset import (
    PatchDataset,
    Sample,
    _mask_for_patch,
    collect_real,
    collect_synthetic,
    positive_weights,
)
from src.data.generate import write_manifest
from src.labeling.app import append_label
from src.schema import LabelRecord, SyntheticRecord
from tests import factories as fx


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


@pytest.fixture
def page_file(tmp_path):
    path = tmp_path / "page.png"
    fx.save(fx.text_page(width=1200, height=1600), path, dpi=300)
    return path


def write_mask(path, shape, region) -> None:
    mask = np.zeros(shape, dtype=np.uint8)
    y0, y1, x0, x1 = region
    mask[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", mask)[1].tofile(str(path))


# --- пересечение патча с маской ---------------------------------------------


def test_mask_overlap_scales_to_page() -> None:
    """Маска хранится уменьшенной — бокс патча надо приводить к её масштабу."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255  # верхняя половина

    full = _mask_for_patch(mask, (0, 0, 1000, 500), (1000, 1000))
    empty = _mask_for_patch(mask, (0, 600, 1000, 1000), (1000, 1000))

    assert full == pytest.approx(1.0, abs=0.05)
    assert empty == pytest.approx(0.0, abs=0.05)


def test_mask_overlap_on_degenerate_box() -> None:
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    assert _mask_for_patch(mask, (0, 0, 1, 1), (1000, 1000)) >= 0.0


# --- правило метки патча ----------------------------------------------------


def test_local_label_only_where_the_mask_is(config: Config, tmp_path, page_file) -> None:
    """Ключевое правило: патч из чистого угла не получает метку блика из центра.

    Без него большинство патчей страницы учили бы сеть, что блик выглядит
    как обычный текст.
    """
    mask_path = tmp_path / "masks" / "glare.png"
    # Блик только в левой верхней четверти.
    write_mask(mask_path, (400, 300), (0, 200, 0, 150))

    sample = Sample(
        path=page_file,
        labels={"glare": True, "blur": True},
        masks={"glare": mask_path},
        source="synthetic",
    )
    dataset = PatchDataset([sample], config, train=False)
    position = config.labels.index("glare")
    blur_position = config.labels.index("blur")

    inside = dataset._patch_labels(sample, (0, 0, 300, 300), (1600, 1200))
    outside = dataset._patch_labels(sample, (900, 1300, 1200, 1600), (1600, 1200))

    assert inside[position] == 1.0
    assert outside[position] == 0.0
    # Глобальная метка наследуется всегда, независимо от положения патча.
    assert inside[blur_position] == 1.0 and outside[blur_position] == 1.0


def test_local_label_without_mask_falls_back_to_page(config: Config, page_file) -> None:
    """У размеченных вручную страниц масок нет — метка относится ко всей странице."""
    sample = Sample(path=page_file, labels={"glare": True}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=False)

    target = dataset._patch_labels(sample, (900, 1300, 1200, 1600), (1600, 1200))
    assert target[config.labels.index("glare")] == 1.0


def test_negative_labels_stay_zero(config: Config, page_file) -> None:
    sample = Sample(path=page_file, labels={"blur": False}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=False)
    assert dataset._patch_labels(sample, (0, 0, 384, 384), (1600, 1200)).sum() == 0.0


# --- выбор патча ------------------------------------------------------------


def test_patch_prefers_text(config: Config, tmp_path) -> None:
    """Патч без текста бесполезен: резкость на пустом поле бумаги не измерима."""
    page = np.full((1600, 1200), fx.PAPER, dtype=np.uint8)
    page[100:500, 100:500] = fx.text_page(width=400, height=400, line_height=18, margin=20)
    path = tmp_path / "sparse.png"
    fx.save(page, path, dpi=300)

    sample = Sample(path=path, labels={}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=True)

    boxes = [dataset._pick_box(page, np.random.default_rng(seed)) for seed in range(12)]
    with_text = sum(1 for x0, y0, _, _ in boxes if x0 < 500 and y0 < 500)
    assert with_text >= 8


def test_patch_size_is_always_the_configured_one(config: Config, page_file) -> None:
    sample = Sample(path=page_file, labels={"blur": True}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=True)

    tensor, target = dataset[0]
    assert tuple(tensor.shape) == (3, config.data.patch_size, config.data.patch_size)
    assert tuple(target.shape) == (config.n_labels,)


def test_tiny_page_is_upscaled_to_patch(config: Config, tmp_path) -> None:
    """Страница мельче патча не должна ронять загрузчик."""
    path = tmp_path / "tiny.png"
    fx.save(fx.text_page(width=200, height=260, line_height=8, line_gap=5, margin=12), path, 300)

    sample = Sample(path=path, labels={}, masks={}, source="real")
    tensor, _ = PatchDataset([sample], config, train=False)[0]
    assert tuple(tensor.shape) == (3, config.data.patch_size, config.data.patch_size)


def test_validation_is_deterministic(config: Config, page_file) -> None:
    """Иначе кривая val дрожит от смены патчей, а не от обучения."""
    sample = Sample(path=page_file, labels={"blur": True}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=False, seed=7)

    first, _ = dataset[0]
    second, _ = dataset[0]
    assert bool((first == second).all())


def test_length_counts_patches_per_page(config: Config, page_file) -> None:
    samples = [Sample(path=page_file, labels={}, masks={}, source="real")] * 5
    assert len(PatchDataset(samples, config, train=True)) == 5 * config.dataset.patches_per_page
    assert len(PatchDataset(samples, config, train=False)) == 5


def test_empty_dataset_is_rejected(config: Config) -> None:
    with pytest.raises(ValueError, match="пуст"):
        PatchDataset([], config)


# --- сборка источников ------------------------------------------------------


def test_collect_synthetic_excludes_held_out(config: Config, tmp_path) -> None:
    """Исключение val/test, а не отбор по train: сплит покрывает лишь часть корпуса."""
    records = [
        SyntheticRecord(
            image=f"images/{i}.jpg", reference="r.jpg", document=f"d{i}", corpus="t", labels={}
        )
        for i in range(4)
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)

    everything = collect_synthetic(manifest, tmp_path)
    without_held_out = collect_synthetic(manifest, tmp_path, {"d0", "d2"})

    assert len(everything) == 4
    assert {s.path.name for s in without_held_out} == {"1.jpg", "3.jpg"}


def test_collect_real_filters_by_image(config: Config, tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    for i in range(3):
        append_label(
            path, LabelRecord(image=f"{i}.jpg", document=f"d{i}", corpus="t", labels={"blur": True})
        )

    assert len(collect_real(path, tmp_path)) == 3
    assert len(collect_real(path, tmp_path, {"1.jpg"})) == 1


# --- pos_weight -------------------------------------------------------------


def test_positive_weights_reflect_imbalance(config: Config, tmp_path) -> None:
    samples = [
        Sample(path=tmp_path / "x.png", labels={"blur": i < 10}, masks={}, source="real")
        for i in range(100)
    ]
    weights = positive_weights(samples, config.labels)
    # 10 положительных из 100 -> отрицательных вдевятеро больше
    assert weights[config.labels.index("blur")] == pytest.approx(9.0, abs=0.01)


def test_label_without_examples_gets_weight_one(config: Config, tmp_path) -> None:
    """Иначе деление на ноль, а учить всё равно нечему."""
    samples = [Sample(path=tmp_path / "x.png", labels={}, masks={}, source="real")]
    assert set(positive_weights(samples, config.labels)) == {1.0}
