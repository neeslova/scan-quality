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
    page_positive_rates,
    patch_label_dilution,
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
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :] = True  # верхняя половина
    total = int(mask.sum())

    full, _ = _mask_for_patch(mask, total, (0, 0, 1000, 500), (1000, 1000))
    empty, _ = _mask_for_patch(mask, total, (0, 600, 1000, 1000), (1000, 1000))

    assert full == pytest.approx(1.0, abs=0.05)
    assert empty == pytest.approx(0.0, abs=0.05)


def test_thin_defect_is_caught_by_the_second_share() -> None:
    """Полоса во всю страницу не способна накрыть 15% квадрата 384x384.

    Из-за этого `streaks` и `cropped` не получали метку почти никогда — 0.8% и
    1.7% патчей замером, — и сеть двух дефектов из десяти не видела вовсе.
    Ловит их вторая доля: сколько самого дефекта попало внутрь патча.
    """
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[:, 500:503] = True  # полоса в три пикселя во всю высоту
    total = int(mask.sum())

    coverage, share = _mask_for_patch(mask, total, (300, 300, 684, 684), (1000, 1000))

    assert coverage < 0.15  # по доле патча метка не прошла бы
    assert share > 0.02  # а по доле маски проходит


def test_mask_overlap_on_degenerate_box() -> None:
    mask = np.ones((10, 10), dtype=bool)
    coverage, share = _mask_for_patch(mask, int(mask.sum()), (0, 0, 1, 1), (1000, 1000))
    assert coverage >= 0.0 and share >= 0.0


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


def test_page_yields_all_its_patches_at_once(config: Config, page_file) -> None:
    """Один декод страницы вместо `patches_per_page`: на двух ядрах это решало всё."""
    sample = Sample(path=page_file, labels={"blur": True}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=True)

    tensor, target = dataset[0]
    patch = config.data.patch_size
    assert tuple(tensor.shape) == (config.dataset.patches_per_page, 3, patch, patch)
    assert tuple(target.shape) == (config.dataset.patches_per_page, config.n_labels)


def test_validation_takes_several_patches_per_page(config: Config, page_file) -> None:
    """Вердикт по странице собирается максимумом по патчам, и одним патчем блик
    в углу не поймать: локальные метки получали AP на уровне случайного угадывания."""
    sample = Sample(path=page_file, labels={}, masks={}, source="real")
    tensor, target = PatchDataset([sample], config, train=False)[0]

    expected = config.dataset.val_patches_per_page
    assert tensor.shape[0] == expected and target.shape[0] == expected


def test_flatten_merges_pages_and_patches(config: Config, page_file) -> None:
    """DataLoader собирает батч из страниц, модель принимает патчи."""
    import torch

    from src.data.dataset import flatten_patches

    samples = [Sample(path=page_file, labels={}, masks={}, source="real")] * 3
    dataset = PatchDataset(samples, config, train=True)
    pages = torch.stack([dataset[i][0] for i in range(3)])
    targets = torch.stack([dataset[i][1] for i in range(3)])

    batch, target = flatten_patches(pages, targets)
    patch = config.data.patch_size
    expected = 3 * config.dataset.patches_per_page
    assert tuple(batch.shape) == (expected, 3, patch, patch)
    assert tuple(target.shape) == (expected, config.n_labels)


def test_tiny_page_is_upscaled_to_patch(config: Config, tmp_path) -> None:
    """Страница мельче патча не должна ронять загрузчик."""
    path = tmp_path / "tiny.png"
    fx.save(fx.text_page(width=200, height=260, line_height=8, line_gap=5, margin=12), path, 300)

    sample = Sample(path=path, labels={}, masks={}, source="real")
    tensor, _ = PatchDataset([sample], config, train=False)[0]
    patch = config.data.patch_size
    assert tuple(tensor.shape) == (config.dataset.val_patches_per_page, 3, patch, patch)


def test_validation_is_deterministic(config: Config, page_file) -> None:
    """Иначе кривая val дрожит от смены патчей, а не от обучения."""
    sample = Sample(path=page_file, labels={"blur": True}, masks={}, source="real")
    dataset = PatchDataset([sample], config, train=False, seed=7)

    first, _ = dataset[0]
    second, _ = dataset[0]
    assert bool((first == second).all())


def test_length_counts_pages(config: Config, page_file) -> None:
    """Длина в страницах: патчи страницы выдаются одним элементом."""
    samples = [Sample(path=page_file, labels={}, masks={}, source="real")] * 5
    assert len(PatchDataset(samples, config, train=True)) == 5
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


def test_page_rate_reflects_imbalance(config: Config, tmp_path) -> None:
    samples = [
        Sample(path=tmp_path / "x.png", labels={"blur": i < 10}, masks={}, source="real")
        for i in range(100)
    ]
    rates = page_positive_rates(samples, config.labels)
    assert rates[config.labels.index("blur")] == pytest.approx(0.10, abs=0.001)


def test_label_without_examples_has_zero_rate(config: Config, tmp_path) -> None:
    """Дальше по цепочке такая метка получает вес 1.0: учить всё равно нечему."""
    samples = [Sample(path=tmp_path / "x.png", labels={}, masks={}, source="real")]
    assert set(page_positive_rates(samples, config.labels)) == {0.0}


def test_unseen_label_keeps_the_page_estimate(config: Config, page_file) -> None:
    """Метка, не попавшая в выборку, получает разбавление 1.0.

    Иначе `unreadable` с тремя примерами на восемь тысяч страниц не попала бы
    в выборку из четырёхсот, получила бы долю 0 и вес 1.0 вместо потолка —
    то есть починка локальных меток сломала бы редкие.
    """
    samples = [Sample(path=page_file, labels={}, masks={}, source="real")] * 4
    dataset = PatchDataset(samples, config, train=True)

    assert set(patch_label_dilution(dataset, sample_pages=2, seed=1)) == {1.0}
