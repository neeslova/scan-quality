"""Обучение: отбор лучшей эпохи, лог метрик, сводные метрики."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from src.models.evaluate import LabelMetrics, macro, per_label, summary
from src.models.train import CSV_FIELDS, append_csv, update_best

# --- отбор лучшей эпохи -----------------------------------------------------


def test_best_updates_only_on_improvement() -> None:
    best, improved = update_best(0.5, 0.6)
    assert (best, improved) == (0.6, True)

    best, improved = update_best(0.6, 0.55)
    assert (best, improved) == (0.6, False)

    best, improved = update_best(0.6, 0.6)
    assert (best, improved) == (0.6, False)


def test_nan_never_becomes_best() -> None:
    """Метрика бывает не определена — она не должна побеждать."""
    assert update_best(0.4, float("nan")) == (0.4, False)
    assert update_best(float("-inf"), float("nan")) == (float("-inf"), False)


def test_first_epoch_always_improves() -> None:
    best, improved = update_best(float("-inf"), 0.01)
    assert improved and best == 0.01


def test_best_survives_a_resume() -> None:
    """Ключевая проверка: после обрыва сессии лучшая модель не должна затираться.

    В чекпоинт пишется УЖЕ обновлённое значение, иначе продолжение стартует
    с best = -inf и первая же эпоха перезаписывает best.ckpt худшей моделью.
    """
    best = float("-inf")
    for value in (0.30, 0.45, 0.40):
        best, _ = update_best(best, value)
    saved = best  # что попало бы в last.ckpt

    restored = saved
    restored, improved = update_best(restored, 0.42)
    assert not improved
    assert restored == pytest.approx(0.45)


# --- потолок pos_weight -----------------------------------------------------


def test_pos_weight_is_capped(tmp_path) -> None:
    """Метка с тремя примерами на восемь тысяч получала вес 2686.

    Её слагаемое съедало всю функцию потерь: на реальном прогоне train-потеря
    была 0.52, а val-потеря 23.7 — почти целиком из этого веса.
    """
    from src.config import load_config
    from src.data.dataset import PatchDataset, Sample
    from src.models.train import pos_weight_tensor
    from tests import factories as fx

    config = load_config()
    path = tmp_path / "page.png"
    fx.save(fx.text_page(width=400, height=520), path, dpi=300)

    # 3 положительных примера unreadable на 8000 страниц — как в реальном прогоне.
    samples = [
        Sample(path=path, labels={"unreadable": i < 3}, masks={}, source="real")
        for i in range(8000)
    ]
    weights = pos_weight_tensor(config, PatchDataset(samples, config, train=True), "cpu").numpy()

    position = config.labels.index("unreadable")
    assert weights[position] == pytest.approx(config.train.pos_weight_max)
    assert weights.max() <= config.train.pos_weight_max


def test_explicit_pos_weight_is_also_capped(tmp_path) -> None:
    from src.config import load_config
    from src.models.train import pos_weight_tensor

    config = load_config()
    explicit = [1000.0] * config.n_labels
    config = config.model_copy(
        update={"train": config.train.model_copy(update={"pos_weight": explicit})}
    )

    weights = pos_weight_tensor(config, [], "cpu").numpy()
    assert weights.max() <= config.train.pos_weight_max


# --- лог метрик -------------------------------------------------------------


def test_csv_appends_with_one_header(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    for epoch in range(3):
        append_csv(path, {"epoch": epoch, "macro_ap": 0.1 * epoch})

    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    assert path.read_text(encoding="utf-8").count("epoch,train_loss") == 1


def test_csv_tolerates_missing_fields(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    append_csv(path, {"epoch": 0})
    with path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert set(row) == set(CSV_FIELDS)


# --- сводные метрики --------------------------------------------------------


def test_macro_skips_labels_without_examples() -> None:
    """Метка, которой нет в выборке, не должна штрафовать модель нулём."""
    metrics = [
        LabelMetrics("blur", 0.8, 0.8, 0.8, 0.9, support=10),
        LabelMetrics("glare", 0.0, 0.0, 0.0, float("nan"), support=0),
    ]
    assert macro(metrics, "f1") == pytest.approx(0.8)
    assert macro(metrics, "average_precision") == pytest.approx(0.9)


def test_macro_skips_labels_that_come_from_ocr() -> None:
    """`unreadable` в пайплайне приходит от OCR-слоя, а не от сети.

    Голова общая и метку предсказывает, но отбирать по ней лучшую эпоху нельзя:
    в train три примера на восемь тысяч страниц. В таблице метка показана
    отдельно, в макро-среднее не входит.
    """
    metrics = [
        LabelMetrics("blur", 0.8, 0.8, 0.8, 0.9, support=10),
        LabelMetrics("unreadable", 0.1, 0.1, 0.1, 0.1, support=3),
    ]
    assert macro(metrics, "average_precision") == pytest.approx(0.5)
    assert macro(metrics, "average_precision", ["unreadable"]) == pytest.approx(0.9)


def test_macro_on_nothing_is_nan() -> None:
    metrics = [LabelMetrics("blur", 0.0, 0.0, 0.0, float("nan"), support=0)]
    assert np.isnan(macro(metrics, "f1"))


def test_per_label_counts_support_and_scores() -> None:
    labels = ["a", "b"]
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.7], [0.1, 0.3]])

    metrics = per_label(y_true, y_score, labels, threshold=0.5)

    by_name = {m.label: m for m in metrics}
    assert by_name["a"].support == 2
    assert by_name["a"].precision == pytest.approx(1.0)
    assert by_name["a"].recall == pytest.approx(1.0)
    assert by_name["b"].support == 1


def test_summary_has_all_four_macros() -> None:
    metrics = [LabelMetrics("blur", 0.5, 0.6, 0.55, 0.7, support=4)]
    assert set(summary(metrics)) == {
        "macro_f1",
        "macro_ap",
        "macro_precision",
        "macro_recall",
    }


# --- агрегация патчей в страницу --------------------------------------------


class _ConstantPatches:
    """Модель-заглушка: отдаёт заранее заданные вероятности, по строке на патч."""

    def __init__(self, probabilities) -> None:
        import torch

        probs = torch.tensor(probabilities, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        self.logits = torch.log(probs / (1 - probs))

    def eval(self):
        return self

    def __call__(self, batch):
        return self.logits


def test_page_score_is_max_for_local_and_mean_for_global() -> None:
    """Мерить надо страницу, а не патч — так же, как схлопывает приложение.

    У реальной страницы масок нет, её метка достаётся всем патчам, и патч из
    центра отвечал бы за блик в углу. Локальные метки получали от этого AP
    на уровне своей доли в выборке, то есть уровень случайного угадывания.
    """
    import torch

    from src.config import load_config
    from src.models.evaluate import predict

    config = load_config()
    glare = config.labels.index("glare")  # локальная -> max
    blur = config.labels.index("blur")  # глобальная -> mean

    first = [0.0] * config.n_labels
    second = [0.0] * config.n_labels
    first[glare], second[glare] = 0.9, 0.1
    first[blur], second[blur] = 0.9, 0.1

    targets = torch.zeros(1, 2, config.n_labels)
    targets[0, :, glare] = 1.0
    loader = [(torch.zeros(1, 2, 1), targets)]

    y_true, y_score = predict(_ConstantPatches([first, second]), loader, config)

    assert y_score.shape == (1, config.n_labels)
    assert y_score[0, glare] == pytest.approx(0.9, abs=1e-4)
    assert y_score[0, blur] == pytest.approx(0.5, abs=1e-4)
    # Истина у страницы одна на все патчи — максимум возвращает её без изменений.
    assert y_true[0, glare] == pytest.approx(1.0)


# --- загрузчики --------------------------------------------------------------


def test_validation_batch_counts_patches_not_pages(tmp_path) -> None:
    """Батч измеряется в страницах, а патчей со страницы на валидации больше.

    Валидационный загрузчик остался бы с батчем в 32 страницы по 8 патчей —
    256 патчей за проход вместо тридцати двух, молча и только на валидации.
    """
    import json
    from argparse import Namespace

    from src.config import load_config
    from src.data.generate import write_manifest
    from src.labeling.app import append_label
    from src.models.train import build_loaders
    from src.schema import LabelRecord, SyntheticRecord

    config = load_config()
    for name, images in (("train", ["t.jpg"]), ("val", ["v.jpg"]), ("test", [])):
        payload = {"documents": [f"d_{name}"], "images": images}
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    labels = tmp_path / "labels.jsonl"
    for image, document in (("t.jpg", "d_train"), ("v.jpg", "d_val")):
        append_label(
            labels,
            LabelRecord(image=image, document=document, corpus="t", labels={"blur": True}),
        )

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [SyntheticRecord(image="s.jpg", reference="t.jpg", document="d_s", corpus="t", labels={})],
        manifest,
    )

    args = Namespace(
        splits=tmp_path,
        labels=labels,
        data=tmp_path,
        manifest=manifest,
        synthetic=tmp_path,
        source="mixed",
    )
    train_loader, val_loader = build_loaders(args, config)

    patches_in_train = train_loader.batch_size * config.dataset.patches_per_page
    patches_in_val = val_loader.batch_size * config.dataset.val_patches_per_page

    assert patches_in_train == config.train.batch_size
    assert patches_in_val <= config.train.batch_size

    # Дообучение на реальных: синтетика в набор не попадает вовсе.
    only_real, _ = build_loaders(Namespace(**{**vars(args), "source": "real"}), config)
    assert len(only_real.dataset.samples) == 1
    assert {s.source for s in only_real.dataset.samples} == {"real"}
    assert len(train_loader.dataset.samples) == 2  # синтетика + реальная


def test_cosine_fits_the_epochs_actually_requested() -> None:
    """По `config.train.epochs` график при `--epochs 10` прошёл бы треть цикла."""
    from src.config import load_config
    from src.models.model import build_model
    from src.models.train import make_optimizer

    config = load_config()
    model = build_model(config.model.backbone, config.n_labels, pretrained=False)

    _, scheduler = make_optimizer(model, config, epochs=10)
    assert scheduler.T_max == 10

    _, default = make_optimizer(model, config)
    assert default.T_max == config.train.epochs
