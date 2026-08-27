"""Сеть за onnxruntime: страница -> скоры по меткам.

Torch здесь нет намеренно. Он стоит только в extra `train`, а приложение обязано
работать на базовых зависимостях — значит инференс идёт через onnxruntime, и
нормировка патча повторяется вручную ровно той же формулой, что в обучении.
Разойдись они, сеть получила бы распределение, на котором не училась.

Патчи берутся регулярной сеткой и отбираются по наличию текста: на пустом поле
бумаги резкость и шум не измеримы. Обратно в скор страницы они схлопываются той
же агрегацией, что и на валидации, — локальные максимумом, глобальные средним.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.special import expit

from src.config import Config
from src.imaging import IMAGENET_MEAN, IMAGENET_STD
from src.io.patches import Patch, aggregate, grid, select_informative

logger = logging.getLogger(__name__)


def normalize_patches(gray: np.ndarray, patches: list[Patch], patch_size: int) -> np.ndarray:
    """Патчи -> тензор (N, 3, patch, patch), нормированный как в обучении.

    Три канала — повторённый серый, а не раскраска: страница чёрно-белая, а
    backbone ждёт три входа.
    """
    crops = []
    for patch in patches:
        crop = patch.crop(gray)
        if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
            crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        crops.append(crop.astype(np.float32))

    values = (np.stack(crops) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.repeat(values[:, None, :, :], 3, axis=1).astype(np.float32)


@dataclass(frozen=True)
class PagePrediction:
    """Результат одного прогона страницы: патчи и вероятности по ним."""

    patches: list[Patch]
    probabilities: np.ndarray
    labels: Sequence[str]
    local: Sequence[str]

    def scores(self) -> dict[str, float]:
        """Скор страницы по каждой метке: локальные максимумом, глобальные средним."""
        columns = {
            label: self.probabilities[:, position].tolist()
            for position, label in enumerate(self.labels)
        }
        return {label: round(value, 4) for label, value in aggregate(columns, self.local).items()}

    def top_patches(self, label: str, count: int = 3) -> list[int]:
        """Индексы патчей, где метка сработала сильнее всего — для heatmap в С7."""
        position = list(self.labels).index(label)
        order = np.argsort(self.probabilities[:, position])[::-1][:count]
        return [self.patches[int(i)].index for i in order]


class PatchPredictor:
    """Обёртка над quality.onnx. Модель загружается один раз и переиспользуется."""

    def __init__(self, model_path: Path, config: Config, batch: int = 8) -> None:
        import onnxruntime as ort

        self.config = config
        self.batch = batch
        self.patch = config.data.patch_size
        self.labels = list(config.labels)
        self._local = list(config.data.aggregation.local)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        logger.info("Модель загружена: %s", model_path)

    def _logits(self, batch: np.ndarray) -> np.ndarray:
        """Прогон пачками: страница A4 даёт 54 патча, и все сразу — лишняя память."""
        outputs = []
        for start in range(0, len(batch), self.batch):
            chunk = batch[start : start + self.batch]
            outputs.append(self.session.run(["logits"], {"patch": chunk})[0])
        return np.concatenate(outputs)

    def predict(self, gray: np.ndarray) -> PagePrediction:
        """Один прогон страницы. Всё остальное считается из его результата.

        Именно один: и скор метки, и её самые горячие патчи выводятся из одних
        и тех же вероятностей. Отдельные методы, каждый со своим прогоном,
        превращали страницу в три инференса вместо одного.
        """
        height, width = gray.shape[:2]
        boxes = grid(
            width,
            height,
            self.patch,
            self.config.data.grid.rows,
            self.config.data.grid.cols,
        )
        chosen = select_informative(gray, boxes, self.config.dataset.min_ink_frac)

        logits = self._logits(normalize_patches(gray, chosen, self.patch))
        return PagePrediction(
            patches=chosen,
            # expit, а не 1/(1+exp(-x)): на логите -700 наивная формула
            # переполняется и сыплет предупреждениями посреди инференса.
            probabilities=expit(logits),
            labels=self.labels,
            local=self._local,
        )

    def scores(self, gray: np.ndarray) -> dict[str, float]:
        """Скор страницы по каждой метке. Обёртка для одиночного вызова."""
        return self.predict(gray).scores()


def load_predictor(config: Config, model_path: Optional[Path] = None) -> Optional[PatchPredictor]:
    """Предсказатель или None, если модели нет.

    None — рабочее состояние, а не ошибка: без обученной сети система обязана
    работать на CV-метриках и OCR, просто без двух меток, которые даёт сеть.
    """
    path = Path(model_path or config.paths.onnx_model)
    if not path.is_file():
        logger.warning("Модель %s не найдена — метки сети выдаваться не будут", path)
        return None
    try:
        return PatchPredictor(path, config)
    except Exception as error:  # noqa: BLE001 - причин много, а падать нельзя
        logger.warning("Модель %s не загрузилась (%s) — работаем без неё", path, error)
        return None


_CACHE: dict[str, Optional[PatchPredictor]] = {}


def shared_predictor(config: Config, model_path: Optional[Path] = None) -> Optional[PatchPredictor]:
    """Один экземпляр на процесс: сессия onnxruntime поднимается сотни миллисекунд.

    Отсутствие модели кэшируется тоже — иначе на каждой странице батча пришлось бы
    заново проверять диск и писать в лог одно и то же предупреждение.
    """
    key = str(model_path or config.paths.onnx_model)
    if key not in _CACHE:
        _CACHE[key] = load_predictor(config, model_path)
    return _CACHE[key]
