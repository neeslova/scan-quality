"""torch Dataset: патчи 384×384 и метки, выведенные из масок.

Главное здесь — правило метки патча, и оно не сводится к «наследовать от страницы».

  * **Глобальный дефект** (размытие, шум, контраст, перекос, разрешение) —
    свойство всего прогона сканера, патч наследует метку страницы как есть.
  * **Локальный дефект** (блик, тень, полосы, обрез) — метку получает только патч,
    который реально накрывает область. Иначе патч из чистого угла обучал бы сеть,
    что блик выглядит как обычный текст, — а таких патчей на странице большинство.

Патчи берутся из полного разрешения. A4 при 300 dpi это 2480×3508, и ресайз всей
страницы в 224 px превращает текст в серый шум: `blur` и `low_resolution` после
этого физически недетектируемы (журнал, №4).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import Config
from src.data.generate import read_manifest
from src.data.split import read_labels
from src.imaging import IMAGENET_MEAN, IMAGENET_STD, binarize_ink
from src.io.loader import load_page

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """Одна страница-источник: где лежит, какие метки, где маски локальных."""

    path: Path
    labels: dict[str, bool]
    masks: dict[str, Path]
    source: str  # synthetic | real


def _mask_for_patch(
    mask: np.ndarray, total: int, box: tuple[int, int, int, int], shape
) -> tuple[float, float]:
    """Пересечение патча и маски дефекта двумя долями сразу.

    Возвращает (доля патча под маской, доля маски внутри патча). Одной доли
    патча не хватает: тонкий дефект физически не способен накрыть заметную часть
    квадрата 384x384, см. `_patch_labels`. Маска хранится уменьшенной, поэтому
    бокс масштабируем; обе доли — отношения, и масштаб из них сокращается.
    """
    x0, y0, x1, y1 = box
    height, width = shape
    scale_x = mask.shape[1] / width
    scale_y = mask.shape[0] / height
    mx0, mx1 = int(x0 * scale_x), max(int(x1 * scale_x), int(x0 * scale_x) + 1)
    my0, my1 = int(y0 * scale_y), max(int(y1 * scale_y), int(y0 * scale_y) + 1)
    window = mask[my0:my1, mx0:mx1]
    if window.size == 0 or total == 0:
        return 0.0, 0.0
    inside = int(np.count_nonzero(window))
    return inside / window.size, inside / total


def collect_synthetic(manifest: Path, root: Path, exclude_documents: Optional[set[str]] = None):
    """Страницы синтетики за вычетом отложенных документов.

    Именно ИСКЛЮЧЕНИЕ, а не отбор по train. Сплит покрывает только 300 размеченных
    страниц, а эталоны синтетики берутся из всего корпуса вне val/test — в те
    шестьдесят документов train они почти не попадают, и фильтр «оставить только
    train» выбросил бы почти всю синтетику.

    Генератор уже исключает val/test при отборе эталонов; проверка здесь вторая
    линия обороны — цена ошибки слишком велика, чтобы полагаться на одну.
    """
    samples = []
    for record in read_manifest(manifest):
        if exclude_documents is not None and record.document in exclude_documents:
            logger.warning("Документ %s из val/test попал в синтетику — пропуск", record.document)
            continue
        samples.append(
            Sample(
                path=root / record.image,
                labels=dict(record.labels),
                masks={label: root / rel for label, rel in record.masks.items()},
                source="synthetic",
            )
        )
    return samples


def collect_real(labels_path: Path, root: Path, images: Optional[set[str]] = None):
    """Размеченные вручную страницы. У них масок нет — метки относятся ко всей странице."""
    samples = []
    for record in read_labels(labels_path):
        if images is not None and record.image not in images:
            continue
        samples.append(
            Sample(
                path=root / record.image,
                labels=dict(record.labels),
                masks={},
                source="real",
            )
        )
    return samples


def load_split(splits_dir: Path, name: str) -> tuple[set[str], set[str]]:
    with (splits_dir / f"{name}.json").open("r", encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    return set(payload["documents"]), set(payload["images"])


class PatchDataset:
    """Патчи со страниц. Совместим с torch DataLoader, но torch импортируется лениво.

    Ленивый импорт нарочно: локально torch стоит только CPU-версией и нужен для
    отладки, а весь остальной пайплайн (метрики, разметка, синтетика) обязан
    работать вообще без него.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        config: Config,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("Набор пуст: нечего обучать")
        self.samples = list(samples)
        self.config = config
        self.labels = list(config.labels)
        self.train = train
        self.patch = config.data.patch_size
        self.per_page = (
            config.dataset.patches_per_page if train else config.dataset.val_patches_per_page
        )
        self._local = set(config.data.aggregation.local)
        self._seed = seed
        self._cache: dict[Path, Optional[tuple[np.ndarray, int]]] = {}

    def _rng(self, index: int) -> np.random.Generator:
        # Детерминированность важна для валидации: одни и те же патчи каждую эпоху,
        # иначе кривая val дрожит от смены патчей, а не от обучения.
        return np.random.default_rng(self._seed * 1_000_003 + index)

    def _read(self, path: Path) -> np.ndarray:
        page = load_page(
            path,
            target_dpi=self.config.data.target_dpi,
            dpi_fallback=self.config.data.dpi_fallback,
            allow_upscale=self.config.data.allow_upscale,
        )
        return page.gray

    def _pick_box(self, gray: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
        """Патч с текстом. На пустом поле бумаги учить нечему."""
        height, width = gray.shape
        size = min(self.patch, height, width)
        best: Optional[tuple[int, int, int, int]] = None
        best_ink = -1.0

        for _ in range(self.config.dataset.max_patch_attempts):
            x0 = int(rng.integers(0, max(1, width - size + 1)))
            y0 = int(rng.integers(0, max(1, height - size + 1)))
            box = (x0, y0, x0 + size, y0 + size)
            crop = gray[y0 : y0 + size, x0 : x0 + size]
            ink = float(np.count_nonzero(binarize_ink(crop)) / crop.size)
            if ink >= self.config.dataset.min_ink_frac:
                return box
            if ink > best_ink:
                best_ink, best = ink, box

        return best if best is not None else (0, 0, size, size)

    def _mask(self, path: Path) -> Optional[tuple[np.ndarray, int]]:
        """Бинарная маска и её площадь.

        Нечитаемую маску кэшируем как None: иначе битый файл перечитывался бы
        с диска на каждый патч каждой эпохи.
        """
        if path in self._cache:
            return self._cache[path]
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            logger.warning("Маска не читается: %s", path)
            self._cache[path] = None
            return None
        mask = raw > 127
        entry = (mask, int(np.count_nonzero(mask)))
        self._cache[path] = entry
        return entry

    def _patch_labels(self, sample: Sample, box, shape) -> np.ndarray:
        target = np.zeros(len(self.labels), dtype=np.float32)
        for position, label in enumerate(self.labels):
            if not sample.labels.get(label):
                continue
            if label in self._local and label in sample.masks:
                entry = self._mask(sample.masks[label])
                if entry is None:
                    continue
                coverage, share = _mask_for_patch(entry[0], entry[1], box, shape)
                # Два критерия по ИЛИ. Одной доли патча мало: маска `streaks` —
                # это тонкие линии, `cropped` — узкая полоса у края кадра, и 15%
                # квадрата 384x384 им недостижимы арифметически. Патч не получал
                # метку почти никогда (0.8% и 1.7% замером), и сеть двух дефектов
                # из десяти не видела вовсе. Вторая доля спрашивает обратное —
                # попал ли в патч заметный кусок самого дефекта. Для крупной
                # маски она строже первой и потому сама уходит с дороги.
                if (
                    coverage < self.config.dataset.min_mask_overlap
                    and share < self.config.dataset.min_mask_share
                ):
                    continue
            target[position] = 1.0
        return target

    def _augment(self, crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Только яркость и контраст.

        Геометрию трогать нельзя: перекос и обрез — сами по себе метки, и случайный
        поворот подделывал бы разметку. Отражения тоже: текст в зеркале не текст.
        """
        cfg = self.config.dataset
        brightness = float(rng.uniform(-cfg.brightness, cfg.brightness)) * 255.0
        contrast = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
        mean = float(crop.mean())
        out = (crop.astype(np.float32) - mean) * contrast + mean + brightness
        return np.clip(out, 0, 255)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """Все патчи одной страницы за один раз: (N, 3, H, W) и (N, меток).

        Именно все сразу, а не по одному. Чтобы взять патч, надо декодировать
        страницу целиком — многомегапиксельный JPEG, — и при выдаче по одному
        патчу одна и та же страница декодировалась `patches_per_page` раз за
        эпоху. На двух ядрах Colab это упирало эпоху в 16 минут при простаивающем
        GPU. Один декод на страницу вместо четырёх.
        """
        import torch

        sample = self.samples[index % len(self.samples)]
        gray = self._read(sample.path)

        patches = []
        targets = []
        for step in range(self.per_page):
            rng = self._rng(index * self.per_page + step)
            box = self._pick_box(gray, rng)
            crop = gray[box[1] : box[3], box[0] : box[2]]
            targets.append(self._patch_labels(sample, box, gray.shape))

            if crop.shape[0] != self.patch or crop.shape[1] != self.patch:
                crop = cv2.resize(crop, (self.patch, self.patch), interpolation=cv2.INTER_AREA)

            values = self._augment(crop, rng) if self.train else crop.astype(np.float32)
            values = (values / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            patches.append(values)

        # Backbone ждёт три канала; серую страницу повторяем, а не красим.
        stack = torch.from_numpy(np.stack(patches)).unsqueeze(1).repeat(1, 3, 1, 1)
        return stack, torch.from_numpy(np.stack(targets))


def flatten_patches(batch, target):
    """(страниц, патчей, 3, H, W) -> (страниц*патчей, 3, H, W).

    DataLoader собирает батч из страниц, а модель принимает патчи — распрямляем.
    """
    return batch.flatten(0, 1), target.flatten(0, 1)


def patch_label_dilution(dataset: PatchDataset, sample_pages: int, seed: int = 0) -> np.ndarray:
    """Во сколько раз метка на патче реже, чем на странице. Оценка по выборке.

    Локальную метку патч получает, только если реально накрывает дефект: замером
    на синтетике её получают 58% патчей у `streaks` и 7.5% у `cropped`. Без этой
    поправки `pos_weight` считался бы по страницам и занижал вес на порядок
    ровно там, где метка и так редкая.

    Метка, ни разу не встреченная в выборке, получает 1.0: разбавление мы просто
    не измерили, и занижать вес по неизмеренному хуже, чем оставить страничную
    оценку. Так сохраняется поведение для `unreadable` — три примера на восемь
    тысяч страниц в выборку из четырёхсот почти наверняка не попадут.

    Выборкой, а не полным проходом: оценка требует декодирования страниц, и на
    восьми тысячах это лишний проход по всему корпусу перед первой же эпохой.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset), size=min(sample_pages, len(dataset)), replace=False)

    on_patches = np.zeros(len(dataset.labels), dtype=np.float64)
    possible = np.zeros(len(dataset.labels), dtype=np.float64)
    for index in chosen:
        sample = dataset.samples[int(index)]
        _, target = dataset[int(index)]
        array = np.asarray(target)
        on_patches += array.sum(axis=0)
        possible += [
            array.shape[0] if sample.labels.get(label) else 0.0 for label in dataset.labels
        ]
    return np.where(possible > 0, on_patches / np.maximum(possible, 1.0), 1.0)


def page_positive_rates(samples: Sequence[Sample], labels: Sequence[str]) -> np.ndarray:
    """Доля страниц с меткой. Основа для `pos_weight`, поправка — в
    `patch_label_dilution`."""
    total = max(1, len(samples))
    return np.array(
        [sum(1 for s in samples if s.labels.get(label)) / total for label in labels],
        dtype=np.float64,
    )
