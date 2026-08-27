"""Загрузка и валидация конфигурации проекта: configs/*.yaml -> pydantic-схема.

Единственное место, где YAML превращается в объекты. Все остальные модули принимают
готовый :class:`Config` и не читают файлы конфигурации сами.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


class _Section(BaseModel):
    """Общие настройки для всех секций: опечатка в yaml должна падать, а не молчать."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class GridConfig(_Section):
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)

    @property
    def n_patches(self) -> int:
        return self.rows * self.cols


class AggregationConfig(_Section):
    """Правило схлопывания предсказаний по патчам: локальные — max, глобальные — mean."""

    local: list[str]
    # `global` — ключевое слово Python, поэтому поле с алиасом.
    global_: list[str] = Field(alias="global", serialization_alias="global")


class DataConfig(_Section):
    patch_size: int = Field(gt=0)
    target_dpi: int = Field(gt=0)
    dpi_fallback: Literal["a4", "none"] = "a4"
    allow_upscale: bool = False
    grid: GridConfig
    aggregation: AggregationConfig


class ScoreMapping(_Section):
    """Линейный перевод сырой метрики в скор 0..1 по двум якорям."""

    metric: str
    good: float
    bad: float

    @model_validator(mode="after")
    def _check_anchors(self) -> ScoreMapping:
        if self.good == self.bad:
            raise ValueError(f"cv.scores[{self.metric}]: good и bad не должны совпадать")
        return self


class CVConfig(_Section):
    glare_threshold: int = Field(ge=0, le=255)
    glare_min_cluster_frac: float = Field(gt=0.0, lt=1.0)
    glare_flat_window_frac: float = Field(gt=0.0, lt=1.0)
    glare_flat_std: float = Field(gt=0.0)
    glare_min_excess: int = Field(ge=0, le=255)

    ink_block_frac: float = Field(gt=0.0, lt=1.0)
    ink_offset: int = Field(ge=0, le=255)
    ink_denoise_frac: float = Field(gt=0.0, lt=1.0)

    crop_band_frac: float = Field(gt=0.0, lt=1.0)
    crop_frame_span_frac: float = Field(gt=0.0, le=1.0)

    bitonal_mid_low: int = Field(ge=0, le=255)
    bitonal_mid_high: int = Field(ge=0, le=255)
    bitonal_max_mid_frac: float = Field(gt=0.0, lt=1.0)

    shadow_rel_threshold: float = Field(gt=0.0, lt=1.0)
    shadow_background_frac: float = Field(gt=0.0, lt=1.0)

    skew_max_angle: float = Field(gt=0.0, le=45.0)
    skew_coarse_step: float = Field(gt=0.0)
    skew_fine_step: float = Field(gt=0.0)
    skew_work_height: int = Field(gt=0)

    min_ink_frac: float = Field(ge=0.0, lt=1.0)
    line_min_row_ink_frac: float = Field(ge=0.0, lt=1.0)
    line_max_height_frac: float = Field(gt=0.0, le=1.0)
    line_profile_smooth_frac: float = Field(ge=0.0, lt=1.0)
    streak_smooth_frac: float = Field(gt=0.0, lt=1.0)

    scores: dict[str, ScoreMapping]

    @model_validator(mode="after")
    def _check_steps(self) -> CVConfig:
        if self.skew_fine_step > self.skew_coarse_step:
            raise ValueError("cv: skew_fine_step должен быть не больше skew_coarse_step")
        if self.bitonal_mid_low >= self.bitonal_mid_high:
            raise ValueError("cv: требуется bitonal_mid_low < bitonal_mid_high")
        return self


class AnchorPair(_Section):
    """Пара якорей без указания метрики: имя метрики задаётся ключом секции."""

    good: float
    bad: float

    @model_validator(mode="after")
    def _check_anchors(self) -> AnchorPair:
        if self.good == self.bad:
            raise ValueError("good и bad не должны совпадать")
        return self


class OCRConfig(_Section):
    engine: Literal["easyocr", "tesseract"] = "easyocr"
    languages: list[str] = Field(min_length=1)
    work_side: int = Field(gt=0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    extra_chars: str
    unreadable: dict[str, AnchorPair]
    min_boxes: int = Field(ge=0)
    unreadable_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_signals(self) -> OCRConfig:
        expected = {"mean_confidence", "garbage_ratio", "nonword_ratio", "readable_share"}
        if set(self.unreadable) != expected:
            raise ValueError(f"ocr.unreadable: нужны ровно {sorted(expected)}")
        return self


class LabelingConfig(_Section):
    auto_labels: list[str] = Field(default_factory=list)
    suggest_threshold: float = Field(ge=0.0, le=1.0)
    sample_total: int = Field(gt=0)
    random_share: float = Field(ge=0.0, le=1.0)
    sample_seed: int


class SplitConfig(_Section):
    ratios: dict[str, float]
    seed: int
    document_id: Literal["bates7", "stem", "parent"] = "bates7"

    @model_validator(mode="after")
    def _check_ratios(self) -> SplitConfig:
        expected = {"train", "val", "test"}
        if set(self.ratios) != expected:
            raise ValueError(f"split.ratios: нужны ровно {sorted(expected)}")
        if any(value <= 0.0 for value in self.ratios.values()):
            raise ValueError("split.ratios: все доли должны быть положительными")
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split.ratios: сумма должна быть 1.0, получено {total}")
        return self


class RangePair(_Section):
    """[значение при severity=0, значение при severity=1]. Порядок может убывать."""

    min: float
    max: float


class ReferenceConfig(_Section):
    max_defect_score: float = Field(ge=0.0, le=1.0)
    count: int = Field(gt=0)


class SynthConfig(_Section):
    seed: int
    target_total: int = Field(gt=0)
    max_defects_per_image: int = Field(gt=0)
    min_label_share: float = Field(ge=0.0, lt=1.0)
    base_probability: float = Field(gt=0.0, le=1.0)
    severity: RangePair
    reference: ReferenceConfig
    # Ключ — имя деградации, значение — {параметр: [начало, конец]}.
    params: dict[str, dict[str, list[float]]]

    @model_validator(mode="after")
    def _check_params(self) -> SynthConfig:
        for name, block in self.params.items():
            for key, bounds in block.items():
                if len(bounds) != 2:
                    raise ValueError(f"synth.params.{name}.{key}: нужно ровно два значения")
                if bounds[0] == bounds[1]:
                    raise ValueError(f"synth.params.{name}.{key}: границы совпадают")
        if self.severity.min > self.severity.max:
            raise ValueError("synth.severity: min должен быть не больше max")
        return self

    def span(self, name: str, key: str) -> tuple[float, float]:
        bounds = self.params[name][key]
        return (float(bounds[0]), float(bounds[1]))


class DatasetConfig(_Section):
    patches_per_page: int = Field(gt=0)
    val_patches_per_page: int = Field(gt=0)
    min_ink_frac: float = Field(ge=0.0, lt=1.0)
    max_patch_attempts: int = Field(gt=0)
    min_mask_overlap: float = Field(ge=0.0, le=1.0)
    min_mask_share: float = Field(ge=0.0, le=1.0)
    brightness: float = Field(ge=0.0, lt=1.0)
    contrast: float = Field(ge=0.0, lt=1.0)
    workers: int = Field(ge=0)


class ModelConfig(_Section):
    backbone: str
    pretrained: bool = True
    dropout: float = Field(ge=0.0, lt=1.0)


class TrainConfig(_Section):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    optimizer: str
    scheduler: str
    seed: int
    pos_weight: Union[str, list[float]] = "auto"
    pos_weight_max: float = Field(gt=0.0)
    pos_weight_sample: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_pos_weight(self) -> TrainConfig:
        if isinstance(self.pos_weight, str) and self.pos_weight != "auto":
            raise ValueError("train.pos_weight: допустимо 'auto' или список чисел")
        return self


class PathsConfig(_Section):
    raw: Path
    labeled: Path
    synthetic: Path
    splits: Path
    models: Path
    reports: Path
    onnx_model: Path


class ExplainConfig(_Section):
    """Внешний API как декоратор над готовым отчётом. По умолчанию выключен."""

    enabled: bool = False
    model: str
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0.0)


class CalibrateConfig(_Section):
    # Опорные точки шкалы метки. Намеренно отвязаны от verdict.tau_*: иначе
    # якоря строятся относительно порога, а порог подбирается по якорям — круг.
    anchor_low: float = Field(ge=0.0, le=1.0)
    anchor_high: float = Field(ge=0.0, le=1.0)
    target_recall: float = Field(gt=0.0, le=1.0)
    confident_recall: float = Field(gt=0.0, le=1.0)
    recall: dict[str, float] = Field(default_factory=dict)
    min_support: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> CalibrateConfig:
        if self.confident_recall >= self.target_recall:
            raise ValueError(
                "calibrate: требуется confident_recall < target_recall, иначе точка "
                "«бесспорно плохо» не строже точки «подозрительно»: "
                f"{self.confident_recall} >= {self.target_recall}"
            )
        if self.anchor_low >= self.anchor_high:
            raise ValueError(
                "calibrate: требуется anchor_low < anchor_high, иначе шкала метки "
                f"переворачивается: {self.anchor_low} >= {self.anchor_high}"
            )
        return self


class VerdictConfig(_Section):
    tau_low: float = Field(ge=0.0, le=1.0)
    tau_high: float = Field(ge=0.0, le=1.0)
    tau_unreadable: float = Field(ge=0.0, le=1.0)
    # Тот же AnchorPair, что у CV-метрик: «0 в good, 1 в bad» — одна и та же
    # механика, только уровнем выше, над скором метки, а не над сырой метрикой.
    anchors: dict[str, AnchorPair] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_order(self) -> VerdictConfig:
        if self.tau_low >= self.tau_high:
            raise ValueError(
                f"verdict: требуется tau_low < tau_high, получено {self.tau_low} >= {self.tau_high}"
            )
        return self


class SourcesConfig(_Section):
    """Кто выдаёт какую метку. Заполняется по замеру, а не по замыслу."""

    cv: list[str] = Field(default_factory=list)
    cnn: list[str] = Field(default_factory=list)
    ocr: list[str] = Field(default_factory=list)

    def all_labels(self) -> list[str]:
        return [*self.cv, *self.cnn, *self.ocr]

    def of(self, label: str) -> str:
        for name in ("cv", "cnn", "ocr"):
            if label in getattr(self, name):
                return name
        raise KeyError(f"метке {label} не назначен источник")


class Config(_Section):
    labels: list[str]
    sources: SourcesConfig
    paths: PathsConfig
    data: DataConfig
    cv: CVConfig
    ocr: OCRConfig
    labeling: LabelingConfig
    split: SplitConfig
    synth: SynthConfig
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    verdict: VerdictConfig
    calibrate: CalibrateConfig
    explain: ExplainConfig

    @property
    def ocr_derived(self) -> list[str]:
        """Метки не от сети: в макро-среднее обучения они не входят (решение №38)."""
        return list(self.sources.ocr)

    @model_validator(mode="after")
    def _check_labels(self) -> Config:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels: есть дубликаты")

        local = set(self.data.aggregation.local)
        global_ = set(self.data.aggregation.global_)

        overlap = local & global_
        if overlap:
            raise ValueError(
                f"data.aggregation: метки одновременно в local и global: {sorted(overlap)}"
            )

        covered = local | global_
        expected = set(self.labels)
        if covered != expected:
            missing = sorted(expected - covered)
            unknown = sorted(covered - expected)
            raise ValueError(
                "data.aggregation должна покрывать ровно labels; "
                f"не назначены: {missing}, лишние: {unknown}"
            )

        # CV-baseline закрывает не все метки (unreadable приходит из OCR, С3),
        # но выдумывать метки, которых нет в таксономии, он не должен.
        unknown_cv = sorted(set(self.cv.scores) - expected)
        if unknown_cv:
            raise ValueError(f"cv.scores: метки вне таксономии: {unknown_cv}")

        assigned = self.sources.all_labels()
        duplicates = sorted({name for name in assigned if assigned.count(name) > 1})
        if duplicates:
            raise ValueError(f"sources: метка у двух источников сразу: {duplicates}")
        if set(assigned) != expected:
            missing = sorted(expected - set(assigned))
            unknown = sorted(set(assigned) - expected)
            raise ValueError(
                "sources должна покрывать ровно labels; "
                f"без источника: {missing}, лишние: {unknown}"
            )

        unknown_auto = sorted(set(self.labeling.auto_labels) - expected)
        if unknown_auto:
            raise ValueError(f"labeling.auto_labels: метки вне таксономии: {unknown_auto}")

        unknown_anchors = sorted(set(self.verdict.anchors) - expected)
        if unknown_anchors:
            raise ValueError(f"verdict.anchors: метки вне таксономии: {unknown_anchors}")

        unknown_recall = sorted(set(self.calibrate.recall) - expected)
        if unknown_recall:
            raise ValueError(f"calibrate.recall: метки вне таксономии: {unknown_recall}")
        return self

    @property
    def manual_labels(self) -> list[str]:
        """Метки, которые ставит человек: всё, кроме выводимых автоматически."""
        auto = set(self.labeling.auto_labels)
        return [label for label in self.labels if label not in auto]

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    def is_local(self, label: str) -> bool:
        """True — агрегируем по патчам через max, False — через mean."""
        return label in self.data.aggregation.local


def resolve_path(path: Path, root: Optional[Path] = None) -> Path:
    """Относительные пути из конфига считаем от корня репозитория."""
    if path.is_absolute():
        return path
    return (root or PROJECT_ROOT) / path


def deep_merge(base: dict, overlay: Mapping) -> dict:
    """Рекурсивно накладывает overlay на base. Словари сливаются, остальное заменяется.

    Список заменяется целиком, а не дополняется: иначе оверлей не смог бы,
    например, сократить набор меток — только расширить.
    """
    result = dict(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Ожидался словарь на верхнем уровне {path}, получено {type(raw)}")
    return raw


def load_config(
    path: Union[str, Path, None] = None,
    overlays: Union[str, Path, Sequence[Union[str, Path]], None] = None,
) -> Config:
    """Читает YAML и валидирует его. Без аргумента — configs/base.yaml.

    `overlays` — конфиги корпуса, накладываются поверх базового по порядку. Нужны
    потому, что якоря CV-метрик не универсальны: значение, которое на офисных сканах
    означает дефект, на архивных может быть нормой (см. журнал решений, №22).
    Оверлей задаёт только то, что отличается, — обычно блок `cv.scores`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(cfg_path)

    if overlays is None:
        paths: list[Path] = []
    elif isinstance(overlays, (str, Path)):
        paths = [Path(overlays)]
    else:
        paths = [Path(p) for p in overlays]

    for overlay_path in paths:
        raw = deep_merge(raw, _read_yaml(overlay_path))
        logger.debug("Наложен оверлей: %s", overlay_path)

    config = Config.model_validate(raw)
    logger.debug("Конфиг загружен: %s (%d меток)", cfg_path, config.n_labels)
    return config
