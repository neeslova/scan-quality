"""Загрузка и валидация конфигурации проекта: configs/*.yaml -> pydantic-схема.

Единственное место, где YAML превращается в объекты. Все остальные модули принимают
готовый :class:`Config` и не читают файлы конфигурации сами.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

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

    local: List[str]
    # `global` — ключевое слово Python, поэтому поле с алиасом.
    global_: List[str] = Field(alias="global", serialization_alias="global")


class DataConfig(_Section):
    patch_size: int = Field(gt=0)
    target_dpi: int = Field(gt=0)
    grid: GridConfig
    aggregation: AggregationConfig


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
    pos_weight: Union[str, List[float]] = "auto"

    @model_validator(mode="after")
    def _check_pos_weight(self) -> "TrainConfig":
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


class VerdictConfig(_Section):
    tau_low: float = Field(ge=0.0, le=1.0)
    tau_high: float = Field(ge=0.0, le=1.0)
    tau_unreadable: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> "VerdictConfig":
        if self.tau_low >= self.tau_high:
            raise ValueError(
                f"verdict: требуется tau_low < tau_high, получено "
                f"{self.tau_low} >= {self.tau_high}"
            )
        return self


class Config(_Section):
    labels: List[str]
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    verdict: VerdictConfig

    @model_validator(mode="after")
    def _check_labels(self) -> "Config":
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
        return self

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


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Читает YAML и валидирует его. Без аргумента — configs/base.yaml."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Ожидался словарь на верхнем уровне {cfg_path}, получено {type(raw)}")

    config = Config.model_validate(raw)
    logger.debug("Конфиг загружен: %s (%d меток)", cfg_path, config.n_labels)
    return config
