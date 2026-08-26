"""Контракт системы: структуры на границах пайплайна.

`QualityReport` — то, что уходит наружу (Gradio, CLI, встраивание в поток проверки).
Меняем осторожно и вместе с `schema_version`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Verdict = Literal["good", "acceptable", "bad"]
ScoreSource = Literal["cnn", "cv", "ocr", "stub"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DefectScore(_Model):
    """Вероятность одного дефекта и то, кто её выдал."""

    label: str
    score: float = Field(ge=0.0, le=1.0)
    source: ScoreSource = "cnn"
    # Для локальных дефектов — индексы патчей, где сработало сильнее всего.
    top_patches: list[int] = Field(default_factory=list)


class OCRResult(_Model):
    engine: str
    mean_confidence: float = Field(ge=0.0, le=1.0)
    garbage_ratio: float = Field(ge=0.0, le=1.0)
    text_density: float = Field(ge=0.0)
    n_boxes: int = Field(ge=0)


class QualityReport(_Model):
    """Машиночитаемый результат по одной странице."""

    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = "stub"

    image: str
    width: int = Field(ge=0)
    height: int = Field(ge=0)

    verdict: Verdict
    quality_score: float = Field(ge=0.0, le=1.0)
    defects: list[DefectScore] = Field(default_factory=list)

    cv_metrics: dict[str, float] = Field(default_factory=dict)
    ocr: Optional[OCRResult] = None

    heatmap_path: Optional[str] = None
    explanation: Optional[str] = None
    elapsed_ms: float = Field(default=0.0, ge=0.0)

    def scores(self) -> dict[str, float]:
        """Плоский вид {метка: вероятность} — удобно для правил и таблиц."""
        return {d.label: d.score for d in self.defects}

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=False)
