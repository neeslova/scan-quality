"""Оркестратор: изображение -> QualityReport.

С0: настоящих предсказаний ещё нет — `analyze` отдаёт детерминированную заглушку,
чтобы был сквозной путь «файл -> JSON» и было что показать в Gradio.
Реальные ветки подключаются по спринтам:
  С1 — CV-метрики, С3 — OCR, С5/С6 — CNN через onnxruntime.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from src.config import Config, VerdictConfig, load_config
from src.schema import DefectScore, QualityReport, Verdict

logger = logging.getLogger(__name__)


def decide_verdict(scores: Mapping[str, float], cfg: VerdictConfig) -> Verdict:
    """Правило вердикта из PLAN.md §4.

    Асимметрия намеренная: пропустить плохой скан дороже, чем отклонить хороший,
    поэтому `unreadable` имеет собственный, более низкий порог.
    """
    if scores.get("unreadable", 0.0) > cfg.tau_unreadable:
        return "bad"
    if any(score > cfg.tau_high for score in scores.values()):
        return "bad"
    if any(score > cfg.tau_low for score in scores.values()):
        return "acceptable"
    return "good"


def quality_score(scores: Mapping[str, float]) -> float:
    """Сводный балл 0..1: 1 — идеальный скан. Пока просто «один минус худший дефект»."""
    if not scores:
        return 1.0
    return round(1.0 - max(scores.values()), 4)


def _image_size(path: Path) -> Tuple[int, int]:
    """Размер картинки; без Pillow (голое окружение) возвращаем нули, а не падаем."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow не установлен — размер изображения не определён")
        return (0, 0)

    try:
        with Image.open(path) as img:
            return img.size
    except Exception as exc:  # noqa: BLE001 — заглушка не должна ронять приложение
        logger.warning("Не удалось прочитать %s: %s", path, exc)
        return (0, 0)


def _stub_scores(labels: Sequence[str], path: Path) -> Dict[str, float]:
    """Псевдослучайные, но стабильные для файла вероятности.

    Seed — от содержимого файла: один и тот же скан всегда даёт один и тот же отчёт,
    иначе заглушку невозможно отлаживать.
    """
    digest = hashlib.sha256(path.read_bytes()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    # Bias к нулю: у «типичного» скана большинство меток близко к нулю.
    return {label: round(rng.betavariate(1.2, 6.0), 3) for label in labels}


def analyze(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
) -> QualityReport:
    """Пока — заглушка (С0). Сигнатура зафиксирована, наполнение придёт по спринтам."""
    started = time.perf_counter()
    cfg = config or load_config()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    scores = _stub_scores(cfg.labels, path)
    width, height = _image_size(path)

    defects = [
        DefectScore(label=label, score=score, source="stub") for label, score in scores.items()
    ]
    defects.sort(key=lambda d: d.score, reverse=True)

    return QualityReport(
        pipeline_version="stub",
        image=path.name,
        width=width,
        height=height,
        verdict=decide_verdict(scores, cfg.verdict),
        quality_score=quality_score(scores),
        defects=defects,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
