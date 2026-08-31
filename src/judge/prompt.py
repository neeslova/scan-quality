"""Сборка промпта судьи из конфига проекта.

Шаблон задан в TASK и правится только по явному запросу: метрики разных прогонов
сравнимы лишь при неизменном промпте. Меняется в нём ровно то, что подставляется
из проекта — список осей и пороговые правила.

**Правила вердикта выводятся из `verdict.tau_*`, а не переписываются руками.**
Иначе промпт и пайплайн разъезжаются при первом же пересчёте порогов, и судья
начинает судить по одним числам, а система — по другим. Перевод шкалы прямой:
судья ставит 0..4, скор пайплайна — это оценка, делённая на 4.
"""

from __future__ import annotations

import json
from typing import Optional

from src.config import Config
from src.judge.schema import SEVERITY_MAX

# Определения осей — из таблицы дефектов PLAN.md, раздел «Разметка».
# Формулировки короткие: судье нужен признак, а не пересказ методики.
AXIS_DEFINITIONS = {
    "blur": "out-of-focus or motion smear; character edges are not sharp",
    "glare": "local blown-out highlight; an area is washed to plain white",
    "shadow": "darkening from a fold, book binding, or the lid edge",
    "skew": "text lines tilted by more than about 2 degrees",
    "cropped": "a document edge is cut off; part of the text is outside the frame",
    "low_resolution": "characters are not separable; line height is tiny",
    "low_contrast": "faded print; grey text on a grey background",
    "noise": "noise, grain, speckle, or dirt on the scanner glass",
    "streaks": "streaks, roller marks, or copier artifacts",
    "unreadable": "the text cannot be made out at all",
}

SYSTEM = (
    "You are a document scan quality inspector. You judge IMAGE QUALITY only.\n"
    "Never comment on the document's meaning, topic, or content correctness."
)


def axis_lines(config: Config) -> str:
    """По строке на ось: имя и короткое определение.

    Порядок и состав берутся из `config.labels` — это источник истины. Ось без
    определения не пропускается молча: судья, которому не объяснили признак,
    начнёт выдумывать, и расхождение будет не с чем связать.
    """
    lines = []
    for label in config.labels:
        definition = AXIS_DEFINITIONS.get(label)
        if definition is None:
            raise KeyError(f"для оси {label} нет определения в AXIS_DEFINITIONS")
        lines.append(f"- {label}: {definition}")
    return "\n".join(lines)


def verdict_rules(config: Config) -> str:
    """Пороговые правила словами, выведенные из конфига.

    Пороги заданы в шкале 0..1, судья отвечает в шкале 0..4, поэтому каждый порог
    переводится в минимальную целую оценку, которая его превышает. Строгое
    неравенство сохраняется: в пайплайне сравнение идёт через `>`.
    """
    scale = float(SEVERITY_MAX)

    def threshold_to_severity(value: float) -> int:
        """Наименьшая целая оценка, строго превышающая порог."""
        severity = int(value * scale) + 1
        return min(severity, SEVERITY_MAX)

    unreadable = threshold_to_severity(config.verdict.tau_unreadable)
    high = threshold_to_severity(config.verdict.tau_high)
    low = threshold_to_severity(config.verdict.tau_low)

    return "\n".join(
        [
            f"- If unreadable >= {unreadable}, the verdict is bad.",
            f"- Else if any axis >= {high}, the verdict is bad.",
            f"- Else if any axis >= {low}, the verdict is acceptable.",
            "- Otherwise the verdict is good.",
        ]
    )


def output_example(config: Config) -> str:
    """Пример валидного ответа ровно с теми ключами, которых мы ждём."""
    example = {
        "scores": dict.fromkeys(config.labels, 0),
        "rotation_deg": 0,
        "blank_page": False,
        "legible_fraction": 1.0,
    }
    return json.dumps(example, indent=2, ensure_ascii=False)


def build_prompt(config: Config, few_shot: Optional[str] = None) -> str:
    """Полный текст запроса. `few_shot` вставляется перед правилами вердикта.

    Место под примеры оставлено намеренно пустым: три-пять размеченных изображений
    (плохое, пограничное, хорошее) влияют на согласие с человеком сильнее любых
    правок формулировок, и подбирать их должен тот, чьё мнение считается эталоном.
    """
    parts = [
        "Rate the scan on each axis. Use the scale strictly:",
        f"0 = not present / perfect, 1 = slight, 2 = moderate, 3 = severe, "
        f"{SEVERITY_MAX} = makes text unreadable",
        "",
        "Axes:",
        axis_lines(config),
        "",
        "Also report:",
        "- rotation_deg: one of 0, 90, 180, 270 (dominant text orientation)",
        "- blank_page: true if there is no meaningful text",
        "- legible_fraction: estimated share of the page's text a human could read, 0.0-1.0",
        "",
    ]

    if few_shot:
        parts.extend([few_shot, ""])

    parts.extend(
        [
            "Verdict rules (apply mechanically, do not override with intuition):",
            verdict_rules(config),
            "",
            "Judge only what is visible. Do not guess about parts you cannot see.",
            "Output ONLY valid JSON, no markdown fences, no explanation:",
            "",
            output_example(config),
        ]
    )
    return "\n".join(parts)
