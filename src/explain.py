"""Текстовое объяснение готового отчёта через внешний API. Опционально.

**Внешний API здесь — декоратор, а не источник вердикта** (решение №2). Он
получает УЖЕ посчитанный отчёт и только пересказывает его словами. Вердикт,
вероятности и пороги к моменту вызова определены полностью, и ответ модели на
них не влияет никак: если вызов не удался, отчёт остаётся тем же самым, просто
без абзаца текста. Автоматическая проверка потока сканов не может зависеть от
доступности стороннего сервиса.

Отсюда все решения в модуле:

  * `anthropic` не в базовых зависимостях, а в extra `explain`, и импортируется
    лениво. Без него весь остальной пайплайн работает как раньше;
  * любая ошибка — нет пакета, нет ключа, нет сети, лимит, 500 — возвращает
    None с записью в лог. Наружу не летит ничего;
  * **наружу уходят только числа и метки, не изображение.** Tobacco3482 —
    документы табачных процессов с настоящими именами и адресами, и отправлять
    сам скан в чужой сервис нельзя. Отправляется вердикт, список меток со
    скорами и источниками, список неизмеренного.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import Config
from src.schema import QualityReport

logger = logging.getLogger(__name__)

SYSTEM = (
    "Ты помогаешь оператору сканирования. По готовому машинному отчёту о качестве "
    "скана объясни на русском, что с ним не так и что делать: пересканировать, "
    "проверить глазами или пропустить. Пиши коротко, три-четыре предложения, без "
    "списков и заголовков.\n\n"
    "Отчёт уже посчитан, и менять его выводы нельзя. Вердикт и числа даны — "
    "объясняй их, а не пересматривай. Если метка помечена как неизмеренная, так и "
    "скажи: её не проверяли, а не «дефекта нет»."
)


def report_digest(report: QualityReport, config: Config) -> str:
    """Компактная выжимка отчёта. Только числа и метки — изображение не уходит."""
    lines = [
        f"Файл: {report.image}",
        f"Вердикт: {report.verdict} (сводный балл качества {report.quality_score})",
        "Метки:",
    ]
    for defect in report.defects:
        lines.append(f"  {defect.label}: {defect.score:.2f} (источник: {defect.source})")
    if report.not_applicable:
        lines.append("Не измерено (источник не смог оценить): " + ", ".join(report.not_applicable))
    lines.append(
        f"Порог «есть замечания» {config.verdict.tau_low}, "
        f"порог «плохо» {config.verdict.tau_high}."
    )
    return "\n".join(lines)


def explain(report: QualityReport, config: Config) -> Optional[str]:
    """Абзац текста по отчёту или None, если объяснить не удалось.

    None — рабочий исход, а не ошибка: отчёт полон и без него.
    """
    cfg = config.explain
    if not cfg.enabled:
        return None

    try:
        import anthropic
    except ModuleNotFoundError:
        logger.warning(
            "Пакет anthropic не установлен — объяснение пропущено. "
            'Поставить: pip install -e ".[explain]"'
        )
        return None

    try:
        client = anthropic.Anthropic(timeout=float(cfg.timeout_s))
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM,
            # Пересказ готовых чисел — простая задача, и глубоко думать над ней
            # незачем: низкое усилие тут экономит и время, и деньги.
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": report_digest(report, config)}],
        )
    except anthropic.AuthenticationError:
        logger.warning("Ключ ANTHROPIC_API_KEY не принят — объяснение пропущено")
        return None
    except anthropic.RateLimitError:
        logger.warning("Лимит запросов исчерпан — объяснение пропущено")
        return None
    except anthropic.APIConnectionError:
        # Ровно тот случай, ради которого модуль опционален: сети нет.
        logger.warning("Нет связи с API — объяснение пропущено, отчёт не изменился")
        return None
    except anthropic.APIStatusError as error:
        logger.warning("API ответил %s — объяснение пропущено", error.status_code)
        return None
    except Exception as error:  # noqa: BLE001 - декоратор не имеет права ронять отчёт
        logger.warning("Объяснение не получено (%s) — отчёт не изменился", error)
        return None

    if response.stop_reason == "refusal":
        logger.warning("Модель отказалась отвечать — объяснение пропущено")
        return None

    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    return text or None


def with_explanation(report: QualityReport, config: Config) -> QualityReport:
    """Тот же отчёт, но с текстом, если его удалось получить."""
    text = explain(report, config)
    return report if text is None else report.model_copy(update={"explanation": text})
