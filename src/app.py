"""Gradio-приложение. С0: одиночный режим, картинка -> JSON-отчёт (заглушка).

Батч, PDF и Grad-CAM появятся в С7 — здесь только сквозной путь и DoD спринта С0.
Запуск: python -m src.app  (или scanq-app после `pip install -e .`)
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.pipeline import analyze

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    "good": "✅ good — скан пригоден",
    "acceptable": "⚠️ acceptable — есть замечания",
    "bad": "⛔ bad — пересканировать",
}


def _import_gradio():
    # Gradio при импорте стучится в api.gradio.app. Система обязана работать без сети —
    # выключаем телеметрию до импорта, иначе старт зависит от интернета.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover — зависит от окружения
        if exc.name != "gradio":
            raise
        raise SystemExit(
            "Gradio не установлен. Поставь зависимости:\n"
            '  py -m pip install -U pip setuptools\n  py -m pip install -e ".[dev]"'
        ) from exc
    return gr


def _run(image_path: Optional[str], config: Config) -> tuple[str, dict]:
    """Хендлер кнопки: путь к файлу -> (человекочитаемый вердикт, JSON-отчёт)."""
    if not image_path:
        return "Загрузите изображение", {}

    report = analyze(image_path, config)
    logger.info("%s -> %s (%.0f мс)", report.image, report.verdict, report.elapsed_ms)
    return VERDICT_LABELS[report.verdict], report.model_dump(mode="json")


def build_demo(config: Config):
    """Собирает интерфейс. Вынесено отдельно, чтобы можно было монтировать в тестах."""
    gr = _import_gradio()

    with gr.Blocks(title="Качество сканов") as demo:
        gr.Markdown(
            "# Оценка качества сканов\n"
            "**Спринт С0: вердикт фиктивный.** Проверяем сквозной путь "
            "«файл → QualityReport → JSON»."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Скан (jpg/png)", height=420)
                run_btn = gr.Button("Проверить", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Textbox(label="Вердикт", interactive=False)
                report = gr.JSON(label="QualityReport")

        run_btn.click(fn=lambda p: _run(p, config), inputs=image, outputs=[verdict, report])
        image.change(fn=lambda p: _run(p, config), inputs=image, outputs=[verdict, report])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio-приложение оценки качества сканов")
    parser.add_argument("--config", type=Path, default=None, help="путь к yaml-конфигу")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="публичная ссылка Gradio")
    parser.add_argument(
        "--no-browser", action="store_true", help="не открывать браузер (проверки, CI)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="прогнать один файл в консоль и выйти, без запуска UI",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)

    if args.image is not None:
        report = analyze(args.image, config)
        print(report.to_json())
        return

    # Gradio на Windows иначе кладёт временные файлы в неудобное место.
    tempfile.tempdir = tempfile.gettempdir()
    build_demo(config).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
