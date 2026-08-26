"""Gradio-приложение: скан -> вердикт, вероятности дефектов, CV-метрики, JSON.

С1: вердикт считается CV-baseline'ом (без обучения). Батч, PDF и Grad-CAM — С7.
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
from src.schema import QualityReport

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    "good": "✅ good — скан пригоден",
    "acceptable": "⚠️ acceptable — есть замечания",
    "bad": "⛔ bad — пересканировать",
}

# Что показываем в таблице метрик и как подписываем. Остальное остаётся в JSON.
METRIC_ROWS: list[tuple[str, str]] = [
    ("tenengrad_norm", "Резкость, нормированная на контраст"),
    ("tenengrad", "Резкость (Tenengrad)"),
    ("laplacian_var", "Резкость (дисперсия лапласиана)"),
    ("noise_sigma", "Шум, σ"),
    ("ink_paper_gap", "Разрыв бумага/чернила"),
    ("rms_contrast", "RMS-контраст"),
    ("line_height_px", "Высота строки, px"),
    ("source_line_height_px", "Высота строки в исходном файле, px"),
    ("skew_deg", "Перекос, °"),
    ("min_margin_frac", "Минимальное поле, доля"),
    ("glare_cluster_frac", "Пересвет, доля площади"),
    ("shadow_frac", "Тень, доля площади"),
    ("streak_energy", "Полосы, энергия"),
    ("mid_tone_frac", "Доля средних тонов"),
    ("dpi", "Рабочий dpi"),
    ("n_informative_patches", "Патчей с текстом"),
]

EMPTY = ("Загрузите изображение", {}, [], {})


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


def _metrics_table(report: QualityReport) -> list[list[object]]:
    return [
        [title, report.cv_metrics[key]] for key, title in METRIC_ROWS if key in report.cv_metrics
    ]


def _run(image_path: Optional[str], config: Config):
    """Хендлер: путь к файлу -> (вердикт, вероятности, таблица метрик, полный JSON)."""
    if not image_path:
        return EMPTY

    report = analyze(image_path, config)
    verdict = VERDICT_LABELS[report.verdict]
    if report.not_applicable:
        verdict += f"   ·   не измерено: {', '.join(report.not_applicable)}"
    return verdict, report.scores(), _metrics_table(report), report.model_dump(mode="json")


def build_demo(config: Config):
    """Собирает интерфейс. Вынесено отдельно, чтобы можно было монтировать в тестах."""
    gr = _import_gradio()

    with gr.Blocks(title="Качество сканов") as demo:
        gr.Markdown(
            "# Оценка качества сканов\n"
            "**Спринт С1: вердикт считает CV-baseline** — детерминированные метрики "
            "без обучения. Это точка отсчёта, с которой дальше сравнивается CNN."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Скан (jpg/png/tiff)", height=460)
                run_btn = gr.Button("Проверить", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Textbox(label="Вердикт", interactive=False)
                defects = gr.Label(label="Вероятности дефектов", num_top_classes=5)
                metrics = gr.Dataframe(
                    headers=["Метрика", "Значение"],
                    label="CV-метрики",
                    interactive=False,
                    wrap=True,
                )
                report = gr.JSON(label="QualityReport")

        outputs = [verdict, defects, metrics, report]
        run_btn.click(fn=lambda p: _run(p, config), inputs=image, outputs=outputs)
        image.change(fn=lambda p: _run(p, config), inputs=image, outputs=outputs)

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
        print(analyze(args.image, config).to_json())
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
