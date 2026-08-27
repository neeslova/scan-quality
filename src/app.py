"""Gradio-приложение: скан -> вердикт, вероятности дефектов, карта, JSON.

Три источника дают разные метки (решения №39-40), и интерфейс это показывает:
у каждой вероятности написано, кто её выдал. Иначе пользователь читает десять
чисел как одинаковые, хотя семь из них — детерминированные CV-метрики, две —
сеть, одна — OCR.

Карта дефекта строится не Grad-CAM'ом, а от источника метки — см. `src/localize.py`.
Локализуемы не все метки: у расфокуса и перекоса нет места на странице.

Запуск: python -m src.app  (или scanq-app после `pip install -e .`)
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, load_config
from src.io.loader import load_page, load_pages
from src.localize import heatmap, localizable, overlay
from src.pipeline import analyze, build_report
from src.schema import QualityReport

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    "good": "✅ good — скан пригоден",
    "acceptable": "⚠️ acceptable — есть замечания",
    "bad": "⛔ bad — пересканировать",
}

SOURCE_LABELS = {"cv": "CV-метрика", "cnn": "сеть", "ocr": "OCR", "stub": "заглушка"}

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

EMPTY = ("Загрузите скан", {}, [], {}, None, None)


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


def defect_rows(report: QualityReport, config: Config) -> list[list[object]]:
    """Таблица дефектов с источником: сеть, CV-метрика и OCR — не одно и то же."""
    rows = [
        [d.label, round(d.score, 3), SOURCE_LABELS.get(d.source, d.source)] for d in report.defects
    ]
    # Неизмеренное показываем строкой, а не молчанием: пустое место читается как
    # «дефекта нет», хотя означает «мерить было нечем» (решение №21).
    rows += [
        [label, "не измерено", SOURCE_LABELS.get(config.sources.of(label), "")]
        for label in report.not_applicable
    ]
    return rows


def verdict_line(report: QualityReport) -> str:
    line = VERDICT_LABELS[report.verdict]
    if report.not_applicable:
        line += f"   ·   не измерено: {', '.join(report.not_applicable)}"
    return line


def _run(path: Optional[str], with_ocr: bool, config: Config):
    """Хендлер одиночного режима.

    Возвращает ещё и саму страницу: карту дефекта считаем по требованию, когда
    пользователь выберет метку, а не на каждый скан — у сети это лишний проход.
    """
    if not path:
        return EMPTY

    page = load_page(
        path,
        target_dpi=config.data.target_dpi,
        dpi_fallback=config.data.dpi_fallback,
        allow_upscale=config.data.allow_upscale,
    )
    import time

    report = build_report(page, config, time.perf_counter(), with_ocr, _predictor(config))

    shown = [label for label in localizable(config) if label in report.scores()]
    return (
        verdict_line(report),
        report.scores(),
        _metrics_table(report),
        report.model_dump(mode="json"),
        page.gray,
        shown,
    )


def _predictor(config: Config):
    from src.models.infer import shared_predictor

    return shared_predictor(config)


def _heatmap(gray: Optional[np.ndarray], label: Optional[str], config: Config):
    """Карта по требованию. Метка без карты — честное «нечего показать»."""
    if gray is None or not label:
        return None
    prediction = None
    if config.sources.of(label) == "cnn":
        predictor = _predictor(config)
        prediction = predictor.predict(gray) if predictor is not None else None

    heat = heatmap(label, gray, config, prediction)
    if heat is None:
        return None
    return overlay(gray, heat)


def batch_rows(paths: list[str], with_ocr: bool, config: Config) -> list[list[object]]:
    """Батч: строка на страницу. Многостраничный PDF даёт несколько строк."""
    import time

    predictor = _predictor(config)
    rows: list[list[object]] = []
    for path in paths or []:
        try:
            pages = load_pages(
                path,
                target_dpi=config.data.target_dpi,
                dpi_fallback=config.data.dpi_fallback,
                allow_upscale=config.data.allow_upscale,
            )
            for page in pages:
                report = build_report(page, config, time.perf_counter(), with_ocr, predictor)
                worst = report.defects[0] if report.defects else None
                rows.append(
                    [
                        report.image,
                        report.verdict,
                        report.quality_score,
                        f"{worst.label} {worst.score:.2f}" if worst else "—",
                        ", ".join(report.not_applicable) or "—",
                    ]
                )
        except Exception as error:  # noqa: BLE001 - битый файл не должен ронять батч
            rows.append([Path(path).name, "ошибка", 0.0, str(error)[:60], "—"])
    return rows


def build_demo(config: Config):
    """Собирает интерфейс. Вынесено отдельно, чтобы можно было монтировать в тестах."""
    gr = _import_gradio()

    with gr.Blocks(title="Качество сканов") as demo:
        gr.Markdown(
            "# Оценка качества сканов\n"
            "Метки приходят из трёх источников: **CV-метрики** (резкость, перекос, "
            "блик, тень, полосы, обрез, разрешение), **сеть** (контраст и шум — там, "
            "где CV-слой не применим) и **OCR** (`unreadable`). Источник каждой "
            "вероятности показан в таблице."
        )

        with gr.Tab("Один скан"):
            page_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="filepath", label="Скан (jpg/png/tiff)", height=420)
                    pdf = gr.File(label="…или PDF", file_types=[".pdf"], height=90)
                    ocr_flag = gr.Checkbox(
                        value=False, label="Считать unreadable (OCR, заметно медленнее)"
                    )
                    run_btn = gr.Button("Проверить", variant="primary")
                with gr.Column(scale=1):
                    verdict = gr.Textbox(label="Вердикт", interactive=False)
                    defects = gr.Label(label="Вероятности дефектов", num_top_classes=6)
                    label_pick = gr.Dropdown(
                        label="Показать на скане", choices=[], interactive=True
                    )
                    heat_view = gr.Image(label="Где дефект", height=420)
                    metrics = gr.Dataframe(
                        headers=["Метрика", "Значение"],
                        label="CV-метрики",
                        interactive=False,
                        wrap=True,
                    )
                    report_json = gr.JSON(label="QualityReport")

            def run_single(image_path, pdf_file, ocr_on):
                path = image_path or (pdf_file.name if pdf_file else None)
                line, scores, table, payload, gray, shown = _run(path, ocr_on, config)
                return (
                    line,
                    scores,
                    table,
                    payload,
                    gray,
                    gr.update(choices=shown, value=shown[0] if shown else None),
                )

            outputs = [verdict, defects, metrics, report_json, page_state, label_pick]
            run_btn.click(fn=run_single, inputs=[image, pdf, ocr_flag], outputs=outputs)
            image.change(fn=run_single, inputs=[image, pdf, ocr_flag], outputs=outputs)

            label_pick.change(
                fn=lambda gray, label: _heatmap(gray, label, config),
                inputs=[page_state, label_pick],
                outputs=heat_view,
            )

        with gr.Tab("Батч"):
            gr.Markdown(
                "Несколько файлов разом. Для целой папки удобнее CLI:\n"
                "`python -m src.cli --input <папка> --csv out.csv --workers 5`"
            )
            files = gr.File(label="Сканы или PDF", file_count="multiple")
            batch_ocr = gr.Checkbox(value=False, label="Считать unreadable (медленно)")
            batch_btn = gr.Button("Обработать", variant="primary")
            batch_table = gr.Dataframe(
                headers=["Файл", "Вердикт", "Качество", "Худшая метка", "Не измерено"],
                label="Результат по страницам",
                interactive=False,
                wrap=True,
            )
            batch_btn.click(
                fn=lambda items, ocr_on: batch_rows(
                    [item.name for item in items or []], ocr_on, config
                ),
                inputs=[files, batch_ocr],
                outputs=batch_table,
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio-приложение оценки качества сканов")
    parser.add_argument("--config", type=Path, default=None, help="путь к yaml-конфигу")
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="оверлей конфига под корпус, напр. configs/corpora/yenisei.yaml",
    )
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
    config = load_config(args.config, args.corpus)

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
