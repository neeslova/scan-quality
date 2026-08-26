"""Корреляция CV-метрик с фактической деградацией OCR.

Это пункт 3 раздела «Метрики для защиты»: доказать, что «плохой скан» по мнению
системы действительно плохо распознаётся. Без такой связи вся оценка качества
остаётся вкусовщиной — она меряла бы не пригодность документа, а похожесть на
представление разметчика о плохом скане.

Считается ранговая корреляция Спирмена, а не Пирсона: связь между скором дефекта
и уверенностью OCR монотонная, но не линейная — у сильного размытия уверенность
упирается в пол и дальше падать некуда.

Данные берутся из готовых файлов, ничего не пересчитывается:
  prelabels.jsonl — скоры CV по каждой метке и сводный quality_score;
  labels.jsonl    — результаты OCR в поле derived (после src.ocr.annotate).

Запуск:
    python -m src.ocr.correlate --labels data/labeled/labels.jsonl \\
        --prelabels data/labeled/prelabels_tobacco.jsonl --out reports
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.config import Config, load_config
from src.data.split import read_labels
from src.labeling.prelabel import read_prelabels

logger = logging.getLogger(__name__)

# Палитра проверена scripts/validate_palette.js: расхождение по CVD 21.6, обычное 32.3.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
EXPECTED = "#2a78d6"  # связь в ожидаемую сторону
UNEXPECTED = "#e34948"  # связь против ожидания

# Какой сигнал OCR считаем «фактической ошибкой распознавания».
OCR_SIGNALS = {
    "ocr_confidence": "уверенность OCR",
    "ocr_nonword": "доля неправдоподобных слов",
}


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ранговая корреляция и p-value. Возвращает (nan, nan) на вырожденных данных."""
    from scipy import stats

    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return (float("nan"), float("nan"))
    result = stats.spearmanr(x, y)
    return (float(result.statistic), float(result.pvalue))


# Префикс колонок с ручными метками: их считаем отдельно от скоров CV.
HUMAN = "рука:"


def collect(labels_path: Path, prelabels_path: Path, config: Config) -> dict[str, np.ndarray]:
    """Сводит скоры CV, ручные метки и результаты OCR по общим страницам.

    Ручные метки нужны как контрольная группа. Если скор CV не связан с
    деградацией OCR, а ручная метка того же дефекта связана — сломана метрика.
    Если и ручная не связана — на этом корпусе дефект просто не мешает движку,
    и претензия не к нам.
    """
    prelabels = {r.image: r for r in read_prelabels(prelabels_path)}
    records = [r for r in read_labels(labels_path) if r.image in prelabels and r.derived]
    if not records:
        raise SystemExit("Нет страниц, где есть и предразметка CV, и результаты OCR")

    columns: dict[str, list[float]] = {name: [] for name in OCR_SIGNALS}
    for label in config.cv.scores:
        columns[label] = []
    for label in config.manual_labels:
        columns[HUMAN + label] = []
    columns["quality_score"] = []

    for record in records:
        pre = prelabels[record.image]
        for name in OCR_SIGNALS:
            columns[name].append(record.derived.get(name, float("nan")))
        for label in config.cv.scores:
            columns[label].append(pre.scores.get(label, float("nan")))
        for label in config.manual_labels:
            value = record.labels.get(label)
            columns[HUMAN + label].append(float(value) if value is not None else float("nan"))
        columns["quality_score"].append(pre.quality_score)

    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def metric_names(config: Config) -> list[str]:
    return [
        *config.cv.scores,
        "quality_score",
        *(HUMAN + label for label in config.manual_labels),
    ]


def correlations(data: dict[str, np.ndarray], config: Config, signal: str) -> list[tuple]:
    """(имя, rho, p, n) по каждой метрике против выбранного сигнала OCR."""
    target = data[signal]
    rows = []
    for name in metric_names(config):
        values = data[name]
        mask = np.isfinite(values) & np.isfinite(target)
        rho, pvalue = spearman(values[mask], target[mask])
        rows.append((name, rho, pvalue, int(mask.sum())))
    # Сильнейшая связь сверху, независимо от знака.
    rows.sort(key=lambda row: abs(row[1]) if np.isfinite(row[1]) else -1, reverse=True)
    return rows


def _expected_sign(name: str, signal: str) -> int:
    """Куда метрика ДОЛЖНА двигать сигнал OCR, если система меряет то, что заявлено.

    Скор дефекта (и ручная метка дефекта) растёт — уверенность распознавания
    падает. У `quality_score` знак обратный: он тем выше, чем скан лучше.
    """
    if signal == "ocr_confidence":
        return +1 if name == "quality_score" else -1
    return -1 if name == "quality_score" else +1


def plot(rows: list[tuple], signal: str, path: Path, n_pages: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = [row for row in rows if np.isfinite(row[1])]
    if not finite:
        logger.warning("Нечего рисовать: все корреляции вырождены")
        return

    names = [row[0] for row in finite][::-1]
    values = [row[1] for row in finite][::-1]
    colors = [
        EXPECTED if np.sign(rho) == _expected_sign(name, signal) else UNEXPECTED
        for name, rho in zip(names, values)
    ]

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(names) + 1.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(names, values, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1.2, zorder=4)

    for index, value in enumerate(values):
        offset = 0.02 if value >= 0 else -0.02
        ax.text(
            value + offset,
            index,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color=INK,
        )

    ax.set_xlim(-1.05, 1.05)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("ранговая корреляция Спирмена", color=MUTED, fontsize=9)
    ax.set_title(
        f"Связь скоров CV с сигналом «{OCR_SIGNALS[signal]}»\n"
        f"{n_pages} размеченных страниц · синим — связь в ожидаемую сторону",
        color=INK,
        fontsize=11,
        loc="left",
        pad=14,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    logger.info("График: %s", path)


def write_report(
    data: dict[str, np.ndarray], config: Config, out_dir: Path, charts: list[Path]
) -> Path:
    n_pages = len(data["quality_score"])
    lines = [
        "# Корреляция CV-метрик с деградацией OCR",
        "",
        f"Страниц в анализе: **{n_pages}**. Корреляция ранговая (Спирмен): связь",
        "монотонная, но не линейная — у сильного размытия уверенность OCR упирается",
        "в пол и дальше падать некуда.",
        "",
        "Знак читается так: скор дефекта растёт — уверенность распознавания должна",
        "падать, поэтому **ожидаемая связь отрицательная**. У `quality_score`",
        "наоборот: он тем выше, чем скан лучше.",
        "",
    ]

    for signal, title in OCR_SIGNALS.items():
        rows = correlations(data, config, signal)
        lines += [
            f"## Против сигнала «{title}»",
            "",
            "| Метрика | ρ | p | n | направление |",
            "|---|---:|---:|---:|---|",
        ]
        for name, rho, pvalue, count in rows:
            if not np.isfinite(rho):
                lines.append(f"| `{name}` | — | — | {count} | не измерено |")
                continue
            agrees = np.sign(rho) == _expected_sign(name, signal)
            mark = "ожидаемое" if agrees else "**против ожидания**"
            stars = "" if pvalue >= 0.05 else (" ***" if pvalue < 0.001 else " *")
            lines.append(f"| `{name}` | {rho:+.3f}{stars} | {pvalue:.2g} | {count} | {mark} |")
        lines.append("")

    lines += ["`*` p < 0.05, `***` p < 0.001.", ""]
    for chart in charts:
        lines += [f"![{chart.stem}]({chart.name})", ""]

    path = out_dir / "ocr_correlation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Корреляция CV-метрик с деградацией OCR")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)
    data = collect(args.labels, args.prelabels, config)
    n_pages = len(data["quality_score"])

    charts: list[Path] = []
    for signal in OCR_SIGNALS:
        chart = args.out / f"ocr_correlation_{signal}.png"
        plot(correlations(data, config, signal), signal, chart, n_pages)
        if chart.is_file():
            charts.append(chart)

    report = write_report(data, config, args.out, charts)
    print(f"\nстраниц: {n_pages}")
    for signal, title in OCR_SIGNALS.items():
        print(f"\n{title}:")
        for name, rho, pvalue, count in correlations(data, config, signal)[:10]:
            if not np.isfinite(rho):
                continue
            agrees = "" if np.sign(rho) == _expected_sign(name, signal) else "  <- против ожидания"
            print(f"  {name:22s} rho {rho:+.3f}  p {pvalue:.2g}  n {count}{agrees}")
    print(f"\nотчёт: {report}")


if __name__ == "__main__":
    main()
