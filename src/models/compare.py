"""Сравнение трёх систем на одной выборке: CV-метрики, сеть, гибрид.

Это таблица для записки (С8) и ответ на вопрос «что дал каждый компонент».
Все три считаются за ОДИН проход по страницам: гибрид не пересчитывает ничего
заново, он лишь выбирает по каждой метке источник из уже посчитанного. Проход
один и потому, что отложенный тест открывается ровно один раз.

**Основная величина — AP, а не F1.** AP не зависит от порога и потому сравним
между системами; F1 у CV-слоя и сети пришлось бы считать при пороге 0.5, который
для них никто не калибровал, и такое сравнение говорило бы о выборе порога, а не
о качестве. Калиброванные пороги есть только у гибрида — его P/R/F1 приводятся
отдельно, ниже таблицы AP.

Метка учитывается только там, где источник её ВЫДАЛ. На битональном скане
CV-слой не измеряет контраст и шум вовсе (решение №21), и подставлять туда ноль
значило бы засчитать «не мерил» за «дефекта нет».
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, load_config
from src.models.evaluate import LabelMetrics, per_label

logger = logging.getLogger(__name__)

SYSTEMS = ("cv", "cnn", "hybrid")


@dataclass
class PageRow:
    """Одна страница: что сказала каждая система и что стоит в разметке."""

    image: str
    truth: dict[str, bool] = field(default_factory=dict)
    cv: dict[str, float] = field(default_factory=dict)
    cnn: dict[str, float] = field(default_factory=dict)
    # Приведённые якорями скоры: по ним считаются вердикт и P/R/F1.
    hybrid: dict[str, float] = field(default_factory=dict)
    # Те же метки до приведения. AP считается по НИМ (решение №43): приведение
    # монотонно, но зажимает края в 0 и 1, а к таким связкам AP чувствителен.
    hybrid_raw: dict[str, float] = field(default_factory=dict)
    verdict: str = "good"

    def defect_count(self, exclude: tuple[str, ...] = ()) -> int:
        return sum(1 for label, on in self.truth.items() if on and label not in exclude)


def collect(samples, config: Config, with_ocr: bool = False) -> list[PageRow]:
    """Один проход по страницам -> оценки всех трёх систем."""
    from src.io.loader import load_page
    from src.metrics.baseline import analyze_page
    from src.models.infer import shared_predictor
    from src.pipeline import build_report

    predictor = shared_predictor(config)
    if predictor is None:
        logger.warning("Модели нет: колонки сети и гибрида будут неполными")

    rows: list[PageRow] = []
    for index, sample in enumerate(samples, 1):
        page = load_page(
            sample.path,
            target_dpi=config.data.target_dpi,
            dpi_fallback=config.data.dpi_fallback,
            allow_upscale=config.data.allow_upscale,
        )
        _, cv_scores, _ = analyze_page(page, config)
        cnn_scores = predictor.scores(page.gray) if predictor is not None else {}
        report = build_report(page, config, time.perf_counter(), with_ocr, predictor)

        rows.append(
            PageRow(
                image=sample.path.name,
                truth={label: bool(sample.labels.get(label)) for label in config.labels},
                cv=dict(cv_scores),
                cnn=dict(cnn_scores),
                hybrid=report.scores(),
                hybrid_raw={
                    d.label: (d.raw if d.raw is not None else d.score) for d in report.defects
                },
                verdict=report.verdict,
            )
        )
        if index % 25 == 0:
            logger.info("%d/%d", index, len(samples))
    return rows


def metrics_for(
    rows: list[PageRow], system: str, label: str, config: Config
) -> Optional[LabelMetrics]:
    """Метрики одной метки у одной системы или None, если она её не выдавала."""
    pairs = [
        (1.0 if row.truth.get(label) else 0.0, getattr(row, system)[label])
        for row in rows
        if label in getattr(row, system)
    ]
    if not pairs or not any(truth for truth, _ in pairs):
        return None

    y_true = np.array([[truth] for truth, _ in pairs], dtype=float)
    y_score = np.array([[score] for _, score in pairs], dtype=float)
    return per_label(y_true, y_score, [label], config.verdict.tau_low)[0]


AP_COLUMNS = {"cv": "cv", "cnn": "cnn", "hybrid": "hybrid_raw"}


def format_ap_table(rows: list[PageRow], config: Config) -> str:
    """Таблица AP по трём системам. Прочерк — источник метку не выдавал.

    Два макро-средних, и оба нужны. Считать одно по тем меткам, что система
    выдала, — ловушка: CV-слой отказывается ровно от двух самых трудных меток,
    и среднее по остальным семи польстило бы ему просто за отказ отвечать.
    Поэтому отдельно приводится среднее по общим меткам (строго сопоставимое)
    и среднее по всем девяти, где невыданная метка засчитывается на уровне
    случайного угадывания — то есть ровно тем, чего стоит отсутствие ответа.
    """
    lines = [
        f"{'метка':16s}{'источник':>9s}{'CV':>9s}{'сеть':>9s}{'гибрид':>9s}{'n+':>6s}",
        "-" * 58,
    ]
    scored: dict[str, dict[str, float]] = {name: {} for name in SYSTEMS}
    chance: dict[str, float] = {}
    graded = [label for label in config.labels if label not in config.ocr_derived]

    for label in config.labels:
        cells = []
        support = sum(1 for row in rows if row.truth.get(label))
        chance[label] = support / len(rows) if rows else 0.0
        for system in SYSTEMS:
            item = metrics_for(rows, AP_COLUMNS[system], label, config)
            if item is None or np.isnan(item.average_precision):
                cells.append("   —")
                continue
            cells.append(f"{item.average_precision:.3f}")
            scored[system][label] = item.average_precision

        star = "*" if label in config.ocr_derived else " "
        lines.append(
            f"{label + star:16s}{config.sources.of(label):>9s}"
            f"{cells[0]:>9s}{cells[1]:>9s}{cells[2]:>9s}{support:6d}"
        )

    common = [label for label in graded if all(label in scored[name] for name in SYSTEMS)]
    lines.append("-" * 58)
    lines.append(
        f"{'macro, общие':16s}{f'({len(common)})':>9s}"
        + "".join(
            f"{np.mean([scored[name][label] for label in common]):9.3f}" if common else "        —"
            for name in SYSTEMS
        )
    )
    lines.append(
        f"{'macro, все':16s}{f'({len(graded)})':>9s}"
        + "".join(
            f"{np.mean([scored[name].get(label, chance[label]) for label in graded]):9.3f}"
            for name in SYSTEMS
        )
    )
    lines.append("")
    lines.append("«общие» — метки, которые выдают все три системы: сравнение строго честное.")
    lines.append("«все» — девять меток; невыданная засчитана на уровне случайного угадывания.")
    if any(label in config.ocr_derived for label in config.labels):
        lines.append("* в макро не входит: метка выводится из OCR по построению (решение №7)")
    return "\n".join(lines)


def format_hybrid_quality(rows: list[PageRow], config: Config) -> str:
    """P/R/F1 гибрида при калиброванных порогах — только у него они есть."""
    lines = [
        f"{'метка':16s}{'P':>8s}{'R':>8s}{'F1':>8s}{'n+':>6s}",
        "-" * 46,
    ]
    scores = []
    for label in config.labels:
        item = metrics_for(rows, "hybrid", label, config)
        if item is None:
            continue
        lines.append(
            f"{label:16s}{item.precision:8.3f}{item.recall:8.3f}{item.f1:8.3f}{item.support:6d}"
        )
        if label not in config.ocr_derived:
            scores.append((item.precision, item.recall, item.f1))
    if scores:
        arr = np.array(scores)
        lines.append("-" * 46)
        lines.append(
            f"{'макро':16s}{arr[:, 0].mean():8.3f}{arr[:, 1].mean():8.3f}{arr[:, 2].mean():8.3f}"
        )
    return "\n".join(lines)


def format_verdicts(rows: list[PageRow], config: Config) -> str:
    """Вердикт против числа дефектов в разметке."""
    exclude = tuple(config.ocr_derived)
    lines = [f"{'дефектов':12s}{'n':>5s}{'good':>8s}{'acceptable':>12s}{'bad':>7s}", "-" * 44]
    for bucket, name in ((0, "0 (чистые)"), (1, "1"), (2, "2"), (3, "3+")):
        # Последняя корзина собирает хвост: три дефекта и больше.
        group = [
            row
            for row in rows
            if (row.defect_count(exclude) >= bucket)
            if bucket == 3 or row.defect_count(exclude) == bucket
        ]
        counts = dict.fromkeys(("good", "acceptable", "bad"), 0)
        for row in group:
            counts[row.verdict] += 1
        lines.append(
            f"{name:12s}{len(group):5d}{counts['good']:8d}"
            f"{counts['acceptable']:12d}{counts['bad']:7d}"
        )
    return "\n".join(lines)


def worst_errors(rows: list[PageRow], config: Config, count: int = 20) -> list[tuple[PageRow, str]]:
    """Худшие ошибки обоих видов: половина пропусков, половина ложных тревог.

    Разделение обязательно. Пропуск дороже ложной тревоги (раздел 4), и по одной
    общей шкале стоимости полный пропуск всегда обгоняет любую ложную тревогу —
    список из двадцати оказывался бы сплошь пропусками, а второй тип ошибок
    в разбор не попадал бы вовсе. Порядок внутри каждой половины — по величине
    промаха; сами половины идут пропусками вперёд, потому что они дороже.
    """
    misses: list[tuple[float, PageRow, str]] = []
    alarms: list[tuple[float, PageRow, str]] = []

    for row in rows:
        for kind, bucket, threshold in (
            ("пропуск", misses, config.verdict.tau_low),
            ("ложная тревога", alarms, config.verdict.tau_high),
        ):
            worst_label, worst_gap = "", 0.0
            for label, score in row.hybrid.items():
                if label in config.ocr_derived:
                    continue
                truth = bool(row.truth.get(label))
                if (kind == "пропуск") != truth:
                    continue
                gap = (threshold - score) if truth else (score - threshold)
                if gap > worst_gap:
                    worst_label, worst_gap = label, gap
            if worst_label:
                bucket.append(
                    (
                        worst_gap,
                        row,
                        f"{worst_label} ({kind}, скор {row.hybrid[worst_label]:.2f})",
                    )
                )

    half = max(1, count // 2)
    picked = []
    for bucket in (misses, alarms):
        bucket.sort(key=lambda item: item[0], reverse=True)
        picked.extend((row, why) for _, row, why in bucket[:half])
    return picked


def _with_caption(tile: np.ndarray, text: str) -> np.ndarray:
    """Подпись под плиткой. Через PIL, а не cv2.putText: тот не умеет кириллицу
    и печатает вместо неё вопросительные знаки."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:  # pragma: no cover — зависит от системы
        font = ImageFont.load_default()

    height = tile.shape[0]
    draw.rectangle([0, height - 24, tile.shape[1], height], fill=255)
    draw.text((5, height - 22), text, fill=0, font=font)
    return np.asarray(image)


def error_sheet(
    errors: list[tuple[PageRow, str]],
    root: Path,
    out_path: Path,
    tile: int = 460,
    columns: int = 5,
) -> Optional[Path]:
    """Контактный лист худших ошибок: страница и подпись, чем именно промахнулись.

    Список имён файлов ошибку не показывает — по нему нельзя понять, права ли
    разметка и что сеть могла увидеть. Глазами это видно за секунду, поэтому
    в записку идут картинки, а не только таблица.
    """
    import cv2

    if not errors:
        return None

    def caption_of(text: str) -> str:
        label = text.split(" (")[0]
        return f"{label} — {'пропуск' if 'пропуск' in text else 'ложная тревога'}"

    tiles = []
    for row, why in errors:
        matches = list(root.rglob(row.image))
        if not matches:
            logger.warning("Страница %s не найдена под %s", row.image, root)
            continue
        data = np.fromfile(str(matches[0]), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        scale = (tile - 26) / max(image.shape)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.full((tile, tile), 245, dtype=np.uint8)
        canvas[: small.shape[0], : small.shape[1]] = small
        # Подпись снизу: без неё лист — просто двадцать похожих серых страниц.
        tiles.append(_with_caption(canvas, caption_of(why)))

    if not tiles:
        return None

    grid = []
    for start in range(0, len(tiles), columns):
        line = tiles[start : start + columns]
        while len(line) < columns:
            line.append(np.full((tile, tile), 255, dtype=np.uint8))
        grid.append(np.hstack(line))

    sheet = np.vstack(grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", sheet)[1].tofile(str(out_path))
    return out_path


def main() -> None:
    from src.data.dataset import collect_real, load_split

    parser = argparse.ArgumentParser(description="CV против сети против гибрида")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--part", default="val", choices=("train", "val", "test"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--with-ocr", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить отчёт")
    parser.add_argument(
        "--sheet", type=Path, default=None, help="куда положить контактный лист худших ошибок"
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="сохранить сырые оценки по страницам: разбор потом не потребует "
        "повторного прогона, а тест открывается один раз",
    )
    parser.add_argument(
        "--yes-open-the-test",
        action="store_true",
        help="подтверждение для --part test: он открывается ОДИН раз, в С8",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.part == "test" and not args.yes_open_the_test:
        raise SystemExit(
            "Отложенный тест открывается один раз и только в С8. "
            "Если это он — добавьте --yes-open-the-test."
        )

    config = load_config(args.config, args.corpus)
    _, images = load_split(args.splits, args.part)
    samples = collect_real(args.labels, args.data, images)
    rows = collect(samples, config, args.with_ocr)

    if args.dump is not None:
        import json
        from dataclasses import asdict

        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(
            json.dumps([asdict(row) for row in rows], ensure_ascii=False), encoding="utf-8"
        )
        logger.info("сырые оценки сохранены в %s", args.dump)

    blocks = [
        f"# Сравнение систем: {args.part}, {len(rows)} страниц",
        "",
        "## AP по меткам (порог не участвует)",
        "```",
        format_ap_table(rows, config),
        "```",
        "",
        f"## Гибрид при калиброванных порогах {config.verdict.tau_low}/{config.verdict.tau_high}",
        "```",
        format_hybrid_quality(rows, config),
        "```",
        "",
        "## Вердикт против разметки",
        "```",
        format_verdicts(rows, config),
        "```",
        "",
        "## Двадцать худших ошибок",
        "",
        "Пропуск дефекта взвешен вдвое дороже ложной тревоги (раздел 4).",
        "",
    ]
    errors = worst_errors(rows, config)
    for row, why in errors:
        blocks.append(f"- `{row.image}` — {why}, вердикт {row.verdict}")

    if args.sheet is not None:
        written = error_sheet(errors, args.data, args.sheet)
        if written is not None:
            blocks.append("")
            blocks.append(f"Контактный лист: `{written}`")

    text = "\n".join(blocks)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nсохранено в {args.out}")


if __name__ == "__main__":
    main()
