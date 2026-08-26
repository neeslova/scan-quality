"""Генерация синтетики: деградации поверх чистых реальных сканов.

Три вещи, ради которых этот модуль вообще существует отдельно от `degrade`:

1. **Отбор эталонов.** Портить можно только чистые страницы, и брать их можно
   только из документов ВНЕ val и test. Иначе тестовая страница вернётся в
   обучение своей деградированной копией — та же бумага, тот же прогон сканера.
   Это риск утечки №1, просто через синтетику (журнал, №29).
2. **Контроль частот.** Редкая метка, которой в выборке меньше 15%, не выучивается,
   и `pos_weight` её не вытягивает. Вероятность дефекта поднимается по ходу
   генерации для тех меток, что отстают от квоты.
3. **Наследование документа.** У сгенерированной страницы document тот же, что у
   эталона, — сплит синтетики обязан согласовываться со сплитом реальных.

Запуск:
    python -m src.data.generate --data data/raw/tobacco3482/data \\
        --prelabels data/labeled/prelabels_tobacco.jsonl \\
        --labels data/labeled/labels.jsonl --splits data/splits \\
        --out data/synthetic --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import Config, load_config
from src.data.degrade import DEGRADATIONS, LOCAL, apply
from src.data.split import read_labels
from src.io.loader import load_page
from src.labeling.prelabel import read_prelabels
from src.schema import PrelabelRecord, SyntheticRecord

logger = logging.getLogger(__name__)

# Маски храним уменьшенными: для проверки «попал ли патч в область дефекта»
# полное разрешение не нужно, а места они заняли бы больше самих страниц.
MASK_LONG_SIDE = 256
JPEG_QUALITY = 88


def held_out_documents(splits_dir: Path) -> set[str]:
    """Документы val и test. Эталоны из них брать нельзя ни при каких условиях."""
    held: set[str] = set()
    for name in ("val", "test"):
        path = splits_dir / f"{name}.json"
        if not path.is_file():
            raise SystemExit(f"Нет файла сплита: {path}. Сначала запустите src.data.split")
        with path.open("r", encoding="utf-8") as fh:
            held.update(json.load(fh)["documents"])
    return held


def select_references(
    prelabels: list[PrelabelRecord],
    labels_path: Optional[Path],
    held_out: set[str],
    config: Config,
) -> list[PrelabelRecord]:
    """Чистые страницы из документов вне val и test.

    Приоритет у тех, что человек своими руками отметил как чистые: это самое
    надёжное свидетельство отсутствия дефекта, какое у нас есть. Дальше добираем
    по CV-скорам — там, где ни один дефект не превысил порога.
    """
    limit = config.synth.reference.max_defect_score
    candidates = [r for r in prelabels if r.document not in held_out]

    human_clean: set[str] = set()
    if labels_path is not None and labels_path.is_file():
        for record in read_labels(labels_path):
            if record.document in held_out:
                continue
            manual = [record.labels.get(label) for label in config.manual_labels]
            if manual and not any(value for value in manual if value is not None):
                human_clean.add(record.image)

    def rank(record: PrelabelRecord) -> tuple[int, float]:
        # 0 — подтверждено человеком, 1 — только по CV.
        return (0 if record.image in human_clean else 1, record.suspicion)

    clean = [
        r
        for r in candidates
        if r.image in human_clean or all(score <= limit for score in r.scores.values())
    ]
    clean.sort(key=rank)

    chosen = clean[: config.synth.reference.count]
    logger.info(
        "эталонов: %d (подтверждено человеком %d) из %d кандидатов",
        len(chosen),
        sum(1 for r in chosen if r.image in human_clean),
        len(candidates),
    )
    return chosen


def pick_labels(
    rng: np.random.Generator,
    config: Config,
    counts: Counter,
    produced: int,
    total: int,
) -> list[str]:
    """Какие дефекты наложить на очередной эталон.

    Вероятность метки поднимается тем сильнее, чем больше она отстаёт от квоты:
    к концу прогона недобравшие метки начинают выпадать почти всегда, и доля
    выравнивается сама, без второго прохода.
    """
    quota = config.synth.min_label_share * total
    remaining = max(1, total - produced)
    picked: list[str] = []

    for label in DEGRADATIONS:
        deficit = max(0.0, quota - counts[label])
        probability = max(config.synth.base_probability, min(0.9, deficit / remaining))
        if rng.random() < probability:
            picked.append(label)

    if not picked:
        # Чистых страниц в синтетике не делаем: чистые у нас и так реальные.
        picked = [str(rng.choice(list(DEGRADATIONS)))]
    if len(picked) > config.synth.max_defects_per_image:
        picked = list(rng.choice(picked, size=config.synth.max_defects_per_image, replace=False))
    return picked


def downscale_mask(mask: np.ndarray, long_side: int = MASK_LONG_SIDE) -> np.ndarray:
    """Уменьшает маску с семантикой «есть ли дефект в этой области».

    Это max-pooling, а не усреднение, и разница принципиальна: полоса шириной
    в один пиксель при уменьшении в десять раз усредняется до значения 25 и при
    любом разумном пороге исчезает. Маски полос пропадали бы молча, а всплыло бы
    это только на обучении — как «streaks не локализуется».
    """
    scale = long_side / max(mask.shape)
    if scale >= 1.0:
        return mask
    kernel = max(1, int(round(1.0 / scale)))
    grown = cv2.dilate(mask, np.ones((kernel, kernel), np.uint8))
    return cv2.resize(grown, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def _write_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", downscale_mask(mask))[1].tofile(str(path))


def generate(
    root: Path,
    references: list[PrelabelRecord],
    out_dir: Path,
    config: Config,
    total: int,
    render: bool = True,
) -> list[SyntheticRecord]:
    """Планирует набор и, если `render`, сразу его рисует.

    Без рендера получается только рецепт: какие эталоны, какие дефекты, какой
    силы и с каким зерном. Это мгновенно и весит мегабайты вместо гигабайт —
    страницы восстанавливаются на месте функцией `reproduce`.
    """
    if not references:
        raise SystemExit("Не нашлось ни одного чистого эталона вне val и test")

    planner = np.random.default_rng(config.synth.seed)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    records: list[SyntheticRecord] = []
    started = time.perf_counter()

    for index in range(total):
        # Отдельное зерно на страницу: с ним запись манифеста становится полным
        # рецептом, и страницу можно восстановить, не таская её через Drive.
        seed = int(config.synth.seed) * 1_000_003 + index
        rng = np.random.default_rng(seed)
        reference = references[index % len(references)]

        # План (какие дефекты и насколько сильные) тянем из общего генератора:
        # он видит счётчики и выравнивает частоты по всему прогону.
        labels = pick_labels(planner, config, counts, index, total)
        severities = {
            label: round(
                float(planner.uniform(config.synth.severity.min, config.synth.severity.max)), 3
            )
            for label in labels
        }
        name = f"{index:06d}_{Path(reference.image).stem}"
        image_rel = f"images/{name}.jpg"
        mask_paths = {label: f"masks/{name}_{label}.png" for label in labels if label in LOCAL}
        width = height = 0

        if render:
            try:
                page = load_page(
                    root / reference.image,
                    target_dpi=config.data.target_dpi,
                    dpi_fallback=config.data.dpi_fallback,
                    allow_upscale=config.data.allow_upscale,
                )
            except Exception as exc:  # noqa: BLE001 — битый эталон не должен ронять прогон
                logger.warning("%s пропущен: %s", reference.image, exc)
                continue

            image, masks = apply(page.gray, labels, severities, config, rng)
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tofile(
                str(out_dir / image_rel)
            )
            for label, relative in list(mask_paths.items()):
                mask = masks.get(label)
                if mask is None:
                    del mask_paths[label]
                    continue
                _write_mask(mask, out_dir / relative)
            width, height = int(image.shape[1]), int(image.shape[0])

        records.append(
            SyntheticRecord(
                image=image_rel,
                reference=reference.image,
                document=reference.document,
                corpus=reference.corpus,
                labels={label: label in labels for label in DEGRADATIONS},
                severities=severities,
                seed=seed,
                masks=mask_paths,
                width=width,
                height=height,
            )
        )
        counts.update(labels)

        if (index + 1) % 50 == 0:
            rate = (index + 1) / (time.perf_counter() - started)
            left = (total - index - 1) / rate if rate else 0
            print(
                f"  {index + 1}/{total}  {rate:.1f} стр/с  осталось ~{left/60:.0f} мин",
                file=sys.stderr,
                flush=True,
            )

    return records


def read_manifest(path: Path) -> list[SyntheticRecord]:
    # utf-8-sig, а не utf-8: инструменты Windows дописывают BOM в начало файла,
    # и парсер JSON падает на первой же строке с невнятным «expected value».
    return [
        SyntheticRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def reproduce(
    root: Path,
    records: list[SyntheticRecord],
    out_dir: Path,
    config: Config,
) -> int:
    """Восстанавливает страницы по манифесту — побитово, из эталонов и зёрен.

    Ради этого в записи и хранится `seed`. Синтетика весит гигабайты, эталоны —
    сотни мегабайт: в Colab едет рецепт, а страницы собираются на месте за минуты.
    """
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    restored = 0

    for index, record in enumerate(records, 1):
        try:
            page = load_page(
                root / record.reference,
                target_dpi=config.data.target_dpi,
                dpi_fallback=config.data.dpi_fallback,
                allow_upscale=config.data.allow_upscale,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s пропущен: %s", record.reference, exc)
            continue

        rng = np.random.default_rng(record.seed)
        labels = record.positive
        image, masks = apply(page.gray, labels, record.severities, config, rng)

        target = out_dir / record.image
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tofile(str(target))
        for label, relative in record.masks.items():
            mask = masks.get(label)
            if mask is not None:
                _write_mask(mask, out_dir / relative)
        restored += 1

        if index % 100 == 0:
            print(f"  восстановлено {index}/{len(records)}", file=sys.stderr, flush=True)

    return restored


def write_manifest(records: list[SyntheticRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")


def contact_sheets(
    records: list[SyntheticRecord], out_dir: Path, reports_dir: Path, per_label: int = 10
) -> list[Path]:
    """Сетка примеров по каждому дефекту — глазами проверить, что блик похож на блик."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tile = 420
    columns = 5
    gap = 6

    for label in DEGRADATIONS:
        picked = [r for r in records if r.labels.get(label)][:per_label]
        if not picked:
            continue
        tiles = []
        for record in picked:
            data = np.fromfile(str(out_dir / record.image), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            scale = tile / max(image.shape)
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            canvas = np.full((tile, tile), 235, dtype=np.uint8)
            canvas[: small.shape[0], : small.shape[1]] = small
            tiles.append(canvas)

        if not tiles:
            continue

        # Раскладываем сеткой, а не в одну строку: десять плиток подряд читаются
        # только как узкая лента, и разглядеть дефект в них невозможно.
        rows = []
        for start in range(0, len(tiles), columns):
            row = tiles[start : start + columns]
            while len(row) < columns:
                row.append(np.full((tile, tile), 255, dtype=np.uint8))
            rows.append(np.concatenate([_pad(t, gap) for t in row], axis=1))
        sheet = np.concatenate(rows, axis=0)

        path = reports_dir / f"synth_{label}.png"
        cv2.imencode(".png", sheet)[1].tofile(str(path))
        written.append(path)
    return written


def _pad(tile: np.ndarray, gap: int) -> np.ndarray:
    return cv2.copyMakeBorder(tile, gap, gap, gap, gap, cv2.BORDER_CONSTANT, value=255)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетики поверх чистых сканов")
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--prelabels", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="labels.jsonl для эталонов")
    parser.add_argument("--splits", type=Path, required=True, help="папка с train/val/test.json")
    parser.add_argument("--out", type=Path, required=True, help="куда писать синтетику")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--limit", type=int, default=None, help="сколько сгенерировать")
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="восстановить страницы по готовому манифесту вместо новой генерации",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="только рецепт: манифест без картинок (мегабайты вместо гигабайт)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config, args.corpus)

    if args.from_manifest is not None:
        records = read_manifest(args.from_manifest)
        restored = reproduce(args.data, records, args.out, config)
        sheets = contact_sheets(records, args.out, args.reports)
        print(f"\nвосстановлено {restored} из {len(records)} страниц по манифесту")
        print(f"контрольные сетки: {len(sheets)} шт. в {args.reports}")
        return

    held_out = held_out_documents(args.splits)
    prelabels = read_prelabels(args.prelabels)
    references = select_references(prelabels, args.labels, held_out, config)

    total = args.limit if args.limit is not None else config.synth.target_total
    render = not args.no_render
    records = generate(args.data, references, args.out, config, total, render=render)
    write_manifest(records, args.out / "manifest.jsonl")
    sheets = contact_sheets(records, args.out, args.reports) if render else []

    # Проверка, ради которой всё и затевалось: ни один документ из val/test
    # не должен просочиться в синтетику.
    leaked = {r.document for r in records} & held_out
    if leaked:
        raise SystemExit(f"Утечка: документы val/test попали в синтетику: {sorted(leaked)[:5]}")

    print(f"\nсгенерировано: {len(records)} страниц из {len(references)} эталонов")
    print(f"утечки val/test: нет ({len(held_out)} документов исключено)")
    print(f"\n{'метка':16s}{'страниц':>10s}{'доля':>9s}{'статус':>12s}")
    for label in DEGRADATIONS:
        count = sum(1 for r in records if r.labels.get(label))
        share = count / len(records) if records else 0.0
        status = "ок" if share >= config.synth.min_label_share else "ниже квоты"
        print(f"{label:16s}{count:10d}{share:9.1%}{status:>12s}")
    print(f"\nманифест: {args.out / 'manifest.jsonl'}")
    if render:
        print(f"контрольные сетки: {len(sheets)} шт. в {args.reports}")
    else:
        print("картинки не рисовались — восстановить: --from-manifest")


if __name__ == "__main__":
    main()
