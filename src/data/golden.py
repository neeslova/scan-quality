"""Эталонный набор good/bad из разложенных по папкам сканов.

Датасет `data/Data iz tg` разложен руками: подпапка `Good` и подпапка `bad`.
Это уже готовый ground truth, и ручная разметка для бинарного среза не нужна —
нужно только перевести раскладку в тот же построчный формат, которым живёт
остальной проект.

Две вещи, из-за которых это не однострочник.

**PDF разворачивается постранично.** Метка стоит на файле, а страницы внутри
одного документа могут отличаться: в сканах из мессенджера первая страница
часто ровная, а последняя снята под углом. Поэтому каждая страница получает
свою запись и метку файла, а `document` у них общий — иначе страницы одного
документа разъедутся по разным половинам сплита и оценка будет завышена.
Число страниц берём из заголовка PDF, не рендеря их: рендер всех страниц ради
подсчёта стоил бы минуты на ровном месте.

**Регистр имени папки не значим.** Разложено как `Good`/`bad`, и полагаться
на то, что так будет всегда, не стоит.

Запуск:
    python -m src.data.golden --data "data/Data iz tg" --corpus tg \
        --out data/labeled/golden_tg.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from src.config import load_config
from src.data.split import document_id
from src.io.loader import IMAGE_SUFFIXES, PDF_SUFFIXES
from src.schema import GoldenRecord

logger = logging.getLogger(__name__)

# Имя папки (в нижнем регистре) -> метка. Всё остальное игнорируется.
FOLDER_LABELS = {"good": "good", "bad": "bad"}

SUFFIXES = {*IMAGE_SUFFIXES, *PDF_SUFFIXES}


def _page_count(path: Path) -> int:
    """Сколько страниц в файле. Для картинки — одна, для PDF — по заголовку."""
    if path.suffix.lower() not in PDF_SUFFIXES:
        return 1
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover — зависит от окружения
        raise SystemExit("Для чтения PDF нужен PyMuPDF: pip install pymupdf") from exc

    try:
        with fitz.open(str(path)) as doc:
            return int(doc.page_count)
    except Exception as error:  # noqa: BLE001 — битый файл не повод бросать папку
        logger.warning("%s: не удалось прочитать (%s), пропущен", path.name, error)
        return 0


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """sha256 файла. Читаем кусками: в корпусе есть сканы по десятку мегабайт."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(
    root: Path,
    corpus: str,
    strategy: str = "stem",
    annotator: str = "",
) -> list[GoldenRecord]:
    """Обходит папки классов и разворачивает каждый файл в записи по страницам.

    Дубликаты ищутся по содержимому, а не по имени. В корпусе есть два разных
    скана с именем `scale_1200.png` — один в good, другой в bad, и сравнение по
    имени объявило бы их конфликтом разметки, выбросив вполне годную страницу.
    Совпадение же хешей означает буквально один и тот же файл: в двух классах
    сразу это противоречие, и такая страница не может быть эталоном.
    """
    records: list[GoldenRecord] = []
    seen: dict[str, tuple[str, str]] = {}  # sha256 -> (метка, путь)

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        label = FOLDER_LABELS.get(folder.name.lower())
        if label is None:
            logger.info("Папка %s не является классом — пропущена", folder.name)
            continue

        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            if path.suffix.lower() not in SUFFIXES:
                logger.debug("%s: формат не поддерживается, пропущен", path.name)
                continue

            digest = file_sha256(path)
            relative = path.relative_to(root).as_posix()

            if digest in seen:
                previous_label, previous_path = seen[digest]
                if previous_label != label:
                    logger.warning(
                        "%s и %s — один файл в разных классах, обе копии исключены",
                        previous_path,
                        relative,
                    )
                    records = [r for r in records if r.sha256 != digest]
                    seen[digest] = (label, previous_path)
                else:
                    logger.info("%s повторяет %s, пропущен", relative, previous_path)
                continue
            seen[digest] = (label, relative)

            pages = _page_count(path)
            for index in range(pages):
                records.append(
                    GoldenRecord(
                        image=relative,
                        page=index,
                        document=document_id(path, strategy),
                        corpus=corpus,
                        label=label,
                        source="folder",
                        annotator=annotator,
                        sha256=digest,
                    )
                )

    return records


def write_golden(records: list[GoldenRecord], path: Path) -> None:
    """Пишет набор построчно. Файл перезаписывается: он выводится из папок целиком."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")


def read_golden(path: Path) -> list[GoldenRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [GoldenRecord.model_validate_json(line) for line in fh if line.strip()]


def summarize(records: list[GoldenRecord]) -> str:
    good = sum(1 for r in records if r.label == "good")
    bad = len(records) - good
    documents = len({r.document for r in records})
    files = len({r.image for r in records})
    return (
        f"страниц: {len(records)} (good {good}, bad {bad}); "
        f"файлов: {files}; документов: {documents}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="корень с папками good/bad")
    parser.add_argument("--corpus", required=True, help="имя корпуса в записях")
    parser.add_argument("--out", type=Path, required=True, help="куда писать jsonl")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--document-id",
        default="stem",
        choices=["stem", "parent", "bates7"],
        help="как группировать страницы в документы для сплита",
    )
    parser.add_argument("--annotator", default="", help="кто разложил папки")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_config(args.config)  # валидируем конфиг: падать лучше сразу

    if not args.data.is_dir():
        raise SystemExit(f"Не папка: {args.data}")

    records = collect(args.data, args.corpus, args.document_id, args.annotator)
    if not records:
        raise SystemExit(f"В {args.data} не найдено файлов в папках good/bad")

    write_golden(records, args.out)
    logger.info("%s -> %s", summarize(records), args.out)


if __name__ == "__main__":
    main()
