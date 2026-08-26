"""Gradio-разметчик: страница -> галочки по дефектам -> строка в labels.jsonl.

Устройство рассчитано на четыре часа монотонной работы, поэтому:

- чекбоксы предзаполнены черновыми метками CV — снять галочку быстрее, чем поставить;
- горячие клавиши на цифрах, стрелки для перехода, пробел — «страница чистая»;
- запись на диск после КАЖДОЙ страницы, JSONL и только дозапись: обрыв стоит
  одной страницы, а не всей сессии;
- повторная разметка страницы дописывает новую строку, а не переписывает старую —
  историю правок видно, при чтении берётся последняя запись.

Запуск:
    python -m src.labeling.app --data data/raw/tobacco3482/data \\
        --queue data/labeled/queue.json --labels data/labeled/labels.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

from src.config import Config, load_config
from src.schema import LabelRecord, PrelabelRecord

logger = logging.getLogger(__name__)

HOTKEYS_JS = """
() => {
  const isTyping = () => {
    const el = document.activeElement;
    return el && (el.tagName === 'INPUT' && el.type === 'text' || el.tagName === 'TEXTAREA');
  };
  document.addEventListener('keydown', (e) => {
    if (isTyping()) return;
    const click = (id) => {
      const el = document.querySelector('#' + id + ' input, #' + id);
      if (el) { el.click(); return true; }
      return false;
    };
    if (e.key >= '1' && e.key <= '9') {
      if (click('cb' + e.key)) e.preventDefault();
    } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
      if (click('btn_next')) e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
      if (click('btn_prev')) e.preventDefault();
    } else if (e.key === ' ') {
      if (click('btn_clean')) e.preventDefault();
    }
  });
}
"""


def load_queue(path: Path) -> list[PrelabelRecord]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [PrelabelRecord.model_validate(item) for item in raw]


def load_done(path: Path) -> dict[str, LabelRecord]:
    """Уже размеченные страницы. При повторах побеждает последняя запись."""
    done: dict[str, LabelRecord] = {}
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = LabelRecord.model_validate_json(line)
            except ValueError as exc:
                logger.warning("%s:%d — строка пропущена: %s", path, number, exc)
                continue
            done[record.image] = record
    return done


def append_label(path: Path, record: LabelRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def first_unlabeled(queue: list[PrelabelRecord], done: dict[str, LabelRecord]) -> int:
    for index, record in enumerate(queue):
        if record.image not in done:
            return index
    return 0


def _info_markdown(record: PrelabelRecord, config: Config, done: dict[str, LabelRecord]) -> str:
    scored = sorted(record.scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ", ".join(f"`{label}` {score:.2f}" for label, score in scored[:4] if score > 0.05)
    lines = [
        f"**{record.image}**",
        f"документ `{record.document}` · {record.width}×{record.height} · корпус `{record.corpus}`",
        f"CV подсказывает: {top or 'ничего заметного'}",
    ]
    if record.not_applicable:
        lines.append(
            f"⚠️ не измерено: {', '.join(record.not_applicable)} — решай глазами, "
            "галочка не предзаполнена"
        )
    if record.image in done:
        previous = done[record.image]
        lines.append(f"✓ уже размечено: {', '.join(previous.positive) or 'чисто'}")
    return "\n\n".join(lines)


def build_annotator(
    queue: list[PrelabelRecord],
    root: Path,
    labels_path: Path,
    config: Config,
    annotator: str,
):
    import gradio as gr

    manual = config.manual_labels
    done = load_done(labels_path)
    start_index = first_unlabeled(queue, done)

    def render(index: int):
        index = max(0, min(index, len(queue) - 1))
        record = queue[index]
        previous = done.get(record.image)
        # Уже размеченную страницу показываем с её метками, новую — с черновыми.
        values = [
            (previous.labels.get(label, False) if previous else record.suggested.get(label, False))
            for label in manual
        ]
        progress = f"### {index + 1} / {len(queue)}  ·  размечено {len(done)}"
        return [
            str(root / record.image),
            _info_markdown(record, config, done),
            progress,
            previous.notes if previous else "",
            index,
            time.time(),
            *values,
        ]

    def save(index: int, started: float, notes: str, *values):
        index = max(0, min(index, len(queue) - 1))
        record = queue[index]
        labelled = LabelRecord(
            image=record.image,
            document=record.document,
            corpus=record.corpus,
            labels=dict(zip(manual, [bool(v) for v in values])),
            prelabel=record.scores,
            annotator=annotator,
            timestamp=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_s=round(max(0.0, time.time() - started), 1),
            notes=notes.strip(),
        )
        append_label(labels_path, labelled)
        done[record.image] = labelled
        return labelled

    def save_and_next(index: int, started: float, notes: str, *values):
        save(index, started, notes, *values)
        return render(min(index + 1, len(queue) - 1))

    def save_clean(index: int, started: float, notes: str, *values):
        save(index, started, notes, *[False] * len(manual))
        return render(min(index + 1, len(queue) - 1))

    def go(index: int, delta: int):
        return render(max(0, min(index + delta, len(queue) - 1)))

    with gr.Blocks(title="Разметка сканов", js=HOTKEYS_JS) as demo:
        gr.Markdown(
            "# Разметка качества сканов\n"
            "Галочки предзаполнены черновой разметкой CV — проверь и поправь. "
            "Клавиши: **1–9** переключить дефект, **пробел** — страница чистая, "
            "**→ / Enter** — сохранить и дальше, **←** — назад."
        )
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(type="filepath", label="Скан", height=760, show_label=False)
            with gr.Column(scale=1):
                progress = gr.Markdown()
                info = gr.Markdown()
                checkboxes = [
                    gr.Checkbox(label=f"{i}. {label}", elem_id=f"cb{i}")
                    for i, label in enumerate(manual, 1)
                ]
                notes = gr.Textbox(label="Заметка", lines=2, placeholder="необязательно")
                with gr.Row():
                    prev_btn = gr.Button("← Назад", elem_id="btn_prev")
                    clean_btn = gr.Button("Чистая (пробел)", elem_id="btn_clean")
                    next_btn = gr.Button("Сохранить →", variant="primary", elem_id="btn_next")

        index_state = gr.State(start_index)
        started_state = gr.State(time.time())
        outputs = [image, info, progress, notes, index_state, started_state, *checkboxes]
        inputs = [index_state, started_state, notes, *checkboxes]

        next_btn.click(fn=save_and_next, inputs=inputs, outputs=outputs)
        clean_btn.click(fn=save_clean, inputs=inputs, outputs=outputs)
        prev_btn.click(fn=lambda i: go(i, -1), inputs=index_state, outputs=outputs)
        demo.load(fn=lambda i: render(i), inputs=index_state, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Разметчик качества сканов")
    parser.add_argument("--data", type=Path, required=True, help="корень корпуса")
    parser.add_argument("--queue", type=Path, required=True, help="queue.json из prelabel")
    parser.add_argument("--labels", type=Path, required=True, help="labels.jsonl (дозапись)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--annotator", default=os.environ.get("USERNAME", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    config = load_config(args.config, args.corpus)
    queue = load_queue(args.queue)
    if not queue:
        raise SystemExit(f"Очередь {args.queue} пуста")

    done = load_done(args.labels)
    print(f"очередь: {len(queue)} страниц, уже размечено {len(done)}")

    demo = build_annotator(queue, args.data, args.labels, config, args.annotator)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(args.data.resolve())],
    )


if __name__ == "__main__":
    main()
