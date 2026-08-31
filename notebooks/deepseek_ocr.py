"""Генератор ноутбука `deepseek_ocr.ipynb`. Запускать локально после правок.

Ноутбук держим сгенерированным, а не редактируем руками: в .ipynb ячейки лежат
списком строк с метаданными, и ручная правка в редакторе даёт нечитаемый diff.
Здесь же виден весь текст ноутбука подряд.

    python notebooks/deepseek_ocr.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/neeslova/scan-quality.git"
# Слой DeepSeek живёт в рабочей ветке; в main его нет, и клон по умолчанию
# приедет без `src/ocr/deepseek.py`. Ветку указываем явно.
BRANCH = "s5-cv-vs-cnn"

MARKDOWN_INTRO = """# DeepSeek-OCR: прогон корпуса (Colab)

**Правило то же, что в `train.ipynb`: в ноутбуке нет логики.** Весь код — в
`src/ocr/deepseek.py`, ноутбук только запускает.

Что здесь происходит и почему именно так:

* модель читает каждую страницу **дважды** — в низком и высоком разрешении.
  Расхождение двух прочтений и есть главный сигнал этапа (self-consistency):
  на плохом скане генеративная модель каждый раз выдумывает своё. Разметка для
  этого не нужна, поэтому сигнал применим ко всему корпусу;
* сохраняется **сырой текст**, а не метрики. Прогон стоит часов GPU, а формулы
  сигналов ещё будут меняться — всё, что можно пересчитать локально, здесь не
  считается;
* прогон **переживает обрыв сессии**. Готовое адресуется по sha256 файла, при
  повторном запуске уже посчитанное пропускается. Бесплатный Colab отключается
  по таймауту, и это нормальный режим работы, а не авария.

Перед запуском: Runtime → Change runtime type → **GPU**.

**Ноутбук правится только переоткрытием из GitHub.** Ячейка установки делает
`git pull`, и он обновляет `src/`, но не текст ячеек: они живут во вкладке
браузера, а не в клоне. Если фикс касается ноутбука — закройте вкладку и
откройте ссылку заново, иначе будете выполнять старый код поверх нового `src`.

Где лежат данные и результат — решается в первой ячейке флагом `USE_DRIVE`.
По умолчанию всё на диске Colab: Drive делит 15 ГБ с почтой и фото, и при
переполнении аккаунта Colab начинает показывать баннер о квоте, а запись
отказывает.
"""

MARKDOWN_HARDWARE = """## 1. Железо и пути

Важно, какая именно карта досталась. FlashAttention-2 требует Ampere и новее;
на T4 (Turing) он не собирается, и модуль сам переключается на штатное внимание.
Ячейка ниже просто показывает, с чем предстоит работать.

Тип весов от карты при этом **не** зависит: remote-код модели зашивает
bfloat16, и на T4 он идёт через эмуляцию. Это медленнее, но выбора нет —
подробности в `model_dtype`.

Смотрим и на **ОЗУ**, а не только на видеопамять: сеанс на бесплатном Colab
убивает именно она. Карта тут не самое узкое место.

Здесь же решается, где живут данные и результат. **Drive не обязателен.** Он
переживает удаление среды, но делит 15 ГБ квоты с Gmail и Google Фото, и когда
аккаунт за квотой, Colab показывает баннер, а запись отказывает — кодом это не
обходится. Локальный диск Colab даёт 113 ГБ без квот и переживает падение ядра;
теряется он только вместе со средой. По умолчанию работаем на нём.
"""

CODE_HARDWARE = """from pathlib import Path

# True — данные и результат в Drive (переживает удаление среды, но упирается
# в квоту аккаунта). False — локальный диск Colab: 113 ГБ, никаких квот и
# никакого баннера, но всё исчезает вместе со средой.
USE_DRIVE = False

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/scanq')
else:
    ROOT = Path('/content/scanq')

ROOT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'Data iz tg'
OUT = ROOT / 'deepseek_tg.jsonl'
WORK = Path('/content/work')
print('корпус:', DATA)
print('результат:', OUT)

!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
!free -g | head -2
!df -h /content | tail -1
"""

MARKDOWN_DATA = """## 1a. Корпус на диск

Нужна только при `USE_DRIVE = False`. Перетащите архив `Data iz tg.zip` в
файловый браузер слева, в папку `/content`, и выполните ячейку — 146 МБ
заливаются за пару минут. Повторный запуск ничего не делает: корпус уже на месте.

Внутри архива должна лежать сама папка `Data iz tg` с `Good/` и `bad/`.
"""

CODE_DATA = """if not DATA.is_dir():
    !unzip -q '/content/Data iz tg.zip' -d {ROOT}

pages = [p for p in DATA.rglob('*') if p.is_file()]
print('файлов в корпусе:', len(pages))
"""

MARKDOWN_INSTALL = """## 2. Установка

Версии зафиксированы карточкой модели: `transformers` новее 4.46.3 ломает её
remote-код. `flash-attn` ставится **только на Ampere+** — на T4 его сборка
займёт двадцать минут и закончится ошибкой.

`accelerate` здесь не декоративен: без него не работает `low_cpu_mem_usage`,
которым веса льются пошардово, — а без этого сеанс умирает по ОЗУ (см. ниже).

Клонируется **рабочая ветка**, не `main`: слой DeepSeek в `main` ещё не влит.
Ячейка печатает последний коммит — если в нём нет ожидаемой работы, дальше идти
незачем, прогон всё равно упадёт на импорте.

Ячейку можно перезапускать: если клон уже есть, он подтягивается, а не роняет
ячейку. Это не косметика — после падения по ОЗУ ядро перезапускается, а диск
Colab остаётся, и одноразовый `clone` молча оставил бы старый код.
"""

CODE_INSTALL = """import os

# Клонируем, если папки нет, и подтягиваем, если есть. Иначе ячейка одноразовая:
# после падения по ОЗУ ядро перезапускается, а диск Colab переживает это, и
# `git clone` падает на непустой папке — оставляя ровно тот код, из-за которого
# сеанс и умер.
if os.path.isdir('/content/scan-quality/.git'):
    !git -C /content/scan-quality fetch -q origin BRANCH_NAME
    !git -C /content/scan-quality reset --hard -q FETCH_HEAD
else:
    !git clone -q -b BRANCH_NAME REPO_URL /content/scan-quality

!git -C /content/scan-quality log -1 --oneline
!ls /content/scan-quality/src/ocr/deepseek.py  # нет файла -> ветка не та
!pip install -q transformers==4.46.3 tokenizers==0.20.3 accelerate einops addict easydict pymupdf

import torch

major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else 'CPU'
print('compute capability:', cap)

if major >= 8:
    print('Ampere+: ставим flash-attn')
    !pip install -q flash-attn==2.7.3 --no-build-isolation
else:
    print('Turing или CPU: flash-attn пропускаем, пойдёт eager attention')
"""

MARKDOWN_PROBE = """## 3. Проба на одной странице

Документация модели не описывает, что возвращает `infer`: строку с текстом или
только запись в `output_path`. Модуль поддерживает оба варианта, но проверить
это надо **до** многочасового прогона, а не в его середине.

Заодно первый запуск скачивает веса (несколько ГБ) — пусть это случится здесь.

**Про ОЗУ — узкое место всего прогона.** У бесплатного Colab её 12.7 ГБ,
а видеопамяти T4 — 15 ГБ, и умирает сеанс именно по ОЗУ. Одного `torch_dtype`
не хватило: чекпойнт лежит **одним шардом на 6.7 ГБ**, поэтому `low_cpu_mem_usage`
со своей загрузкой «по шарду за раз» не выигрывает ничего, а копия тензоров
вместе с mmap того же файла даёт около 13 ГБ. Поэтому `load()` передаёт ещё и
`device_map`: accelerate раскладывает веса прямо на карту, потензорно, и полной
копии в ОЗУ не возникает.

Ячейка печатает ОЗУ до и после загрузки — если запас меньше гигабайта, полный
прогон лучше не начинать.

**Веса остаются на диске Colab и в Drive не уводятся.** Их 6.7 ГБ, а
бесплатный Drive — 15 ГБ на всё, вместе с корпусами; попытка сложить их туда
упирается в квоту и оставляет обрезанный файл. Кэшировать их там и незачем:
качаются они чуть больше минуты (116 МБ/с), а чтение из Drive через FUSE
медленнее локального диска. Диск Colab переживает падение ядра и теряется
только при удалении среды.

В Drive держим ровно то, что дорого потерять: корпуса и `deepseek_*.jsonl`
с результатами.
"""

CODE_PROBE = """%cd /content/scan-quality
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(message)s')

from pathlib import Path
from src.ocr.deepseek import DeepSeekOCR, attention_implementation, model_dtype

print('attention:', attention_implementation(), '| dtype:', model_dtype())

sample = sorted(p for p in (DATA / 'Good').glob('*') if p.suffix.lower() in {'.jpg', '.png'})[0]
print('пробная страница:', sample.name)

print('ОЗУ до загрузки:')
!free -g | sed -n '2p'

engine = DeepSeekOCR()
text = engine.read(sample, 'tiny', WORK)

print('ОЗУ после загрузки:')
!free -g | sed -n '2p'

import torch
print('видеопамять под весами: %.1f ГБ' % (torch.cuda.memory_allocated() / 2**30))
print('символов:', len(text))
print(text[:500])
"""

MARKDOWN_BENCH = """## 4. Замер скорости

Двадцать страниц, чтобы посчитать бюджет прогона по факту, а не по догадке.
Умножьте `с/страница` из лога на размер корпуса — и станет видно, влезает ли
полный прогон в сессию или его надо резать выборкой.

Напомню объёмы: `Data iz tg` — 204 страницы, Tobacco3482 — 3482, Yenisei — 1802.
Каждая читается дважды.
"""

CODE_BENCH = """from src.ocr.deepseek import run

run(DATA, OUT, modes=('tiny', 'base'), limit=20, workdir=WORK)
"""

MARKDOWN_FULL = """## 5. Полный прогон

Ячейку можно перезапускать сколько угодно: посчитанное пропускается. Если
сессия отвалилась — просто выполните её снова, прогон продолжится с места обрыва.

Корпуса гоняем по одному, начиная со своего: он маленький и на нём быстрее
станет ясно, что сигналы вообще работают.
"""

CODE_FULL = """run(DATA, OUT, modes=('tiny', 'base'), workdir=WORK)

# Следующие корпуса — по очереди, когда первый закрыт:
# run(ROOT / 'tobacco3482', ROOT / 'deepseek_tobacco.jsonl',
#     modes=('tiny', 'base'), workdir=WORK)
"""

MARKDOWN_RESCUE = """## 6. Забрать результат

При `USE_DRIVE = False` файл лежит на диске Colab и исчезнет вместе со средой.
Забирайте его **до** конца работы, а на длинном прогоне — время от времени.

Докатка от этого не страдает: чтобы продолжить после обрыва, залейте jsonl
обратно в `/content/scanq/` рядом с корпусом. Готовое адресуется по sha256, и
уже прочитанные страницы будут пропущены.
"""

CODE_RESCUE = """if not USE_DRIVE:
    from google.colab import files
    files.download(str(OUT))
"""

MARKDOWN_DONE = """## 7. Что дальше

Скачанный `deepseek_*.jsonl` кладём локально в `data/labeled/` и считаем сигналы
на своей машине — GPU для этого не нужен:

```powershell
python -m src.ocr.deepseek_signals --texts data/labeled/deepseek_tg.jsonl \\
    --golden data/labeled/golden_tg.jsonl --out reports/deepseek_tg.md
```
"""

CODE_SUMMARY = """import json, collections

rows = [json.loads(line) for line in open(OUT, encoding='utf-8')]
status = collections.Counter(r['status'] for r in rows)
print('строк:', len(rows), dict(status))

ok = [r for r in rows if r['status'] == 'ok']
if ok:
    for mode in ok[0]['elapsed_s']:
        times = [r['elapsed_s'][mode] for r in ok if mode in r['elapsed_s']]
        print(f'{mode}: {sum(times)/len(times):.1f} с/страница')
    empty = sum(1 for r in ok if not any(t.strip() for t in r['texts'].values()))
    print('пустых прочтений:', empty)
"""


def cell(kind: str, source: str) -> dict:
    lines = source.strip("\n").split("\n")
    payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": payload}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": payload,
    }


def build() -> dict:
    cells = [
        cell("markdown", MARKDOWN_INTRO),
        cell("markdown", MARKDOWN_HARDWARE),
        cell("code", CODE_HARDWARE),
        cell("markdown", MARKDOWN_DATA),
        cell("code", CODE_DATA),
        cell("markdown", MARKDOWN_INSTALL),
        cell("code", CODE_INSTALL.replace("REPO_URL", REPO).replace("BRANCH_NAME", BRANCH)),
        cell("markdown", MARKDOWN_PROBE),
        cell("code", CODE_PROBE),
        cell("markdown", MARKDOWN_BENCH),
        cell("code", CODE_BENCH),
        cell("markdown", MARKDOWN_FULL),
        cell("code", CODE_FULL),
        cell("code", CODE_SUMMARY),
        cell("markdown", MARKDOWN_RESCUE),
        cell("code", CODE_RESCUE),
        cell("markdown", MARKDOWN_DONE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    target = Path(__file__).with_suffix(".ipynb")
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записан {target}")
