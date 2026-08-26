# Оценка качества сканов документов

Принимает скан (jpg/png/pdf) и выдаёт вердикт `good` / `acceptable` / `bad`, список
дефектов с вероятностями, локализацию (heatmap) и машиночитаемый JSON.

Ядро — CNN multi-label по патчам 384×384 + детерминированные CV-метрики + OCR-слой.
Внешняя сеть не нужна: без интернета система работает полностью.

Полное ТЗ, архитектура и план спринтов — в [PLAN.md](PLAN.md).

## Статус

| Спринт | Что | Статус |
|---|---|---|
| С0 | каркас, конфиг, схема, заглушка приложения | ✅ |
| С1 | CV-метрики (baseline) | — |
| С2 | разметка | — |
| С3 | OCR-слой | — |
| С4 | синтетика | — |
| С5 | обучение | — |
| С6 | калибровка и экспорт в ONNX | — |
| С7 | приложение и CLI | — |
| С8 | оценка и документация | — |

## Установка (Windows, CPU)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip setuptools
py -m pip install -e ".[dev]"
```

Обучение локально не ставим — оно живёт в Colab (`notebooks/train.ipynb`),
зависимости там: `pip install -e ".[train]"`.

## Запуск

```powershell
python -m src.app                      # Gradio на http://127.0.0.1:7860
python -m src.app --image scan.jpg     # один файл в консоль, без UI
pytest                                 # тесты
ruff check . && black --check .        # линт
```

## Структура

Всё содержательное — в `src/`, все параметры — в `configs/base.yaml`.
Магических чисел в коде нет: метки, пороги, гиперпараметры берутся из конфига.

```
configs/base.yaml   единая точка правды
src/config.py       yaml -> pydantic
src/schema.py       QualityReport — контракт наружу
src/pipeline.py     оркестратор image -> QualityReport
src/app.py          Gradio
src/metrics/        CV-метрики (С1)
src/ocr/            OCR и метка unreadable (С3)
src/data/           сплит, деградации, синтетика (С2/С4)
src/models/         обучение, калибровка, ONNX (С5/С6)
```

`data/`, `models/`, `runs/`, `reports/` — в `.gitignore`.
