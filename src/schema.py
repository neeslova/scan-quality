"""Контракт системы: структуры на границах пайплайна.

`QualityReport` — то, что уходит наружу (Gradio, CLI, встраивание в поток проверки).
Меняем осторожно и вместе с `schema_version`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Verdict = Literal["good", "acceptable", "bad"]
ScoreSource = Literal["cnn", "cv", "ocr", "stub"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DefectScore(_Model):
    """Вероятность одного дефекта и то, кто её выдал."""

    label: str
    # Скор в ОБЩЕЙ шкале: приведён якорями из `verdict.anchors` (С6). Именно его
    # сравнивают с порогами вердикта, потому что скоры трёх источников иначе
    # несравнимы между собой.
    score: float = Field(ge=0.0, le=1.0)
    # То, что выдал сам источник, до приведения. Нужен для разбора ошибок:
    # по нему видно, метка не сработала или её шкала посчитана неверно.
    raw: Optional[float] = None
    source: ScoreSource = "cnn"
    # Для локальных дефектов — индексы патчей, где сработало сильнее всего.
    top_patches: list[int] = Field(default_factory=list)


class OCRResult(_Model):
    engine: str
    mean_confidence: float = Field(ge=0.0, le=1.0)
    # Доля символов вне алфавита языка. С движком, у которого закрытый алфавит
    # (EasyOCR), всегда 0 — там работает nonword_ratio.
    garbage_ratio: float = Field(ge=0.0, le=1.0)
    # Доля токенов, не похожих на слова: выживает при закрытом алфавите.
    nonword_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    # Доля площади текста, прочитанная уверенно и осмысленно. Устойчива к
    # локальной помехе: печать поверх текста портит свой участок, а не страницу.
    readable_share: float = Field(default=1.0, ge=0.0, le=1.0)
    text_density: float = Field(ge=0.0)
    n_boxes: int = Field(ge=0)


class QualityReport(_Model):
    """Машиночитаемый результат по одной странице."""

    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = "stub"

    image: str
    width: int = Field(ge=0)
    height: int = Field(ge=0)

    verdict: Verdict
    quality_score: float = Field(ge=0.0, le=1.0)
    defects: list[DefectScore] = Field(default_factory=list)

    cv_metrics: dict[str, float] = Field(default_factory=dict)
    # Метки, которые на этой странице измерить нечем (например, low_contrast и noise
    # на битональном скане). Их отсутствие в defects — не «дефекта нет», а «не измерено».
    not_applicable: list[str] = Field(default_factory=list)
    ocr: Optional[OCRResult] = None

    heatmap_path: Optional[str] = None
    explanation: Optional[str] = None
    elapsed_ms: float = Field(default=0.0, ge=0.0)

    def scores(self) -> dict[str, float]:
        """Плоский вид {метка: вероятность} — удобно для правил и таблиц."""
        return {d.label: d.score for d in self.defects}

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=False)


class PrelabelRecord(_Model):
    """Черновая разметка страницы по CV-метрикам — вход для ручного разметчика."""

    image: str  # путь относительно корня корпуса
    document: str  # id документа: сплит идёт по нему, не по странице
    corpus: str

    scores: dict[str, float] = Field(default_factory=dict)
    suggested: dict[str, bool] = Field(default_factory=dict)
    # Метки, которые на этой странице измерить нечем: чекбокс не предзаполняем,
    # решает человек.
    not_applicable: list[str] = Field(default_factory=list)

    verdict: Verdict = "good"
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)

    @property
    def suspicion(self) -> float:
        """Насколько страница подозрительна: по ней сортируем очередь разметки."""
        return max(self.scores.values(), default=0.0)


class LabelRecord(_Model):
    """Одна размеченная человеком страница. Хранится строкой в labels.jsonl.

    `prelabel` сохраняется намеренно: по нему потом видно, сколько галочек
    разметчик снял или добавил, то есть насколько CV-предразметка помогала
    и где она систематически врёт.
    """

    image: str
    document: str
    corpus: str

    labels: dict[str, bool] = Field(default_factory=dict)
    prelabel: dict[str, float] = Field(default_factory=dict)
    # Скоры меток, выведенных автоматически (unreadable из OCR): бинарная метка
    # в `labels` теряет градацию, а для анализа порога она нужна.
    derived: dict[str, float] = Field(default_factory=dict)

    annotator: str = ""
    timestamp: str = ""
    duration_s: float = Field(default=0.0, ge=0.0)
    notes: str = ""

    @property
    def positive(self) -> list[str]:
        return sorted(label for label, present in self.labels.items() if present)


class GoldenRecord(_Model):
    """Эталонный бинарный вердикт по одной странице — то, с чем сверяются модели.

    Отдельный тип, а не `LabelRecord`, потому что это принципиально другая
    величина. `LabelRecord` отвечает на вопрос «какие дефекты видит разметчик»
    (десять независимых галочек), а здесь записан итог: годится страница или нет.
    Из набора галочек этот итог не выводится однозначно — решает то же правило
    порогов, что и в пайплайне, а значит вывод был бы не эталоном, а мнением
    системы о себе самой.

    Источник метки хранится рядом со значением: `folder` — раскладка по папкам,
    сделанная человеком заранее, `manual` — разметка в приложении. Смешивать их
    в одном файле можно, но при разборе ошибок надо знать, откуда метка.
    """

    image: str  # путь относительно корня корпуса, со слэшами
    page: int = Field(default=0, ge=0)  # страница внутри PDF; для картинок всегда 0
    document: str  # страницы одного PDF — один документ, иначе сплит потечёт
    corpus: str

    label: Literal["good", "bad"]
    source: Literal["folder", "manual"] = "folder"
    annotator: str = ""
    notes: str = ""
    # sha256 файла. Им же адресуются результаты дорогих прогонов (OCR, VLM):
    # переименованный или перемещённый файл не должен считаться заново, а два
    # разных файла с одинаковым именем не должны слиться в один.
    sha256: str = ""

    @property
    def key(self) -> str:
        """Ключ страницы: файл плюс номер. По нему сверяются прогоны."""
        return f"{self.image}#{self.page}"


class SyntheticRecord(_Model):
    """Одна сгенерированная страница. Метки известны по построению."""

    image: str
    reference: str  # чистая страница, из которой сделана
    # Документ наследуется от эталона: иначе деградированная копия тестовой
    # страницы вернулась бы в обучение и утечка №1 прошла бы через синтетику.
    document: str
    corpus: str

    labels: dict[str, bool] = Field(default_factory=dict)
    severities: dict[str, float] = Field(default_factory=dict)
    # Зерно генератора этой страницы. С ним запись — полный рецепт: по эталону,
    # меткам, силам и зерну страница восстанавливается побитово. Поэтому в Colab
    # везём эталоны и манифест (сотни мегабайт), а не готовые картинки (гигабайты).
    seed: int = 0
    # Метка -> путь к маске области. Только для локальных дефектов: патч из
    # чистого угла не должен получить метку `glare` из-за блика в центре.
    masks: dict[str, str] = Field(default_factory=dict)

    # Нули, если страница ещё не отрисована: в режиме «только рецепт» размер
    # неизвестен, он определяется при восстановлении.
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)

    @property
    def positive(self) -> list[str]:
        return sorted(label for label, present in self.labels.items() if present)
