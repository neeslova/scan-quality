"""Контракт ответа VLM-судьи.

Схема строгая намеренно. Судья — внешняя модель, и единственная защита от того,
что она вернёт правдоподобный мусор, — это проверка структуры до того, как числа
куда-то попадут. `extra="forbid"` ловит выдуманные оси, диапазоны ловят оценки
вне шкалы, а обязательность полей — молчаливые пропуски.

**Ни у одного поля нет значения по умолчанию.** Это главное решение модуля.
Дефолт здесь означал бы, что неудавшийся вызов тихо превращается в «дефектов
нет», и весь брак уезжает в `good` — ровно та ошибка, которую в задании назвали
самой частой. Не ответила модель — запись получает `status="failed"` и не
участвует в метриках вовсе.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Шкала судьи. Ноль — дефекта нет, четыре — текст из-за него не читается.
SEVERITY_MIN = 0
SEVERITY_MAX = 4

JudgeStatus = Literal["ok", "failed"]


class JudgeAnswer(BaseModel):
    """То, что модель обязана вернуть. Всё лишнее или отсутствующее — ошибка."""

    model_config = ConfigDict(extra="forbid")

    # Оценки по осям проекта. Ключи проверяются отдельно, против config.labels:
    # схема не может знать их заранее, а список меток — источник истины в конфиге.
    scores: dict[str, int]
    rotation_deg: Literal[0, 90, 180, 270]
    blank_page: bool
    legible_fraction: float = Field(ge=0.0, le=1.0)

    def validate_axes(self, expected: list[str]) -> JudgeAnswer:
        """Проверяет, что оси ровно те, что заданы проектом.

        Лишняя ось означает, что модель придумала свой дефект, пропущенная — что
        она промолчала об одном из наших. И то и другое делает ответ негодным:
        сравнивать прогоны с разным набором осей нельзя.
        """
        got = set(self.scores)
        want = set(expected)
        if got != want:
            missing = sorted(want - got)
            extra = sorted(got - want)
            raise ValueError(
                "оси не совпадают с проектом; не хватает: {}; лишние: {}".format(
                    ", ".join(missing) or "—", ", ".join(extra) or "—"
                )
            )
        for name, value in self.scores.items():
            if not SEVERITY_MIN <= value <= SEVERITY_MAX:
                raise ValueError(f"{name}={value} вне шкалы {SEVERITY_MIN}..{SEVERITY_MAX}")
        return self

    def as_unit_scores(self) -> dict[str, float]:
        """Оценки в шкале 0..1 — той, в которой работают пороги пайплайна.

        Деление на максимум шкалы, а не подгонка: severity 4 означает «текст
        нечитаем», и это ровно единица в шкале скоров.
        """
        return {name: value / SEVERITY_MAX for name, value in self.scores.items()}


class JudgeRecord(BaseModel):
    """Строка результата по одной странице: ответ судьи либо честный отказ."""

    model_config = ConfigDict(extra="forbid")

    image: str
    page: int = Field(default=0, ge=0)
    sha256: str = ""

    status: JudgeStatus
    answer: Optional[JudgeAnswer] = None
    # Вердикт, выведенный из оценок теми же порогами, что и в пайплайне.
    verdict: Optional[str] = None
    error: str = ""
    model: str = ""
    elapsed_s: float = Field(default=0.0, ge=0.0)
    # Сырой ответ сохраняется только при провале разбора: по нему видно, модель
    # ответила чепухой или обвязка не справилась с корректным ответом.
    raw: str = ""

    @property
    def key(self) -> str:
        return f"{self.sha256}#{self.page}"
