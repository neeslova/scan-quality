"""Бэкенды VLM-судьи: провайдер меняется одной строкой конфига.

Наружу все отдают одно и то же — текст ответа модели на пару «изображение +
промпт». Ни разбор JSON, ни валидация схемы сюда не входят: бэкенд отвечает
только за доставку запроса, и подменить его должно быть можно, не трогая
ничего выше.

Изображение уходит во внешний сервис — в отличие от `src/explain.py`, где
наружу идут только числа. Это разрешено не везде: корпус выбирается в конфиге
прогона, и запрет на отправку конкретного корпуса задаётся там же.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Что умеет принимать типичный VLM. TIFF и GIF среди них нет — их конвертируем.
SUPPORTED_MEDIA = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class BackendError(RuntimeError):
    """Вызов не удался. Отличается от негодного ответа: тот ловится схемой."""


class JudgeBackend(Protocol):
    name: str

    def ask(self, image: bytes, media_type: str, prompt: str, system: str) -> str:
        """Отправляет изображение с вопросом и возвращает текст ответа."""


def encode_image(path: Path, max_side: int = 1568) -> tuple[bytes, str]:
    """Готовит изображение к отправке: поддерживаемый формат и разумный размер.

    Уменьшение до `max_side` — не экономия, а условие вменяемого ответа: модели
    сами ужимают вход, и картинка в 4000 пикселей приедет к ним уменьшенной, но
    уже без нашего контроля над качеством ресайза. Даунскейл только вниз: делать
    из мелкого скана крупный бессмысленно, а `low_resolution` при этом перестал
    бы быть виден.

    TIFF, GIF и прочее, чего провайдеры не принимают, пересохраняется в PNG.
    """
    from PIL import Image

    suffix = path.suffix.lower()
    with Image.open(path) as image:
        image = image.convert("RGB")
        longest = max(image.size)
        if longest > max_side:
            scale = max_side / longest
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        elif suffix in SUPPORTED_MEDIA and longest <= max_side:
            # Формат годится и размер в норме — отдаём файл как есть, без
            # перекодирования: лишний цикл JPEG добавил бы артефактов, которые
            # судья честно засчитает как дефект скана.
            return path.read_bytes(), SUPPORTED_MEDIA[suffix]

        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"


def extract_json(text: str) -> str:
    """Достаёт объект JSON из ответа модели.

    Промпт требует голый JSON, но модели регулярно оборачивают его в ```json```
    или предваряют фразой. Разбирать это здесь дешевле, чем терять страницу:
    ошибкой считается только отсутствие объекта как такового.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        without_fence = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        stripped = without_fence.removeprefix("json").strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("в ответе нет объекта JSON")
    return stripped[start : end + 1]


class AnthropicBackend:
    """Claude. Пакет `anthropic` ставится вместе с extra `explain`."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int, timeout_s: float) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:  # pragma: no cover — зависит от окружения
                raise BackendError(
                    'Пакет anthropic не установлен: pip install -e ".[explain]"'
                ) from exc
            self._client = anthropic.Anthropic(timeout=self._timeout_s)
        return self._client

    def ask(self, image: bytes, media_type: str, prompt: str, system: str) -> str:
        client = self._ensure_client()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                # Температура ноль: судья должен быть воспроизводим. Прогон,
                # который нельзя повторить, не годится ни для метрик, ни для
                # разбора расхождений с человеком.
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64.b64encode(image).decode("ascii"),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as error:  # noqa: BLE001 — наружу отдаём один тип ошибки
            raise BackendError(f"{type(error).__name__}: {error}") from error

        if getattr(response, "stop_reason", None) == "refusal":
            raise BackendError("модель отказалась отвечать")
        return "".join(block.text for block in response.content if block.type == "text")


# Реестр бэкендов. Добавить провайдера — значит дописать класс и строку сюда;
# в остальном коде не меняется ничего.
BACKENDS = {AnthropicBackend.name: AnthropicBackend}


def get_backend(name: str, model: str, max_tokens: int, timeout_s: float) -> JudgeBackend:
    factory = BACKENDS.get(name)
    if factory is None:
        raise ValueError(f"Неизвестный бэкенд судьи: {name}; есть: {', '.join(BACKENDS)}")
    return factory(model=model, max_tokens=max_tokens, timeout_s=timeout_s)
