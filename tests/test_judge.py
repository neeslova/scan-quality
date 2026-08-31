"""VLM-судья: схема ответа, промпт, ретрай и запрет на подстановку дефолтов."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.config import load_config
from src.judge.backends import BackendError, encode_image, extract_json, get_backend
from src.judge.prompt import AXIS_DEFINITIONS, build_prompt, verdict_rules
from src.judge.run import MAX_ATTEMPTS, decide_verdict, judge_page
from src.judge.schema import SEVERITY_MAX, JudgeAnswer
from tests import factories as fx


@pytest.fixture()
def config():
    return load_config()


def _answer(config, **overrides) -> dict:
    payload = {
        "scores": dict.fromkeys(config.labels, 0),
        "rotation_deg": 0,
        "blank_page": False,
        "legible_fraction": 1.0,
    }
    payload.update(overrides)
    return payload


class FakeBackend:
    """Бэкенд, отдающий заготовленные ответы по очереди."""

    name = "fake"

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    def ask(self, image, media_type, prompt, system):
        self.calls += 1
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture()
def scan(tmp_path):
    path = tmp_path / "скан.png"
    fx.save(fx.text_page(width=400, height=500), path)
    return path


def test_prompt_contains_every_project_axis(config) -> None:
    """Оси в промпте — из конфига, а не из головы: список меток зафиксирован."""
    prompt = build_prompt(config)
    for label in config.labels:
        assert f"- {label}:" in prompt


def test_every_axis_has_a_definition(config) -> None:
    """Ось без определения судья начнёт трактовать по-своему."""
    assert set(config.labels) <= set(AXIS_DEFINITIONS)


def test_prompt_rules_match_pipeline_thresholds(config) -> None:
    """Главная проверка: правила в промпте и код вердикта должны совпадать.

    Пороги пересчитываются калибровкой, и промпт, написанный руками, разъехался
    бы с системой молча — судья судил бы по одним числам, пайплайн по другим.
    Здесь для каждой оценки шкалы сверяется, что обещанное промптом равно тому,
    что посчитает `decide_verdict`.
    """
    rules = verdict_rules(config)

    def threshold_from(rules_text: str, needle: str, outcome: str) -> int:
        """Порог из той строки правил, что ведёт к нужному вердикту.

        Разбирать надо по паре «признак + исход»: «any axis >=» встречается и в
        правиле про bad, и в правиле про acceptable, и первое попавшееся число
        относилось бы не к тому правилу.
        """
        for line in rules_text.splitlines():
            if needle in line and line.rstrip(".").endswith(outcome):
                return int(line.split(needle)[1].split(",")[0])
        raise AssertionError(f"в правилах нет строки {needle!r} -> {outcome!r}:\n{rules_text}")

    unreadable_bad = threshold_from(rules, "unreadable >= ", "bad")
    acceptable_at = threshold_from(rules, "any axis >= ", "acceptable")

    # Оценка, объявленная в промпте как «bad по unreadable», действительно bad.
    scores = dict.fromkeys(config.labels, 0.0)
    scores["unreadable"] = unreadable_bad / SEVERITY_MAX
    assert decide_verdict(scores, config) == "bad"

    # На ступень ниже порога — уже не bad по этому правилу.
    scores["unreadable"] = (unreadable_bad - 1) / SEVERITY_MAX
    assert decide_verdict(scores, config) != "bad"

    # Оценка, объявленная как «acceptable», не должна давать good.
    scores = dict.fromkeys(config.labels, 0.0)
    scores["blur"] = acceptable_at / SEVERITY_MAX
    assert decide_verdict(scores, config) in {"acceptable", "bad"}

    scores["blur"] = (acceptable_at - 1) / SEVERITY_MAX
    assert decide_verdict(scores, config) == "good"


def test_answer_rejects_unknown_axis(config) -> None:
    """Придуманная судьёй ось делает прогон несравнимым с другими."""
    payload = _answer(config)
    payload["scores"]["коррозия"] = 2

    with pytest.raises(ValueError, match="лишние"):
        JudgeAnswer.model_validate(payload).validate_axes(config.labels)


def test_answer_rejects_missing_axis(config) -> None:
    payload = _answer(config)
    del payload["scores"]["blur"]

    with pytest.raises(ValueError, match="не хватает"):
        JudgeAnswer.model_validate(payload).validate_axes(config.labels)


def test_answer_rejects_score_out_of_scale(config) -> None:
    payload = _answer(config)
    payload["scores"]["blur"] = 7

    with pytest.raises(ValueError, match="вне шкалы"):
        JudgeAnswer.model_validate(payload).validate_axes(config.labels)


def test_answer_rejects_extra_top_level_field(config) -> None:
    payload = _answer(config)
    payload["confidence"] = 0.9

    with pytest.raises(ValidationError):
        JudgeAnswer.model_validate(payload)


def test_severity_maps_onto_pipeline_scale(config) -> None:
    """Шкала 0..4 переводится в скоры 0..1 делением, без подгонки."""
    payload = _answer(config)
    payload["scores"]["blur"] = SEVERITY_MAX
    answer = JudgeAnswer.model_validate(payload)

    assert answer.as_unit_scores()["blur"] == 1.0


def test_json_extracted_from_markdown_fence() -> None:
    """Модели регулярно оборачивают ответ в ```json — терять из-за этого страницу глупо."""
    payload = {"scores": {}, "rotation_deg": 0}
    fenced = "```json\n" + json.dumps(payload) + "\n```"

    assert json.loads(extract_json(fenced)) == payload


def test_json_extracted_after_preamble() -> None:
    assert json.loads(extract_json('Here is the result: {"a": 1} hope it helps')) == {"a": 1}


def test_extract_json_fails_loudly_without_object() -> None:
    with pytest.raises(ValueError, match="нет объекта JSON"):
        extract_json("I cannot judge this image.")


def test_retry_recovers_from_broken_json(config, scan) -> None:
    """Испорченный JSON — случайность, и один повтор её лечит."""
    backend = FakeBackend("не json вовсе", json.dumps(_answer(config)))

    record = judge_page(backend, scan, build_prompt(config), config, model_name="test")

    assert record.status == "ok"
    assert backend.calls == 2


def test_failure_is_recorded_not_defaulted(config, scan) -> None:
    """Самое важное: провал не превращается в нули.

    Подставленные оценки по умолчанию отправили бы весь брак в `good`, и по
    сводным цифрам это не заметить.
    """
    backend = FakeBackend(BackendError("сеть недоступна"))

    record = judge_page(backend, scan, build_prompt(config), config, model_name="test")

    assert record.status == "failed"
    assert record.answer is None
    assert record.verdict is None
    assert "сеть недоступна" in record.error


def test_retry_happens_exactly_once(config, scan) -> None:
    """Устойчиво негодный ответ — повод остановиться, а не платить пять раз."""
    backend = FakeBackend(BackendError("тайм-аут"))

    judge_page(backend, scan, build_prompt(config), config, model_name="test")

    assert backend.calls == MAX_ATTEMPTS


def test_invalid_axes_also_trigger_failure(config, scan) -> None:
    """Ответ разобрался, но оси чужие — это тоже отказ, а не «почти годится»."""
    wrong = _answer(config)
    wrong["scores"] = {"blur": 1}
    backend = FakeBackend(json.dumps(wrong))

    record = judge_page(backend, scan, build_prompt(config), config, model_name="test")

    assert record.status == "failed"
    assert record.answer is None


def test_verdict_is_derived_from_the_answer(config, scan) -> None:
    payload = _answer(config)
    payload["scores"]["unreadable"] = SEVERITY_MAX
    backend = FakeBackend(json.dumps(payload))

    record = judge_page(backend, scan, build_prompt(config), config, model_name="test")

    assert record.status == "ok"
    assert record.verdict == "bad"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестный бэкенд"):
        get_backend("не-существует", model="m", max_tokens=10, timeout_s=1.0)


def test_tiff_is_converted_for_sending(tmp_path) -> None:
    """TIFF провайдеры не принимают — иначе страница молча выпала бы из прогона."""
    path = tmp_path / "скан.tif"
    fx.save(fx.text_page(width=300, height=400), path)

    data, media_type = encode_image(path)

    assert media_type == "image/png"
    assert data[:4] == b"\x89PNG"


def test_large_image_is_downscaled(tmp_path) -> None:
    """Ресайз делаем сами: иначе его сделает провайдер, но без нашего контроля."""
    from PIL import Image

    path = tmp_path / "big.png"
    fx.save(fx.text_page(width=3000, height=4000), path)

    data, _ = encode_image(path, max_side=800)
    import io

    with Image.open(io.BytesIO(data)) as image:
        assert max(image.size) == 800


def test_small_image_is_sent_as_is(tmp_path) -> None:
    """Апскейл запрещён: он стёр бы признак low_resolution."""
    path = tmp_path / "small.png"
    fx.save(fx.text_page(width=200, height=300), path)

    data, media_type = encode_image(path, max_side=1568)

    assert media_type == "image/png"
    assert data == path.read_bytes()
