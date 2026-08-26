"""Девять деградаций поверх чистого реального скана.

Ключевое решение проекта (журнал, №3): страницы не рисуются с нуля, а портятся
поверх настоящих. Так сохраняется домен — та же бумага, тот же шрифт, тот же
прогон сканера, — а метки известны по построению.

Локальные дефекты возвращают **маску области**: без неё нельзя корректно разметить
патчи. Блик занимает восьмую часть страницы, и патч из чистого угла не должен
получить метку `glare` только потому, что она стоит у всей страницы.

Все диапазоны параметров — в `configs/base.yaml`, секция `synth.params`.
Сила дефекта задаётся одним числом 0..1 и линейно раскладывается по диапазонам:
границы могут идти и по убыванию (`low_resolution`: чем сильнее, тем меньше масштаб).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Degraded:
    image: np.ndarray
    # None — дефект глобальный, метка относится ко всей странице.
    mask: Optional[np.ndarray] = None


def _lerp(bounds: tuple[float, float], severity: float) -> float:
    low, high = bounds
    return low + (high - low) * severity


def _odd(value: float) -> int:
    return max(1, int(value) | 1)


def _paper_level(gray: np.ndarray) -> int:
    """Уровень бумаги: медиана. Текст всегда в меньшинстве."""
    return int(np.median(gray))


# --- глобальные дефекты -----------------------------------------------------


def blur(gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator) -> Degraded:
    """Расфокус или смаз — движок выбирается случайно, как в жизни."""
    if rng.random() < 0.5:
        ksize = _odd(_lerp(config.synth.span("blur_gauss", "ksize"), severity))
        return Degraded(cv2.GaussianBlur(gray, (ksize, ksize), 0))

    length = _odd(_lerp(config.synth.span("blur_motion", "length"), severity))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    angle = float(rng.uniform(0, 180))
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return Degraded(cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT))


def noise(gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator) -> Degraded:
    """Зерно плюс соль-перец: у сканера бывает и то, и другое одновременно."""
    sigma = _lerp(config.synth.span("noise", "sigma"), severity)
    out = gray.astype(np.float32) + rng.normal(0.0, sigma, gray.shape).astype(np.float32)

    salt_prob = _lerp(config.synth.span("noise", "salt_prob"), severity)
    if salt_prob > 0:
        picks = rng.random(gray.shape)
        out[picks < salt_prob / 2] = 0
        out[picks > 1 - salt_prob / 2] = 255
    return Degraded(np.clip(out, 0, 255).astype(np.uint8))


def low_contrast(
    gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator
) -> Degraded:
    """Блёклая печать: диапазон сжимается к уровню бумаги, а не к среднему.

    К среднему сжимать нельзя — страница потемнела бы целиком, а выцветает
    именно краска: бумага остаётся светлой, текст к ней приближается.
    """
    factor = _lerp(config.synth.span("low_contrast", "factor"), severity)
    paper = float(_paper_level(gray))
    out = paper + (gray.astype(np.float32) - paper) * factor
    return Degraded(np.clip(out, 0, 255).astype(np.uint8))


def low_resolution(
    gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator
) -> Degraded:
    """Скан снят в низком разрешении: страница реально становится мельче.

    Именно мельче, а не «уменьшили и вернули». Загрузчик апскейл запрещает
    (решение №20), и метрика высоты строки меряет рабочее разрешение — значит
    низкое разрешение должно остаться низким.
    """
    scale = _lerp(config.synth.span("low_resolution", "scale"), severity)
    height = max(32, int(gray.shape[0] * scale))
    width = max(32, int(gray.shape[1] * scale))
    return Degraded(cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA))


def skew(gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator) -> Degraded:
    """Перекос страницы. Знак случайный: заваливают в обе стороны."""
    angle = _lerp(config.synth.span("skew", "angle_deg"), severity)
    angle *= 1.0 if rng.random() < 0.5 else -1.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_paper_level(gray),
    )
    return Degraded(rotated)


# --- локальные дефекты (возвращают маску) -----------------------------------


def glare(gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator) -> Degraded:
    """Пересвет: эллипс, выбитый в белое, с мягкими краями.

    Мягкость обязательна: резкая граница белого пятна сама выглядит как дефект
    другого рода, и сеть выучила бы контур эллипса вместо потери информации.
    """
    height, width = gray.shape
    radius_frac = _lerp(config.synth.span("glare", "radius_frac"), severity)
    softness = _lerp(config.synth.span("glare", "softness_frac"), severity)

    center = (int(rng.uniform(0.15, 0.85) * width), int(rng.uniform(0.15, 0.85) * height))
    axes = (
        max(8, int(width * radius_frac)),
        max(8, int(height * radius_frac * rng.uniform(0.6, 1.1))),
    )

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 255, thickness=-1)
    blur_k = _odd(min(axes) * softness)
    soft = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0

    out = gray.astype(np.float32) * (1.0 - soft) + 255.0 * soft
    return Degraded(np.clip(out, 0, 255).astype(np.uint8), mask)


def shadow(
    gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator
) -> Degraded:
    """Затемнение у одного края — как от переплёта или крышки сканера."""
    height, width = gray.shape
    depth = _lerp(config.synth.span("shadow", "depth"), severity)
    width_frac = _lerp(config.synth.span("shadow", "width_frac"), severity)
    side = int(rng.integers(0, 4))

    axis_len = width if side in (0, 1) else height
    band = max(1, int(axis_len * width_frac))
    ramp = np.ones(axis_len, dtype=np.float32)
    fade = np.linspace(1.0 - depth, 1.0, band, dtype=np.float32)
    if side in (0, 2):
        ramp[:band] = fade
    else:
        ramp[-band:] = fade[::-1]

    field = ramp[None, :] if side in (0, 1) else ramp[:, None]
    out = np.clip(gray.astype(np.float32) * field, 0, 255).astype(np.uint8)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    if side == 0:
        mask[:, :band] = 255
    elif side == 1:
        mask[:, -band:] = 255
    elif side == 2:
        mask[:band, :] = 255
    else:
        mask[-band:, :] = 255
    return Degraded(out, mask)


def streaks(
    gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator
) -> Degraded:
    """Полосы валиков: несколько колонок или строк стабильно темнее по всей странице."""
    height, width = gray.shape
    count = int(_lerp(config.synth.span("streaks", "count"), severity))
    depth = _lerp(config.synth.span("streaks", "depth"), severity)
    thickness = max(1, int(_lerp(config.synth.span("streaks", "width"), severity)))
    vertical = rng.random() < 0.7  # у валиков полосы чаще вертикальные

    out = gray.astype(np.int16)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    axis_len = width if vertical else height
    for start in rng.choice(max(1, axis_len - thickness), size=count, replace=False):
        stop = int(start) + thickness
        # Полосы неодинаковы: одинаковая глубина выглядит как сетка, а не как грязь.
        local = depth * float(rng.uniform(0.5, 1.0))
        if vertical:
            out[:, int(start) : stop] -= int(local)
            mask[:, int(start) : stop] = 255
        else:
            out[int(start) : stop, :] -= int(local)
            mask[int(start) : stop, :] = 255
    return Degraded(np.clip(out, 0, 255).astype(np.uint8), mask)


def cropped(
    gray: np.ndarray, severity: float, config: Config, rng: np.random.Generator
) -> Degraded:
    """Срез края вместе с текстом: кадр обрезается, страница становится меньше.

    Маска отмечает полосу у нового края — там штрихи рассечены, и патчи оттуда
    несут метку, а остальная страница нет.
    """
    height, width = gray.shape
    cut_frac = _lerp(config.synth.span("cropped", "cut_frac"), severity)
    side = int(rng.integers(0, 4))

    if side == 0:
        cut = max(1, int(width * cut_frac))
        out = gray[:, cut:].copy()
    elif side == 1:
        cut = max(1, int(width * cut_frac))
        out = gray[:, : width - cut].copy()
    elif side == 2:
        cut = max(1, int(height * cut_frac))
        out = gray[cut:, :].copy()
    else:
        cut = max(1, int(height * cut_frac))
        out = gray[: height - cut, :].copy()

    mask = np.zeros(out.shape, dtype=np.uint8)
    band = max(1, int(min(out.shape) * 0.05))
    if side == 0:
        mask[:, :band] = 255
    elif side == 1:
        mask[:, -band:] = 255
    elif side == 2:
        mask[:band, :] = 255
    else:
        mask[-band:, :] = 255
    return Degraded(out, mask)


DegradeFn = Callable[[np.ndarray, float, Config, np.random.Generator], Degraded]

DEGRADATIONS: dict[str, DegradeFn] = {
    "blur": blur,
    "glare": glare,
    "shadow": shadow,
    "skew": skew,
    "cropped": cropped,
    "low_resolution": low_resolution,
    "low_contrast": low_contrast,
    "noise": noise,
    "streaks": streaks,
}

# Дефекты, у которых есть маска области. Совпадает с data.aggregation.local.
LOCAL = frozenset({"glare", "shadow", "streaks", "cropped"})


def apply(
    gray: np.ndarray,
    labels: list[str],
    severities: dict[str, float],
    config: Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Optional[np.ndarray]]]:
    """Накладывает несколько дефектов подряд и возвращает маски локальных.

    Порядок применения не случаен: геометрия (`skew`, `cropped`, `low_resolution`)
    идёт первой, иначе маски уже наложенных локальных дефектов пришлось бы
    поворачивать и обрезать вместе с изображением.
    """
    geometric = [name for name in ("cropped", "skew", "low_resolution") if name in labels]
    rest = [name for name in labels if name not in geometric]

    masks: dict[str, Optional[np.ndarray]] = {}
    out = gray
    for name in geometric + rest:
        result = DEGRADATIONS[name](out, severities[name], config, rng)
        out = result.image
        # Маски ранее наложенных дефектов подгоняем под новый размер.
        for done, mask in list(masks.items()):
            if mask is not None and mask.shape != out.shape:
                masks[done] = cv2.resize(mask, out.shape[::-1], interpolation=cv2.INTER_NEAREST)
        masks[name] = result.mask if name in LOCAL else None
    return out, masks
