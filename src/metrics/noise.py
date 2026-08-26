"""Оценка шума: σ по высокочастотному отклику и доля высокочастотной энергии."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.imaging import LAPLACE_MASK, MASK_NORM, estimate_noise_sigma

# Робастная оценка живёт в src/imaging: тем же порогом пользуется бинаризация чернил.
noise_sigma = estimate_noise_sigma


def _highpass(gray: np.ndarray) -> np.ndarray:
    response = cv2.filter2D(
        gray.astype(np.float64), ddepth=-1, kernel=LAPLACE_MASK, borderType=cv2.BORDER_REFLECT
    )
    return response[1:-1, 1:-1]  # рамка: там свёртка неполная


def noise_sigma_immerkaer(gray: np.ndarray) -> float:
    """Оригинальная оценка Immerkær — оставлена для сравнения в записке."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    response = _highpass(gray)
    height, width = response.shape
    return float(np.sum(np.abs(response)) * math.sqrt(0.5 * math.pi) / (MASK_NORM * width * height))


def high_freq_energy(gray: np.ndarray) -> float:
    """Доля энергии в верхней половине спектра. Растёт и от шума, и от зерна."""
    if gray.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft2(gray.astype(np.float64) - float(np.mean(gray))))
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    half = spectrum.shape[1] // 2
    return float(np.sum(spectrum[:, half:]) / total)
