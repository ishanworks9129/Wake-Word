"""Training-time augmentation (plan 5.4): rooms, noise, gain, microphone band-limiting, speed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, resample, sosfilt

from . import SAMPLE_RATE
from .config import AugmentConfig


class NoiseBank:
    """Concatenated int16 noise recordings; random segments are drawn for mixing."""

    def __init__(self, audio: np.ndarray):
        if audio.size < SAMPLE_RATE:
            raise ValueError("noise bank needs at least 1 s of audio")
        self.audio = audio.astype(np.int16)

    def segment(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n >= self.audio.size:
            reps = int(np.ceil(n / self.audio.size))
            return np.tile(self.audio, reps)[:n].astype(np.float32) / 32768.0
        start = int(rng.integers(0, self.audio.size - n))
        return self.audio[start:start + n].astype(np.float32) / 32768.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.audio)

    @staticmethod
    def load(path: Path) -> "NoiseBank":
        return NoiseBank(np.load(path))


def colored_noise(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """White, pink and brown noise, int16; licence-free filler for the noise bank."""
    n = int(seconds * SAMPLE_RATE)
    parts = []
    for exponent in (0.0, 1.0, 2.0):
        spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
        f = np.arange(n // 2 + 1, dtype=np.float64)
        f[0] = 1.0
        x = np.fft.irfft(spectrum / f ** (exponent / 2), n)
        parts.append(x / (np.abs(x).max() + 1e-9) * 0.3)
    return (np.concatenate(parts) * 32767).astype(np.int16)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


class Augmenter:
    def __init__(self, cfg: AugmentConfig, noise: NoiseBank, rirs: list[np.ndarray], window_samples: int, seed: int):
        self.cfg = cfg
        self.noise = noise
        self.rirs = rirs
        self.window = window_samples
        self.rng = np.random.default_rng(seed)

    def __call__(self, clip: np.ndarray, end_offset: int | None = None) -> np.ndarray:
        """Places a clip in a window_samples window so it ends end_offset samples before the window end,
        then augments. Returns int16 [window_samples]."""
        rng, cfg = self.rng, self.cfg
        x = clip.astype(np.float32) / 32768.0

        if rng.random() < cfg.speed_prob:
            rate = rng.uniform(*cfg.speed_range)
            x = resample(x, max(1, int(len(x) / rate))).astype(np.float32)

        if self.rirs and rng.random() < cfg.rir_prob:
            rir = self.rirs[int(rng.integers(len(self.rirs)))]
            wet = fftconvolve(x, rir)[: len(x) + SAMPLE_RATE // 4]
            x = (wet * (np.abs(x).max() / (np.abs(wet).max() + 1e-9))).astype(np.float32)

        if end_offset is None:
            end_offset = int(rng.uniform(*cfg.trailing_seconds) * SAMPLE_RATE)
        x = x[-(self.window - end_offset):] if len(x) > self.window - end_offset else x
        out = np.zeros(self.window, dtype=np.float32)
        end = self.window - end_offset
        out[end - len(x):end] = x
        speech_rms = _rms(x)

        if rng.random() < cfg.noise_prob:
            snr = rng.uniform(*cfg.snr_db)
            noise = self.noise.segment(self.window, rng)
            noise *= speech_rms / (_rms(noise) * 10 ** (snr / 20))
            out += noise
        out += rng.normal(scale=1e-4, size=self.window).astype(np.float32)  # never exact digital silence

        if rng.random() < cfg.band_limit_prob:
            low = rng.uniform(80, 400)
            high = rng.uniform(3400, 7600)
            out = sosfilt(butter(4, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos"), out).astype(np.float32)

        out *= 10 ** (rng.uniform(*cfg.gain_db) / 20)
        peak = np.abs(out).max()
        if peak > 0.99 and rng.random() < 0.7:
            out *= 0.99 / peak  # mostly avoid clipping; keep some clipped examples
        return (np.clip(out, -1, 1) * 32767).astype(np.int16)
