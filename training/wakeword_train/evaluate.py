"""Streaming evaluation with the shipped fire rules, DET sweep, and sensitivity calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import CLASSIFIER_FRAMES, FRAME_SAMPLES, SAMPLE_RATE
from .config import EvalConfig
from .store import FRAMES_PER_HOUR
from eval.metrics import poisson_upper_bound


def count_fires(scores: np.ndarray, threshold: float, consecutive: int, refractory_seconds: float) -> np.ndarray:
    """Frame indices that fire, for a constant threshold.

    Identical to training/eval/detector.py (and the C# WakeWordDetector) with no adaptive offset:
    N consecutive frames at or above threshold fire; frames inside the refractory period after a fire
    are ignored and break runs. Vectorised over runs so threshold sweeps over ~1M frames stay fast.
    """
    refractory_samples = round(refractory_seconds * SAMPLE_RATE)
    blocked_frames = -(-refractory_samples // FRAME_SAMPLES)  # frames after a fire that are still refractory
    above = np.concatenate([[False], scores >= threshold, [False]])
    edges = np.diff(above.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    fires: list[int] = []
    last = None
    for s, e in zip(starts.tolist(), ends.tolist()):
        j = s
        while True:
            if last is not None:
                j = max(j, last + max(blocked_frames, 1))
            f = j + consecutive - 1
            if f >= e:
                break
            fires.append(f)
            last = f
            j = f + 1
    return np.asarray(fires, dtype=np.int64)


@torch.no_grad()
def window_scores(model: torch.nn.Module, windows: np.ndarray | torch.Tensor, device: str, batch: int = 65536) -> np.ndarray:
    out = []
    for i in range(0, len(windows), batch):
        x = torch.as_tensor(np.asarray(windows[i:i + batch]) if not torch.is_tensor(windows) else windows[i:i + batch])
        out.append(model(x.to(device, torch.float32)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


@torch.no_grad()
def stream_scores(model: torch.nn.Module, stream: np.ndarray | torch.Tensor, device: str, batch: int = 65536) -> np.ndarray:
    """Score for every position of a [T, 96] embedding stream: score k uses embeddings k..k+15."""
    s = torch.as_tensor(stream) if not torch.is_tensor(stream) else stream
    s = s.to(device)
    n = s.shape[0] - CLASSIFIER_FRAMES + 1
    out = []
    for i in range(0, max(n, 0), batch):
        idx = torch.arange(i, min(i + batch, n), device=device)[:, None] + torch.arange(CLASSIFIER_FRAMES, device=device)
        out.append(model(s[idx].float()).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


@dataclass
class DetPoint:
    threshold: float
    recall: float
    fa_per_hour: float
    false_accepts: int


@dataclass
class PhraseEvaluation:
    phrase: str
    negative_hours: float
    positives: int
    det: list[DetPoint]
    sensitivity_table: list[tuple[float, float]]  # (sensitivity, threshold)
    default_sensitivity: float
    default_threshold: float
    default_recall: float
    default_fa_per_hour: float

    def to_json(self) -> dict:
        d = asdict(self)
        d["sensitivity_table"] = [list(p) for p in self.sensitivity_table]
        return d


# Down to the detectors' min_threshold (0.01): real voices score lower than the synthetic validation clips.
THRESHOLDS = np.unique(np.round(np.concatenate([
    np.arange(0.01, 0.02, 0.0025), np.arange(0.02, 0.99, 0.01), np.arange(0.99, 0.9995, 0.001)]), 4))


def sweep(
    positive_scores: list[np.ndarray],
    negative_scores: np.ndarray,
    cfg: EvalConfig,
    thresholds: np.ndarray = THRESHOLDS,
) -> list[DetPoint]:
    hours = len(negative_scores) / FRAMES_PER_HOUR
    points = []
    for t in thresholds:
        detected = sum(
            1 for s in positive_scores if len(count_fires(s, t, cfg.consecutive_frames, cfg.refractory_seconds)) > 0
        )
        fa = len(count_fires(negative_scores, t, cfg.consecutive_frames, cfg.refractory_seconds))
        points.append(DetPoint(float(t), detected / max(1, len(positive_scores)), fa / hours if hours else math.inf, fa))
    return points


def target_fa_per_hour(sensitivity: float, cfg: EvalConfig) -> float:
    lo, hi = math.log10(cfg.fa_per_hour_at_sensitivity_0), math.log10(cfg.fa_per_hour_at_sensitivity_1)
    return 10 ** (lo + sensitivity * (hi - lo))


def calibrate(det: list[DetPoint], cfg: EvalConfig, negative_hours: float) -> list[tuple[float, float]]:
    """Sensitivity 0..1 (step 0.1) -> the lowest threshold whose false-accept rate is within that sensitivity's
    target at 95% confidence (Poisson upper bound, as in Section 3).

    Like Picovoice's sensitivity: higher means fewer misses and more false accepts. A target the validation
    audio is too short to prove gets the strictest threshold, never a guess.
    """
    by_threshold = sorted(det, key=lambda p: p.threshold)
    table = []
    for s in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        target = target_fa_per_hour(float(s), cfg)
        ok = [p.threshold for p in by_threshold if negative_hours > 0
              and poisson_upper_bound(p.false_accepts) / negative_hours <= target]
        table.append((float(s), ok[0] if ok else by_threshold[-1].threshold))
    # Monotonic: more sensitive never means a higher threshold.
    for i in range(1, len(table)):
        table[i] = (table[i][0], min(table[i][1], table[i - 1][1]))
    return table


def threshold_for(sensitivity: float, table: list[tuple[float, float]]) -> float:
    """Linear interpolation in the calibration table; the apps do the same."""
    s = min(max(sensitivity, 0.0), 1.0)
    for (s0, t0), (s1, t1) in zip(table, table[1:]):
        if s <= s1:
            return t0 + (t1 - t0) * (s - s0) / (s1 - s0)
    return table[-1][1]


def evaluate_phrase(
    phrase: str,
    positive_scores: list[np.ndarray],
    negative_scores: np.ndarray,
    cfg: EvalConfig,
) -> PhraseEvaluation:
    det = sweep(positive_scores, negative_scores, cfg)
    table = calibrate(det, cfg, len(negative_scores) / FRAMES_PER_HOUR)
    t = threshold_for(cfg.default_sensitivity, table)
    nearest = min(det, key=lambda p: abs(p.threshold - t))
    return PhraseEvaluation(
        phrase=phrase,
        negative_hours=len(negative_scores) / FRAMES_PER_HOUR,
        positives=len(positive_scores),
        det=det,
        sensitivity_table=table,
        default_sensitivity=cfg.default_sensitivity,
        default_threshold=round(t, 4),
        default_recall=nearest.recall,
        default_fa_per_hour=nearest.fa_per_hour,
    )
