"""Python mirror of WakeWord.Core.Detection.WakeWordDetector.

Section 3 metrics must be measured with the same fire rules that ship, so this module and the C#
detector both run testdata/golden/detector_cases.json. Change one, change the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

FRAME_SAMPLES = 1280
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AdaptiveThreshold:
    points: tuple[tuple[float, float], ...] = ()
    min_threshold: float = 0.2
    max_threshold: float = 0.95

    def threshold_for(self, base: float, noise_floor_dbfs: float) -> float:
        offset = 0.0
        pts = self.points
        if pts:
            if noise_floor_dbfs <= pts[0][0]:
                offset = pts[0][1]
            elif noise_floor_dbfs >= pts[-1][0]:
                offset = pts[-1][1]
            else:
                for (lo_db, lo_off), (hi_db, hi_off) in zip(pts, pts[1:]):
                    if noise_floor_dbfs <= hi_db:
                        t = (noise_floor_dbfs - lo_db) / (hi_db - lo_db)
                        offset = lo_off + t * (hi_off - lo_off)
                        break
        return min(max(base + offset, self.min_threshold), self.max_threshold)


@dataclass(frozen=True)
class DetectorOptions:
    base_threshold: float = 0.5
    consecutive_frames: int = 3
    refractory_seconds: float = 1.5
    adaptive: AdaptiveThreshold = field(default_factory=AdaptiveThreshold)

    @staticmethod
    def from_json(o: dict) -> "DetectorOptions":
        a = o.get("adaptive", {})
        return DetectorOptions(
            base_threshold=o["base_threshold"],
            consecutive_frames=o["consecutive_frames"],
            refractory_seconds=o["refractory_seconds"],
            adaptive=AdaptiveThreshold(
                points=tuple((float(p[0]), float(p[1])) for p in a.get("points", [])),
                min_threshold=a.get("min_threshold", 0.2),
                max_threshold=a.get("max_threshold", 0.95),
            ),
        )


class Detector:
    def __init__(self, options: DetectorOptions, sample_rate: int = SAMPLE_RATE):
        if options.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be at least 1")
        self._o = options
        # Same rounding as C# Math.Round (banker's rounding); Python's round() matches.
        self._refractory_samples = round(options.refractory_seconds * sample_rate)
        self._run = 0
        self._last_fire: int | None = None

    def process(self, score: float, noise_floor_dbfs: float, frame_end_sample: int) -> bool:
        threshold = self._o.adaptive.threshold_for(self._o.base_threshold, noise_floor_dbfs)
        if self._last_fire is not None and frame_end_sample - self._last_fire < self._refractory_samples:
            self._run = 0
            return False
        if score < threshold:
            self._run = 0
            return False
        self._run += 1
        if self._run < self._o.consecutive_frames:
            return False
        self._run = 0
        self._last_fire = frame_end_sample
        return True

    def reset(self) -> None:
        self._run = 0


def fire_frames(
    scores: Sequence[float],
    noise_floor_dbfs: float | Sequence[float],
    options: DetectorOptions,
    frame_samples: int = FRAME_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
) -> list[int]:
    """Indices of frames that fire, for a continuous score stream where frame i ends at (i + 1) * frame_samples."""
    detector = Detector(options, sample_rate)
    fires = []
    for i, score in enumerate(scores):
        floor = noise_floor_dbfs if isinstance(noise_floor_dbfs, (int, float)) else noise_floor_dbfs[i]
        if detector.process(score, floor, (i + 1) * frame_samples):
            fires.append(i)
    return fires


def load_golden(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def golden_cases(path: Path) -> Iterable[tuple[str, list[int], list[int]]]:
    """Yields (name, expected, actual) for each golden case."""
    golden = load_golden(path)
    for case in golden["cases"]:
        actual = fire_frames(
            case["scores"],
            case["noise_floor_dbfs"],
            DetectorOptions.from_json(case["options"]),
            golden["frame_samples"],
            golden["sample_rate"],
        )
        yield case["name"], case["expected_fire_frames"], actual
