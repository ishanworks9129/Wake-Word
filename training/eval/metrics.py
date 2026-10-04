"""Section 3 acceptance metrics.

A band passes only if the 95% upper confidence bound on false accepts per hour is at or under target,
not the raw average, so a lucky 20-hour run cannot pass a 0.2/hour target.

Usage:
    python -m eval.metrics results.json

results.json:
    {"quiet":    {"negative_hours": 100, "false_accepts": 9,  "positives_total": 300, "positives_detected": 293},
     "moderate": {...}, "loud": {...}}
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BandTarget:
    max_fa_per_hour: float
    max_miss_rate: float
    min_negative_hours: float


TARGETS: dict[str, BandTarget] = {
    "quiet": BandTarget(max_fa_per_hour=0.2, max_miss_rate=0.03, min_negative_hours=100),
    "moderate": BandTarget(max_fa_per_hour=0.5, max_miss_rate=0.08, min_negative_hours=50),
    "loud": BandTarget(max_fa_per_hour=1.0, max_miss_rate=0.20, min_negative_hours=30),
}


def poisson_cdf(k: int, lam: float) -> float:
    if lam == 0:
        return 1.0
    log_lam = math.log(lam)
    return sum(math.exp(i * log_lam - lam - math.lgamma(i + 1)) for i in range(k + 1))


def poisson_upper_bound(k: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on a Poisson mean after observing k events."""
    if k < 0:
        raise ValueError("k must be non-negative")
    alpha = 1 - confidence
    lo, hi = 0.0, k + 10 * math.sqrt(k + 1) + 10
    for _ in range(200):
        mid = (lo + hi) / 2
        if poisson_cdf(k, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def max_false_accepts_to_pass(target_per_hour: float, hours: float, confidence: float = 0.95) -> int:
    """Largest observed false-accept count that still passes; -1 if the test set is too small to pass at all."""
    k = -1
    while poisson_upper_bound(k + 1, confidence) / hours <= target_per_hour:
        k += 1
    return k


@dataclass
class BandResult:
    name: str
    negative_hours: float
    false_accepts: int
    positives_total: int
    positives_detected: int
    fa_per_hour: float = 0.0
    fa_upper_per_hour: float = 0.0
    miss_rate: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_band(
    name: str,
    negative_hours: float,
    false_accepts: int,
    positives_total: int,
    positives_detected: int,
    target: BandTarget | None = None,
) -> BandResult:
    target = target or TARGETS[name]
    r = BandResult(name, negative_hours, false_accepts, positives_total, positives_detected)
    r.fa_per_hour = false_accepts / negative_hours
    r.fa_upper_per_hour = poisson_upper_bound(false_accepts) / negative_hours
    r.miss_rate = 1 - positives_detected / positives_total if positives_total else 1.0

    if negative_hours < target.min_negative_hours:
        r.failures.append(f"only {negative_hours:g} h of negatives; need {target.min_negative_hours:g} h")
    if r.fa_upper_per_hour > target.max_fa_per_hour:
        r.failures.append(f"false accepts {r.fa_upper_per_hour:.3f}/h (95% bound) > {target.max_fa_per_hour}/h")
    if r.miss_rate > target.max_miss_rate:
        r.failures.append(f"miss rate {r.miss_rate:.1%} > {target.max_miss_rate:.0%}")
    return r


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    results = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    ok = True
    print(f"{'band':<9} {'hours':>6} {'FA':>4} {'FA/h':>6} {'FA/h 95%':>9} {'miss':>6}  result")
    for name in TARGETS:
        if name not in results:
            print(f"{name:<9} missing")
            ok = False
            continue
        r = evaluate_band(name, **results[name])
        ok &= r.passed
        verdict = "PASS" if r.passed else "FAIL: " + "; ".join(r.failures)
        print(f"{name:<9} {r.negative_hours:>6g} {r.false_accepts:>4} {r.fa_per_hour:>6.3f} {r.fa_upper_per_hour:>9.3f} {r.miss_rate:>6.1%}  {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
