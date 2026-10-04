"""Validates DATASET_MANIFEST.csv (plan 4.1).

Every audio file used for training or evaluation has one row. The check fails on:
  - non-commercial (NC) or no-derivatives (ND) licenses, and any license not on the allow list
  - share-alike licenses, unless --allow-share-alike (needs a legal decision first)
  - missing provenance (url, license_url, retrieved date) or non-positive durations
  - duplicate paths
  - a positive-recording speaker appearing in both the test split and train/dev (plan 5.2: held-out speakers)
  - own recordings without a signed contributor release id (plan 4.2)

Usage:
    python -m data.manifest DATASET_MANIFEST.csv [--allow-share-alike]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

COLUMNS = (
    "path", "kind", "split", "source", "url", "license", "license_url",
    "retrieved", "duration_seconds", "speaker_id", "release_id",
)
KINDS = {"negative", "hard_negative", "positive", "noise", "rir"}
SPLITS = {"train", "dev", "test"}

ALLOWED_LICENSES = {
    "CC0-1.0", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0", "MIT", "PDDL-1.0", "Public-Domain",
    "Contributor-Release",  # our own recordings under the signed release in plan 4.2
}
SHARE_ALIKE_LICENSES = {"CC-BY-SA-3.0", "CC-BY-SA-4.0"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    hours: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    hours_by_license: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def ok(self) -> bool:
        return not self.errors


def _license_problem(license_id: str, allow_share_alike: bool) -> str | None:
    upper = license_id.upper()
    if "-NC" in upper or upper.startswith("NC"):
        return f"non-commercial license {license_id}"
    if "-ND" in upper:
        return f"no-derivatives license {license_id}"
    if license_id in SHARE_ALIKE_LICENSES:
        return None if allow_share_alike else f"share-alike license {license_id} needs legal sign-off (--allow-share-alike)"
    if license_id not in ALLOWED_LICENSES:
        return f"license {license_id!r} is not on the allow list"
    return None


def validate(rows: list[dict[str, str]], allow_share_alike: bool = False) -> Report:
    report = Report()
    seen_paths: set[str] = set()
    speaker_splits: dict[str, set[str]] = defaultdict(set)

    for n, row in enumerate(rows, start=2):  # row 1 is the header
        where = f"row {n} ({row.get('path', '?')})"
        missing = [c for c in COLUMNS if c not in row]
        if missing:
            report.errors.append(f"{where}: missing columns {missing}")
            continue

        path = row["path"].strip()
        if not path:
            report.errors.append(f"{where}: empty path")
        elif path in seen_paths:
            report.errors.append(f"{where}: duplicate path")
        seen_paths.add(path)

        kind, split = row["kind"].strip(), row["split"].strip()
        if kind not in KINDS:
            report.errors.append(f"{where}: kind {kind!r} not in {sorted(KINDS)}")
        if split not in SPLITS:
            report.errors.append(f"{where}: split {split!r} not in {sorted(SPLITS)}")

        license_id = row["license"].strip()
        if problem := _license_problem(license_id, allow_share_alike):
            report.errors.append(f"{where}: {problem}")
        if license_id == "Contributor-Release" and not row["release_id"].strip():
            report.errors.append(f"{where}: own recording without a signed release_id")

        for col in ("source", "url", "license_url"):
            if not row[col].strip():
                report.errors.append(f"{where}: empty {col}")
        try:
            date.fromisoformat(row["retrieved"].strip())
        except ValueError:
            report.errors.append(f"{where}: retrieved {row['retrieved']!r} is not an ISO date")

        try:
            seconds = float(row["duration_seconds"])
            if seconds <= 0:
                raise ValueError
        except ValueError:
            report.errors.append(f"{where}: duration_seconds must be a positive number")
            seconds = 0.0

        if kind == "positive":
            speaker = row["speaker_id"].strip()
            if not speaker:
                report.errors.append(f"{where}: positive recording without speaker_id")
            else:
                speaker_splits[speaker].add(split)

        report.hours[(kind, split)] += seconds / 3600
        report.hours_by_license[license_id] += seconds / 3600

    for speaker, splits in sorted(speaker_splits.items()):
        if "test" in splits and splits - {"test"}:
            report.errors.append(f"speaker {speaker} is in test and {sorted(splits - {'test'})}; test speakers must be held out")

    return report


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    report = validate(read_rows(Path(args[0])), allow_share_alike="--allow-share-alike" in argv)

    print("Hours by kind and split:")
    for (kind, split), hours in sorted(report.hours.items()):
        print(f"  {kind:<14} {split:<6} {hours:10.2f}")
    print("Hours by license:")
    for license_id, hours in sorted(report.hours_by_license.items()):
        print(f"  {license_id:<20} {hours:10.2f}")
    for e in report.errors:
        print(f"ERROR {e}")
    print(f"{len(report.errors)} error(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
