"""Tests a finished model package on your own recordings, on this machine: the audio never leaves it.

    python -m wakeword_train.test_package --package ../models/wakeword --folder "D:/meetings" --out D:/wakeword_test

Reports false accepts per hour at every sensitivity in the package's calibration, with the detector rules the
apps use (no adaptive offset), and lists every fire at the most sensitive setting in fires.csv to check by ear.
Embeddings are cached in --out, so testing a newer package on the same recordings only rescores them.
Recordings must contain no wake phrase; a meeting where someone says it should be left out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eval.metrics import poisson_upper_bound

from . import CLASSIFIER_FRAMES
from .evaluate import count_fires
from .features import FeatureExtractor
from .store import FRAMES_PER_HOUR
from .testset import clock, decode_media, media_files, score_seconds


def onnx_scores(session: ort.InferenceSession, input_name: str, emb: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Score for every position of a [T, 96] stream, as the apps compute it (embeddings k..k+15)."""
    n = emb.shape[0] - CLASSIFIER_FRAMES + 1
    if n <= 0:
        return np.empty(0, np.float32)
    offsets = np.arange(CLASSIFIER_FRAMES)
    out = []
    for i in range(0, n, batch):
        starts = np.arange(i, min(i + batch, n))
        out.append(session.run(None, {input_name: emb[starts[:, None] + offsets].astype(np.float32)})[0][:, 0])
    return np.concatenate(out)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embeddings(files: list[Path], folder: Path, fx: FeatureExtractor, features_id: str, cache: Path) -> list[tuple[str, np.ndarray]]:
    cache.mkdir(parents=True, exist_ok=True)
    streams, started = [], time.time()
    for i, f in enumerate(files):
        st = f.stat()
        key = hashlib.sha1(f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}|{features_id}".encode()).hexdigest()[:20]
        c = cache / f"{key}.npy"
        if c.exists():
            emb = np.load(c)
        else:
            emb = fx.embed_long(decode_media(f)).astype(np.float16)
            with open(c.with_suffix(".tmp"), "wb") as out:
                np.save(out, emb)
            c.with_suffix(".tmp").replace(c)
        streams.append((f.relative_to(folder).as_posix(), emb))
        hours = sum(e.shape[0] for _, e in streams) / FRAMES_PER_HOUR
        print(f"  {i + 1}/{len(files)} recordings, {hours:.1f} h ({(time.time() - started) / 60:.0f} min)", flush=True)
    return streams


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="unpacked model package (models.json and .onnx files)")
    ap.add_argument("--folder", required=True, type=Path, help="recordings with no wake phrase in them, searched recursively")
    ap.add_argument("--out", required=True, type=Path, help="where the report, fires.csv and the embedding cache go")
    args = ap.parse_args(argv)

    pkg = json.loads((args.package / "models.json").read_text(encoding="utf-8"))
    files = media_files(args.folder)
    if not files:
        raise SystemExit(f"no audio or video files in {args.folder}")
    feats = pkg["features"]
    mel, emb_model = args.package / feats["melspectrogram"], args.package / feats["embedding"]
    fx = FeatureExtractor(str(mel), str(emb_model))
    print(f"embedding {len(files)} recordings from {args.folder}")
    streams = embeddings(files, args.folder, fx, file_hash(mel)[:12] + file_hash(emb_model)[:12], args.out / "cache")
    hours = sum(e.shape[0] for _, e in streams) / FRAMES_PER_HOUR

    results, fires = {}, []
    for kw in pkg["keywords"]:
        session = ort.InferenceSession(str(args.package / kw["model"]), providers=["CPUExecutionProvider"])
        det = kw["detector"]
        scores = [(rel, onnx_scores(session, kw["input"], e)) for rel, e in streams]

        def fire_frames(sc: np.ndarray, threshold: float) -> np.ndarray:
            return count_fires(sc, threshold, det["consecutive_frames"], det["refractory_seconds"])

        rows = []
        for s, t in kw["sensitivity_table"]:
            n = sum(len(fire_frames(sc, t)) for _, sc in scores)
            rows.append({"sensitivity": s, "threshold": t, "false_accepts": n,
                         "fa_per_hour": n / hours, "fa_per_hour_upper": poisson_upper_bound(n) / hours})
        lowest = min(t for _, t in kw["sensitivity_table"])
        for rel, sc in scores:
            for k in fire_frames(sc, lowest).tolist():
                fires.append({"keyword": kw["id"], "file": rel, "time": clock(score_seconds(k)), "score": round(float(sc[k]), 3)})
        results[kw["id"]] = {"phrase": kw["phrase"], "default_sensitivity": kw["default_sensitivity"], "by_sensitivity": rows}

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "fires.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["keyword", "file", "time", "score"])
        w.writeheader()
        w.writerows(sorted(fires, key=lambda r: (r["file"], r["time"], r["keyword"])))
    summary = {"package_run": pkg.get("run"), "package_created": pkg.get("created"), "recordings": len(files),
               "hours": hours, "keywords": results}
    (args.out / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ids = list(results)
    lines = [f"# Test on your own recordings: {len(files)} recordings, {hours:.1f} h", "",
             f"Package `{pkg.get('run')}` created {pkg.get('created')}. False accepts per hour, 95% upper bound in "
             "brackets; plan Section 3 targets at most 0.2 (quiet) to 1.0 (loud).", "",
             "| Sensitivity | " + " | ".join(results[i]["phrase"] for i in ids) + " |", "| --- |" + " --- |" * len(ids)]
    for j, r in enumerate(results[ids[0]]["by_sensitivity"]):
        cells = [results[i]["by_sensitivity"][j] for i in ids]
        lines.append(f"| {r['sensitivity']:.1f} | " + " | ".join(
            f"{c['false_accepts']} = {c['fa_per_hour']:.2f} ({c['fa_per_hour_upper']:.2f})" for c in cells) + " |")
    lines += ["", f"{len(fires)} fires at the most sensitive setting are listed in fires.csv: listen to each to check "
              "whether someone actually said the wake phrase."]
    report = "\n".join(lines) + "\n"
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
