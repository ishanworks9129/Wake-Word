"""Tests a finished model package on your own recordings, on this machine: the audio never leaves it.

    python -m wakeword_train.test_package --package ../models/wakeword --folder "D:/meetings" --out D:/wakeword_test
    python -m wakeword_train.test_package --package <newer package> --out D:/wakeword_test   # recordings no longer needed

Reports false accepts per hour at every sensitivity in the package's calibration, with the detector rules the
apps use (no adaptive offset), and lists every fire at the most sensitive setting in fires.csv to check by ear.
Embeddings are cached in --out (with cache/index.json naming each recording), so a newer package is tested on the
same recordings in minutes, and without --folder once they are processed: the audio itself can then be deleted.
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


def load_index(cache: Path) -> dict[str, dict]:
    """cache/index.json: embedding file key -> {"file": recording's path in the folder, "features": feature models id}."""
    path = cache / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_index(cache: Path, index: dict[str, dict]) -> None:
    tmp = cache / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(cache / "index.json")


def embeddings(files: list[Path], folder: Path, fx: FeatureExtractor, features_id: str, cache: Path) -> list[tuple[str, np.ndarray]]:
    cache.mkdir(parents=True, exist_ok=True)
    index = load_index(cache)
    streams, started = [], time.time()
    for i, f in enumerate(files):
        st = f.stat()
        rel = f.relative_to(folder).as_posix()
        key = hashlib.sha1(f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}|{features_id}".encode()).hexdigest()[:20]
        c = cache / f"{key}.npy"
        if c.exists():
            emb = np.load(c)
        else:
            emb = fx.embed_long(decode_media(f)).astype(np.float16)
            with open(c.with_suffix(".tmp"), "wb") as out:
                np.save(out, emb)
            c.with_suffix(".tmp").replace(c)
        for old in [k for k, v in index.items() if v["file"] == rel and v["features"] == features_id and k != key]:
            (cache / f"{old}.npy").unlink(missing_ok=True)  # the recording changed since it was processed
            del index[old]
        index[key] = {"file": rel, "features": features_id}
        save_index(cache, index)
        streams.append((rel, emb))
        hours = sum(e.shape[0] for _, e in streams) / FRAMES_PER_HOUR
        print(f"  {i + 1}/{len(files)} recordings, {hours:.1f} h ({(time.time() - started) / 60:.0f} min)", flush=True)
    return streams


def cached_embeddings(cache: Path, features_id: str) -> list[tuple[str, np.ndarray]]:
    """Recordings processed earlier with these feature models, without needing the recordings themselves."""
    index = load_index(cache)
    entries = sorted((v["file"], k) for k, v in index.items() if v["features"] == features_id and (cache / f"{k}.npy").exists())
    return [(rel, np.load(cache / f"{k}.npy")) for rel, k in entries]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="unpacked model package (models.json and .onnx files)")
    ap.add_argument("--folder", type=Path,
                    help="recordings with no wake phrase in them, searched recursively; omit to reuse those already processed into --out")
    ap.add_argument("--out", required=True, type=Path, help="where the report, fires.csv and the embedding cache go")
    args = ap.parse_args(argv)

    pkg = json.loads((args.package / "models.json").read_text(encoding="utf-8"))
    feats = pkg["features"]
    mel, emb_model = args.package / feats["melspectrogram"], args.package / feats["embedding"]
    features_id = file_hash(mel)[:12] + file_hash(emb_model)[:12]
    if args.folder:
        files = media_files(args.folder)
        if not files:
            raise SystemExit(f"no audio or video files in {args.folder}")
        print(f"embedding {len(files)} recordings from {args.folder}")
        streams = embeddings(files, args.folder, FeatureExtractor(str(mel), str(emb_model)), features_id, args.out / "cache")
    else:
        streams = cached_embeddings(args.out / "cache", features_id)
        if not streams:
            raise SystemExit(f"no recordings processed with this package's feature models in {args.out / 'cache'}; pass --folder")
        print(f"using {len(streams)} recordings already processed in {args.out / 'cache'}")
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
    summary = {"package_run": pkg.get("run"), "package_created": pkg.get("created"), "recordings": len(streams),
               "hours": hours, "keywords": results}
    (args.out / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ids = list(results)
    lines = [f"# Test on your own recordings: {len(streams)} recordings, {hours:.1f} h", "",
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
