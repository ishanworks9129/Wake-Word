"""ONNX export, parity checks, golden vectors and the model package the apps load."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import CLASSIFIER_FRAMES, EMBEDDING_DIM, FRAME_SAMPLES, SAMPLE_RATE
from .config import Config
from .features import MEL_STEP, MEL_WINDOW, FeatureExtractor

PACKAGE_FORMAT = "wakeword-models/1"


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    model = model.cpu().eval()
    dummy = torch.zeros(1, CLASSIFIER_FRAMES, EMBEDDING_DIM)
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=["input"], output_names=["score"],
        dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def check_parity(model: torch.nn.Module, path: Path, windows: np.ndarray, tolerance: float = 1e-4) -> float:
    """Max absolute difference between PyTorch and ONNX Runtime scores on real windows."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = windows.astype(np.float32)
    onnx_scores = session.run(None, {"input": x})[0]
    with torch.no_grad():
        torch_scores = model.cpu().eval()(torch.from_numpy(x)).numpy()
    diff = float(np.abs(onnx_scores - torch_scores).max())
    if diff > tolerance:
        raise RuntimeError(f"ONNX parity failed for {path.name}: max diff {diff}")
    return diff


def golden_vectors(fx: FeatureExtractor, classifiers: dict[str, Path], audio: np.ndarray) -> dict:
    """Runs the exact runtime streaming pipeline over audio and records every intermediate the apps must match."""
    sessions = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in classifiers.items()}
    stream = fx.streaming()
    chunks = []
    n = len(audio) // FRAME_SAMPLES
    for i in range(n):
        produced = stream.push(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
        x = stream.classifier_input()
        entry = {"chunk": i, "embedding_produced": produced}
        if produced:
            entry["embedding_first8"] = [round(float(v), 5) for v in stream.embeddings[-1][:8]]
        if x is not None:
            entry["scores"] = {k: round(float(s.run(None, {"input": x})[0][0, 0]), 6) for k, s in sessions.items()}
        chunks.append(entry)
    return {
        "description": "Runtime feature + classifier golden vectors. Feed audio in 1280-sample chunks; "
                       "C# and web implementations must match scores within 1e-3.",
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "audio_int16": audio[: n * FRAME_SAMPLES].astype(int).tolist(),
        "chunks": chunks,
    }


# The pipeline's "val" split is the plan's dev split (5.3): it picks thresholds. "test" is kept for real recordings.
MANIFEST_SPLIT = {"train": "train", "val": "dev"}


def write_dataset_manifest(
    cfg: Config,
    hours: dict[str, float],
    path: Path,
    tts_seconds: dict[tuple[int, str, str], float],
    rir_seconds: float,
) -> None:
    """Corpus-level rows that pass training/data/manifest.py: one per voice and use, negative source and RIR set.

    tts_seconds maps (voice index, kind, split) to seconds of synthetic clips; kind is "positive" or
    "hard_negative" (near-misses), split "train" or "dev" (held-out speakers). Sources that contributed
    no audio are left out.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    rows = []

    def row(path, kind, split, source, url, licensed, seconds, speaker=""):
        rows.append({"path": path, "kind": kind, "split": split, "source": source, "url": url,
                     "license": licensed.license, "license_url": licensed.license_url, "retrieved": today,
                     "duration_seconds": round(seconds, 1), "speaker_id": speaker, "release_id": ""})

    for (i, kind, split), seconds in sorted(tts_seconds.items()):
        v = cfg.tts.voices[i]
        if seconds > 0:
            # Synthetic speakers are tracked per voice; held-out ones get their own id so they never share a split.
            speaker = (v.name + ("/held-out" if split == "dev" else "")) if kind == "positive" else ""
            row(f"tts/{v.name}/{kind}/{split}", kind, split, f"Piper voice {v.name}", v.model_url, v, seconds, speaker)
    for src in cfg.negatives:
        if hours.get(src.name, 0.0) > 0:
            row(f"negatives/{src.name}", "negative", MANIFEST_SPLIT[src.split], src.name, src.source_url or src.urls[0],
                src, hours[src.name] * 3600)
    if rir_seconds > 0:
        row("rir/mit", "rir", "train", "MIT Acoustical Reverberation Scene Statistics Survey", cfg.rir_url, cfg.rir, rir_seconds)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_package(
    cfg: Config,
    out_dir: Path,
    feature_models: dict[str, Path],
    classifiers: dict[str, Path],
    evaluations: dict[str, dict],
    hours: dict[str, float],
    golden: dict,
    notes: list[str],
    tts_seconds: dict[tuple[int, str, str], float],
    rir_seconds: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, p in {**feature_models, **classifiers}.items():
        shutil.copy(p, out_dir / p.name)

    keywords = []
    for phrase in cfg.phrases:
        ev = evaluations[phrase.id]
        keywords.append({
            "id": phrase.id,
            "phrase": phrase.display,
            "model": classifiers[phrase.id].name,
            "input": "input",
            "output": "score",
            "default_sensitivity": cfg.eval.default_sensitivity,
            "sensitivity_table": ev["sensitivity_table"],
            # Thresholds come from the calibration table, so the app's clamp must not cut them off.
            "detector": {"consecutive_frames": cfg.eval.consecutive_frames, "refractory_seconds": cfg.eval.refractory_seconds,
                         "min_threshold": 0.01, "max_threshold": 0.9999},
            "transcript_variants": phrase.transcript_variants,
            "validation": {
                "threshold": ev["default_threshold"],
                "recall": ev["default_recall"],
                "fa_per_hour": ev["default_fa_per_hour"],
                "negative_hours": ev["negative_hours"],
                "positives": ev["positives"],
            },
        })

    manifest = {
        "format": PACKAGE_FORMAT,
        "run": cfg.run_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "features": {
            "melspectrogram": feature_models["melspectrogram"].name,
            "embedding": feature_models["embedding"].name,
            "mel_transform": "x / 10 + 2",
            "mel_context_samples": 480,
            "embedding_window": MEL_WINDOW,
            "embedding_step": MEL_STEP,
            "classifier_frames": CLASSIFIER_FRAMES,
        },
        "keywords": keywords,
        "training_hours": {k: round(v, 2) for k, v in hours.items()},
        "notes": notes,
    }
    (out_dir / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "golden.json").write_text(json.dumps(golden), encoding="utf-8")
    write_dataset_manifest(cfg, hours, out_dir / "DATASET_MANIFEST.csv", tts_seconds, rir_seconds)
    (out_dir / "METRICS.md").write_text(metrics_markdown(cfg, evaluations, hours, notes), encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            z.write(p, p.name)
    return zip_path


def metrics_markdown(cfg: Config, evaluations: dict[str, dict], hours: dict[str, float], notes: list[str]) -> str:
    lines = [f"# Wake word models: {cfg.run_name}", ""]
    lines += [f"- {n}" for n in notes] + [""]
    lines += ["| Keyword | Sensitivity | Threshold | Recall (held-out synthetic voices) | False accepts / hour | Negative hours |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in cfg.phrases:
        ev = evaluations[p.id]
        lines.append(f"| {p.display} | {ev['default_sensitivity']} | {ev['default_threshold']:.3f} | "
                     f"{ev['default_recall']:.1%} | {ev['default_fa_per_hour']:.2f} | {ev['negative_hours']:.1f} |")
    lines += ["", "## Sensitivity calibration", "", "| Sensitivity | " + " | ".join(p.display for p in cfg.phrases) + " |",
              "| --- |" + " --- |" * len(cfg.phrases)]
    tables = {p.id: dict((round(s, 1), t) for s, t in evaluations[p.id]["sensitivity_table"]) for p in cfg.phrases}
    for s in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        lines.append(f"| {s:.1f} | " + " | ".join(f"{tables[p.id][float(s)]:.3f}" for p in cfg.phrases) + " |")
    lines += ["", "## Training audio (hours)", ""] + [f"- {k}: {v:.1f}" for k, v in hours.items()]
    return "\n".join(lines) + "\n"
