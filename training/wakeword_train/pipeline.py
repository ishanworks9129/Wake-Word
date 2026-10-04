"""Resumable training pipeline.

    python -m wakeword_train.pipeline --config configs/uno.yaml --work /content/drive/MyDrive/wakeword/uno --steps all

Steps (in order): setup, preview, tts, negatives, features, train, evaluate, export.
Each step writes work/steps/<step>.done; finished steps are skipped unless --force. Long steps (tts,
negatives) also resume part-way, so a dropped Colab session loses minutes, not hours.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import yaml

from . import SAMPLE_RATE
from .augment import Augmenter, NoiseBank, colored_noise
from .config import AugmentConfig, Config, load_config
from .features import FeatureExtractor
from .sources import download, iter_tar_audio, load_rirs
from .store import EmbeddingStore, FRAMES_PER_HOUR
from .tts import ClipSet, PiperSynth, generate, is_held_out

STEPS = ["setup", "preview", "tts", "negatives", "features", "train", "evaluate", "export"]
TTS_PART = 2000  # clips per resumable part
EVAL_WINDOW = 4 * SAMPLE_RATE  # validation positives: clip ends 1 s before the end of a 4 s window
EVAL_TRAILING = SAMPLE_RATE


class Run:
    def __init__(self, cfg: Config, work: Path):
        self.cfg = cfg
        self.work = work
        for d in ("assets", "steps", "clips", "noise", "features", "models", "eval", "export", "preview"):
            (work / d).mkdir(parents=True, exist_ok=True)

    # ---- paths and shared resources

    def asset(self, name: str) -> Path:
        return self.work / "assets" / name

    def voice_paths(self) -> list[tuple[Path, Path]]:
        return [(self.asset(f"voices/{v.name}.onnx"), self.asset(f"voices/{v.name}.onnx.json")) for v in self.cfg.tts.voices]

    def features(self) -> FeatureExtractor:
        return FeatureExtractor(str(self.asset("melspectrogram.onnx")), str(self.asset("embedding_model.onnx")),
                                self.cfg.feature_threads, self.cfg.features_use_cuda)

    def synths(self) -> list[PiperSynth]:
        return [PiperSynth(m, c, self.cfg.tts.use_cuda) for m, c in self.voice_paths()]

    def store(self, split: str) -> EmbeddingStore:
        return EmbeddingStore(self.work / "negatives" / split)

    def augmenter(self, seed: int, cfg: AugmentConfig | None = None, window: int | None = None) -> Augmenter:
        noise = NoiseBank.load(self.work / "noise" / "bank.npy")
        rirs = load_rirs(self.asset("rir.zip"))
        return Augmenter(cfg or self.cfg.augment, noise, rirs, window or self.cfg.window_samples, seed)

    def clips(self, phrase_id: str, kind: str) -> ClipSet:
        return ClipSet.load(self.work / "clips" / f"{phrase_id}_{kind}.npz")

    def hours(self) -> dict[str, float]:
        out = {}
        for split in ("train", "val"):
            store = self.store(split)
            for src in self.cfg.negatives:
                if src.split == split:
                    out[src.name] = store.hours(src.name)
        return out

    def done(self, step: str) -> bool:
        return (self.work / "steps" / f"{step}.done").exists()

    def mark(self, step: str, summary: dict) -> None:
        (self.work / "steps" / f"{step}.done").write_text(json.dumps(summary, indent=2, default=str))

    # ---- steps

    def setup(self) -> dict:
        cfg = self.cfg
        download(cfg.melspectrogram_url, self.asset("melspectrogram.onnx"))
        download(cfg.embedding_url, self.asset("embedding_model.onnx"))
        download(cfg.rir_url, self.asset("rir.zip"))
        for v, (model, config) in zip(cfg.tts.voices, self.voice_paths()):
            download(v.model_url, model)
            download(v.config_url, config)
        rirs = load_rirs(self.asset("rir.zip"))
        fx = self.features()
        probe = fx.embed(np.zeros(cfg.window_samples, dtype=np.int16))
        if probe.shape != (16, 96):
            raise RuntimeError(f"feature models produced {probe.shape} for a {cfg.window_samples}-sample window; expected (16, 96)")
        return {"rirs": len(rirs), "voices": [v.name for v in cfg.tts.voices]}

    def preview(self, per_text: int = 2) -> dict:
        """A few samples per TTS spelling, to check pronunciation before generating thousands."""
        rng = np.random.default_rng(0)
        files = []
        for synth, (path, _) in zip(self.synths(), self.voice_paths()):
            for p in self.cfg.phrases:
                for t, text in enumerate(p.tts_texts):
                    for k in range(per_text):
                        audio = synth.synth(text, int(rng.integers(synth.num_speakers)), 1.0, 0.667, 0.8)
                        out = self.work / "preview" / f"{p.id}_{t}_{k}_{path.stem}.wav"
                        write_wav(out, audio)
                        files.append(str(out))
        return {"files": files}

    def tts(self) -> dict:
        cfg = self.cfg
        synths = None
        summary = {}
        for p in cfg.phrases:
            for kind, texts, n in (("pos", p.tts_texts, cfg.tts.positives_per_phrase),
                                   ("adv", p.adversarial_texts, cfg.tts.adversarial_per_phrase)):
                final = self.work / "clips" / f"{p.id}_{kind}.npz"
                if final.exists():
                    summary[f"{p.id}_{kind}"] = len(ClipSet.load(final))
                    continue
                parts_dir = self.work / "clips" / f"{p.id}_{kind}_parts"
                parts_dir.mkdir(exist_ok=True)
                n_parts = -(-n // TTS_PART)
                for part in range(n_parts):
                    part_path = parts_dir / f"part_{part:03d}.npz"
                    if part_path.exists():
                        continue
                    synths = synths or self.synths()
                    count = min(TTS_PART, n - part * TTS_PART)
                    rng = np.random.default_rng([cfg.train.seed, hash_id(p.id), 0 if kind == "pos" else 1, part])
                    started = time.time()
                    clips = generate(synths, texts, count, cfg.tts, rng)
                    clips.save(part_path)
                    print(f"{p.id} {kind}: part {part + 1}/{n_parts} ({len(clips)} clips, {time.time() - started:.0f}s)")
                parts = [ClipSet.load(f) for f in sorted(parts_dir.glob("part_*.npz"))]
                merged = ClipSet.from_clips(
                    [c.clip(i) for c in parts for i in range(len(c))],
                    np.concatenate([c.speakers for c in parts]),
                    np.concatenate([c.texts for c in parts]),
                    texts,
                )
                merged.save(final)
                summary[f"{p.id}_{kind}"] = len(merged)
        return summary

    def negatives(self) -> dict:
        cfg = self.cfg
        fx = self.features()
        noise_dir = self.work / "noise" / "parts"
        noise_dir.mkdir(parents=True, exist_ok=True)
        noise_samples = sum(np.load(f, mmap_mode="r").shape[0] for f in noise_dir.glob("*.npy"))
        noise_target = int(cfg.augment.noise_bank_hours * 3600 * SAMPLE_RATE)
        noise_buf: list[np.ndarray] = []

        def flush_noise():
            nonlocal noise_buf
            if noise_buf:
                np.save(noise_dir / f"part_{len(list(noise_dir.glob('*.npy'))):04d}.npy", np.concatenate(noise_buf))
                noise_buf = []

        for src in cfg.negatives:
            store = self.store(src.split)
            for url in src.urls:
                if store.hours(src.name) >= src.max_hours:
                    break
                skip = store.source_progress(src.name)["next_member"].get(url, 0)
                accept = (lambda name, inc=src.include: not inc or any(s in name for s in inc))
                print(f"{src.name}: streaming {url} from member {skip} ({store.hours(src.name):.1f}/{src.max_hours} h)")
                last_report = time.time()
                for index, name, audio in iter_tar_audio(url, accept, skip_members=skip):
                    emb = fx.embed(audio)
                    if emb.shape[0]:
                        store.append(src.name, url, index, emb)
                    if noise_samples < noise_target and any(s in name for s in src.noise_bank):
                        noise_buf.append(audio)
                        noise_samples += audio.shape[0]
                        if sum(a.shape[0] for a in noise_buf) > 5 * 60 * SAMPLE_RATE:
                            flush_noise()
                    if time.time() - last_report > 60:
                        print(f"  {src.name}: {store.hours(src.name) + store._buf_frames / FRAMES_PER_HOUR:.2f} h")
                        last_report = time.time()
                    if store.hours(src.name) + store._buf_frames / FRAMES_PER_HOUR >= src.max_hours:
                        break
                store.flush()
                flush_noise()

        parts = [np.load(f) for f in sorted(noise_dir.glob("*.npy"))]
        rng = np.random.default_rng(cfg.train.seed)
        bank = np.concatenate(parts + [colored_noise(120, rng)])
        NoiseBank(bank).save(self.work / "noise" / "bank.npy")
        hours = self.hours()
        for split in ("train", "val"):
            if self.store(split).hours() == 0:
                raise RuntimeError(f"no {split} negative audio was collected; check the negatives sources")
        return {"hours": hours, "noise_bank_hours": bank.shape[0] / SAMPLE_RATE / 3600}

    def features_step(self) -> dict:
        cfg = self.cfg
        fx = self.features()
        summary = {}
        for p in cfg.phrases:
            aug = self.augmenter(seed=hash_id(p.id))
            pos = self.clips(p.id, "pos")
            held = is_held_out(pos.speakers, cfg.tts.holdout_every_nth_speaker)
            jobs = {
                "pos": (pos.subset(~held), cfg.augment.copies_per_clip),
                "adv": (self.clips(p.id, "adv"), cfg.augment.copies_per_clip),
                "valpos": (pos.subset(held), 1),
            }
            for kind, (clips, copies) in jobs.items():
                out = self.work / "features" / f"{p.id}_{kind}.npy"
                if out.exists():
                    summary[f"{p.id}_{kind}"] = int(np.load(out, mmap_mode="r").shape[0])
                    continue
                feats = []
                for start in range(0, len(clips), 512):
                    windows = np.stack([aug(clips.clip(i)) for _ in range(copies)
                                        for i in range(start, min(start + 512, len(clips)))])
                    feats.append(fx.embed_batch(windows).astype(np.float16))
                    print(f"{p.id} {kind}: {min(start + 512, len(clips))}/{len(clips)} clips")
                data = np.concatenate(feats) if feats else np.empty((0, 16, 96), np.float16)
                np.save(out, data)
                summary[f"{p.id}_{kind}"] = int(data.shape[0])
        return summary

    def train_step(self) -> dict:
        import torch

        from .trainer import train

        cfg = self.cfg
        neg = self.store("train").load()
        val_neg = self.store("val").load()
        summary = {}
        for p in cfg.phrases:
            out = self.work / "models" / f"{p.id}.pt"
            if out.exists():
                continue
            print(f"training {p.display}: negatives {neg.shape[0] / FRAMES_PER_HOUR:.1f} h, validation {val_neg.shape[0] / FRAMES_PER_HOUR:.1f} h")
            model, history = train(
                np.load(self.work / "features" / f"{p.id}_pos.npy"),
                np.load(self.work / "features" / f"{p.id}_adv.npy"),
                neg,
                np.load(self.work / "features" / f"{p.id}_valpos.npy"),
                val_neg,
                cfg.train,
                cfg.eval,
            )
            torch.save({"state": model.state_dict(), "layer_size": cfg.train.layer_size, "n_blocks": cfg.train.n_blocks}, out)
            (self.work / "models" / f"{p.id}_history.json").write_text(json.dumps(history, indent=1))
            summary[p.id] = history[-1] if history else {}
        return summary

    def load_model(self, phrase_id: str):
        import torch

        from .model import WakeWordDNN

        ckpt = torch.load(self.work / "models" / f"{phrase_id}.pt", map_location="cpu")
        model = WakeWordDNN(ckpt["layer_size"], ckpt["n_blocks"])
        model.load_state_dict(ckpt["state"])
        return model.eval()

    def eval_audio(self, phrase_id: str) -> list[np.ndarray]:
        """Held-out-speaker positives, each in a 4 s noisy window ending 1 s after the phrase."""
        cfg = self.cfg
        eval_aug = AugmentConfig(
            copies_per_clip=1, snr_db=cfg.eval.val_positive_snr_db, noise_prob=1.0, rir_prob=0.3,
            gain_db=(-6.0, 6.0), band_limit_prob=0.2, speed_prob=0.0, trailing_seconds=(1.0, 1.0),
        )
        aug = self.augmenter(seed=hash_id(phrase_id) + 7, cfg=eval_aug, window=EVAL_WINDOW)
        pos = self.clips(phrase_id, "pos")
        held = pos.subset(is_held_out(pos.speakers, cfg.tts.holdout_every_nth_speaker))
        return [aug(held.clip(i), end_offset=EVAL_TRAILING) for i in range(len(held))]

    def evaluate_step(self) -> dict:
        from .evaluate import evaluate_phrase, stream_scores

        cfg = self.cfg
        fx = self.features()
        val_neg = self.store("val").load()
        summary = {}
        for p in cfg.phrases:
            model = self.load_model(p.id)
            audio = self.eval_audio(p.id)
            if not audio:
                raise RuntimeError(f"{p.id}: no held-out positives; generate more clips or lower holdout_every_nth_speaker")
            feats = fx.embed_batch(np.stack(audio))
            positive_scores = [stream_scores(model, f, "cpu") for f in feats]
            negative_scores = stream_scores(model, val_neg, "cpu")
            ev = evaluate_phrase(p.id, positive_scores, negative_scores, cfg.eval)
            (self.work / "eval" / f"{p.id}.json").write_text(json.dumps(ev.to_json(), indent=1))
            summary[p.id] = {"threshold": ev.default_threshold, "recall": ev.default_recall,
                             "fa_per_hour": ev.default_fa_per_hour, "negative_hours": ev.negative_hours}
            print(f"{p.display}: sensitivity {ev.default_sensitivity} -> threshold {ev.default_threshold:.3f}, "
                  f"recall {ev.default_recall:.1%}, {ev.default_fa_per_hour:.2f} false accepts/hour "
                  f"over {ev.negative_hours:.1f} h")
        return summary

    def export_step(self) -> dict:
        from .export import check_parity, export_onnx, golden_vectors, write_package

        cfg = self.cfg
        out = self.work / "export" / "wakeword_models"
        out.mkdir(parents=True, exist_ok=True)
        classifiers, parity = {}, {}
        for p in cfg.phrases:
            model = self.load_model(p.id)
            path = self.work / "export" / f"{p.id}.onnx"
            export_onnx(model, path)
            windows = np.load(self.work / "features" / f"{p.id}_valpos.npy")[:256]
            parity[p.id] = check_parity(model, path, windows)
            classifiers[p.id] = path
        golden = golden_vectors(self.features(), classifiers, self.eval_audio(cfg.phrases[0].id)[0])
        evaluations = {p.id: json.loads((self.work / "eval" / f"{p.id}.json").read_text()) for p in cfg.phrases}
        notes = [
            "Provisional v0: positives are synthetic (Piper) only; negatives are read speech, music and noise.",
            "Recall is measured on held-out synthetic voices and FA/hour on read speech, so real-world numbers will differ. "
            "Re-measure on the real-recording test set (plan 5.3) before release.",
            "All training data is commercially licensed; see DATASET_MANIFEST.csv.",
        ]
        zip_path = write_package(
            cfg, out,
            {"melspectrogram": self.asset("melspectrogram.onnx"), "embedding": self.asset("embedding_model.onnx")},
            classifiers, evaluations, self.hours(), golden, notes,
        )
        print(f"model package: {zip_path}")
        return {"zip": str(zip_path), "onnx_parity_max_diff": parity}


def hash_id(s: str) -> int:
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little") % (2**31)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def run(config: Path, work: Path, steps: list[str], force: bool = False, overrides: dict | None = None) -> None:
    cfg = load_config(config, overrides)
    r = Run(cfg, work)
    handlers = {
        "setup": r.setup, "preview": r.preview, "tts": r.tts, "negatives": r.negatives,
        "features": r.features_step, "train": r.train_step, "evaluate": r.evaluate_step, "export": r.export_step,
    }
    for step in steps:
        if r.done(step) and not force:
            print(f"[{step}] already done")
            continue
        print(f"[{step}] starting")
        started = time.time()
        summary = handlers[step]()
        r.mark(step, {"seconds": round(time.time() - started), **(summary or {})})
        print(f"[{step}] done in {time.time() - started:.0f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--steps", default="all", help="comma-separated steps, or 'all'")
    ap.add_argument("--force", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. train.steps=500")
    args = ap.parse_args(argv)
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps {unknown}; choose from {STEPS}")
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    run(args.config, args.work, steps, args.force, overrides)


if __name__ == "__main__":
    main()
