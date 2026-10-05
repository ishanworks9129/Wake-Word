"""Resumable training pipeline.

    python -m wakeword_train.pipeline --config configs/uno.yaml --work /content/drive/MyDrive/wakeword/uno --steps all

Steps (in order): setup, preview, tts, negatives, features, train, evaluate, export.
Each step writes work/steps/<step>.done; finished steps are skipped unless --force. Long steps (tts,
negatives) also resume part-way, so a dropped Colab session loses minutes, not hours.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from . import SAMPLE_RATE
from .augment import Augmenter, NoiseBank, colored_noise
from .config import AugmentConfig, Config, load_config
from .features import MIN_EMBED_SAMPLES, FeatureExtractor
from .sources import download, iter_file_audio, iter_tar_audio, load_rirs
from .store import EmbeddingStore, FRAMES_PER_HOUR
from .tts import ClipSet, PiperSynth, generate, is_held_out

STEPS = ["setup", "preview", "tts", "negatives", "features", "train", "evaluate", "test", "export"]
SHARED_DIRS = ("assets", "noise", "negatives")
EVAL_WINDOW = 4 * SAMPLE_RATE  # validation positives: clip ends 1 s before the end of a 4 s window
EVAL_TRAILING = SAMPLE_RATE


class Run:
    def __init__(self, cfg: Config, work: Path, negatives_split: str | None = None):
        self.cfg = cfg
        self.work = work
        self.negatives_split = negatives_split  # limits the negatives step to "train" or "val" sources
        # Background audio, noise and downloaded models can be shared by many runs (Wake Word Studio).
        self.shared = Path(cfg.shared_dir) if cfg.shared_dir else work
        for d in ("steps", "clips", "features", "models", "eval", "export", "preview"):
            (work / d).mkdir(parents=True, exist_ok=True)
        for d in SHARED_DIRS:
            (self.shared / d).mkdir(parents=True, exist_ok=True)

    # ---- paths and shared resources

    def asset(self, name: str) -> Path:
        return self.shared / "assets" / name

    def voice_paths(self) -> list[tuple[Path, Path]]:
        return [(self.asset(f"voices/{v.name}.onnx"), self.asset(f"voices/{v.name}.onnx.json")) for v in self.cfg.tts.voices]

    def features(self) -> FeatureExtractor:
        return FeatureExtractor(str(self.asset("melspectrogram.onnx")), str(self.asset("embedding_model.onnx")),
                                self.cfg.feature_threads, self.cfg.features_use_cuda)

    def synths(self) -> list[PiperSynth]:
        return [PiperSynth(m, c, self.cfg.tts.use_cuda) for m, c in self.voice_paths()]

    def store(self, split: str) -> EmbeddingStore:
        return EmbeddingStore(self.shared / "negatives" / split)

    def augmenter(self, seed: int, cfg: AugmentConfig | None = None, window: int | None = None) -> Augmenter:
        noise = NoiseBank.load(self.shared / "noise" / "bank.npy")
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
        download(cfg.cmudict_url, self.asset("cmudict.dict"))
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
        """Synthesises every missing part, in parallel when tts.workers > 1, then merges parts per phrase and kind."""
        cfg = self.cfg
        size = cfg.tts.part_size
        jobs, groups = [], []
        for p in cfg.phrases:
            for kind, texts, n in (("pos", p.tts_texts, cfg.tts.positives_per_phrase),
                                   ("adv", p.adversarial_texts, cfg.tts.adversarial_per_phrase)):
                final = self.work / "clips" / f"{p.id}_{kind}.npz"
                parts_dir = self.work / "clips" / f"{p.id}_{kind}_parts"
                groups.append((p.id, kind, texts, final, parts_dir))
                if final.exists():
                    continue
                parts_dir.mkdir(exist_ok=True)
                n_parts = -(-n // size)
                for part in range(n_parts):
                    part_path = parts_dir / f"part_{part:03d}.npz"
                    if not part_path.exists():
                        seed = [cfg.train.seed, hash_id(p.id), 0 if kind == "pos" else 1, part]
                        label = f"{p.id} {kind}: part {part + 1}/{n_parts}"
                        jobs.append((texts, min(size, n - part * size), cfg.tts, seed, str(part_path), label))

        voices = [(str(m), str(c)) for m, c in self.voice_paths()]
        started = time.time()
        if cfg.tts.workers > 1 and len(jobs) > 1:
            ctx = multiprocessing.get_context("spawn")  # ONNX Runtime sessions must not be forked
            workers = min(cfg.tts.workers, len(jobs))
            threads = max(1, (os.cpu_count() or workers) // workers)
            with ProcessPoolExecutor(workers, mp_context=ctx,
                                     initializer=_init_tts_worker, initargs=(voices, cfg.tts.use_cuda, threads)) as pool:
                for future in as_completed([pool.submit(_tts_part, job) for job in jobs]):
                    label, count, seconds = future.result()
                    print(f"{label} ({count} clips, {seconds:.0f}s; {time.time() - started:.0f}s elapsed)", flush=True)
        elif jobs:
            _init_tts_worker(voices, cfg.tts.use_cuda)
            for job in jobs:
                label, count, seconds = _tts_part(job)
                print(f"{label} ({count} clips, {seconds:.0f}s)", flush=True)

        summary = {}
        for phrase_id, kind, texts, final, parts_dir in groups:
            if not final.exists():
                parts = [ClipSet.load(f) for f in sorted(parts_dir.glob("part_*.npz"))]
                ClipSet.from_clips(
                    [c.clip(i) for c in parts for i in range(len(c))],
                    np.concatenate([c.speakers for c in parts]),
                    np.concatenate([c.texts for c in parts]),
                    texts,
                ).save(final)
            summary[f"{phrase_id}_{kind}"] = len(ClipSet.load(final))
        return summary

    def negatives(self) -> dict:
        cfg = self.cfg
        fx = self.features()
        noise_dir = self.shared / "noise" / "parts"
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
            if self.negatives_split and src.split != self.negatives_split:
                continue
            store = self.store(src.split)
            for key, items in self._negative_streams(src, store):
                last_report = time.time()
                for index, name, audio in items:
                    # Whole recordings can be hours long (VoxPopuli sessions): keep only what the budget needs.
                    have = store.hours(src.name) + store._buf_frames / FRAMES_PER_HOUR
                    audio = audio[: int((src.max_hours - have) * 3600 * SAMPLE_RATE) + MIN_EMBED_SAMPLES]
                    emb = fx.embed_long(audio)
                    if emb.shape[0]:
                        store.append(src.name, key, index, emb)
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
                else:
                    store.mark_exhausted(src.name, key)
                store.flush()
                flush_noise()
                if src.download_first:
                    self._archive_copy(src, key).unlink(missing_ok=True)

        parts = [np.load(f) for f in sorted(noise_dir.glob("*.npy"))]
        rng = np.random.default_rng(cfg.train.seed)
        bank = np.concatenate(parts + [colored_noise(120, rng)])
        NoiseBank(bank).save(self.shared / "noise" / "bank.npy")
        hours = self.hours()
        for split in ("train", "val"):
            if self.store(split).hours() == 0:
                raise RuntimeError(f"no {split} negative audio was collected; check the negatives sources")
        return {"hours": hours, "noise_bank_hours": bank.shape[0] / SAMPLE_RATE / 3600}

    def _negative_streams(self, src, store: EmbeddingStore):
        """(progress key, audio iterator) for each part of a source still to read."""
        if src.kind == "files":
            parts = [("files", lambda skip: iter_file_audio(src.urls, skip_members=skip))]
        elif src.kind == "tar":
            accept = (lambda name: not src.include or any(s in name for s in src.include))
            parts = [(url, lambda skip, url=url: iter_tar_audio(url, accept, skip_members=skip, local=self._fetch(src, url)))
                     for url in src.urls]
        else:
            raise ValueError(f"{src.name}: unknown negatives kind {src.kind!r}; use 'tar' or 'files'")
        for key, open_part in parts:
            if store.hours(src.name) >= src.max_hours:
                return
            if store.exhausted(src.name, key):
                continue
            skip = store.source_progress(src.name)["next_member"].get(key, 0)
            where = f"{len(src.urls)} files from file {skip}" if src.kind == "files" else f"{key} from member {skip}"
            print(f"{src.name}: streaming {where} ({store.hours(src.name):.1f}/{src.max_hours} h)")
            yield key, open_part(skip)

    def _archive_copy(self, src, url: str) -> Path:
        # Local disk, not the work folder: in Colab that is Google Drive, and GBs written there can drop the mount.
        return Path(tempfile.gettempdir()) / "wakeword_downloads" / src.name / url.rstrip("/").rsplit("/", 1)[-1]

    def _fetch(self, src, url: str) -> Path | None:
        if not src.download_first:
            return None
        path = self._archive_copy(src, url)
        print(f"{src.name}: downloading {url} to disk first (resumes if interrupted)")
        download(url, path)
        return path

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
        cap = int(cfg.train.max_negative_hours * FRAMES_PER_HOUR) or None
        neg = self.store("train").load(cap)
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

    def tts_seconds(self) -> dict[tuple[int, str, str], float]:
        """(voice index, manifest kind, split) -> seconds of generated clips, for DATASET_MANIFEST.csv."""
        out: dict[tuple[int, str, str], float] = {}
        for p in self.cfg.phrases:
            for kind, manifest_kind in (("pos", "positive"), ("adv", "hard_negative")):
                path = self.work / "clips" / f"{p.id}_{kind}.npz"
                if not path.exists():
                    continue
                with np.load(path) as z:  # reads only these two arrays, not the audio
                    lengths, speakers = np.diff(z["offsets"]), z["speakers"]
                # Held-out speakers' positives are the dev set; every near-miss clip is trained on.
                held = is_held_out(speakers, self.cfg.tts.holdout_every_nth_speaker) if kind == "pos" else np.zeros(len(speakers), bool)
                for voice in np.unique(speakers // 100000):
                    for split, mask in (("train", ~held), ("dev", held)):
                        sel = (speakers // 100000 == voice) & mask
                        key = (int(voice), manifest_kind, split)
                        out[key] = out.get(key, 0.0) + float(lengths[sel].sum()) / SAMPLE_RATE
        return out

    def rir_seconds(self) -> float:
        return sum(len(r) for r in load_rirs(self.asset("rir.zip"))) / SAMPLE_RATE

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

    def test_step(self) -> dict:
        """False accepts on your own recordings (cfg.test.folder) at every calibrated sensitivity.

        Writes eval/<phrase>_test.json (totals, which go in the package) and eval/test_fires.csv (file and
        time of every fire at the most sensitive setting, to check by ear; it stays out of the package).
        """
        from eval.metrics import poisson_upper_bound

        from .evaluate import count_fires, stream_scores
        from .testset import clock, decode_media, media_files, score_seconds

        cfg = self.cfg
        folder = Path(cfg.test.folder) if cfg.test.folder else None
        files = media_files(folder) if folder and folder.is_dir() else []
        if not files:
            print(f"no recordings to test on ({cfg.test.folder or 'test.folder is not set'}); skipping")
            return {"skipped": True}
        fx = self.features()
        cache = self.work / "test" / "features"  # embeddings only, keyed by a hash: no audio or names kept
        cache.mkdir(parents=True, exist_ok=True)
        streams = []
        for i, f in enumerate(files):
            rel = f.relative_to(folder).as_posix()
            c = cache / (hashlib.sha1(rel.encode()).hexdigest()[:16] + ".npy")
            if c.exists():
                emb = np.load(c)
            else:
                emb = fx.embed_long(decode_media(f)).astype(np.float16)
                with open(c.with_suffix(".tmp"), "wb") as out:
                    np.save(out, emb)
                c.with_suffix(".tmp").replace(c)
            streams.append((rel, emb))
            print(f"  {i + 1}/{len(files)} recordings, {sum(e.shape[0] for _, e in streams) / FRAMES_PER_HOUR:.1f} h")
        hours = sum(e.shape[0] for _, e in streams) / FRAMES_PER_HOUR

        def fire_frames(scores: np.ndarray, threshold: float) -> np.ndarray:
            return count_fires(scores, threshold, cfg.eval.consecutive_frames, cfg.eval.refractory_seconds)

        summary, fires = {}, []
        for p in cfg.phrases:
            model = self.load_model(p.id)
            table = json.loads((self.work / "eval" / f"{p.id}.json").read_text())["sensitivity_table"]
            scores = [(rel, stream_scores(model, emb, "cpu")) for rel, emb in streams]
            rows = []
            for s, t in table:
                n = sum(len(fire_frames(sc, t)) for _, sc in scores)
                rows.append({"sensitivity": s, "threshold": t, "false_accepts": n,
                             "fa_per_hour": n / hours, "fa_per_hour_upper": poisson_upper_bound(n) / hours})
            lowest = min(t for _, t in table)
            for rel, sc in scores:
                for k in fire_frames(sc, lowest).tolist():
                    fires.append({"keyword": p.id, "file": rel, "time": clock(score_seconds(k)), "score": round(float(sc[k]), 3)})
            result = {"files": len(files), "hours": hours, "model_sha256": self.model_hash(p.id), "by_sensitivity": rows}
            (self.work / "eval" / f"{p.id}_test.json").write_text(json.dumps(result, indent=1))
            default = next(r for r in rows if r["sensitivity"] == cfg.eval.default_sensitivity)
            summary[p.id] = {k: default[k] for k in ("threshold", "false_accepts", "fa_per_hour", "fa_per_hour_upper")}
            print(f"{p.display}: {default['false_accepts']} false accepts in {hours:.1f} h of your recordings at sensitivity "
                  f"{cfg.eval.default_sensitivity} ({default['fa_per_hour']:.2f}/h, at most {default['fa_per_hour_upper']:.2f}/h)")
        with open(self.work / "eval" / "test_fires.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["keyword", "file", "time", "score"])
            w.writeheader()
            w.writerows(sorted(fires, key=lambda r: (r["file"], r["time"], r["keyword"])))
        print(f"{len(fires)} fires at the most sensitive setting listed in {self.work / 'eval' / 'test_fires.csv'}: "
              "listen to each to check nobody actually said the wake phrase")
        return {"hours": hours, "files": len(files), **summary}

    def model_hash(self, phrase_id: str) -> str:
        return hashlib.sha256((self.work / "models" / f"{phrase_id}.pt").read_bytes()).hexdigest()

    def current_tests(self, evaluations: dict[str, dict]) -> dict[str, dict]:
        """Test results that still match the models and calibration; stale ones (after a retrain) are left out."""
        out = {}
        for p in self.cfg.phrases:
            path = self.work / "eval" / f"{p.id}_test.json"
            if not path.exists():
                continue
            test = json.loads(path.read_text())
            thresholds = [r["threshold"] for r in test["by_sensitivity"]]
            if test["model_sha256"] == self.model_hash(p.id) and thresholds == [t for _, t in evaluations[p.id]["sensitivity_table"]]:
                out[p.id] = test
            else:
                print(f"{p.id}: test results are from older models or calibration; re-run the test step to include them")
        return out if len(out) == len(self.cfg.phrases) else {}

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
        hours = self.hours()
        def used(split: str) -> str:
            return ", ".join(f"{src.name} ({hours.get(src.name, 0):.0f} h)" for src in cfg.negatives
                             if src.split == split and hours.get(src.name, 0) > 0)

        val_sources = used("val")
        notes = [
            f"Provisional v0: positives are synthetic (Piper) only; training negatives are {used('train')}.",
            f"Recall is measured on held-out synthetic voices and FA/hour on {val_sources}, so real-world numbers will differ. "
            "Re-measure on the real-recording test set (plan 5.3) before release.",
            "All training data is commercially licensed; see DATASET_MANIFEST.csv.",
        ]
        zip_path = write_package(
            cfg, out,
            {"melspectrogram": self.asset("melspectrogram.onnx"), "embedding": self.asset("embedding_model.onnx")},
            classifiers, evaluations, hours, golden, notes, self.tts_seconds(), self.rir_seconds(),
            self.current_tests(evaluations),
        )
        print(f"model package: {zip_path}")
        return {"zip": str(zip_path), "onnx_parity_max_diff": parity}


_TTS_SYNTHS: list[PiperSynth] | None = None


def _init_tts_worker(voices: list[tuple[str, str]], use_cuda: bool, threads: int = 0) -> None:
    global _TTS_SYNTHS
    if threads:
        # Piper builds its own SessionOptions with every core; with several workers that oversubscribes the CPU
        # badly (8 workers x 16 threads), so each worker gets its share of the cores instead.
        import onnxruntime

        base = onnxruntime.SessionOptions

        def limited() -> onnxruntime.SessionOptions:
            options = base()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            return options

        onnxruntime.SessionOptions = limited  # type: ignore[assignment]
    _TTS_SYNTHS = [PiperSynth(Path(m), Path(c), use_cuda) for m, c in voices]


def _tts_part(job: tuple) -> tuple[str, int, float]:
    texts, count, tts_cfg, seed, part_path, label = job
    started = time.time()
    clips = generate(_TTS_SYNTHS, texts, count, tts_cfg, np.random.default_rng(seed))
    clips.save(Path(part_path))
    return label, len(clips), time.time() - started


def hash_id(s: str) -> int:
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little") % (2**31)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def run(config: Path, work: Path, steps: list[str], force: bool = False, overrides: dict | None = None,
        negatives_split: str | None = None) -> None:
    cfg = load_config(config, overrides)
    r = Run(cfg, work, negatives_split)
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
    ap.add_argument("--negatives-split", choices=["train", "val"], help="limit the negatives step to one split's sources")
    args = ap.parse_args(argv)
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown steps {unknown}; choose from {STEPS}")
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    run(args.config, args.work, steps, args.force, overrides, args.negatives_split)


if __name__ == "__main__":
    main()
