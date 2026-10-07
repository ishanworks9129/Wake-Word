"""Unit tests for the training pipeline pieces that need no downloads (numpy, scipy, torch, onnxruntime)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.detector import AdaptiveThreshold, DetectorOptions, fire_frames
from wakeword_train.augment import Augmenter, NoiseBank, colored_noise
from wakeword_train.config import AugmentConfig, EvalConfig, load_config
from wakeword_train.evaluate import DetPoint, calibrate, count_fires, threshold_for
from wakeword_train.store import EmbeddingStore, sample_windows

ROOT = Path(__file__).resolve().parents[1]


class CountFiresMatchesShippedDetector(unittest.TestCase):
    def test_random_streams(self):
        rng = np.random.default_rng(0)
        for trial in range(300):
            n = int(rng.integers(1, 400))
            scores = rng.random(n) ** rng.uniform(0.2, 3)
            t = float(rng.uniform(0.05, 0.95))
            consecutive = int(rng.integers(1, 6))
            refractory = float(rng.choice([0.0, 0.08, 0.1, 0.5, 1.5, 3.0]))
            expected = fire_frames(scores, -70.0, DetectorOptions(t, consecutive, refractory, AdaptiveThreshold(min_threshold=0.0, max_threshold=1.0)))
            actual = count_fires(scores, t, consecutive, refractory).tolist()
            self.assertEqual(expected, actual, f"trial {trial}: t={t} N={consecutive} r={refractory}")


class CalibrationTests(unittest.TestCase):
    def test_sensitivity_maps_to_monotonic_thresholds(self):
        hours = 200.0
        det = [DetPoint(t, 1 - t, k / hours, k) for t, k in [(0.1, 4000), (0.3, 800), (0.5, 120), (0.7, 40), (0.9, 3), (0.99, 0)]]
        table = calibrate(det, EvalConfig(), hours)
        thresholds = [t for _, t in table]
        self.assertEqual(sorted(thresholds, reverse=True), thresholds)
        self.assertEqual(0.9, dict(table)[0.0])  # 0.05/h: 3 FAs in 200 h -> upper bound 0.039/h passes
        self.assertEqual(0.7, dict(table)[0.5])  # 0.5/h: 40 FAs -> upper bound 0.27/h passes; 120 FAs fails
        self.assertEqual(0.3, dict(table)[1.0])  # 5/h: 800 FAs -> upper bound 4.3/h passes
        self.assertAlmostEqual(threshold_for(0.75, table), (dict(table)[0.7] + dict(table)[0.8]) / 2)

    def test_too_little_validation_audio_gives_strict_thresholds(self):
        det = [DetPoint(t, 1 - t, 0.0, 0) for t in (0.1, 0.5, 0.9, 0.99)]
        table = calibrate(det, EvalConfig(), negative_hours=0.1)  # zero FAs, but 0.1 h proves nothing
        self.assertEqual({0.99}, {t for _, t in table})


class AugmenterTests(unittest.TestCase):
    def test_places_clip_at_requested_end_and_keeps_length(self):
        cfg = AugmentConfig(noise_prob=0.0, rir_prob=0.0, band_limit_prob=0.0, speed_prob=0.0, gain_db=(0.0, 0.0))
        aug = Augmenter(cfg, NoiseBank(colored_noise(2, np.random.default_rng(1))), [], 32000, seed=0)
        clip = np.full(8000, 10000, dtype=np.int16)
        out = aug(clip, end_offset=4000)
        self.assertEqual(out.shape, (32000,))
        self.assertTrue(np.all(np.abs(out[20000:28000].astype(int) - 10000) < 50))
        self.assertTrue(np.all(np.abs(out[:19900]) < 50))
        self.assertTrue(np.all(np.abs(out[28100:]) < 50))

    def test_noise_mixed_at_requested_snr(self):
        cfg = AugmentConfig(noise_prob=1.0, snr_db=(10.0, 10.0), rir_prob=0.0, band_limit_prob=0.0, speed_prob=0.0, gain_db=(0.0, 0.0))
        rng = np.random.default_rng(2)
        aug = Augmenter(cfg, NoiseBank(colored_noise(5, rng)), [], 32000, seed=3)
        clip = (np.sin(np.arange(32000) / 5) * 8000).astype(np.int16)
        out = aug(clip, end_offset=0).astype(np.float64)
        noise = out - clip
        snr = 10 * np.log10(np.mean(clip.astype(np.float64) ** 2) / np.mean(noise**2))
        self.assertAlmostEqual(snr, 10.0, delta=0.5)


class ConversionTests(unittest.TestCase):
    def test_float_audio_normalised_to_full_scale_survives_resampling(self):
        from wakeword_train.sources import to_16k_mono_int16

        rng = np.random.default_rng(0)
        for _ in range(50):  # Piper-style: 22.05 kHz, peak exactly 1.0; resampling may overshoot
            x = rng.normal(size=22050).astype(np.float32)
            x /= np.abs(x).max()
            out = to_16k_mono_int16(x, 22050)
            self.assertGreater(np.abs(out.astype(int)).max(), 20000)

    def test_audio_already_16k_mono_int16_is_not_copied(self):
        from wakeword_train.sources import to_16k_mono_int16

        x = (np.sin(np.arange(16000) / 3) * 1000).astype(np.int16)[:, None]  # soundfile's always_2d shape
        out = to_16k_mono_int16(x, 16000)
        self.assertEqual(out.shape, (16000,))
        self.assertTrue(np.shares_memory(out, x))

    def test_int16_audio_keeps_its_scale(self):
        from wakeword_train.sources import to_16k_mono_int16

        x = (np.sin(np.arange(16000) / 3) * 1000).astype(np.int16)
        self.assertTrue(np.array_equal(to_16k_mono_int16(x, 16000), x))


class StoreTests(unittest.TestCase):
    def test_resume_progress_and_windows(self):
        with tempfile.TemporaryDirectory() as d:
            store = EmbeddingStore(Path(d), shard_frames=100)
            for i in range(5):
                store.append("src", "u1", i, np.full((60, 96), i, np.float32))
            store.append("src", "u1", 5, np.full((10, 96), 5, np.float32))  # unflushed
            # A crash now loses only the unflushed tail; progress points after the last flushed member.
            reopened = EmbeddingStore(Path(d), shard_frames=100)
            self.assertEqual(reopened.source_progress("src")["next_member"]["u1"], 4)
            self.assertEqual(reopened.load().shape, (240, 96))
            store.flush()
            reopened = EmbeddingStore(Path(d))
            self.assertEqual(reopened.source_progress("src")["next_member"]["u1"], 6)
            data = reopened.load()
            self.assertEqual(data.shape, (310, 96))
            w = sample_windows(data, 50, np.random.default_rng(0))
            self.assertEqual(w.shape, (50, 16, 96))

    def test_exhausted_parts_survive_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            store = EmbeddingStore(Path(d))
            store.append("src", "u1", 0, np.zeros((5, 96), np.float32))
            store.mark_exhausted("src", "u1")
            reopened = EmbeddingStore(Path(d))
            self.assertTrue(reopened.exhausted("src", "u1"))
            self.assertFalse(reopened.exhausted("src", "u2"))
            self.assertEqual(reopened.load().shape, (5, 96))


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wakeword_train.features import FeatureExtractor

        smoke = ROOT.parent / "testdata" / "models" / "smoke"
        cls.fx = FeatureExtractor(str(smoke / "melspectrogram.onnx"), str(smoke / "embedding_model.onnx"), threads=1)

    def test_long_audio_in_chunks_matches_one_pass(self):
        audio = (np.random.default_rng(0).normal(size=16000 * 20) * 3000).astype(np.int16)
        whole = self.fx.embed(audio)
        for chunk in (16000 * 3, 13000, 12512):
            np.testing.assert_allclose(self.fx.embed_long(audio, chunk_samples=chunk), whole, atol=1e-4)

    def test_package_tester_scores_like_the_apps(self):
        import json

        import onnxruntime as ort

        from wakeword_train.test_package import onnx_scores

        smoke = ROOT.parent / "testdata" / "models" / "smoke"
        golden = json.loads((smoke / "golden.json").read_text())
        pkg = json.loads((smoke / "models.json").read_text())
        audio = np.array(golden["audio_int16"], dtype=np.int16)
        expected = [c["scores"] for c in golden["chunks"] if "scores" in c]  # streaming runtime, one per chunk
        emb = self.fx.embed(audio)
        for kw in pkg["keywords"]:
            session = ort.InferenceSession(str(smoke / kw["model"]), providers=["CPUExecutionProvider"])
            got = onnx_scores(session, kw["input"], emb)
            # Whole-file features (training, evaluation, this tester) differ from the apps' chunk-by-chunk ones by
            # ~1e-4; on the steepest edge of a detection that moves a score by a few thousandths.
            np.testing.assert_allclose(got[:len(expected)], [e[kw["id"]] for e in expected], atol=1e-2)

    def test_audio_too_short_for_one_embedding_is_skipped(self):
        from wakeword_train.features import MIN_EMBED_SAMPLES

        for n in (0, 418, 511, MIN_EMBED_SAMPLES - 1):  # the mel model rejects under 512 samples
            self.assertEqual(self.fx.embed(np.zeros(n, np.int16)).shape, (0, 96))
            self.assertEqual(self.fx.embed_long(np.zeros(n, np.int16)).shape, (0, 96))
        self.assertEqual(self.fx.embed(np.zeros(MIN_EMBED_SAMPLES, np.int16)).shape, (1, 96))


class TarSourceTests(unittest.TestCase):
    def test_connection_reset_mid_archive_reconnects_and_resumes(self):
        import io
        import tarfile
        import wave
        from unittest import mock

        import urllib3

        from wakeword_train import sources

        def wav(n):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(np.full(n, n, dtype=np.int16).tobytes())
            return buf.getvalue()

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as t:
            for i, n in enumerate((1000, 2000, 3000)):
                data = wav(n)
                info = tarfile.TarInfo(f"a/{i}.wav")
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        data = archive.getvalue()

        class Resets(io.BytesIO):  # the server drops the connection partway through the second file
            def read(self, size=-1):
                if self.tell() > 4000:
                    raise urllib3.exceptions.ProtocolError("Connection broken", ConnectionResetError(104, "reset"))
                return super().read(size)

        responses = iter([Resets(data), io.BytesIO(data)])

        def get(url, stream, timeout):
            r = mock.MagicMock()
            r.__enter__.return_value.raw = next(responses)
            return r

        with mock.patch.object(sources.requests, "get", side_effect=get), mock.patch.object(sources.time, "sleep"):
            got = [(i, a.shape[0]) for i, _, a in sources.iter_tar_audio("https://x/a.tar", lambda name: True)]
        self.assertEqual([(0, 1000), (1, 2000), (2, 3000)], got)

        with tempfile.TemporaryDirectory() as d:  # a downloaded copy reads the same, with no network
            local = Path(d) / "a.tar"
            local.write_bytes(data)
            with mock.patch.object(sources.requests, "get", side_effect=AssertionError("no network")):
                got = [(i, a.shape[0]) for i, _, a in sources.iter_tar_audio("https://x/a.tar", lambda n: True, 1, local=local)]
            self.assertEqual([(1, 2000), (2, 3000)], got)


class PackageTesterTests(unittest.TestCase):
    def test_retest_from_cache_after_the_recordings_are_deleted(self):
        import json
        import shutil
        import wave

        from wakeword_train.test_package import main

        smoke = ROOT.parent / "testdata" / "models" / "smoke"
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            rec, out = Path(d) / "meetings", Path(d) / "out"
            (rec / "team").mkdir(parents=True)
            for name, secs in (("team/standup.wav", 40), ("retro.wav", 30)):
                with wave.open(str(rec / name), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes((rng.normal(size=16000 * secs) * 3000).astype(np.int16).tobytes())
            main(["--package", str(smoke), "--folder", str(rec), "--out", str(out)])
            first = json.loads((out / "results.json").read_text())

            # Re-processing a changed recording replaces its old entry rather than counting it twice.
            with wave.open(str(rec / "retro.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((rng.normal(size=16000 * 30) * 3000).astype(np.int16).tobytes())
            main(["--package", str(smoke), "--folder", str(rec), "--out", str(out)])
            index = json.loads((out / "cache" / "index.json").read_text())
            self.assertEqual(["retro.wav", "team/standup.wav"], sorted(v["file"] for v in index.values()))
            self.assertEqual(2, len(list((out / "cache").glob("*.npy"))))
            second = json.loads((out / "results.json").read_text())
            fires2 = (out / "fires.csv").read_text()

            shutil.rmtree(rec)  # the audio is gone; the cache alone reproduces the test
            main(["--package", str(smoke), "--out", str(out)])
            self.assertEqual(second, json.loads((out / "results.json").read_text()))
            self.assertEqual(fires2, (out / "fires.csv").read_text())
            self.assertEqual((2, 2), (first["recordings"], second["recordings"]))
            self.assertAlmostEqual(70 / 3600, second["hours"], delta=0.002)


    def test_frozen_split_tests_only_the_held_out_recordings_and_exports_the_rest(self):
        import json
        import wave

        from wakeword_train.test_package import main

        smoke = ROOT.parent / "testdata" / "models" / "smoke"
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            rec, out, exp = Path(d) / "meetings", Path(d) / "out", Path(d) / "train_features"
            rec.mkdir()
            for i in range(6):
                with wave.open(str(rec / f"m{i}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes((rng.normal(size=16000 * 20) * 3000).astype(np.int16).tobytes())
            main(["--package", str(smoke), "--folder", str(rec), "--out", str(out), "--train-share", "0.5", "--export-train", str(exp)])
            split = json.loads((out / "split.json").read_text())
            self.assertEqual(set(split["train"]) | set(split["test"]), {f"m{i}.wav" for i in range(6)})
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertEqual(len(split["train"]), len(list(exp.glob("*.npy"))))
            self.assertFalse(any("m" in f.stem and ".wav" in f.stem for f in exp.glob("*.npy")))  # hashed names only
            self.assertEqual(len(split["test"]), json.loads((out / "results.json").read_text())["recordings"])
            with self.assertRaises(SystemExit):  # the split is frozen
                main(["--package", str(smoke), "--out", str(out), "--train-share", "0.3"])
            main(["--package", str(smoke), "--out", str(out)])
            self.assertEqual(split, json.loads((out / "split.json").read_text()))
            self.assertEqual(len(split["test"]), json.loads((out / "results.json").read_text())["recordings"])


class EmbeddingsSourceTests(unittest.TestCase):
    def test_negatives_step_reads_exported_features_without_audio(self):
        import shutil

        from wakeword_train.config import NegativeSource
        from wakeword_train.pipeline import Run

        smoke = ROOT.parent / "testdata" / "models" / "smoke"
        with tempfile.TemporaryDirectory() as d:
            feats = Path(d) / "feats"
            feats.mkdir()
            for i, n in enumerate((4500, 9000)):  # 0.1 h and 0.2 h
                np.save(feats / f"{i:02d}.npy", np.full((n, 96), i, np.float16))
            cfg = load_config(ROOT / "configs" / "smoke.yaml")
            cfg.negatives = [NegativeSource(name="meetings_train", kind="embeddings", urls=[str(feats)], split="train",
                                            max_hours=0.25, license="Internal-Consent", license_url="x", release_id="r")]
            r = Run(cfg, Path(d) / "work", negatives_split="train")
            for f in ("melspectrogram.onnx", "embedding_model.onnx"):
                shutil.copy(smoke / f, r.asset(f))
            with self.assertRaises(RuntimeError):  # no validation audio in this test
                r.negatives()
            data = r.store("train").load()
            self.assertEqual((11250, 96), data.shape)  # capped at 0.25 h
            self.assertEqual({0.0, 1.0}, set(np.unique(data).tolist()))


class VoiceGroupTests(unittest.TestCase):
    def test_holdout_follows_each_groups_rule(self):
        from wakeword_train.tts import GROUP_SPEAKERS, held_out_mask

        cfg = load_config(ROOT / "configs" / "uno.yaml").tts  # groups: 1 none, 2 all, 3 every nth speaker
        speakers = np.array([0, 7, 10, 100020,                            # main voices: every 10th speaker
                             GROUP_SPEAKERS, GROUP_SPEAKERS + 300000,     # group 1: never held out
                             2 * GROUP_SPEAKERS, 2 * GROUP_SPEAKERS + 3,  # group 2: always held out
                             3 * GROUP_SPEAKERS + 20, 3 * GROUP_SPEAKERS + 21])
        self.assertEqual([True, False, True, True, False, False, True, True, True, False],
                         held_out_mask(speakers, cfg).tolist())
        self.assertEqual("en_US-libritts_r-medium", cfg.voice(0).name)
        self.assertEqual("en_US-norman-medium", cfg.voice(13).name)
        self.assertEqual("parler-tts-mini-v1", cfg.voice(30).name)

    def test_clips_combine_groups_and_features_notice_new_ones(self):
        from wakeword_train.pipeline import Run
        from wakeword_train.tts import GROUP_SPEAKERS, ClipSet

        cfg = load_config(ROOT / "configs" / "uno.yaml")
        with tempfile.TemporaryDirectory() as d:
            r = Run(cfg, Path(d))
            pid = cfg.phrases[0].id

            def save(name, n, offset):
                clips = [np.full(800 + i, i + 1, np.int16) for i in range(n)]
                ClipSet.from_clips(clips, offset + np.arange(n), np.zeros(n, int), ["hey uno"]).save(r.work / "clips" / name)

            save(f"{pid}_pos.npz", 3, 0)
            save(f"{pid}_adv.npz", 2, 0)
            before = r.clips_signature(pid)
            self.assertEqual(before, r.clips_signature_main(pid))  # old runs: features stay valid
            save(f"{pid}_pos@parler.npz", 4, 3 * GROUP_SPEAKERS)
            combined = r.clips(pid, "pos")
            self.assertEqual(7, len(combined))
            self.assertEqual([0, 1, 2] + [3 * GROUP_SPEAKERS + i for i in range(4)], combined.speakers.tolist())
            self.assertNotEqual(before, r.clips_signature(pid))  # a new group means new features

    def test_manifest_lists_group_voices(self):
        from data.manifest import read_rows, validate
        from wakeword_train.export import write_dataset_manifest

        cfg = load_config(ROOT / "configs" / "uno.yaml")
        hours = {src.name: 0.0 if src.license == "Internal-Consent" else 10.0 for src in cfg.negatives}
        tts = {(0, "positive", "train"): 3600.0, (12, "positive", "train"): 900.0, (20, "positive", "dev"): 300.0,
               (30, "positive", "train"): 1800.0, (30, "positive", "dev"): 200.0, (30, "hard_negative", "train"): 1700.0}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.csv"
            write_dataset_manifest(cfg, hours, path, tts, rir_seconds=270.0)
            rows = read_rows(path)
            self.assertEqual([], validate(rows).errors)
        paths = {r["path"]: r for r in rows}
        self.assertEqual("Public-Domain", paths["tts/en_US-john-medium/positive/train"]["license"])
        self.assertEqual("dev", paths["tts/en_US-ljspeech-medium/positive/dev"]["split"])
        self.assertEqual("Parler-TTS parler-tts-mini-v1", paths["tts/parler-tts-mini-v1/hard_negative/train"]["source"])


class ParlerGenerationTests(unittest.TestCase):
    def test_batches_drop_clips_that_ramble_to_the_cap_and_report_progress(self):
        from wakeword_train.tts import generate_parler

        class FakeParler:  # stands in for ParlerSynth: no model needed
            num_speakers, max_seconds = 5, 5.0

            def synth_batch(self, items, seed):
                # every third clip "rambles" to 5 s; the rest are 1 s
                return [np.full(16000 * (5 if i % 3 == 0 else 1), 3000, np.int16) for i, _ in enumerate(items)]

        seen = []
        clips = generate_parler(FakeParler(), ["hey uno"], 12, np.random.default_rng(0), speaker_offset=3_000_000,
                                batch=6, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(8, len(clips))  # 2 of every 6 rambled
        self.assertTrue(all(len(clips.clip(i)) == 16000 for i in range(len(clips))))
        self.assertTrue(all(s >= 3_000_000 for s in clips.speakers))
        self.assertEqual([(6, 12), (12, 12)], seen)


class FileSourceTests(unittest.TestCase):
    def test_missing_files_are_skipped_and_resume_by_index(self):
        import io
        import wave
        from unittest import mock

        from wakeword_train import sources

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.arange(1600, dtype=np.int16).tobytes())

        def get(url, timeout):
            return mock.Mock(status_code=404 if "gone" in url else 200, content=buf.getvalue(), raise_for_status=lambda: None)

        urls = ["https://x/a.wav", "https://x/gone.wav", "https://x/c.wav"]
        with mock.patch.object(sources.requests, "get", side_effect=get):
            got = [(i, u, a.shape[0]) for i, u, a in sources.iter_file_audio(urls)]
            self.assertEqual([(0, urls[0], 1600), (2, urls[2], 1600)], got)
            self.assertEqual([2], [i for i, _, _ in sources.iter_file_audio(urls, skip_members=2)])


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_load(self):
        for name in ("uno.yaml", "smoke.yaml"):
            cfg = load_config(ROOT / "configs" / name)
            self.assertEqual({p.id for p in cfg.phrases}, {"hey_uno", "hello_uno"})
            self.assertTrue(all("NC" not in s.license.upper() for s in cfg.negatives))
            self.assertTrue(all(s.kind in ("tar", "files", "embeddings") for s in cfg.negatives))

    def test_exported_dataset_manifest_passes_the_license_check(self):
        from data.manifest import read_rows, validate
        from wakeword_train.export import write_dataset_manifest

        cfg = load_config(ROOT / "configs" / "uno.yaml")
        for src in cfg.negatives:
            if src.license == "Internal-Consent":  # filled in by whoever holds the consent record
                src.license_url, src.release_id = "https://intranet/consent", "HR-1"
        hours = {src.name: 10.0 for src in cfg.negatives}
        hours[cfg.negatives[0].name] = 0.0  # a source that contributed nothing is left out, not listed at 0 s
        tts = {(0, "positive", "train"): 3600.0, (0, "positive", "dev"): 400.0, (0, "hard_negative", "train"): 3000.0}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "DATASET_MANIFEST.csv"
            write_dataset_manifest(cfg, hours, path, tts, rir_seconds=270.0)
            rows = read_rows(path)
            report = validate(rows)
        self.assertEqual([], report.errors)
        self.assertNotIn(f"negatives/{cfg.negatives[0].name}", [r["path"] for r in rows])
        self.assertEqual({"train", "dev"}, {r["split"] for r in rows})

    def test_internal_recordings_need_a_consent_reference(self):
        from data.manifest import read_rows, validate
        from wakeword_train.export import write_dataset_manifest

        cfg = load_config(ROOT / "configs" / "uno.yaml", {"test.license_url": "https://intranet/consent"})
        hours = {src.name: 0.0 if src.license == "Internal-Consent" else 10.0 for src in cfg.negatives}  # just the test row
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.csv"
            write_dataset_manifest(cfg, hours, path, {(0, "positive", "train"): 60.0}, 270.0, test_hours=30.0)
            errors = validate(read_rows(path)).errors
            self.assertEqual(1, len(errors))
            self.assertIn("consent", errors[0])
            cfg.test.release_id = "HR-2026-114"
            write_dataset_manifest(cfg, hours, path, {(0, "positive", "train"): 60.0}, 270.0, test_hours=30.0)
            rows = read_rows(path)
            self.assertEqual([], validate(rows).errors)
        test_row = next(r for r in rows if r["split"] == "test")
        self.assertEqual(("test/own-recordings", "108000.0"), (test_row["path"], test_row["duration_seconds"]))

    def test_overrides_and_unknown_keys(self):
        cfg = load_config(ROOT / "configs" / "smoke.yaml", {"train.steps": 7})
        self.assertEqual(cfg.train.steps, 7)
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs" / "smoke.yaml", {"train.stepz": 7})


if __name__ == "__main__":
    unittest.main()
