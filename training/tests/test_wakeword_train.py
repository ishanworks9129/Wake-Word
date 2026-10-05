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
            self.assertTrue(all(s.kind in ("tar", "files") for s in cfg.negatives))

    def test_overrides_and_unknown_keys(self):
        cfg = load_config(ROOT / "configs" / "smoke.yaml", {"train.steps": 7})
        self.assertEqual(cfg.train.steps, 7)
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs" / "smoke.yaml", {"train.stepz": 7})


if __name__ == "__main__":
    unittest.main()
