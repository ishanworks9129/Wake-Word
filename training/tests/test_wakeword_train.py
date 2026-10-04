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


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_load(self):
        for name in ("uno.yaml", "smoke.yaml"):
            cfg = load_config(ROOT / "configs" / name)
            self.assertEqual({p.id for p in cfg.phrases}, {"hey_uno", "hello_uno"})
            self.assertTrue(all("NC" not in s.license.upper() for s in cfg.negatives))

    def test_overrides_and_unknown_keys(self):
        cfg = load_config(ROOT / "configs" / "smoke.yaml", {"train.steps": 7})
        self.assertEqual(cfg.train.steps, 7)
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs" / "smoke.yaml", {"train.stepz": 7})


if __name__ == "__main__":
    unittest.main()
