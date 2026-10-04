"""Synthetic speech with Piper (training-time only; nothing from Piper ships in the apps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .config import TtsConfig
from .sources import to_16k_mono_int16


@dataclass
class ClipSet:
    """Variable-length int16 clips stored as one array plus offsets."""

    audio: np.ndarray  # int16, all clips concatenated
    offsets: np.ndarray  # int64, len = n + 1
    speakers: np.ndarray  # int32 speaker id per clip
    texts: np.ndarray  # int32 index into text_list per clip
    text_list: list[str]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def clip(self, i: int) -> np.ndarray:
        return self.audio[self.offsets[i]:self.offsets[i + 1]]

    def subset(self, mask: np.ndarray) -> "ClipSet":
        idx = np.flatnonzero(mask)
        clips = [self.clip(i) for i in idx]
        return ClipSet.from_clips(clips, self.speakers[idx], self.texts[idx], self.text_list)

    @staticmethod
    def from_clips(clips: list[np.ndarray], speakers, texts, text_list: list[str]) -> "ClipSet":
        lengths = np.array([len(c) for c in clips], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        audio = np.concatenate(clips) if clips else np.empty(0, np.int16)
        return ClipSet(audio.astype(np.int16), offsets, np.asarray(speakers, np.int32), np.asarray(texts, np.int32), list(text_list))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, audio=self.audio, offsets=self.offsets, speakers=self.speakers, texts=self.texts,
                 text_list=np.array(self.text_list, dtype=object))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "ClipSet":
        z = np.load(path, allow_pickle=True)
        return ClipSet(z["audio"], z["offsets"], z["speakers"], z["texts"], list(z["text_list"]))


def trim_silence(audio: np.ndarray, threshold_db: float = -40.0, pad_seconds: float = 0.05) -> np.ndarray:
    """Trims leading/trailing audio quieter than threshold_db relative to the clip's peak (20 ms blocks)."""
    block = SAMPLE_RATE // 50
    n = len(audio) // block
    if n == 0:
        return audio
    x = audio[: n * block].astype(np.float32).reshape(n, block)
    rms = np.sqrt((x**2).mean(axis=1) + 1e-9)
    loud = np.flatnonzero(20 * np.log10(rms / (rms.max() + 1e-9)) > threshold_db)
    if loud.size == 0:
        return audio
    pad = int(pad_seconds * SAMPLE_RATE)
    start = max(0, loud[0] * block - pad)
    end = min(len(audio), (loud[-1] + 1) * block + pad)
    return audio[start:end]


class PiperSynth:
    def __init__(self, model_path: Path, config_path: Path, use_cuda: bool = False):
        from piper import PiperVoice  # imported lazily: only the TTS steps need it

        self.voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate
        self.num_speakers = max(1, self.voice.config.num_speakers)

    def synth(self, text: str, speaker_id: int, length_scale: float, noise_scale: float, noise_w_scale: float) -> np.ndarray:
        from piper import SynthesisConfig

        cfg = SynthesisConfig(
            speaker_id=speaker_id if self.num_speakers > 1 else None,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
        )
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=cfg)]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return trim_silence(to_16k_mono_int16(audio, self.sample_rate))


def generate(
    synths: list[PiperSynth],
    texts: list[str],
    n: int,
    cfg: TtsConfig,
    rng: np.random.Generator,
    progress=None,
) -> ClipSet:
    """n clips cycling through texts, spread across voices and speakers, including held-out speakers.

    Speaker ids are offset per voice (voice k uses ids k*100000 + id) so holdout is tracked across voices.
    """
    clips, speakers, text_idx = [], [], []
    for i in range(n):
        v = i % len(synths)
        synth = synths[v]
        speaker = int(rng.integers(0, synth.num_speakers))
        t = i % len(texts)
        audio = synth.synth(
            texts[t],
            speaker,
            float(rng.uniform(*cfg.length_scale)),
            float(rng.uniform(*cfg.noise_scale)),
            float(rng.uniform(*cfg.noise_w_scale)),
        )
        if len(audio) < SAMPLE_RATE // 10:
            continue
        clips.append(audio)
        speakers.append(v * 100000 + speaker)
        text_idx.append(t)
        if progress:
            progress(i + 1, n)
    return ClipSet.from_clips(clips, speakers, text_idx, texts)


def is_held_out(speakers: np.ndarray, holdout_every_nth: int) -> np.ndarray:
    return (speakers % 100000) % holdout_every_nth == 0
