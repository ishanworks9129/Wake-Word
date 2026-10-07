"""Your own recordings as the frozen test set (plan 5.3): false accepts on audio the models never saw."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import CLASSIFIER_FRAMES, FRAME_SAMPLES, SAMPLE_RATE
from .features import MIN_EMBED_SAMPLES
from .sources import to_16k_mono_int16

MEDIA_EXTENSIONS = (".mp4", ".m4a", ".webm", ".mkv", ".mov", ".mp3", ".aac", ".wav", ".ogg", ".flac")


def media_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)


def decode_media(path: Path) -> np.ndarray:
    """Any audio or video file -> int16 16 kHz mono, through ffmpeg (Meet and Teams save .mp4).

    Without ffmpeg (it ships with Colab), plain audio files still work through soundfile.
    """
    if shutil.which("ffmpeg") is None:
        if Path(path).suffix.lower() in (".wav", ".flac", ".ogg"):
            audio, sr = sf.read(str(path), dtype="int16", always_2d=True)
            return to_16k_mono_int16(audio, sr)
        raise RuntimeError(f"reading {Path(path).suffix} files needs ffmpeg on the PATH: {path}")
    # Run from the file's folder with a bare name: works for a Windows ffmpeg.exe called from WSL too.
    path = Path(path).resolve()
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path.name, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, cwd=path.parent,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.int16)


def score_seconds(k: int) -> float:
    """Where in the recording score k ends: it covers embeddings k..k+15, the last ending at this sample."""
    return ((k + CLASSIFIER_FRAMES - 1) * FRAME_SAMPLES + MIN_EMBED_SAMPLES) / SAMPLE_RATE


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
