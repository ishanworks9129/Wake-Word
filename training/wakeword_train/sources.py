"""Downloads and streaming readers for openly licensed audio."""

from __future__ import annotations

import io
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import requests
import soundfile as sf
import urllib3
from scipy.signal import resample_poly

from . import SAMPLE_RATE

AUDIO_EXTENSIONS = (".flac", ".wav", ".ogg")


def download(url: str, dest: Path, retries: int = 5) -> Path:
    """Downloads with resume; skips if already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            total = int(head.headers.get("content-length", 0))
            if dest.exists() and (total == 0 or dest.stat().st_size == total):
                return dest
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                mode = "ab" if done and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            part.replace(dest)
            return dest
        except (requests.RequestException, OSError) as e:
            if attempt == retries - 1:
                raise
            print(f"download {url} failed ({e}); retrying")
            time.sleep(5 * (attempt + 1))
    return dest


def to_16k_mono_int16(audio: np.ndarray, sr: int) -> np.ndarray:
    """int16 input keeps its scale; float input is taken as [-1, 1] full scale.

    The scale is decided by the input dtype, never by the values: resampling a clip normalised to a peak
    of exactly 1.0 can overshoot slightly, and a value-based check then skipped scaling and produced
    near-silent clips.
    """
    x = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.floating):
        x = x * 32767.0
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g)
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def decode(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    return to_16k_mono_int16(audio, sr)


def iter_tar_audio(
    url: str,
    accept: Callable[[str], bool],
    skip_members: int = 0,
    retries: int = 5,
) -> Iterator[tuple[int, str, np.ndarray]]:
    """Streams a remote .tar/.tar.gz without saving it, yielding (member_index, name, int16 16 kHz audio).

    member_index counts every regular file in archive order, so a crashed run can resume with
    skip_members=<last index + 1>. Connection drops reconnect and skip ahead automatically.
    """
    index = -1
    next_needed = skip_members
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                index = -1
                with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        index += 1
                        if index < next_needed:
                            continue
                        name = member.name
                        if name.lower().endswith(AUDIO_EXTENSIONS) and accept(name):
                            try:
                                audio = decode(tf.extractfile(member).read())
                            except (RuntimeError, sf.LibsndfileError) as e:
                                print(f"skipping undecodable {name}: {e}")
                                next_needed = index + 1
                                continue
                            next_needed = index + 1
                            yield index, name, audio
                        else:
                            next_needed = index + 1
            return
        # Reading r.raw directly raises urllib3's own errors (a reset is ProtocolError), not requests'.
        except (requests.RequestException, urllib3.exceptions.HTTPError, tarfile.ReadError, OSError, EOFError) as e:
            if attempt == retries - 1:
                raise
            print(f"stream {url} dropped at member {next_needed} ({e}); reconnecting")
            time.sleep(10 * (attempt + 1))


def iter_file_audio(urls: list[str], skip_members: int = 0, retries: int = 5) -> Iterator[tuple[int, str, np.ndarray]]:
    """Fetches individual remote audio files, yielding (index in urls, url, int16 16 kHz audio).

    Resumes like iter_tar_audio with skip_members=<last index + 1>. A file that is missing (404) or
    undecodable is skipped with a message rather than ending the run.
    """
    for index in range(skip_members, len(urls)):
        url = urls[index]
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    print(f"skipping missing {url}")
                    break
                r.raise_for_status()
                try:
                    audio = decode(r.content)
                except (RuntimeError, sf.LibsndfileError) as e:
                    print(f"skipping undecodable {url}: {e}")
                    break
                yield index, url, audio
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                print(f"fetching {url} failed ({e}); retrying")
                time.sleep(10 * (attempt + 1))


def load_rirs(zip_path: Path, max_seconds: float = 1.0) -> list[np.ndarray]:
    """Room impulse responses from a zip of wav files, as float32 at 16 kHz, peak-normalised."""
    rirs = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".wav") or "__MACOSX" in name:
                continue
            audio, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
            x = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                g = np.gcd(sr, SAMPLE_RATE)
                x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
            x = x[np.argmax(np.abs(x)):][: int(max_seconds * SAMPLE_RATE)]  # start at the direct path
            peak = np.abs(x).max()
            if peak > 0:
                rirs.append((x / peak).astype(np.float32))
    return rirs
