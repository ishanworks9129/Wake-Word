"""openWakeWord feature extraction: melspectrogram.onnx -> embedding_model.onnx (Google speech_embedding).

Batch functions are used for training. StreamingFeatures defines the exact runtime behaviour that the
C# and web clients must reproduce (checked with golden vectors from export.py).
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort

from . import CLASSIFIER_FRAMES, EMBEDDING_DIM, FRAME_SAMPLES, SAMPLE_RATE

MEL_WINDOW = 76  # mel frames per embedding
MEL_STEP = 8  # mel frames per 1280-sample chunk
MEL_CONTEXT = 160 * 3  # extra samples fed to the mel model per streaming chunk
MEL_FFT, MEL_HOP = 512, 160  # the mel model rejects input under 512 samples; frames = (N - 512) // 160 + 1
MIN_EMBED_SAMPLES = MEL_FFT + (MEL_WINDOW - 1) * MEL_HOP  # shortest audio that yields one embedding


def mel_transform(spec: np.ndarray) -> np.ndarray:
    # Matches openWakeWord: brings the ONNX mel model close to the original TF implementation.
    return spec / 10.0 + 2.0


class FeatureExtractor:
    def __init__(self, melspectrogram_path: str, embedding_path: str, threads: int = 0, use_cuda: bool = False):
        options = ort.SessionOptions()
        n = threads or os.cpu_count() or 1
        options.intra_op_num_threads = n
        options.inter_op_num_threads = 1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self._mel = ort.InferenceSession(melspectrogram_path, options, providers=providers)
        self._emb = ort.InferenceSession(embedding_path, options, providers=providers)
        self._mel_in = self._mel.get_inputs()[0].name
        self._emb_in = self._emb.get_inputs()[0].name

    def melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """int16 audio [N] or [B, N] -> float32 mel [frames, 32] or [B, frames, 32]."""
        if audio.dtype != np.int16:
            raise ValueError(f"audio must be int16, got {audio.dtype}")
        batched = audio.ndim == 2
        x = (audio if batched else audio[None]).astype(np.float32)
        spec = self._mel.run(None, {self._mel_in: x})[0]  # [B, 1, frames, 32]
        spec = mel_transform(spec.reshape(x.shape[0], -1, 32))
        return spec if batched else spec[0]

    def embed_mel_windows(self, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """[B, 76, 32] mel windows -> [B, 96] embeddings."""
        out = np.empty((windows.shape[0], EMBEDDING_DIM), dtype=np.float32)
        for i in range(0, windows.shape[0], batch_size):
            chunk = windows[i:i + batch_size, :, :, None].astype(np.float32)
            out[i:i + batch_size] = self._emb.run(None, {self._emb_in: chunk})[0].reshape(chunk.shape[0], EMBEDDING_DIM)
        return out

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """int16 audio of any length -> [T, 96], one embedding per 1280 samples once 76 mel frames exist."""
        if audio.shape[0] < MIN_EMBED_SAMPLES:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        spec = self.melspectrogram(audio)
        starts = range(0, spec.shape[0] - MEL_WINDOW + 1, MEL_STEP)
        if not starts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        windows = np.stack([spec[s:s + MEL_WINDOW] for s in starts])
        return self.embed_mel_windows(windows)

    def embed_long(self, audio: np.ndarray, chunk_samples: int = SAMPLE_RATE * 600) -> np.ndarray:
        """embed() for recordings hours long, in bounded memory, with an identical result.

        Each chunk starts where the previous chunk's next embedding would start (a multiple of 1280
        samples, so mel frames stay on the same grid), so chunking never drops or duplicates an embedding.
        """
        out, pos = [], 0
        while audio.shape[0] - pos >= MIN_EMBED_SAMPLES:
            emb = self.embed(audio[pos:pos + max(chunk_samples, MIN_EMBED_SAMPLES)])
            out.append(emb)
            pos += emb.shape[0] * FRAME_SAMPLES
        return np.concatenate(out) if out else np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def embed_batch(self, clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """int16 [B, N] equal-length clips -> [B, T, 96]."""
        results = []
        for i in range(0, clips.shape[0], batch_size):
            batch = clips[i:i + batch_size]
            try:
                spec = self.melspectrogram(batch)
            except Exception:  # some ORT builds reject batched mel input; fall back per clip
                spec = np.stack([self.melspectrogram(c) for c in batch])
            starts = list(range(0, spec.shape[1] - MEL_WINDOW + 1, MEL_STEP))
            windows = np.stack([spec[:, s:s + MEL_WINDOW] for s in starts], axis=1)  # [b, T, 76, 32]
            b, t = windows.shape[:2]
            emb = self.embed_mel_windows(windows.reshape(b * t, MEL_WINDOW, 32))
            results.append(emb.reshape(b, t, EMBEDDING_DIM))
        return np.concatenate(results)

    def streaming(self) -> "StreamingFeatures":
        return StreamingFeatures(self)


class StreamingFeatures:
    """Runtime feature pipeline, chunk by chunk, exactly as the apps run it.

    For each 1280-sample chunk: run mel over the chunk plus the previous 480 samples (8 new frames; the
    very first chunk has no context and yields 5), append to the mel buffer, and once 77 mel frames exist
    compute one embedding from the 76 frames before the newest one. A classifier score is available once
    16 embeddings exist.

    Skipping the newest frame puts every window on the same 8-frame grid as FeatureExtractor.embed (training):
    after chunk k the buffer holds frames 0..8k+4, and training windows end at frames 8j+75, so the window
    ending at 8k+3 is training window j = k - 9. It costs 10 ms of latency.
    """

    def __init__(self, fx: FeatureExtractor):
        self._fx = fx
        self._prev = np.empty(0, dtype=np.int16)
        self._mel = np.empty((0, 32), dtype=np.float32)
        self.embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def push(self, chunk: np.ndarray) -> bool:
        """Adds one 1280-sample chunk. Returns True if a new embedding was produced."""
        if chunk.shape != (FRAME_SAMPLES,) or chunk.dtype != np.int16:
            raise ValueError("chunks must be 1280 int16 samples")
        audio = np.concatenate([self._prev[-MEL_CONTEXT:], chunk])
        self._prev = audio
        self._mel = np.concatenate([self._mel, self._fx.melspectrogram(audio)])[-(MEL_WINDOW + 1 + MEL_STEP):]
        if self._mel.shape[0] < MEL_WINDOW + 1:
            return False
        emb = self._fx.embed_mel_windows(self._mel[None, -(MEL_WINDOW + 1):-1])
        self.embeddings = np.concatenate([self.embeddings, emb])[-CLASSIFIER_FRAMES:]
        return True

    def classifier_input(self) -> np.ndarray | None:
        return self.embeddings[None].astype(np.float32) if self.embeddings.shape[0] == CLASSIFIER_FRAMES else None
