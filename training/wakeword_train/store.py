"""Append-only, resumable store of continuous embedding streams (negative audio)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import CLASSIFIER_FRAMES, EMBEDDING_DIM, FRAME_SAMPLES, SAMPLE_RATE

FRAMES_PER_HOUR = 3600 * SAMPLE_RATE / FRAME_SAMPLES  # 45,000


class EmbeddingStore:
    """Directory of float16 [T, 96] shards plus index.json recording per-source progress.

    Shards are only listed in the index after they are fully written, so a crash loses at most one
    unflushed shard, and resume restarts each source after its last flushed member.
    """

    def __init__(self, root: Path, shard_frames: int = 50000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_frames = shard_frames
        self.index_path = self.root / "index.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {"shards": [], "sources": {}}
        self._buf: list[np.ndarray] = []
        self._buf_frames = 0
        self._pending_progress: dict[str, dict] = {}

    def source_progress(self, source: str) -> dict:
        return self.index["sources"].get(source, {"next_member": {}, "frames": 0})

    def exhausted(self, source: str, key: str) -> bool:
        """True once an archive (or a "files" list) was read to the end, so a re-run need not re-stream it."""
        return key in self.source_progress(source).get("exhausted", [])

    def mark_exhausted(self, source: str, key: str) -> None:
        progress = self._pending_progress.setdefault(source, json.loads(json.dumps(self.source_progress(source))))
        progress.setdefault("exhausted", []).append(key)
        self.flush()

    def hours(self, source: str | None = None) -> float:
        if source is None:
            return sum(s["frames"] for s in self.index["shards"]) / FRAMES_PER_HOUR
        return self.source_progress(source)["frames"] / FRAMES_PER_HOUR

    def append(self, source: str, url: str, member_index: int, embeddings: np.ndarray) -> None:
        self._buf.append(embeddings.astype(np.float16))
        self._buf_frames += embeddings.shape[0]
        progress = self._pending_progress.setdefault(source, json.loads(json.dumps(self.source_progress(source))))
        progress["next_member"][url] = member_index + 1
        progress["frames"] += embeddings.shape[0]
        if self._buf_frames >= self.shard_frames:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            self._persist_progress()
            return
        data = np.concatenate(self._buf)
        name = f"shard_{len(self.index['shards']):05d}.npy"
        tmp = self.root / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, data)
        tmp.replace(self.root / name)
        self.index["shards"].append({"file": name, "frames": int(data.shape[0])})
        self._buf, self._buf_frames = [], 0
        self._persist_progress()

    def _persist_progress(self) -> None:
        self.index["sources"].update(self._pending_progress)
        self._pending_progress = {}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=1))
        tmp.replace(self.index_path)

    def load(self, max_frames: int | None = None) -> np.ndarray:
        """All shards concatenated into one float16 [T, 96] array.

        Filled shard by shard into one preallocated array, so peak RAM is the result, not twice it.
        """
        shards, total = [], 0
        for s in self.index["shards"]:
            if max_frames and total >= max_frames:
                break
            shards.append(s)
            total += s["frames"]
        n = min(total, max_frames) if max_frames else total
        out = np.empty((n, EMBEDDING_DIM), dtype=np.float16)
        pos = 0
        for s in shards:
            take = min(s["frames"], n - pos)
            out[pos:pos + take] = np.load(self.root / s["file"], mmap_mode="r")[:take]
            pos += take
        return out


def sample_windows(stream: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n random [16, 96] windows from a [T, 96] stream."""
    starts = rng.integers(0, stream.shape[0] - CLASSIFIER_FRAMES + 1, size=n)
    return stream[starts[:, None] + np.arange(CLASSIFIER_FRAMES)]
