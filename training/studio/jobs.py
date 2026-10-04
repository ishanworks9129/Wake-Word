"""Job store and a single-worker queue that runs the training pipeline for one phrase at a time."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from wakeword_train.config import load_config
from wakeword_train.nearmiss import Pronunciations, phrase_config

JOB_STEPS = ["tts", "features", "train", "evaluate", "export"]
# Rough share of a job's time per step, for the progress bar.
STEP_WEIGHTS = {"tts": 0.35, "features": 0.3, "train": 0.2, "evaluate": 0.1, "export": 0.05}
PHRASE_RE = re.compile(r"^[A-Za-z][A-Za-z' ]{1,38}[A-Za-z]$")


@dataclass
class Job:
    id: str
    phrase: str
    phrase_id: str
    profile: str
    status: str = "queued"  # queued, running, done, failed, cancelled
    step: str = ""
    progress: float = 0.0
    message: str = "Waiting to start"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    metrics: dict = field(default_factory=dict)


def validate_phrase(phrase: str) -> str:
    phrase = " ".join(phrase.split())
    if not PHRASE_RE.match(phrase) or len(phrase.split()) > 5:
        raise ValueError("Use 1-5 words of letters only, 3-40 characters, e.g. \"Hey Computer\".")
    return phrase


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_-]{1,80}", job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def save(self, job: Job) -> None:
        with self._lock:
            path = self.dir(job.id) / "job.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(job), indent=1))
            tmp.replace(path)

    def get(self, job_id: str) -> Job | None:
        try:
            path = self.dir(job_id) / "job.json"
        except KeyError:
            return None
        return Job(**json.loads(path.read_text())) if path.exists() else None

    def all(self) -> list[Job]:
        jobs = [Job(**json.loads(p.read_text())) for p in self.root.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def log_tail(self, job_id: str, lines: int = 40) -> list[str]:
        path = self.dir(job_id) / "log.txt"
        return path.read_text(errors="replace").splitlines()[-lines:] if path.exists() else []

    def package_dir(self, job_id: str) -> Path:
        return self.dir(job_id) / "work" / "export" / "wakeword_models"


Command = Callable[[Path, Path], list[str]]


def pipeline_command(config: Path, work: Path) -> list[str]:
    return [sys.executable, "-u", "-m", "wakeword_train.pipeline", "--config", str(config), "--work", str(work),
            "--steps", ",".join(JOB_STEPS)]


class Studio:
    """Creates jobs and runs them one at a time in a background thread."""

    def __init__(self, store: JobStore, base: Path, template: Path, profiles: dict[str, dict], default_profile: str = "fast",
                 command: Command = pipeline_command, cwd: Path | None = None):
        self.store = store
        self.base = base
        self.template = template
        self.profiles = profiles
        self.default_profile = default_profile
        self.command = command
        self.cwd = cwd or Path(__file__).resolve().parents[1]
        self._pron: Pronunciations | None = None
        self._wake = threading.Event()
        self._stop = False
        self._process: subprocess.Popen | None = None
        self._current: str | None = None
        for job in store.all():  # a restart resumes interrupted work; the pipeline skips finished parts
            if job.status == "running":
                job.status, job.message = "queued", "Resuming after restart"
                store.save(job)
        self._thread = threading.Thread(target=self._loop, name="studio-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self._kill()

    def _kill(self) -> None:
        """Stops the running pipeline and every worker it started (they share its process group)."""
        process = self._process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM) if hasattr(os, "killpg") else process.terminate()
            except ProcessLookupError:
                pass

    def readiness(self) -> dict:
        b = self.base
        required = {
            "feature models": [b / "assets" / "melspectrogram.onnx", b / "assets" / "embedding_model.onnx"],
            "room recordings": [b / "assets" / "rir.zip"],
            "pronouncing dictionary": [b / "assets" / "cmudict.dict"],
            "noise bank": [b / "noise" / "bank.npy"],
            "background audio (train)": [b / "negatives" / "train" / "index.json"],
            "background audio (validation)": [b / "negatives" / "val" / "index.json"],
        }
        missing = [name for name, paths in required.items() if not all(p.exists() for p in paths)]
        voices = sorted(p.name for p in (b / "assets" / "voices").glob("*.onnx")) if (b / "assets" / "voices").exists() else []
        if not voices:
            missing.append("Piper voice")
        hours = {}
        for split in ("train", "val"):
            index = b / "negatives" / split / "index.json"
            if index.exists():
                hours[split] = round(sum(s["frames"] for s in json.loads(index.read_text())["shards"]) / 45000, 1)
        return {"ready": not missing, "missing": missing, "voices": voices, "negative_hours": hours, "profiles": list(self.profiles)}

    def pronunciations(self) -> Pronunciations:
        if self._pron is None:
            self._pron = Pronunciations.load(self.base / "assets" / "cmudict.dict")
        return self._pron

    def create(self, phrase: str, profile: str | None = None) -> Job:
        phrase = validate_phrase(phrase)
        profile = profile or self.default_profile
        if profile not in self.profiles:
            raise ValueError(f"Unknown profile '{profile}'; choose from {sorted(self.profiles)}.")
        state = self.readiness()
        if not state["ready"]:
            raise RuntimeError("Studio is missing: " + ", ".join(state["missing"]))

        entry = phrase_config(phrase, self.pronunciations())
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{entry['id'][:40]}-{uuid.uuid4().hex[:4]}"
        job = Job(id=job_id, phrase=phrase, phrase_id=entry["id"], profile=profile)

        config = yaml.safe_load(self.template.read_text(encoding="utf-8"))
        config["run_name"] = f"studio-{entry['id']}"
        config["phrases"] = [entry]
        config["shared_dir"] = str(self.base)
        for dotted, value in self.profiles[profile].items():
            node = config
            *parents, leaf = dotted.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        config.setdefault("tts", {})["workers"] = max(1, min(8, (os.cpu_count() or 2) - 1))

        path = self.store.dir(job_id) / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        load_config(path)  # fail now, not in the worker, if the config is wrong
        self.store.save(job)
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.store.get(job_id)
        if job and job.status in ("queued", "running"):
            if job.status == "running" and self._current == job_id:
                self._kill()
            job.status, job.message, job.finished = "cancelled", "Cancelled", time.time()
            self.store.save(job)
        return job

    def queue_position(self, job: Job) -> int:
        queued = sorted((j for j in self.store.all() if j.status == "queued"), key=lambda j: j.created)
        return next((i + 1 for i, j in enumerate(queued) if j.id == job.id), 0)

    def _loop(self) -> None:
        while not self._stop:
            queued = sorted((j for j in self.store.all() if j.status == "queued"), key=lambda j: j.created)
            if not queued:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            self._run(queued[0])

    def _run(self, job: Job) -> None:
        job.status, job.started, job.message = "running", job.started or time.time(), "Starting"
        self.store.save(job)
        self._current = job.id
        jdir = self.store.dir(job.id)
        config = load_config(jdir / "config.yaml")
        tts_parts = sum(-(-n // config.tts.part_size) for n in (config.tts.positives_per_phrase, config.tts.adversarial_per_phrase))
        tts_done = 0
        with open(jdir / "log.txt", "a", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                self.command(jdir / "config.yaml", jdir / "work"), cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,  # its own process group, so cancel can stop the synthesis workers too
            )
            for line in self._process.stdout:  # type: ignore[union-attr]
                log.write(line)
                log.flush()
                current = self.store.get(job.id)
                if current and current.status == "cancelled":
                    continue
                if m := re.match(r"\[(\w+)\] (starting|already done)", line):
                    job.step = m.group(1)
                    job.message = STEP_MESSAGES.get(job.step, job.step)
                    within = 0.0
                elif ": part " in line:
                    tts_done += 1
                    within = tts_done / max(1, tts_parts)
                elif m := re.search(r"(\d+)/(\d+) clips", line) or re.search(r"step (\d+)/(\d+)", line):
                    within = int(m.group(1)) / max(1, int(m.group(2)))
                else:
                    continue
                job.progress = round(_progress(job.step, within), 3)
                self.store.save(job)
            code = self._process.wait()

        self._process, self._current = None, None
        if (latest := self.store.get(job.id)) and latest.status == "cancelled":
            return
        job.finished = time.time()
        if code == 0:
            job.status, job.progress, job.message = "done", 1.0, "Ready to test and download"
            job.metrics = _metrics(self.store.package_dir(job.id), job.phrase_id)
        else:
            job.status, job.message = "failed", "Training failed"
            job.error = "\n".join(self.store.log_tail(job.id, 15))
        self.store.save(job)


STEP_MESSAGES = {
    "tts": "Generating voices",
    "features": "Mixing into rooms and noise",
    "train": "Training",
    "evaluate": "Measuring accuracy",
    "export": "Packaging",
}


def _progress(step: str, within: float) -> float:
    done = 0.0
    for s in JOB_STEPS:
        if s == step:
            return done + STEP_WEIGHTS[s] * min(1.0, within)
        done += STEP_WEIGHTS[s]
    return done


def _metrics(package: Path, phrase_id: str) -> dict:
    manifest = package / "models.json"
    if not manifest.exists():
        return {}
    keyword = next((k for k in json.loads(manifest.read_text())["keywords"] if k["id"] == phrase_id), None)
    return keyword["validation"] if keyword else {}
