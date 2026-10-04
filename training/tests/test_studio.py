"""Studio API and job runner, with a fake pipeline so the tests take seconds (needs fastapi, httpx)."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

FAKE_PIPELINE = r'''
import json, sys, time, pathlib, zipfile, yaml
config, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = yaml.safe_load(config.read_text())
pid = cfg["phrases"][0]["id"]
if "fail" in pid:
    print("[tts] starting"); print("Traceback: boom"); sys.exit(1)
if "slow" in pid:  # like a ProcessPoolExecutor worker: a child that outlives a plain kill of the parent
    import subprocess
    work.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (work / "child.pid").write_text(str(child.pid))
    print("[tts] starting", flush=True)
    time.sleep(60)
for step in ["tts", "features", "train", "evaluate", "export"]:
    print(f"[{step}] starting", flush=True)
    if step == "tts":
        print(f"{pid} pos: part 1/2 (40 clips, 1s)", flush=True)
    if step == "train":
        print("step 300/600  loss 0.1", flush=True)
    time.sleep(0.05)
out = work / "export" / "wakeword_models"
out.mkdir(parents=True, exist_ok=True)
(out / "models.json").write_text(json.dumps({"keywords": [{"id": pid, "validation": {"recall": 0.9, "fa_per_hour": 0.4, "threshold": 0.7}}]}))
(out / f"{pid}.onnx").write_bytes(b"onnx")
with zipfile.ZipFile(work / "export" / "wakeword_models.zip", "w") as z:
    z.write(out / "models.json", "models.json")
print("[export] done", flush=True)
'''


def make_base(root: Path, cmudict: str) -> Path:
    base = root / "base"
    for rel in ["assets/melspectrogram.onnx", "assets/embedding_model.onnx", "assets/rir.zip", "assets/voices/v.onnx", "noise/bank.npy"]:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(b"x")
    (base / "assets" / "cmudict.dict").write_text(cmudict)
    for split in ("train", "val"):
        (base / "negatives" / split).mkdir(parents=True)
        (base / "negatives" / split / "index.json").write_text(json.dumps({"shards": [{"file": "a", "frames": 45000}], "sources": {}}))
    return base


class StudioApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from studio.app import create_app
        from studio.jobs import JobStore, Studio

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        base = make_base(root, "hey HH EY1\nhay HH EY1\nhe HH IY1\ncomputer K AH0 M P Y UW1 T ER0\ncommuter K AH0 M Y UW1 T ER0\n")
        script = root / "fake_pipeline.py"
        script.write_text(FAKE_PIPELINE)
        profiles = yaml.safe_load((ROOT / "studio" / "profiles.yaml").read_text())
        self.studio = Studio(JobStore(root / "jobs"), base, ROOT / "configs" / "smoke.yaml", profiles, "tiny",
                             command=lambda config, work: [sys.executable, str(script), str(config), str(work)])
        vad = root / "vad.onnx"
        vad.write_bytes(b"vad")
        self.client = TestClient(create_app(self.studio, None, vad))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def wait(self, job_id, statuses=("done", "failed", "cancelled")):
        for _ in range(200):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in statuses:
                return job
            time.sleep(0.05)
        self.fail(f"job stuck: {job}")

    def test_status_reports_a_ready_base(self):
        s = self.client.get("/api/status").json()
        self.assertTrue(s["ready"], s)
        self.assertEqual({"train": 1.0, "val": 1.0}, s["negative_hours"])

    def test_train_runs_to_done_with_metrics_package_and_download(self):
        r = self.client.post("/api/jobs", json={"phrase": "  Hey   Computer "})
        self.assertEqual(201, r.status_code, r.text)
        job = r.json()
        self.assertEqual(("Hey Computer", "hey_computer"), (job["phrase"], job["phrase_id"]))

        config = yaml.safe_load((self.studio.store.dir(job["id"]) / "config.yaml").read_text())
        phrase = config["phrases"][0]
        self.assertIn("hey commuter", phrase["adversarial_texts"])  # one sound away
        self.assertIn("computer", phrase["adversarial_texts"])  # fragment
        self.assertEqual(str(self.studio.base), config["shared_dir"])
        self.assertEqual(120, config["tts"]["positives_per_phrase"])  # tiny profile applied

        done = self.wait(job["id"])
        self.assertEqual("done", done["status"], done.get("log"))
        self.assertEqual(1.0, done["progress"])
        self.assertEqual(0.9, done["metrics"]["recall"])
        self.assertTrue(any("[train] starting" in line for line in done["log"]))
        self.assertEqual(200, self.client.get(f"/api/jobs/{job['id']}/package/models.json").status_code)
        self.assertEqual(200, self.client.get(f"/api/jobs/{job['id']}/package/hey_computer.onnx").status_code)
        self.assertEqual(404, self.client.get(f"/api/jobs/{job['id']}/package/..%2Fjob.json").status_code)
        dl = self.client.get(f"/api/jobs/{job['id']}/download")
        self.assertEqual("application/zip", dl.headers["content-type"])
        self.assertEqual([job["id"]], [j["id"] for j in self.client.get("/api/jobs").json()])

    def test_cancel_stops_the_pipeline_and_its_workers(self):
        job = self.client.post("/api/jobs", json={"phrase": "Hey Slow"}).json()
        self.wait(job["id"], ("running",))
        time.sleep(0.5)  # let the fake pipeline start its child
        self.assertEqual("cancelled", self.client.delete(f"/api/jobs/{job['id']}").json()["status"])
        pid_file = self.studio.store.dir(job["id"]) / "work" / "child.pid"
        for _ in range(100):
            if pid_file.exists():
                break
            time.sleep(0.05)
        child = int(pid_file.read_text())
        for _ in range(100):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return  # the worker died with its parent
            time.sleep(0.05)
        os.kill(child, 9)
        self.fail("the pipeline's worker survived cancellation")

    def test_failed_job_keeps_the_log(self):
        job = self.client.post("/api/jobs", json={"phrase": "Hey Fail"}).json()
        done = self.wait(job["id"])
        self.assertEqual("failed", done["status"])
        self.assertIn("boom", done["error"])

    def test_rejects_bad_phrases_and_profiles(self):
        for phrase in ["", "a", "hey 123", "one two three four five six", "x" * 50]:
            self.assertEqual(422, self.client.post("/api/jobs", json={"phrase": phrase}).status_code, phrase)
        self.assertEqual(422, self.client.post("/api/jobs", json={"phrase": "Hey There", "profile": "huge"}).status_code)
        self.assertEqual(404, self.client.get("/api/jobs/nope").status_code)

    def test_vad_model_is_served(self):
        self.assertEqual(b"vad", self.client.get("/vad/silero_vad.onnx").content)


class StudioReadinessTests(unittest.TestCase):
    def test_lists_what_is_missing(self):
        from studio.jobs import JobStore, Studio

        with tempfile.TemporaryDirectory() as d:
            studio = Studio(JobStore(Path(d) / "jobs"), Path(d) / "empty", ROOT / "configs" / "smoke.yaml", {"tiny": {}})
            state = studio.readiness()
            self.assertFalse(state["ready"])
            self.assertIn("noise bank", state["missing"])
            with self.assertRaises(RuntimeError):
                studio.create("Hey There", "tiny")


class NearMissTests(unittest.TestCase):
    def test_neighbours_fragments_and_no_plurals_of_the_last_word(self):
        from wakeword_train.nearmiss import Pronunciations, near_misses, phrase_config

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cmu.dict"
            path.write_text("hey HH EY1\nhay HH EY1\nhe HH IY1\ncomputer K AH0 M P Y UW1 T ER0\n"
                            "computers K AH0 M P Y UW1 T ER0 Z\ncommuter K AH0 M Y UW1 T ER0\nab AE1 B # comment\n")
            pron = Pronunciations.load(path)
        self.assertEqual(["he"], pron.neighbours("hey"))  # "hay" sounds identical, so it can't be a negative
        misses = near_misses("Hey Computer", pron)
        self.assertIn("he computer", misses)
        self.assertNotIn("hay computer", misses)
        self.assertIn("hey commuter", misses)
        self.assertIn("hey", misses)
        self.assertNotIn("hey computers", misses)
        self.assertNotIn("hey computer", misses)
        cfg = phrase_config("Hey Computer", pron)
        self.assertEqual("hey_computer", cfg["id"])
        self.assertIn("hey, computer", cfg["tts_texts"])


if __name__ == "__main__":
    unittest.main()
