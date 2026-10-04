"""Wake Word Studio server.

    cd training
    python -m studio.app --base <shared folder> --template configs/uno.yaml

The base folder is a finished pipeline run's shared data: assets/ (feature models, voice, room recordings),
noise/bank.npy and negatives/ (background audio features). The Colab run's Drive folder works as-is.
Then open http://localhost:8765, type a phrase and press Train.
"""

from __future__ import annotations

import argparse
import io
import os
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wakeword_train.sources import download

from .jobs import Job, JobStore, Studio, validate_phrase

REPO = Path(__file__).resolve().parents[2]
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


class TrainRequest(BaseModel):
    phrase: str
    profile: str | None = None


def job_view(studio: Studio, job: Job) -> dict:
    view = {**job.__dict__}
    view["queue_position"] = studio.queue_position(job) if job.status == "queued" else 0
    view["package_url"] = f"/api/jobs/{job.id}/package/" if job.status == "done" else None
    return view


def create_app(studio: Studio, static_dir: Path | None, vad_model: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        studio.start()
        yield
        studio.stop()

    app = FastAPI(title="Wake Word Studio", lifespan=lifespan)
    synth_cache: dict = {}

    @app.get("/api/status")
    def status() -> dict:
        return studio.readiness()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [job_view(studio, j) for j in studio.store.all()]

    @app.post("/api/jobs", status_code=201)
    def train(request: TrainRequest) -> dict:
        try:
            return job_view(studio, studio.create(request.phrase, request.profile))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = studio.store.get(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return {**job_view(studio, job), "log": studio.store.log_tail(job_id)}

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        job = studio.cancel(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return job_view(studio, job)

    @app.get("/api/jobs/{job_id}/package/{name}")
    def package_file(job_id: str, name: str) -> FileResponse:
        try:
            folder = studio.store.package_dir(job_id)
        except KeyError:
            raise HTTPException(404) from None
        if not folder.exists() or name not in os.listdir(folder):  # only files the export wrote; no path tricks
            raise HTTPException(404, "Not ready or no such file.")
        return FileResponse(folder / name)

    @app.get("/api/jobs/{job_id}/download")
    def download_zip(job_id: str) -> FileResponse:
        job = studio.store.get(job_id)
        zip_path = studio.store.dir(job_id) / "work" / "export" / "wakeword_models.zip" if job else None
        if not job or job.status != "done" or not zip_path or not zip_path.exists():
            raise HTTPException(404, "Not ready.")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{job.phrase_id}_wakeword.zip")

    @app.get("/api/preview")
    def preview(phrase: str = Query(...)) -> Response:
        """How the training voices will say the phrase (one random voice per request)."""
        try:
            phrase = validate_phrase(phrase)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        voices = sorted((studio.base / "assets" / "voices").glob("*.onnx"))
        if not voices:
            raise HTTPException(503, "No Piper voice in the base folder.")
        from wakeword_train.tts import PiperSynth

        synth = synth_cache.get("synth") or synth_cache.setdefault("synth", PiperSynth(voices[0], voices[0].with_suffix(".onnx.json")))
        rng = np.random.default_rng()
        audio = synth.synth(phrase.lower(), int(rng.integers(synth.num_speakers)), 1.0, 0.667, 0.8)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())
        return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/vad/silero_vad.onnx")
    def vad() -> FileResponse:
        return FileResponse(vad_model)

    if static_dir and static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse)
        def no_ui() -> str:
            return "<p>The Studio page isn't built yet. Run <code>npm run studio:build</code> in <code>web/</code>, then restart.</p>"

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path, help="shared data from a finished pipeline run")
    ap.add_argument("--template", type=Path, default=Path("configs/uno.yaml"), help="config whose voices, URLs and settings jobs inherit")
    ap.add_argument("--jobs", type=Path, default=Path("studio-jobs"), help="where each phrase's work and models are kept")
    ap.add_argument("--profile", default="fast", help="default profile from studio/profiles.yaml")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to share with the team on your network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--static", type=Path, default=REPO / "web" / "dist-studio")
    ap.add_argument("--vad", type=Path, default=REPO / "models" / "vad" / "silero_vad.onnx")
    args = ap.parse_args()

    if not (args.base / "assets" / "cmudict.dict").exists():
        download(CMUDICT_URL, args.base / "assets" / "cmudict.dict")  # older bases predate the dictionary
    profiles = yaml.safe_load((Path(__file__).with_name("profiles.yaml")).read_text(encoding="utf-8"))
    studio = Studio(JobStore(args.jobs.resolve()), args.base.resolve(), args.template.resolve(), profiles, args.profile)
    state = studio.readiness()
    print(f"Base {args.base}: {'ready' if state['ready'] else 'missing ' + ', '.join(state['missing'])}; background audio {state['negative_hours']} h")
    uvicorn.run(create_app(studio, args.static, args.vad), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
