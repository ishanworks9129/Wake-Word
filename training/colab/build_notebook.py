"""Builds train_wake_words.ipynb: one self-contained Colab notebook carrying the pipeline code and config.

    python colab/build_notebook.py            # from training/

Re-run after changing anything under wakeword_train/, eval/ or configs/uno.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("train_wake_words.ipynb")
CODE_FILES = sorted([*TRAINING.glob("wakeword_train/*.py"), *TRAINING.glob("eval/*.py")])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, hidden: bool = False) -> dict:
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def step(name: str, title: str, extra: str = "") -> dict:
    return code(f"#@title {title}\n!cd /content/training && python -m wakeword_train.pipeline --config configs/uno.yaml --work \"$WORK\" --steps {name}\n{extra}")


def build() -> dict:
    cells = [
        md("""
# Train the "Hey UNO" and "Hello UNO" wake words

This notebook builds both wake-word models from openly licensed data, all of it allowed for commercial use, and packages them for the app.

**Before you start:** *Runtime → Change runtime type → T4 GPU*. You need about **10 GB free in Google Drive**: everything is saved to `MyDrive/wakeword/<run>` (default `uno-v1`) so nothing is lost if Colab disconnects.

**To run:** *Runtime → Run all*. Expect 5–7 hours in total; the free tier may stop sooner, which is fine because every step resumes. If the session drops, reconnect and *Run all* again; finished steps are skipped and long steps resume where they stopped.

| Step | What it does | Time on a T4 (approx.) |
| --- | --- | --- |
| Setup | Downloads the feature models, the Piper voice and room recordings | 2 min |
| Preview | Lets you listen to how the voices say each phrase | 1 min |
| Voices | Generates 20,000 spoken examples per phrase, plus 20,000 near-misses | about 3 h (about 4 min per 2,000) |
| Background audio | Streams about 550 hours of speech, meetings, music and noise and turns it into features | 1.5–3.5 h |
| Features | Mixes the examples into noisy rooms and computes features | 15–30 min |
| Train | Trains one small classifier per phrase | 10–20 min |
| Evaluate | Measures recall and false accepts per hour and calibrates sensitivity | 5 min |
| Export | Writes the ONNX models and `models.json`, and downloads a zip | 1 min |
"""),
        code("""
#@title Connect Google Drive and check the GPU
from google.colab import drive
drive.mount('/content/drive')
import os, shutil, subprocess
RUN = 'uno-v1'  #@param {type:"string"}
WORK = f'/content/drive/MyDrive/wakeword/{RUN}'
os.makedirs(WORK, exist_ok=True)
os.environ['WORK'] = WORK
print('Saving to', WORK)
gpu = shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
print(gpu or 'No GPU: switch the runtime to T4 GPU for training, or set features_use_cuda: false in the configuration to run on CPU.')
"""),
        code("""
#@title Install dependencies
!pip install -q piper-tts==1.8.0 onnx
# piper-tts pulls in CPU onnxruntime; swap in the GPU build (same Python API).
!pip uninstall -y -q onnxruntime onnxruntime-gpu && pip install -q onnxruntime-gpu
import onnxruntime
print('onnxruntime', onnxruntime.__version__, onnxruntime.get_available_providers())
""", hidden=True),
    ]

    files = []
    for path in CODE_FILES:
        rel = path.relative_to(TRAINING).as_posix()
        files.append(f"{rel!r}: {path.read_text(encoding='utf-8')!r}")
    cells.append(code(
        "#@title Write the training pipeline code (generated from the repo's training/ folder; do not edit here)\n"
        "import os\nFILES = {\n" + ",\n".join(files) + "\n}\n"
        "for rel, text in FILES.items():\n"
        "    path = os.path.join('/content/training', rel)\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "    open(path, 'w').write(text)\n"
        "os.makedirs('/content/training/configs', exist_ok=True)  # the %%writefile cell below cannot create folders\n"
        "print(f'wrote {len(FILES)} files')",
        hidden=True,
    ))

    config = (TRAINING / "configs" / "uno.yaml").read_text(encoding="utf-8")
    cells += [
        md("""
## Configuration
Edit here if needed, for example the near-miss phrases or `positives_per_phrase`. The defaults are the v0 run from the plan.
"""),
        code("%%writefile /content/training/configs/uno.yaml\n" + config.replace("\r\n", "\n").rstrip("\n")),
        step("setup", "1. Setup"),
        step("preview", "2. Preview pronunciation",
             "import glob\nfrom IPython.display import Audio, display\n"
             "for f in sorted(glob.glob(f'{WORK}/preview/*.wav')):\n"
             "    print(os.path.basename(f)); display(Audio(f))"),
        md("""
**Listen to the previews.** Each should sound like "Hey OO-noh" or "Hello OO-noh". If a spelling sounds wrong, remove it from `tts_texts` in the configuration cell, run that cell again, then delete `steps/preview.done` in the Drive folder and re-run the preview.
"""),
        step("tts", "3. Generate voices"),
        step("negatives", "4. Background audio (longest step; resumes if interrupted)"),
        step("features", "5. Features"),
        step("train", "6. Train"),
        step("evaluate", "7. Evaluate", "print(open(f'{WORK}/steps/evaluate.done').read())"),
        step("export", "8. Export and download",
             "from google.colab import files\n"
             "zip_path = f'{WORK}/export/wakeword_models.zip'\n"
             "print(open(f'{WORK}/export/wakeword_models/METRICS.md').read())\n"
             "files.download(zip_path)"),
        code("""
#@title 9. (Optional) Pack the base for Wake Word Studio
# Wake Word Studio trains new phrases on your own machine, reusing this run's background audio,
# noise and models. This writes one file to download: about 4-5 GB.
import os, tarfile
out = f'{WORK}/studio_base.tar'
if not os.path.exists(out):
    with tarfile.open(out + '.tmp', 'w') as t:
        for rel in ['assets', 'noise/bank.npy', 'negatives']:
            t.add(os.path.join(WORK, rel), arcname=f'studio_base/{rel}')
    os.replace(out + '.tmp', out)
print(f'{out}: {os.path.getsize(out) / 2**30:.1f} GB. Download it from Google Drive (wakeword/{RUN}/studio_base.tar).')
"""),
        md("""
## (Optional) Recalibrate a finished run
Only for a run trained before the validation audio was extended (meetings and parliamentary sessions, not just audiobooks), such as `uno-v1`. New runs already include it, so skip this.

Set `RUN` in the first cell to that run, run the cells down to *Configuration*, then tick **RECALIBRATE** below and run it. It downloads only the missing validation audio (about 110 hours), re-measures both models, recalibrates sensitivity and downloads a new zip. No retraining: about 30–60 minutes.
"""),
        code("""
#@title 10. (Optional) Recalibrate on the extended validation audio
RECALIBRATE = False  #@param {type:"boolean"}
if RECALIBRATE:
    get_ipython().system('cd /content/training && python -m wakeword_train.pipeline --config configs/uno.yaml --work "$WORK" '
                         '--steps negatives,evaluate,export --force --negatives-split val')
    from google.colab import files
    print(open(f'{WORK}/export/wakeword_models/METRICS.md').read())
    files.download(f'{WORK}/export/wakeword_models.zip')
else:
    print('Skipped. Tick RECALIBRATE and run this cell to recalibrate a finished run.')
"""),
        md("""
## What you get
`wakeword_models.zip` holds `melspectrogram.onnx`, `embedding_model.onnx`, `hey_uno.onnx`, `hello_uno.onnx`, `models.json` (keywords, sensitivity tables and detector settings), `golden.json` (parity vectors for the app), `METRICS.md` and `DATASET_MANIFEST.csv`. Copy it into the repo as `models/` and share `METRICS.md` with the team.

These are **v0** models trained on synthetic voices. Retrain with the real recordings (plan 5.2) before release.
"""),
    ]
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
        "cells": cells,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(CODE_FILES)} code files)")
