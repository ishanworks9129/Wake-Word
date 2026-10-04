# Wake Word → Deepgram

An on-device wake word for .NET MAUI (Android, iOS, Windows, macOS) and web. When it fires, the app opens a Deepgram streaming session. No audio leaves the device before that.

The design is in the v5 plan: https://claude.ai/code/artifact/3a4d4999-713d-4338-89fb-166ab0df5caf. Section numbers in code comments ("plan 8.4") refer to it.

## Layout

| Path | What it is | Status |
| --- | --- | --- |
| `src/WakeWord.Core` | Platform-neutral pipeline: pre-roll ring, VAD gate, detector, Deepgram session, token cache, transcript cleanup, controller | Built, tested |
| `src/WakeWord.TokenBroker` | ASP.NET Core service that issues short-lived Deepgram tokens per user, with rate limits | Built, tested |
| `training/wakeword_train/` | Wake word training pipeline (Piper voices, open negatives, openWakeWord features, ONNX export) | Built, smoke-tested |
| `training/colab/` | One-file Colab notebook that runs the pipeline on a free GPU | Ready to run |
| `training/eval/`, `training/data/` | Section 3 metrics, detector mirror, dataset manifest checks | Built, tested |
| `testdata/golden/` | Cases that the C#, Python and (later) TypeScript detectors must all pass | In use |
| `src/WakeWord.Onnx` | ONNX Runtime implementations of `IWakeWordModel` and `IVoiceActivityDetector` | Next |
| `src/WakeWord.Maui` | MAUI app, per-platform `IAudioCaptureService`, Android foreground service | Next |
| `web/` | TypeScript AudioWorklet capture + `onnxruntime-web` | Later |

## Run the tests

The .NET 10 SDK is required.

```sh
dotnet test                                                  # 38 tests: core + broker
cd training && python -m unittest discover -s tests -t .     # 27 tests; the pipeline tests need numpy, scipy, torch, onnxruntime
```

## Run the token broker locally

```sh
cd src/WakeWord.TokenBroker
dotnet user-secrets init
dotnet user-secrets set Deepgram:ApiKey <server-side key, Member role>
dotnet run
curl -X POST http://localhost:5038/v1/deepgram/token -H "X-Dev-User: me"
```

The `X-Dev-User` header works only in Development. In any other environment the broker refuses to start unless `Auth:Authority` and `Auth:Audience` point at the app's identity provider.

## Training tools

```sh
cd training
python -m data.manifest DATASET_MANIFEST.csv   # rejects NC/ND licenses, missing provenance, leaked test speakers
python -m eval.metrics results.json            # pass/fail per noise band using 95% upper bounds
```

## Train the wake word models

1. Open [colab.research.google.com](https://colab.research.google.com), choose *Upload*, and pick `training/colab/train_wake_words.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
3. Listen to the pronunciation previews in step 2, then let it run (about 2–5 hours). It saves to `MyDrive/wakeword/uno-v0` and resumes after disconnects.
4. Step 8 downloads `wakeword_models.zip`. Unzip it into `models/` in this repo.

The notebook is generated from `training/`; after changing the pipeline or `configs/uno.yaml`, run `python colab/build_notebook.py` from `training/`.

To check the pipeline locally first (Linux or WSL, about 10 minutes, makes a useless model):

```sh
cd training
python -m wakeword_train.pipeline --config configs/smoke.yaml --work /tmp/ww-smoke --steps all
```
