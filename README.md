# Wake Word → Deepgram

An on-device wake word for .NET MAUI (Android, iOS, Windows, macOS) and web. When it fires, the app opens a Deepgram streaming session. No audio leaves the device before that.

The design is in the v5 plan: https://claude.ai/code/artifact/3a4d4999-713d-4338-89fb-166ab0df5caf. Section numbers in code comments ("plan 8.4") refer to it.

## Layout

| Path | What it is | Status |
| --- | --- | --- |
| `src/WakeWord.Core` | Platform-neutral pipeline: pre-roll ring, VAD gate, detector, Deepgram session, token cache, transcript cleanup, controller | Built, tested |
| `src/WakeWord.TokenBroker` | ASP.NET Core service that issues short-lived Deepgram tokens per user, with rate limits | Built, tested |
| `src/WakeWord.Recorder` | Consent-first web page for collecting real recordings (26 prompts per person), with withdrawal and admin export | Built, tested; consent text is a draft |
| `training/wakeword_train/` | Wake word training pipeline (Piper voices, open negatives, openWakeWord features, ONNX export) | Built, smoke-tested |
| `training/colab/` | One-file Colab notebook that runs the pipeline on a free GPU | Ready to run |
| `training/studio/` + `web/studio/` | Wake Word Studio: type a phrase, train it (about 12 min), test it with your mic in the browser, download it | Built, tested end to end |
| `training/eval/`, `training/data/` | Section 3 metrics, detector mirror, dataset manifest checks | Built, tested |
| `testdata/` | Golden detector cases and a smoke model package with streaming golden vectors, shared by C#, Python and (later) TypeScript | In use |
| `models/` | Models the apps ship: Silero VAD now, the Colab output in `models/wakeword/` | VAD in place |
| `src/WakeWord.Onnx` | ONNX Runtime wake word model (streaming features, both keywords), Silero VAD, model package loader, Picovoice-style `KeywordSpotter` | Built, golden-tested against Python |
| `src/WakeWord.Maui` | MAUI app, per-platform `IAudioCaptureService`, Android foreground service | Next |
| `web/` | Browser client in TypeScript: mic capture with resampling, the same streaming model and Silero VAD on `onnxruntime-web` (single-threaded WASM, no COOP/COEP), Deepgram handoff, Picovoice-style `KeywordSpotter`, demo page | Built, golden-tested against Python |

## Run the tests

The .NET 10 SDK is required.

```sh
dotnet test                                                  # 59 tests: core, ONNX runtime, broker, recorder
node --test tests/recorder-js/wav.test.mjs                   # recorder page helpers
cd web && npm install && npm test                            # 35 tests: browser client, same golden data
cd training && python -m unittest discover -s tests -t .     # 20 tests; the pipeline tests need numpy, scipy, torch, onnxruntime
```

## Use the wake word in C#

Picovoice-style, for any app that just needs "did they say it?":

```csharp
using var spotter = KeywordSpotter.Create("models/wakeword", ["hey_uno", "hello_uno"], sensitivities: [0.5, 0.5]);
int keyword = spotter.Process(frame);   // 1280 samples of 16 kHz mono PCM; -1, or the index of the keyword heard
```

Sensitivity runs from 0 (fewest false accepts) to 1 (fewest misses) and is calibrated per keyword by the training run.
For the full always-on flow (VAD gating, pre-roll, Deepgram handoff), use `WakeWordController` with
`OnnxWakeWordModel`, `SileroVad` and `package.KeywordOptions(...)`.

## Use the wake word in the browser

```ts
import { WakeWordListener, CachingTokenProvider, brokerTokenSource, configureOnnxRuntime } from "@uno/wake-word";

configureOnnxRuntime();
const listener = await WakeWordListener.create({
  modelBaseUrl: "/models/wakeword/",
  vadModelUrl: "/models/vad/silero_vad.onnx",
  sensitivities: [0.5, 0.5],
  tokens: new CachingTokenProvider(brokerTokenSource("/v1/deepgram/token", () => ({ Authorization: `Bearer ${idToken}` }))),
});
listener.ondetect = (d) => console.log("heard", d.keyword);
listener.ontranscript = (t) => console.log(t.text);
await listener.start(); // call from a click: browsers only start audio after a user gesture
```

Try it: `cd web && npm run demo`, then open http://localhost:5173 (it serves `models/` from this repo). The microphone
needs https or localhost. Listening stops when the tab closes, and on iOS Safari when the page is backgrounded.

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

## Wake Word Studio (train new phrases, Picovoice-style)

An internal page like Picovoice's console: type a phrase, press **Train**, test it with your microphone in the
browser, and download the package the apps load. Each phrase gets its own trained model; near-miss phrases are
generated automatically from the CMU Pronouncing Dictionary. It runs on your machine in WSL and reuses the
background audio from the Colab run, so only the new phrase has to be processed.

One-time setup:
1. VS Code: *Tasks: Run Task -> Studio: set up WSL environment (once)*.
2. After the Colab run, run its step 9, download `studio_base.tar` from Drive, and unpack it in WSL:
   `wsl -d Ubuntu -- tar -xf /mnt/c/Users/<you>/Downloads/studio_base.tar -C ~/ww`

Then *Tasks: Run Task -> Studio: start* (accept the default base `~/ww/studio_base`) opens http://localhost:8765.
Profiles: **fast** (default; measured 11.7 min per phrase on a 16-core laptop) and **thorough** (closer to the main run, about an hour). Jobs and models are kept in
`~/ww/studio-jobs`; a restart resumes any job that was running.

## Collect real recordings

```sh
cd src/WakeWord.Recorder
dotnet user-secrets init
dotnet user-secrets set Recorder:InviteCode <code you give contributors>
dotnet user-secrets set Recorder:AdminKey <long random secret>
dotnet run
```

Phones only allow the microphone over **https**, so host it somewhere with a certificate (Azure App Service or
any VM behind a reverse proxy), or for a single session expose `dotnet run` through an https tunnel such as
Cloudflare Tunnel. Recordings go to `Recorder:StoragePath` (git-ignored). Download everything with
`GET /api/admin/export` and the `X-Admin-Key` header; `manifest.csv` inside passes `python -m data.manifest`.

**Before anyone outside the team records:** legal must approve `wwwroot/consent/v1-draft.html`. Publish the approved
text as a new version (for example `v1.html`) and set `Recorder:ConsentVersion` to match.

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
