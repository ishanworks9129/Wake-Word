# Models shipped with the apps

| Path | What | License |
| --- | --- | --- |
| `vad/silero_vad.onnx` | [Silero VAD](https://github.com/snakers4/silero-vad) voice activity detector | MIT, see `vad/LICENSE.txt` |
| `wakeword/` | Output of the Colab training run (`wakeword_models.zip`): feature models, `hey_uno.onnx`, `hello_uno.onnx`, `models.json`, `golden.json` | Feature models: Apache-2.0 ([openWakeWord](https://github.com/dscripka/openWakeWord)). Classifiers: ours; training data licences in `DATASET_MANIFEST.csv` |

Unzip the Colab download into `models/wakeword/`. The apps load it with `ModelPackage`, and the C# golden test
should be re-run against its `golden.json` before shipping a new model.
