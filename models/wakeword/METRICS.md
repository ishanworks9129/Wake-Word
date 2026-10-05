# Wake word models: uno-v0

- Provisional v0: positives are synthetic (Piper) only; negatives are read speech, music and noise.
- Recall is measured on held-out synthetic voices and FA/hour on librispeech_val (19 h), voxpopuli_val (40 h), ami_val (71 h), so real-world numbers will differ. Re-measure on the real-recording test set (plan 5.3) before release.
- All training data is commercially licensed; see DATASET_MANIFEST.csv.

| Keyword | Sensitivity | Threshold | Recall (held-out synthetic voices) | False accepts / hour | Negative hours |
| --- | --- | --- | --- | --- | --- |
| Hey UNO | 0.5 | 0.790 | 85.8% | 0.39 | 130.4 |
| Hello UNO | 0.5 | 0.710 | 90.4% | 0.39 | 130.4 |

## Sensitivity calibration

| Sensitivity | Hey UNO | Hello UNO |
| --- | --- | --- |
| 0.0 | 0.999 | 0.999 |
| 0.1 | 0.999 | 0.999 |
| 0.2 | 0.998 | 0.998 |
| 0.3 | 0.990 | 0.990 |
| 0.4 | 0.930 | 0.930 |
| 0.5 | 0.790 | 0.710 |
| 0.6 | 0.550 | 0.400 |
| 0.7 | 0.270 | 0.180 |
| 0.8 | 0.100 | 0.050 |
| 0.9 | 0.050 | 0.020 |
| 1.0 | 0.020 | 0.010 |

## Training audio (hours)

- librispeech_train: 300.0
- musan: 108.9
- librispeech_val: 19.0
- voxpopuli_val: 40.0
- ami_val: 71.5
