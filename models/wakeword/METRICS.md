# Wake word models: uno-v0

- Provisional v0: positives are synthetic (Piper) only; training negatives are librispeech_train (300 h), musan (109 h), voxpopuli_train (60 h), ami_train (55 h).
- Recall is measured on held-out synthetic voices and FA/hour on librispeech_val (19 h), voxpopuli_val (40 h), ami_val (71 h), so real-world numbers will differ. Re-measure on the real-recording test set (plan 5.3) before release.
- All training data is commercially licensed; see DATASET_MANIFEST.csv.

| Keyword | Sensitivity | Threshold | Recall (held-out synthetic voices) | False accepts / hour | Negative hours |
| --- | --- | --- | --- | --- | --- |
| Hey UNO | 0.5 | 0.200 | 85.5% | 0.39 | 130.4 |
| Hello UNO | 0.5 | 0.030 | 89.7% | 0.33 | 130.4 |

## Sensitivity calibration

| Sensitivity | Hey UNO | Hello UNO |
| --- | --- | --- |
| 0.0 | 0.998 | 0.720 |
| 0.1 | 0.996 | 0.520 |
| 0.2 | 0.950 | 0.180 |
| 0.3 | 0.860 | 0.070 |
| 0.4 | 0.510 | 0.050 |
| 0.5 | 0.200 | 0.030 |
| 0.6 | 0.080 | 0.013 |
| 0.7 | 0.040 | 0.010 |
| 0.8 | 0.020 | 0.010 |
| 0.9 | 0.010 | 0.010 |
| 1.0 | 0.010 | 0.010 |

## Training audio (hours)

- librispeech_train: 300.0
- musan: 108.9
- voxpopuli_train: 60.0
- ami_train: 54.9
- librispeech_val: 19.0
- voxpopuli_val: 40.0
- ami_val: 71.5
