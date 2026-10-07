# Wake word models: uno-v0

- Provisional v0: positives are synthetic (Piper) only; training negatives are librispeech_train (300 h), musan (109 h), voxpopuli_train (60 h), ami_train (55 h), meetings_train (25 h).
- Recall is measured on held-out synthetic voices and FA/hour on librispeech_val (19 h), voxpopuli_val (40 h), ami_val (71 h), so real-world numbers will differ. Re-measure on the real-recording test set (plan 5.3) before release.
- All training data is commercially licensed; see DATASET_MANIFEST.csv.

| Keyword | Sensitivity | Threshold | Recall (held-out synthetic voices) | False accepts / hour | Negative hours |
| --- | --- | --- | --- | --- | --- |
| Hey UNO | 0.5 | 0.330 | 84.7% | 0.39 | 130.4 |
| Hello UNO | 0.5 | 0.080 | 92.3% | 0.38 | 130.4 |

## Sensitivity calibration

| Sensitivity | Hey UNO | Hello UNO |
| --- | --- | --- |
| 0.0 | 0.994 | 0.995 |
| 0.1 | 0.990 | 0.990 |
| 0.2 | 0.940 | 0.930 |
| 0.3 | 0.750 | 0.710 |
| 0.4 | 0.530 | 0.320 |
| 0.5 | 0.330 | 0.080 |
| 0.6 | 0.150 | 0.030 |
| 0.7 | 0.070 | 0.013 |
| 0.8 | 0.030 | 0.010 |
| 0.9 | 0.015 | 0.010 |
| 1.0 | 0.010 | 0.010 |

## Training audio (hours)

- librispeech_train: 300.0
- musan: 108.9
- voxpopuli_train: 60.0
- ami_train: 54.9
- meetings_train: 25.1
- librispeech_val: 19.0
- voxpopuli_val: 40.0
- ami_val: 71.5
