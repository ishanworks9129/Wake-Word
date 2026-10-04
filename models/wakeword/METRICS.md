# Wake word models: uno-v0

- Provisional v0: positives are synthetic (Piper) only; negatives are read speech, music and noise.
- Recall is measured on held-out synthetic voices and FA/hour on read speech, so real-world numbers will differ. Re-measure on the real-recording test set (plan 5.3) before release.
- All training data is commercially licensed; see DATASET_MANIFEST.csv.

| Keyword | Sensitivity | Threshold | Recall (held-out synthetic voices) | False accepts / hour | Negative hours |
| --- | --- | --- | --- | --- | --- |
| Hey UNO | 0.5 | 0.020 | 93.2% | 0.05 | 19.0 |
| Hello UNO | 0.5 | 0.020 | 95.4% | 0.00 | 19.0 |

## Sensitivity calibration

| Sensitivity | Hey UNO | Hello UNO |
| --- | --- | --- |
| 0.0 | 0.999 | 0.999 |
| 0.1 | 0.999 | 0.999 |
| 0.2 | 0.999 | 0.999 |
| 0.3 | 0.030 | 0.020 |
| 0.4 | 0.020 | 0.020 |
| 0.5 | 0.020 | 0.020 |
| 0.6 | 0.020 | 0.020 |
| 0.7 | 0.020 | 0.020 |
| 0.8 | 0.020 | 0.020 |
| 0.9 | 0.020 | 0.020 |
| 1.0 | 0.020 | 0.020 |

## Training audio (hours)

- librispeech_train: 300.0
- musan: 108.9
- librispeech_val: 19.0
