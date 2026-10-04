"""Commercial-clean wake word training: Piper positives, openly licensed negatives, openWakeWord features.

Run on Colab via training/colab/train_wake_words.ipynb, or locally with
    python -m wakeword_train.pipeline --config configs/uno.yaml --work <dir>
"""

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms; one embedding per frame
CLASSIFIER_FRAMES = 16  # embeddings per classifier input (1.28 s)
EMBEDDING_DIM = 96
