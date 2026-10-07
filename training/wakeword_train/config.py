"""Typed run configuration, loaded from YAML."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Licensed:
    license: str
    license_url: str


@dataclass
class PhraseConfig:
    id: str  # file-safe id, e.g. "hey_uno"
    display: str  # what users say, e.g. "Hey UNO"
    tts_texts: list[str]  # spellings fed to TTS so it pronounces the phrase correctly
    adversarial_texts: list[str]  # near-misses that must NOT fire
    transcript_variants: list[str] = field(default_factory=list)  # Deepgram mis-hearings to strip in the app


@dataclass
class VoiceConfig(Licensed):
    name: str
    model_url: str
    config_url: str


@dataclass
class TtsConfig:
    voices: list[VoiceConfig]
    positives_per_phrase: int = 20000
    adversarial_per_phrase: int = 20000
    length_scale: tuple[float, float] = (0.75, 1.35)
    noise_scale: tuple[float, float] = (0.4, 1.0)
    noise_w_scale: tuple[float, float] = (0.5, 1.2)
    holdout_every_nth_speaker: int = 10  # speaker_id % n == 0 is held out for validation
    use_cuda: bool = False
    part_size: int = 2000  # clips per resumable part
    workers: int = 1  # parallel synthesis processes; parts are the unit of work


@dataclass
class NegativeSource(Licensed):
    name: str
    # "tar": each url is a .tar/.tar.gz of audio files; "files": each url is one audio file; "embeddings": urls[0] is
    # a folder of .npy features already computed elsewhere (python -m wakeword_train.test_package --export-train)
    kind: str
    urls: list[str]
    split: str  # "train" or "val"
    max_hours: float
    include: list[str] = field(default_factory=list)  # member path substrings to keep; empty keeps all audio
    noise_bank: list[str] = field(default_factory=list)  # member path substrings also added to the augmentation noise bank
    # Fetch each archive to disk before reading it (deleted when done): for servers that reset a stream left
    # idle while long recordings are processed, which would otherwise mean re-streaming from the start.
    download_first: bool = False
    source_url: str = ""
    release_id: str = ""  # for license Internal-Consent: reference to the participants' consent


@dataclass
class AugmentConfig:
    copies_per_clip: int = 2
    snr_db: tuple[float, float] = (0.0, 25.0)
    noise_prob: float = 0.9
    rir_prob: float = 0.5
    gain_db: tuple[float, float] = (-12.0, 6.0)
    band_limit_prob: float = 0.3
    speed_prob: float = 0.3
    speed_range: tuple[float, float] = (0.9, 1.1)
    trailing_seconds: tuple[float, float] = (0.0, 0.4)  # audio after the phrase ends, so N consecutive frames can fire
    noise_bank_hours: float = 6.0


@dataclass
class TrainConfig:
    steps: int = 30000
    batch_positive: int = 256
    batch_adversarial: int = 256
    batch_negative: int = 512
    layer_size: int = 128
    n_blocks: int = 1
    max_negative_weight: float = 1000.0
    learning_rates: tuple[float, ...] = (1e-4, 1e-5, 1e-6)
    phase_fractions: tuple[float, ...] = (1.0, 0.1, 0.1)
    eval_points: int = 20
    average_top_k: int = 5
    seed: int = 1234
    max_negative_hours: float = 0.0  # 0 = use all; caps RAM when the shared store is large


@dataclass
class EvalConfig:
    consecutive_frames: int = 3
    refractory_seconds: float = 1.5
    # Sensitivity 0..1 maps (log scale) onto these false accepts per hour on the validation set.
    fa_per_hour_at_sensitivity_0: float = 0.05
    fa_per_hour_at_sensitivity_1: float = 5.0
    default_sensitivity: float = 0.5
    val_positive_snr_db: tuple[float, float] = (5.0, 20.0)


@dataclass
class TestConfig:
    """Your own recordings with no wake word in them, such as team meetings: the frozen test set (plan 5.3).

    Only ever measured on, never trained or calibrated on, so their false-accept rate stays an honest estimate.
    """

    folder: str = ""  # audio or video files, searched recursively (Meet and Teams .mp4 work); empty = no test step
    license: str = "Internal-Consent"  # participants agreed to training use (training/data/manifest.py)
    license_url: str = ""  # where that consent is recorded
    release_id: str = ""  # reference to the consent or its approval


@dataclass
class Config:
    run_name: str
    phrases: list[PhraseConfig]
    tts: TtsConfig
    negatives: list[NegativeSource]
    rir: Licensed
    rir_url: str
    feature_models: Licensed
    melspectrogram_url: str
    embedding_url: str
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    test: TestConfig = field(default_factory=TestConfig)
    window_samples: int = 32000  # 2.0 s of audio -> exactly 16 embeddings
    # Folder holding assets/, negatives/ and noise/ shared by many runs (Wake Word Studio); empty = the work folder.
    shared_dir: str = ""
    cmudict_url: str = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    feature_threads: int = 0  # 0 = all cores
    features_use_cuda: bool = False  # needs onnxruntime-gpu (the Colab notebook installs it)

    def phrase(self, phrase_id: str) -> PhraseConfig:
        return next(p for p in self.phrases if p.id == phrase_id)


def _build(tp, value):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        known = {f.name for f in dataclasses.fields(tp)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"{tp.__name__}: unknown keys {sorted(unknown)}")
        return tp(**{k: _build(hints[k], v) for k, v in value.items()})
    if origin is list:
        (item,) = typing.get_args(tp)
        return [_build(item, v) for v in value]
    if origin is tuple:
        return tuple(value)
    return value


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for dotted, v in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return _build(Config, data)
