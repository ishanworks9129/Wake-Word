"""Classifier head over 16 x 96 embeddings (same architecture as openWakeWord's "dnn")."""

from __future__ import annotations

import torch
from torch import nn

from . import CLASSIFIER_FRAMES, EMBEDDING_DIM


class WakeWordDNN(nn.Module):
    def __init__(self, layer_size: int = 128, n_blocks: int = 1):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(CLASSIFIER_FRAMES * EMBEDDING_DIM, layer_size),
            nn.LayerNorm(layer_size),
            nn.ReLU(),
        ]
        for _ in range(n_blocks):
            layers += [nn.Linear(layer_size, layer_size), nn.LayerNorm(layer_size), nn.ReLU()]
        layers += [nn.Linear(layer_size, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 16, 96] -> [B, 1] score in [0, 1]."""
        return self.net(x)


def average_state_dicts(states: list[dict]) -> dict:
    return {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
