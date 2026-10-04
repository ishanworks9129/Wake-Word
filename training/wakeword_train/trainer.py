"""Trains one phrase's classifier head on precomputed embeddings."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASSIFIER_FRAMES
from .config import EvalConfig, TrainConfig
from .evaluate import count_fires, stream_scores, window_scores
from .model import WakeWordDNN, average_state_dicts
from .store import FRAMES_PER_HOUR


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _quick_metrics(model, val_pos, val_neg, device, eval_cfg: EvalConfig) -> tuple[float, float]:
    """Recall on held-out positive windows and false accepts/hour (single frame, refractory) at 0.5."""
    model.eval()
    recall = float((window_scores(model, val_pos, device) >= 0.5).mean()) if len(val_pos) else 0.0
    scores = stream_scores(model, val_neg, device)
    fa = len(count_fires(scores, 0.5, 1, eval_cfg.refractory_seconds))
    hours = len(scores) / FRAMES_PER_HOUR
    model.train()
    return recall, fa / hours if hours else float("inf")


def train(
    positives: np.ndarray,
    adversarial: np.ndarray,
    negative_stream: np.ndarray,
    val_positives: np.ndarray,
    val_negative_stream: np.ndarray,
    cfg: TrainConfig,
    eval_cfg: EvalConfig,
    log=print,
) -> tuple[WakeWordDNN, list[dict]]:
    """positives/adversarial: [N, 16, 96]; negative streams: [T, 96]. All float16 or float32."""
    torch.manual_seed(cfg.seed)
    device = _device()
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    pos = torch.as_tensor(positives, dtype=torch.float16, device=device)
    adv = torch.as_tensor(adversarial, dtype=torch.float16, device=device)
    neg = torch.as_tensor(negative_stream, dtype=torch.float16, device=device)
    val_neg = torch.as_tensor(val_negative_stream, dtype=torch.float16, device=device)
    offsets = torch.arange(CLASSIFIER_FRAMES, device=device)

    model = WakeWordDNN(cfg.layer_size, cfg.n_blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rates[0])

    phase_steps = [max(1, int(cfg.steps * f)) for f in cfg.phase_fractions]
    total = sum(phase_steps)
    eval_at = set(np.linspace(total // 2, total, cfg.eval_points, dtype=int).tolist())
    y = torch.cat([
        torch.ones(cfg.batch_positive, device=device),
        torch.zeros(cfg.batch_adversarial + cfg.batch_negative, device=device),
    ])
    is_neg = y == 0

    history: list[dict] = []
    step, started = 0, time.time()
    for lr, n_steps in zip(cfg.learning_rates, phase_steps):
        for g in opt.param_groups:
            g["lr"] = lr
        for s in range(n_steps):
            step += 1
            neg_weight = 1.0 + (cfg.max_negative_weight - 1.0) * s / max(1, n_steps - 1)
            p_idx = torch.randint(0, pos.shape[0], (cfg.batch_positive,), device=device, generator=gen)
            a_idx = torch.randint(0, adv.shape[0], (cfg.batch_adversarial,), device=device, generator=gen)
            n_idx = torch.randint(0, neg.shape[0] - CLASSIFIER_FRAMES + 1, (cfg.batch_negative,), device=device, generator=gen)
            x = torch.cat([pos[p_idx], adv[a_idx], neg[n_idx[:, None] + offsets]]).float()

            pred = model(x).squeeze(1).clamp(1e-7, 1 - 1e-7)
            # Focus on examples still wrong enough to matter (as openWakeWord does).
            mask = torch.where(is_neg, pred >= 0.001, pred < 0.999).float()
            weights = torch.where(is_neg, torch.full_like(y, neg_weight), torch.ones_like(y))
            loss = (F.binary_cross_entropy(pred, y, reduction="none") * weights * mask).sum() / mask.sum().clamp(min=1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step in eval_at:
                recall, fa_hr = _quick_metrics(model, val_positives, val_neg, device, eval_cfg)
                history.append({"step": step, "loss": float(loss.detach()), "recall": recall, "fa_per_hour": fa_hr,
                                "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})})
                log(f"step {step}/{total}  loss {float(loss.detach()):.4f}  recall@0.5 {recall:.3f}  FA/h@0.5 {fa_hr:.2f}  ({time.time() - started:.0f}s)")

    if not history:
        raise RuntimeError("no checkpoints were evaluated; increase steps")
    pool = _candidates(history, cfg.average_top_k)
    final = WakeWordDNN(cfg.layer_size, cfg.n_blocks).to(device)
    choice = "single best checkpoint"
    final.load_state_dict(pool[0]["state"])
    best_single = (pool[0]["recall"], pool[0]["fa_per_hour"])
    if len(pool) > 1:
        averaged = WakeWordDNN(cfg.layer_size, cfg.n_blocks).to(device)
        averaged.load_state_dict(average_state_dicts([h["state"] for h in pool]))
        avg_metrics = _quick_metrics(averaged, val_positives, val_neg, device, eval_cfg)
        if _objective(*avg_metrics) >= _objective(*best_single):
            final, choice = averaged, f"average of {len(pool)} checkpoints"
    recall, fa_hr = _quick_metrics(final, val_positives, val_neg, device, eval_cfg)
    log(f"selected {choice}: recall@0.5 {recall:.3f}  FA/h@0.5 {fa_hr:.2f}")
    return final.cpu().eval(), [{k: v for k, v in h.items() if k != "state"} for h in history]


def _objective(recall: float, fa_hr: float) -> float:
    return recall - 0.02 * fa_hr


def _candidates(history: list[dict], top_k: int) -> list[dict]:
    """Best checkpoints by recall, among those with at most the median false-accept rate; best first."""
    median_fa = float(np.median([h["fa_per_hour"] for h in history]))
    pool = [h for h in history if h["fa_per_hour"] <= median_fa] or history
    return sorted(pool, key=lambda h: _objective(h["recall"], h["fa_per_hour"]), reverse=True)[:top_k]
