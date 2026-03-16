"""
training/pretrain_human.py
===========================
Drop-in joint pretraining (value + behavioral cloning) from human play data.

Expected JSON schema
---------------------
    [
        {"obs": [float, ...], "action": int, "battle_id": "str"},
        ...
    ]

Usage
------
    from training.pretrain_human import pretrain_from_human_data

    model = MaskablePPO(AttentionPointerPolicy, env=train_env, ...)
    model = pretrain_from_human_data(model, "data/human_play.json")
    model.learn(total_timesteps=200_000, ...)
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib import MaskablePPO

from env.battle_state import OBS_SIZE

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_GAMMA       = 0.99
_WIN_REWARD  =  15.0
_LOSS_REWARD = -10.0
_DRAW_REWARD =   0.0
_VALUE_CLIP  =  30.0
_GRAD_CLIP   =   5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_episodes(data_path: str, episode_key: str) -> list[dict]:
    """Group flat JSON records into per-episode dicts.

    Each episode dict has keys ``obs`` (list of arrays), ``actions`` (list
    of ints), and ``outcome`` (int: +1 win, -1 loss, 0 draw).

    :param data_path: Path to the human play JSON file.
    :param episode_key: Record field used to group steps into episodes.
    :returns: List of episode dicts.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        records: list[dict] = [json.loads(line) for line in f if line.strip()]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get(episode_key, "unknown"))].append(rec)

    episodes = []
    skipped  = 0

    for battle_id, steps in grouped.items():
        obs_list, action_list, outcome = [], [], 1

        for step in steps:
            obs_arr = np.asarray(step.get("obs", []), dtype=np.float32)
            action  = step.get("action")

            if obs_arr.shape != (OBS_SIZE,) or action is None or not (0 <= int(action) <= 25):
                skipped += 1
                continue

            obs_list.append(obs_arr)
            action_list.append(int(action))

            if "outcome" in step:           # optional per-episode override
                outcome = int(step["outcome"])

        if len(obs_list) >= 2:
            episodes.append(dict(obs=obs_list, actions=action_list, outcome=outcome))
        else:
            skipped += len(obs_list)

    total_steps = sum(len(e["obs"]) for e in episodes)
    print(
        f"[pretrain] Loaded {len(episodes)} episodes / {total_steps} steps "
        f"(skipped {skipped} invalid)"
    )
    return episodes


def _compute_returns(episodes: list[dict], gamma: float) -> np.ndarray:
    """Compute clipped, normalised discounted returns for every step.

    :param episodes: List of episode dicts from ``_load_episodes``.
    :param gamma: Discount factor.
    :returns: Float32 array of shape ``(N_total_steps,)``.
    """
    terminal_rewards = {1: _WIN_REWARD, -1: _LOSS_REWARD, 0: _DRAW_REWARD}
    all_returns = []

    for ep in episodes:
        T   = len(ep["obs"])
        ret = np.zeros(T, dtype=np.float32)
        g   = float(terminal_rewards[ep["outcome"]])
        for t in reversed(range(T)):
            ret[t] = g
            g      = gamma * g
        all_returns.append(ret)

    returns = np.concatenate(all_returns)
    returns = np.clip(returns, -_VALUE_CLIP, _VALUE_CLIP)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pretrain_from_human_data(
        model: MaskablePPO,
        data_path: str,
        *,
        episode_key: str   = "battle_id",
        n_epochs:    int   = 100,
        batch_size:  int   = 256,
        lr:          float = 3e-4,
        lambda_value: float = 1.0,
        lambda_bc:    float = 1.0,
        gamma:        float = _GAMMA,
) -> MaskablePPO:
    """Jointly pretrain the value head and policy from human play data.

    Trains all policy parameters with a combined loss:
        L = lambda_value · MSE(V(s), G_t)  +  lambda_bc · CE(logits(s), a)

    Value targets G_t are computed as discounted Monte-Carlo returns from a
    synthetic terminal reward (win=+15, loss=-10, draw=0), since step-level
    rewards are not available in human recordings.

    :param model: Freshly constructed ``MaskablePPO`` model (modified in-place).
    :param data_path: Path to the human play JSON file.
    :param episode_key: JSON field used to group steps into episodes.
    :param n_epochs: Number of supervised training epochs.
    :param batch_size: Mini-batch size.
    :param lr: Learning rate.
    :param lambda_value: Weight for the value MSE loss term.
    :param lambda_bc: Weight for the behavioral cloning CE loss term.
    :param gamma: Discount factor for return computation.
    :returns: The same ``model`` with pretrained weights.
    """
    policy = model.policy
    dev    = model.device

    # ── Load & build dataset ──────────────────────────────────────────────────
    episodes = _load_episodes(data_path, episode_key)
    returns  = _compute_returns(episodes, gamma)

    obs_np = np.stack([o for ep in episodes for o in ep["obs"]])
    act_np = np.array([a for ep in episodes for a in ep["actions"]], dtype=np.int64)

    obs_t = torch.tensor(obs_np,  dtype=torch.float32, device=dev)
    act_t = torch.tensor(act_np,  dtype=torch.long,    device=dev)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=dev).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(obs_t, act_t, ret_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ── Optimiser over all parameters ────────────────────────────────────────
    optimizer = optim.Adam(list(policy.parameters()), lr=lr)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"[pretrain] {len(obs_np)} steps | epochs={n_epochs} | λ_v={lambda_value} λ_bc={lambda_bc}")
    policy.train()

    for epoch in range(1, n_epochs + 1):
        total_loss = v_loss = bc_loss = 0.0
        n_batches  = 0

        for batch_obs, batch_act, batch_ret in loader:
            optimizer.zero_grad()

            features = policy.mlp_extractor(batch_obs)
            loss_v   = mse(policy.value_head(features), batch_ret)
            loss_bc  = ce(policy._build_logits(features), batch_act)
            loss     = lambda_value * loss_v + lambda_bc * loss_bc

            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()), _GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            v_loss     += loss_v.item()
            bc_loss    += loss_bc.item()
            n_batches  += 1

        nb = max(n_batches, 1)
        print(
            f"  epoch {epoch:3d}/{n_epochs} | "
            f"total={total_loss/nb:.4f}  value={v_loss/nb:.4f}  bc={bc_loss/nb:.4f}"
        )

    policy.eval()
    print("[pretrain] Done.\n")
    return model
