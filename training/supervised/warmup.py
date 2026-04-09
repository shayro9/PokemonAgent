"""
training/supervised/warmup.py
==============================
Joint pretraining (value + behavioral cloning) from human play data.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sb3_contrib import MaskablePPO

from env.battle_config import BattleConfig
from training.supervised.supervised_config import (
    GAMMA, WIN_REWARD, LOSS_REWARD, DRAW_REWARD, VALUE_CLIP, GRAD_CLIP,
    SupervisedWarmupConfig,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_episodes(data_path: str, episode_key: str) -> list[dict]:
    """Load training data from a ``.npz`` archive and group into per-episode dicts.

    Each episode dict has keys ``obs`` (list of arrays), ``actions`` (list of
    ints), ``masks`` (list of bool arrays), and ``outcome`` (int:
    +1 win, -1 loss, 0 draw).

    :param data_path: Path to a .npz file produced by convert_metamon_json.
    :param episode_key: Unused — kept for API compatibility.
    :returns: List of episode dicts.
    """
    if Path(data_path).suffix != ".npz":
        raise ValueError(
            f"Only .npz files are supported, got: {data_path}"
        )
    return _load_episodes_npz(data_path)


def _load_episodes_npz(data_path: str) -> list[dict]:
    """Load episodes from a compressed ``.npz`` archive."""
    _OBS_SIZE = BattleConfig.gen1().obs_dim

    data        = np.load(data_path)
    obs_all     = data["obs"].astype(np.float32)       # (N, obs_dim)
    actions_all = data["actions"].astype(np.int64)     # (N,)
    ep_idx_all  = data["episode_idx"].astype(np.int32) # (N,)
    outcomes_all = data["outcomes"].astype(np.float32) # (N,) — NaN except terminal

    if obs_all.shape[1] != _OBS_SIZE:
        raise ValueError(
            f"NPZ obs dim {obs_all.shape[1]} does not match "
            f"expected {_OBS_SIZE}. Was this file built with a different config?"
        )

    n_episodes = int(ep_idx_all.max()) + 1
    episodes   = []
    skipped    = 0

    for i in range(n_episodes):
        mask    = ep_idx_all == i
        obs_ep  = obs_all[mask]
        acts_ep = actions_all[mask]
        outs_ep = outcomes_all[mask]

        valid = (acts_ep >= 0) & (acts_ep <= 9)
        if not valid.all():
            skipped += (~valid).sum()
            obs_ep  = obs_ep[valid]
            acts_ep = acts_ep[valid]
            outs_ep = outs_ep[valid]

        if len(obs_ep) < 2:
            skipped += len(obs_ep)
            continue

        terminal = ~np.isnan(outs_ep)
        outcome  = int(outs_ep[terminal][0]) if terminal.any() else 1

        episodes.append(dict(
            obs     = list(obs_ep),
            actions = list(acts_ep),
            masks   = [np.ones(10, dtype=bool)] * len(obs_ep),
            outcome = outcome,
        ))

    total_steps = sum(len(e["obs"]) for e in episodes)
    print(
        f"[pretrain] Loaded {len(episodes)} episodes / {total_steps} steps "
        f"from {Path(data_path).name} (skipped {skipped} invalid)"
    )
    return episodes


def _compute_returns(episodes: list[dict], gamma: float) -> np.ndarray:
    """Compute clipped, normalised discounted returns for every step.

    :param episodes: List of episode dicts from ``_load_episodes``.
    :param gamma: Discount factor.
    :returns: Float32 array of shape ``(N_total_steps,)``.
    """
    terminal_rewards = {1: WIN_REWARD, -1: LOSS_REWARD, 0: DRAW_REWARD}
    all_returns = []

    for ep in episodes:
        T   = len(ep['obs'])
        ret = np.zeros(T, dtype=np.float32)
        g   = float(terminal_rewards[ep['outcome']])
        for t in reversed(range(T)):
            ret[t] = g
            g      = gamma * g
        all_returns.append(ret)

    returns = np.concatenate(all_returns)
    returns = np.clip(returns, -VALUE_CLIP, VALUE_CLIP)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def _episodes_to_tensors(episodes: list[dict], gamma: float, device):
    """Convert a list of episode dicts into training tensors.

    :param episodes: Episode dicts from ``_load_episodes``.
    :param gamma: Discount factor for return computation.
    :param device: Torch device.
    :returns: Tuple of ``(obs_t, act_t, ret_t, mask_t)`` tensors.
    """
    returns = _compute_returns(episodes, gamma)
    obs_np  = np.stack([o for ep in episodes for o in ep['obs']])
    act_np  = np.array([a for ep in episodes for a in ep['actions']], dtype=np.int64)
    mask_np = np.stack([m for ep in episodes for m in ep['masks']])

    obs_t  = torch.tensor(obs_np,  dtype=torch.float32, device=device)
    act_t  = torch.tensor(act_np,  dtype=torch.long,    device=device)
    ret_t  = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
    mask_t = torch.tensor(mask_np, dtype=torch.bool,    device=device)
    return obs_t, act_t, ret_t, mask_t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pretrain_from_human_data(
        model: MaskablePPO,
        data_path_or_config: 'str | SupervisedWarmupConfig',
        *,
        episode_key:  str   = 'battle_id',
        n_epochs:     int   = 10,
        batch_size:   int   = 256,
        lr:           float = 3e-4,
        lambda_value: float = 1.0,
        lambda_bc:    float = 1.0,
        gamma:        float = GAMMA,
        val_split:    float = 0.0,
) -> MaskablePPO:
    """Jointly pretrain the value head and policy from human play data.

    Trains all policy parameters with a combined loss::

        L = lambda_value · MSE(V(s), G_t)  +  lambda_bc · CE(logits(s), a)

    Value targets G_t are computed as discounted Monte-Carlo returns from a
    synthetic terminal reward (win=+15, loss=-10, draw=0).

    Can be called with a plain path string **or** a ``SupervisedWarmupConfig``
    object (keyword args are ignored in the latter case)::

        # Explicit kwargs
        pretrain_from_human_data(model, "data/gen1ou.npz", n_epochs=20)

        # Via config object (all settings in one place)
        pretrain_from_human_data(model, SupervisedWarmupConfig("data/gen1ou.npz", n_epochs=20))

    :param model: Freshly constructed ``MaskablePPO`` model (modified in-place).
    :param data_path_or_config: Path string **or** a ``SupervisedWarmupConfig`` instance.
    :param episode_key: Unused — kept for API compatibility.
    :param n_epochs: Number of supervised training epochs.
    :param batch_size: Mini-batch size.
    :param lr: Learning rate.
    :param lambda_value: Weight for the value MSE loss term.
    :param lambda_bc: Weight for the behavioral cloning CE loss term.
    :param gamma: Discount factor for return computation.
    :param val_split: Fraction of episodes held out for per-epoch validation
        accuracy reporting.  0.0 (default) disables the validation set.
    :returns: The same ``model`` with pretrained weights.
    """
    if isinstance(data_path_or_config, SupervisedWarmupConfig):
        cfg = data_path_or_config
        data_path    = cfg.data_path
        episode_key  = cfg.episode_key
        n_epochs     = cfg.n_epochs
        batch_size   = cfg.batch_size
        lr           = cfg.lr
        lambda_value = cfg.lambda_value
        lambda_bc    = cfg.lambda_bc
        gamma        = cfg.gamma
        val_split    = cfg.val_split
    else:
        data_path = data_path_or_config

    policy = model.policy
    dev    = model.device

    # ── Load & split episodes ─────────────────────────────────────────────────
    all_episodes = _load_episodes(data_path, episode_key)

    if val_split > 0.0:
        indices = list(range(len(all_episodes)))
        random.shuffle(indices)
        n_val          = max(1, int(len(all_episodes) * val_split))
        val_idx        = set(indices[:n_val])
        val_episodes   = [all_episodes[i] for i in sorted(val_idx)]
        train_episodes = [all_episodes[i] for i in sorted(set(indices) - val_idx)]
        val_steps = sum(len(e['obs']) for e in val_episodes)
        print(f'[pretrain] Train: {len(train_episodes)} episodes | Val: {len(val_episodes)} episodes ({val_steps} steps)')
    else:
        train_episodes = all_episodes
        val_episodes   = []

    # ── Build tensors ─────────────────────────────────────────────────────────
    obs_t, act_t, ret_t, mask_t = _episodes_to_tensors(train_episodes, gamma, dev)

    val_obs_t = val_act_t = val_mask_t = None
    if val_episodes:
        val_obs_t, val_act_t, _, val_mask_t = _episodes_to_tensors(val_episodes, gamma, dev)

    loader = DataLoader(
        TensorDataset(obs_t, act_t, ret_t, mask_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ── Optimiser over all parameters ────────────────────────────────────────
    optimizer = optim.Adam(list(policy.parameters()), lr=lr)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    n_train = len(obs_t)
    print(f'[pretrain] {n_train} train steps | epochs={n_epochs} | λ_v={lambda_value} λ_bc={lambda_bc}')
    policy.train()

    for epoch in range(1, n_epochs + 1):
        total_loss = v_loss = bc_loss = 0.0
        n_batches  = 0

        for batch_obs, batch_act, batch_ret, batch_mask in loader:
            optimizer.zero_grad()

            out     = policy._run_extractor(batch_obs)
            loss_v  = mse(policy.value_head(out.features), batch_ret)

            logits  = policy._build_logits(out.features, out.move_hidden, out.team_hidden)
            logits  = logits.masked_fill(~batch_mask, float('-inf'))
            loss_bc = ce(logits, batch_act)

            loss = lambda_value * loss_v + lambda_bc * loss_bc

            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            v_loss     += loss_v.item()
            bc_loss    += loss_bc.item()
            n_batches  += 1

        nb = max(n_batches, 1)

        # Per-epoch validation accuracy
        val_str = ''
        if val_obs_t is not None:
            policy.eval()
            with torch.no_grad():
                v_out    = policy._run_extractor(val_obs_t)
                v_logits = policy._build_logits(v_out.features, v_out.move_hidden, v_out.team_hidden)
                v_logits = v_logits.masked_fill(~val_mask_t, float('-inf'))
                val_acc  = (v_logits.argmax(-1) == val_act_t).float().mean().item()
            policy.train()
            val_str = f'  val_acc={100 * val_acc:.1f}%'

        print(
            f'  epoch {epoch:3d}/{n_epochs} | '
            f'total={total_loss/nb:.4f}  value={v_loss/nb:.4f}  bc={bc_loss/nb:.4f}{val_str}'
        )

    policy.eval()
    print('[pretrain] Done.\n')
    return model
