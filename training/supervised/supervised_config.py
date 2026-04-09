"""
training/supervised/supervised_config.py
=========================================
Constants and configuration dataclass for supervised pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Training hyperparameter defaults
# ---------------------------------------------------------------------------

GAMMA       = 0.99
WIN_REWARD  =  15.0
LOSS_REWARD = -10.0
DRAW_REWARD =   0.0
VALUE_CLIP  =  30.0
GRAD_CLIP   =   5.0


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SupervisedWarmupConfig:
    """All settings for a supervised warmup pretraining step.

    Pass an instance to ``train_model(warmup_config=...)`` to run behavioral
    cloning before RL training begins.  When ``warmup_config`` is ``None``
    (the default), the pipeline is completely unaffected.

    :param data_path: Path to the .npz file produced by convert_metamon_json.
    :param n_epochs: Number of supervised training epochs.
    :param batch_size: Mini-batch size.
    :param lr: Learning rate.
    :param lambda_value: Weight for the value MSE loss term.
    :param lambda_bc: Weight for the behavioral cloning CE loss term.
    :param gamma: Discount factor for return computation.
    :param episode_key: Unused — kept for API compatibility.
    :param val_split: Fraction of episodes to hold out for validation accuracy
        reporting (0.0 = no validation set).
    """
    data_path:    str
    n_epochs:     int   = 10
    batch_size:   int   = 256
    lr:           float = 3e-4
    lambda_value: float = 1.0
    lambda_bc:    float = 1.0
    gamma:        float = GAMMA
    episode_key:  str   = 'battle_id'
    val_split:    float = 0.0
