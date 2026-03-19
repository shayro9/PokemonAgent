"""Reusable MLP builder used across the policy package."""

from typing import List
import torch.nn as nn


def build_mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """
    Stack of Linear → LayerNorm → ReLU blocks, ending with a plain Linear.

    Args:
        in_dim:  Input feature size.
        hidden:  Width of every hidden layer.
        out_dim: Output feature size.
        layers:  Total number of linear layers (minimum 1).
    """
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)

