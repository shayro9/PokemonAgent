"""
AttentionPointerExtractor — observation → trunk features.

Pipeline
--------
1. Slice flat obs  →  context  +  my_moves (4 x MOVE_EMBED_LEN)
2. context MLP     →  ctx_h
3. move MLP (shared weights, permutation-equivariant)  →  move_h  (B, 4, move_hidden)
4. CrossAttention(ctx_h, move_h)  →  attended
5. trunk MLP([ctx_h || attended])  →  features  (B, trunk_hidden)

`move_hidden` and `attn_scores` are saved as instance attributes so the
actor head can build pointer logits without a second forward pass.
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention
from .constants import CONTEXT_LEN, MOVE_LEN, MAX_MOVES
from .mlp import build_mlp


class AttentionPointerExtractor(nn.Module):
    """
    Feature extractor with a permutation-equivariant move encoder.

    Parameters
    ----------
    obs_dim           : total size of the flat observation vector
    context_hidden    : hidden/output size of the context encoder MLP
    move_hidden       : hidden/output size of the move encoder MLP
                        (also the kv_dim for CrossAttention)
    trunk_hidden      : output size of the trunk MLP  (= features_dim for SB3)
    n_attention_heads : number of heads in CrossAttention
    """

    def __init__(
        self,
        obs_dim: int,
        context_hidden: int = 128,
        move_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.context_encoder = build_mlp(CONTEXT_LEN, context_hidden, context_hidden)
        self.move_encoder = build_mlp(MOVE_LEN, move_hidden, move_hidden)
        self.attn = CrossAttention(context_hidden, move_hidden, n_attention_heads)
        self.trunk = build_mlp(context_hidden + move_hidden, trunk_hidden, trunk_hidden)

        # Written during forward(); consumed by AttentionPointerPolicy._build_logits()
        self.move_hidden: Optional[torch.Tensor] = None   # (B, 4, move_hidden)
        self.attn_scores: Optional[torch.Tensor] = None   # (B, n_heads, 4)

        # Required by SB3's MaskableActorCriticPolicy._build()
        self.features_dim = trunk_hidden
        self.latent_dim_pi = trunk_hidden
        self.latent_dim_vf = trunk_hidden

    # SB3 calls these to split actor/critic latents; we share one trunk.
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs     : (batch_size, OBS_SIZE)
        returns : (batch_size, trunk_hidden)
        """
        # 1. Slice ──────────────────────────────────────────────────────
        battle_context    = obs[:, :CONTEXT_LEN]
        my_moves_flat = obs[:, CONTEXT_LEN:]

        # 2. Encode context ─────────────────────────────────────────────
        context_hidden = self.context_encoder(battle_context)                    # (batch_size, context_hidden)

        # 3. Encode moves (shared weights → permutation equivariant) ────
        batch_size = my_moves_flat.shape[0]
        my_moves = my_moves_flat.reshape(batch_size, MAX_MOVES, MOVE_LEN)
        move_hidden = self.move_encoder(
            my_moves.reshape(batch_size * MAX_MOVES, MOVE_LEN)
        ).reshape(batch_size, MAX_MOVES, -1)                              # (batch_size, 4, move_hidden)

        is_padding = (my_moves.abs().sum(dim=-1) == 0)           # (batch_size, 4)  True = empty slot

        # 4. Cross-attention ─────────────────────────────────────────────
        attended, attn_scores = self.attn(context_hidden, move_hidden, key_padding_mask=is_padding)

        self.move_hidden = move_hidden      # (batch_size, 4, move_hidden)
        self.attn_scores = attn_scores      # (batch_size, n_heads, 4)

        # 5. Trunk ───────────────────────────────────────────────────────
        features = self.trunk(torch.cat([context_hidden, attended], dim=-1))  # (batch_size, trunk_hidden)
        return features

