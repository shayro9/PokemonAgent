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

from env.embed import MAX_MOVES, MOVE_EMBED_LEN

from .attention import CrossAttention
from .constants import (
    CONTEXT_BEFORE_MY_MOVES,
    CONTEXT_DIM,
    OPP_MOVES_START,
    OPP_MOVES_LEN,
)
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

        self.context_encoder = build_mlp(CONTEXT_DIM, context_hidden, context_hidden)
        self.move_encoder = build_mlp(MOVE_EMBED_LEN, move_hidden, move_hidden)
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
        obs     : (B, OBS_SIZE)
        returns : (B, trunk_hidden)
        """
        # 1. Slice ──────────────────────────────────────────────────────
        ctx_before    = obs[:, :CONTEXT_BEFORE_MY_MOVES]
        my_moves_flat = obs[:, CONTEXT_BEFORE_MY_MOVES:OPP_MOVES_START]
        opp_moves_flat = obs[:, OPP_MOVES_START:OPP_MOVES_START + OPP_MOVES_LEN]
        ctx_after     = obs[:, OPP_MOVES_START + OPP_MOVES_LEN:]

        context = torch.cat([ctx_before, opp_moves_flat, ctx_after], dim=-1)

        # 2. Encode context ─────────────────────────────────────────────
        ctx_h = self.context_encoder(context)                    # (B, context_hidden)

        # 3. Encode moves (shared weights → permutation equivariant) ────
        B = my_moves_flat.shape[0]
        my_moves = my_moves_flat.reshape(B, MAX_MOVES, MOVE_EMBED_LEN)
        move_h = self.move_encoder(
            my_moves.reshape(B * MAX_MOVES, MOVE_EMBED_LEN)
        ).reshape(B, MAX_MOVES, -1)                              # (B, 4, move_hidden)

        is_padding = (my_moves.abs().sum(dim=-1) == 0)           # (B, 4)  True = empty slot

        # 4. Cross-attention ─────────────────────────────────────────────
        attended, attn_scores = self.attn(ctx_h, move_h, key_padding_mask=is_padding)

        self.move_hidden = move_h       # (B, 4, move_hidden)
        self.attn_scores = attn_scores  # (B, n_heads, 4)

        # 5. Trunk ───────────────────────────────────────────────────────
        features = self.trunk(torch.cat([ctx_h, attended], dim=-1))  # (B, trunk_hidden)
        return features

