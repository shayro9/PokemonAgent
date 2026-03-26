"""
AttentionPointerExtractor — observation → trunk features.

Pipeline
--------
1. Slice observation into: context (excl my_moves) | my_moves | extract my_team (active+bench)
2. Encode context → ctx_h
3. Encode my_moves (shared weights, equivariant) → move_h (B, 4, move_hidden)
4. Encode my_team = [active] + [bench (5)] where each member = pokemon_state + 4*move_state
   → team_h (B, 6, team_hidden), using shared encoder
5. Cross-attend: context → moves → attended_moves
6. Cross-attend: context → team → attended_team  
7. Trunk([ctx_h || attended_moves || attended_team]) → features

Saves move_hidden, team_hidden, attn_scores for policy's _build_logits to consume.
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention
from .constants import (
    CONTEXT_LEN, MOVE_LEN, MAX_MOVES,
    MY_ACTIVE_LEN, MY_ACTIVE_START
)
from .mlp import build_mlp


class AttentionPointerExtractor(nn.Module):
    """
    Feature extractor with permutation-equivariant move and team (switch) encoders.

    Parameters
    ----------
    obs_dim           : total size of the flat observation vector
    context_hidden    : hidden/output size of the context encoder MLP
    move_hidden       : hidden/output size of the move encoder MLP
    team_hidden       : hidden/output size of the team/switch encoder MLP
    trunk_hidden      : output size of the trunk MLP  (= features_dim for SB3)
    n_attention_heads : number of heads in CrossAttention
    """

    def __init__(
        self,
        obs_dim: int,
        context_hidden: int = 128,
        move_hidden: int = 64,
        team_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
    ) -> None:
        super().__init__()

        self.context_encoder = build_mlp(CONTEXT_LEN, context_hidden, context_hidden)
        self.move_encoder = build_mlp(MOVE_LEN, move_hidden, move_hidden)
        # Team encoder takes pokemon_state + 4*move_state as input
        self.team_encoder = build_mlp(MY_ACTIVE_LEN, team_hidden, team_hidden)
        
        self.attn_moves = CrossAttention(context_hidden, move_hidden, n_attention_heads)
        self.attn_team = CrossAttention(context_hidden, team_hidden, n_attention_heads)
        
        self.trunk = build_mlp(
            context_hidden + move_hidden + team_hidden,
            trunk_hidden,
            trunk_hidden
        )

        # Written during forward(); consumed by AttentionPointerPolicy._build_logits()
        self.move_hidden: Optional[torch.Tensor] = None       # (B, 4, move_hidden)
        self.team_hidden: Optional[torch.Tensor] = None       # (B, 6, team_hidden)
        self.attn_scores_moves: Optional[torch.Tensor] = None # (B, n_heads, 4)
        self.attn_scores_team: Optional[torch.Tensor] = None  # (B, n_heads, 6)

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
        # 1. Slice observation ──────────────────────────────────────────
        # Context: everything except my_moves
        battle_context  = obs[:, :CONTEXT_LEN]
        
        # My moves come after context (active pokemon's available moves only)
        my_moves_flat   = obs[:, CONTEXT_LEN:CONTEXT_LEN + MAX_MOVES * MOVE_LEN]
        
        # Extract my_active and my_bench for team encoding
        # my_active (pokemon + 4 moves) is at MY_ACTIVE_START:MY_ACTIVE_START+MY_ACTIVE_LEN in context
        my_active_in_context = obs[:, MY_ACTIVE_START:MY_ACTIVE_START + MY_ACTIVE_LEN]
        
        # my_bench starts after arena + my_active + opp_active
        # Each bench entry is MY_ACTIVE_LEN (pokemon + 4 moves) = 162
        my_bench_start_in_context = 5 + MY_ACTIVE_LEN + 25  # arena + my_active + opp_active
        my_bench_len = MY_ACTIVE_LEN * 5  # 5 bench slots
        my_bench_flat = battle_context[:, my_bench_start_in_context:my_bench_start_in_context + my_bench_len]
        
        # 2. Encode context ─────────────────────────────────────────────
        context_hidden = self.context_encoder(battle_context)  # (B, context_hidden)

        # 3. Encode moves (shared weights → permutation equivariant) ────
        batch_size = my_moves_flat.shape[0]
        my_moves = my_moves_flat.reshape(batch_size, MAX_MOVES, MOVE_LEN)
        move_hidden = self.move_encoder(
            my_moves.reshape(batch_size * MAX_MOVES, MOVE_LEN)
        ).reshape(batch_size, MAX_MOVES, -1)  # (B, 4, move_hidden)

        is_padding_moves = (my_moves.abs().sum(dim=-1) == 0)   # (B, 4)

        # 4. Encode team: active (1) + bench (5) = 6 total ─────────────
        # Each team slot is pokemon_state + 4*move_state = MY_ACTIVE_LEN
        team_flat = torch.cat([my_active_in_context, my_bench_flat], dim=1)  # (B, 6*MY_ACTIVE_LEN)
        n_team_slots = 6
        team = team_flat.reshape(batch_size, n_team_slots, MY_ACTIVE_LEN)
        
        team_hidden = self.team_encoder(
            team.reshape(batch_size * n_team_slots, MY_ACTIVE_LEN)
        ).reshape(batch_size, n_team_slots, -1)  # (B, 6, team_hidden)

        is_padding_team = (team.abs().sum(dim=-1) == 0)  # (B, 6)

        # 5. Cross-attention ────────────────────────────────────────────
        attended_moves, attn_scores_moves = self.attn_moves(
            context_hidden, move_hidden, key_padding_mask=is_padding_moves
        )
        attended_team, attn_scores_team = self.attn_team(
            context_hidden, team_hidden, key_padding_mask=is_padding_team
        )

        self.move_hidden = move_hidden              # (B, 4, move_hidden)
        self.team_hidden = team_hidden              # (B, 6, team_hidden)
        self.attn_scores_moves = attn_scores_moves  # (B, n_heads, 4)
        self.attn_scores_team = attn_scores_team    # (B, n_heads, 6)

        # 6. Trunk ──────────────────────────────────────────────────────
        features = self.trunk(
            torch.cat([context_hidden, attended_moves, attended_team], dim=-1)
        )  # (B, trunk_hidden)
        return features
