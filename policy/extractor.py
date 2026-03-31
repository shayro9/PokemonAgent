"""
AttentionPointerExtractor — observation → trunk features.

Pipeline
--------
1. Slice observation into: context  | my_moves | my_team
2. Encode context → ctx_h
3. Encode my_moves (shared weights, equivariant) → move_h (B, 4, move_hidden)
4. Encode my_team = where each member = pokemon_state + 4*move_state
   → team_h (B, 6, team_hidden)
5. Cross-attend: context → moves → attended_moves
6. Cross-attend: context → team → attended_team  
7. Trunk([ctx_h || attended_moves || attended_team]) → features

Saves move_hidden, team_hidden, attn_scores for policy's _build_logits to consume.
"""

from typing import Optional

import torch
import torch.nn as nn

from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from env.states.state_utils import MAX_TEAM_SIZE
from .attention import CrossAttention
from .constants import (
    CONTEXT_LEN, MOVE_LEN, MAX_MOVES, MY_POKEMON_LEN, ARENA_OPPONENT_LEN, MY_MOVES_START,
    N_SWITCH_ACTIONS,
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

        self.context_encoder    = build_mlp(CONTEXT_LEN, context_hidden, context_hidden)
        self.move_encoder       = build_mlp(MOVE_LEN, move_hidden, move_hidden)
        self.team_encoder       = build_mlp(MY_POKEMON_LEN, team_hidden, team_hidden)
        
        self.attn_moves = CrossAttention(context_hidden, move_hidden, n_attention_heads)
        self.attn_team  = CrossAttention(context_hidden, team_hidden, n_attention_heads)
        
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
        self.features_dim  = trunk_hidden
        self.latent_dim_pi = trunk_hidden
        self.latent_dim_vf = trunk_hidden

    # SB3 calls these to split actor/critic latents; we share one trunk.
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def _slice_observation(self, obs: torch.Tensor):
        B = obs.shape[0]
        arena_opponent_vector = obs[:, :ARENA_OPPONENT_LEN]                # (B, ARENA_OPP_LEN)
        my_pokemon_vector     = obs[:, ARENA_OPPONENT_LEN:-MAX_TEAM_SIZE]  # (B, 6*MY_POKEMON_LEN)
        alive_vector          = obs[:, -MAX_TEAM_SIZE:]                    # (B, 6)

        # Reshape into per-slot blocks — no loop needed
        my_team_flat = my_pokemon_vector.reshape(B, MAX_TEAM_SIZE, MY_POKEMON_LEN)  # (B, 6, MY_POKEMON_LEN)

        # Zero fainted slots in one vectorised op
        is_fainted = (alive_vector == -1).unsqueeze(-1)   # (B, 6, 1)
        my_team_flat = my_team_flat.masked_fill(is_fainted, 0.0)

        # Locate active slot per batch item and gather its features
        active_idx    = (alive_vector == 1).long().argmax(dim=1)        # (B,)
        has_active = (alive_vector == 1).any(dim=1)  # (B,)
        idx_expanded = active_idx.view(B, 1, 1).expand(B, 1, MY_POKEMON_LEN)
        active_section = my_team_flat.gather(1, idx_expanded).squeeze(1)  # (B, MY_POKEMON_LEN)
        active_section = active_section * has_active.unsqueeze(1)          # (B, MY_POKEMON_LEN)

        battle_context = torch.cat([arena_opponent_vector, active_section], dim=1)     # (B, CONTEXT_LEN)
        my_moves_flat  = active_section[:, MY_MOVES_START:]                            # (B, moves_len)

        return battle_context, my_moves_flat, my_team_flat

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs     : (batch_size, OBS_SIZE)
        returns : (batch_size, trunk_hidden)
        """
        # 1. Slice observation ──────────────────────────────────────────
        # Context: arena + all opp (excluding bench moves) + my active (including moves)
        battle_context, my_moves_flat, my_team_flat = self._slice_observation(obs)
        
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
        team = my_team_flat.reshape(batch_size, N_SWITCH_ACTIONS, MY_POKEMON_LEN)
        
        team_hidden = self.team_encoder(
            team.reshape(batch_size * N_SWITCH_ACTIONS, MY_POKEMON_LEN)
        ).reshape(batch_size, N_SWITCH_ACTIONS, -1)  # (B, 6, team_hidden)

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
