"""
Recurrent Pointer Policy — LSTM-First
=======================================
The LSTM sits at the very front of the processing pipeline, receiving the
encoded context vector and producing the hidden state that drives both the
cross-attention query and the final pointer/value heads.

Data flow
---------

  obs (flat)
   │
   ├─ context slice ──► context_encoder MLP ──► ctx_h
   │                                               │
   │                                             LSTM          ← memory here
   │                                               │
   │                                           lstm_out  (B, lstm_hidden)
   │                                               │
   └─ my_moves slice ──► move_encoder MLP ──► move_h  (B, 4, move_hidden)
                                                   │
                              CrossAttention(lstm_out, move_h)
                                                   │
                                               attended  (B, move_hidden)
                                                   │
                           trunk([lstm_out ‖ attended]) ──► features  (B, trunk_hidden)
                                                   │
                              ┌────────────────────┴────────────────────┐
                         pointer_proj                              value_head
                    dot(features, move_h_i)                         → scalar
                    + non_move_head(features)
                         → 4 move logits + 22 non-move logits

Why LSTM first?
---------------
Placing the LSTM on the context encoding means the hidden state captures
battle momentum (stat changes, weather turns, prior choices) *before* the
agent decides which moves are worth attending to.  The cross-attention query
is therefore history-aware, letting the attention weights shift based on what
has already happened in the episode.

Usage
-----
    from recurrent_pointer_policy import RecurrentPointerPolicy

    model = RecurrentPPO(
        RecurrentPointerPolicy,
        env=train_env,
        policy_kwargs=dict(
            context_hidden=128,
            move_hidden=64,
            trunk_hidden=128,
            n_attention_heads=4,
            lstm_hidden_size=256,
            n_lstm_layers=1,
        ),
        ...
    )
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticPolicy,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.type_aliases import Schedule

from policy.attention_policy import (
    _mlp,
    CrossAttention,
    CONTEXT_DIM,
    MOVE_EMBED_LEN,
    MAX_MOVES,
    CONTEXT_BEFORE_MY_MOVES,
    OPP_MOVES_START,
    OPP_MOVES_LEN,
    MOVE_ACTION_START,
    N_MOVE_ACTIONS,
    TOTAL_ACTIONS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Extractor: two-stage, LSTM sits between the stages
# ──────────────────────────────────────────────────────────────────────────────

class LSTMFirstExtractor(nn.Module):
    """
    Feature extractor whose forward pass is deliberately split into two stages
    so the policy can insert the LSTM between them.

    Stage 1 — encode_context(obs)
        Slices the context portion of the observation and passes it through
        the context encoder MLP → ctx_h.  The policy feeds ctx_h into the
        LSTM and gets lstm_out back.

    Stage 2 — attend_and_trunk(lstm_out, obs)
        Slices the move portion of the observation, encodes each move with
        the shared move encoder, runs cross-attention with lstm_out as the
        query, then passes the concatenation through the trunk MLP.
        Stores move_hidden as a side-channel for the pointer head.

    Parameters
    ----------
    context_hidden    : output size of the context encoder MLP
    move_hidden       : output size of the per-move encoder MLP
    trunk_hidden      : output size of the trunk MLP (= features_dim)
    n_attention_heads : number of heads in the cross-attention module
    lstm_hidden_size  : size of the LSTM output fed into Stage 2 as the query
    """

    def __init__(
        self,
        context_hidden: int = 128,
        move_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
        lstm_hidden_size: int = 256,
    ):
        super().__init__()

        # Stage 1: context → ctx_h
        self.context_encoder = _mlp(CONTEXT_DIM, context_hidden, context_hidden)

        # Stage 2: lstm_out (query) + moves (keys/values) → features
        self.move_encoder = _mlp(MOVE_EMBED_LEN, move_hidden, move_hidden)
        self.attn = CrossAttention(
            query_dim=lstm_hidden_size,   # query comes from LSTM output
            kv_dim=move_hidden,
            n_heads=n_attention_heads,
        )
        # Trunk input: lstm_out concatenated with the attended move summary.
        self.trunk = _mlp(lstm_hidden_size + move_hidden, trunk_hidden, trunk_hidden)

        # Side-channels read by the pointer head in the policy.
        self.move_hidden: Optional[torch.Tensor] = None
        self.attn_scores: Optional[torch.Tensor] = None

        # Dimension metadata expected by SB3.
        self.features_dim  = trunk_hidden
        self.latent_dim_pi = trunk_hidden
        self.latent_dim_vf = trunk_hidden

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def encode_context(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Slice the context from obs and encode it.

        obs      : (B, OBS_SIZE)
        returns    ctx_h : (B, context_hidden)  — fed into the LSTM by the policy
        """
        ctx_before = obs[:, :CONTEXT_BEFORE_MY_MOVES]
        opp_moves  = obs[:, OPP_MOVES_START: OPP_MOVES_START + OPP_MOVES_LEN]
        ctx_after  = obs[:, OPP_MOVES_START + OPP_MOVES_LEN:]
        context    = torch.cat([ctx_before, opp_moves, ctx_after], dim=-1)
        return self.context_encoder(context)                     # (B, context_hidden)

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def attend_and_trunk(
        self,
        lstm_out: torch.Tensor,   # (B, lstm_hidden_size)
        obs: torch.Tensor,        # (B, OBS_SIZE)  — needed to slice my_moves
    ) -> torch.Tensor:
        """
        Encode moves, cross-attend with lstm_out as the query, run the trunk.

        Returns features : (B, trunk_hidden)
        Also sets self.move_hidden and self.attn_scores for the pointer head.
        """
        # Slice and reshape my moves: (B, 4, MOVE_EMBED_LEN)
        my_moves_flat = obs[:, CONTEXT_BEFORE_MY_MOVES: OPP_MOVES_START]
        B             = my_moves_flat.shape[0]
        my_moves      = my_moves_flat.reshape(B, MAX_MOVES, MOVE_EMBED_LEN)

        # Encode all move slots with shared weights → (B, 4, move_hidden)
        move_h = self.move_encoder(
            my_moves.reshape(B * MAX_MOVES, MOVE_EMBED_LEN)
        ).reshape(B, MAX_MOVES, -1)

        # Padding mask: all-zero slot = unavailable move
        is_padding = (my_moves.abs().sum(dim=-1) == 0)          # (B, 4)

        # Cross-attention: lstm_out is the query; move_h is keys and values
        attended, attn_scores = self.attn(
            lstm_out, move_h, key_padding_mask=is_padding
        )

        # Cache for the actor head (pointer logits).
        self.move_hidden = move_h        # (B, 4, move_hidden)
        self.attn_scores = attn_scores   # (B, n_heads, 4)

        # Trunk: fuse LSTM memory with the attended move summary
        return self.trunk(torch.cat([lstm_out, attended], dim=-1))  # (B, trunk_hidden)

    # ── Unused but required by SB3 ────────────────────────────────────────────

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "LSTMFirstExtractor.forward() should never be called directly. "
            "Use encode_context() then attend_and_trunk() with the LSTM output in between."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────

class RecurrentPointerPolicy(RecurrentActorCriticPolicy):
    """
    RecurrentPPO policy with LSTM-first data flow.

    Orchestrates the two-stage extractor:
      ctx_h    = extractor.encode_context(obs)
      lstm_out = LSTM(ctx_h)
      features = extractor.attend_and_trunk(lstm_out, obs)

    Then scores actions via the pointer head:
      move_logits    = dot(pointer_proj(features), move_hidden_i)
      nonmove_logits = non_move_head(features)

    Parameters
    ----------
    context_hidden    : hidden size of the context encoder MLP
    move_hidden       : hidden size of the per-move encoder MLP
    trunk_hidden      : hidden size of the trunk MLP
    n_attention_heads : cross-attention heads
    lstm_hidden_size  : LSTM hidden/cell size; also the cross-attention query dim
    n_lstm_layers     : number of stacked LSTM layers
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        context_hidden: int = 128,
        move_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        **kwargs,
    ):
        self._extractor_kwargs = dict(
            context_hidden=context_hidden,
            move_hidden=move_hidden,
            trunk_hidden=trunk_hidden,
            n_attention_heads=n_attention_heads,
            lstm_hidden_size=lstm_hidden_size,
        )
        self._trunk_hidden     = trunk_hidden
        self._move_hidden_size = move_hidden

        kwargs.setdefault("net_arch", [])
        kwargs["lstm_hidden_size"] = lstm_hidden_size
        kwargs["n_lstm_layers"]    = n_lstm_layers
        kwargs.setdefault("shared_lstm", True)
        kwargs.setdefault("enable_critic_lstm", False)

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # ── Fix LSTM input size (must be done AFTER super().__init__ returns) ──
        # RecurrentActorCriticPolicy.__init__ recreates self.lstm_actor *after*
        # _build() finishes, resetting it to input_size=features_dim (raw obs,
        # e.g. 383).  We overwrite it here, at the very end of our __init__,
        # so nothing can stomp it again.
        context_hidden = self._extractor_kwargs["context_hidden"]
        self.lstm_actor = nn.LSTM(
            input_size=context_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=False,
        )

        # Rebuild the optimizer so lstm_actor's new parameters are included.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ── build ─────────────────────────────────────────────────────────────────

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMFirstExtractor(**self._extractor_kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        trunk_hidden = self._trunk_hidden
        move_hidden  = self._move_hidden_size
        n_non_move   = TOTAL_ACTIONS - N_MOVE_ACTIONS  # 22

        # Pointer head: project trunk features into move_hidden space for dot product.
        self.pointer_proj  = nn.Linear(trunk_hidden, move_hidden, bias=False)
        self.non_move_head = nn.Linear(trunk_hidden, n_non_move)
        self.value_head    = nn.Linear(trunk_hidden, 1)

        # Neutralise SB3's default heads.
        self.action_net = nn.Identity()
        self.value_net  = nn.Identity()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _run_lstm(
        self,
        ctx_h: torch.Tensor,           # (B, context_hidden) — Stage 1 output
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,  # (B,) float; 1.0 = episode boundary
    ) -> Tuple[torch.Tensor, RNNStates]:
        """
        Pass encoded context through the shared LSTM.

        Episode boundaries zero out the hidden/cell state for that env before
        the step, matching the behavior of RecurrentPPO's internal logic.
        """
        # episode_starts: (B,) → (1, B, 1) for broadcasting against (n_layers, B, hidden)
        mask = (1.0 - episode_starts).view(1, -1, 1)
        h = mask * lstm_states.pi[0]   # (n_layers, B, lstm_hidden)
        c = mask * lstm_states.pi[1]

        # lstm_actor expects input (seq_len, B, input_size); seq_len=1 at inference
        lstm_out, (h_n, c_n) = self.lstm_actor(ctx_h.unsqueeze(0), (h, c))
        lstm_out = lstm_out.squeeze(0)                         # (B, lstm_hidden)

        new_states = RNNStates(pi=(h_n, c_n), vf=(h_n, c_n))
        return lstm_out, new_states

    def _extract_features(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNStates]:
        """
        Full two-stage extraction with LSTM in the middle.

        Returns
        -------
        features    : (B, trunk_hidden)   — input for the heads
        move_hidden : (B, 4, move_hidden) — pointer keys
        new_states  : updated LSTM hidden/cell state
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device)
        obs = obs.float()

        # Stage 1: context → ctx_h
        ctx_h = self.mlp_extractor.encode_context(obs)           # (B, context_hidden)

        # LSTM: ctx_h → lstm_out  (history-aware context)
        lstm_out, new_states = self._run_lstm(ctx_h, lstm_states, episode_starts)

        # Stage 2: (lstm_out, moves) → features; also caches move_hidden
        features = self.mlp_extractor.attend_and_trunk(lstm_out, obs)

        return features, self.mlp_extractor.move_hidden, new_states

    def _build_logits(
        self,
        features: torch.Tensor,    # (B, trunk_hidden)
        move_hidden: torch.Tensor, # (B, 4, move_hidden)
    ) -> torch.Tensor:
        """Assemble the full (B, 26) logit tensor."""
        ptr_query       = self.pointer_proj(features)                          # (B, move_hidden)
        move_logits     = torch.einsum("bd,bnd->bn", ptr_query, move_hidden)   # (B, 4)
        non_move_logits = self.non_move_head(features)                         # (B, 22)

        return torch.cat([
            non_move_logits[:, :MOVE_ACTION_START],   # actions 0-5
            move_logits,                               # actions 6-9
            non_move_logits[:, MOVE_ACTION_START:],   # actions 10-25
        ], dim=-1)                                     # (B, 26)

    def _make_distribution(self, logits: torch.Tensor, action_masks=None):
        from sb3_contrib.common.maskable.distributions import (
            MaskableCategorical,
            MaskableCategoricalDistribution,
        )
        dist = MaskableCategoricalDistribution(int(self.action_space.n))   # type: ignore
        dist.distribution = MaskableCategorical(logits=logits)             # type: ignore
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    # ── SB3 interface ─────────────────────────────────────────────────────────

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
        action_masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        features, move_hidden, new_states = self._extract_features(
            obs, lstm_states, episode_starts
        )
        logits   = self._build_logits(features, move_hidden)
        values   = self.value_head(features)
        dist     = self._make_distribution(logits, action_masks)
        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob, new_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features, move_hidden, _ = self._extract_features(
            obs, lstm_states, episode_starts
        )
        logits   = self._build_logits(features, move_hidden)
        values   = self.value_head(features)
        dist     = self._make_distribution(logits, action_masks)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        action_masks=None,
    ):
        features, move_hidden, _ = self._extract_features(
            obs, lstm_states, episode_starts
        )
        logits = self._build_logits(features, move_hidden)
        return self._make_distribution(logits, action_masks)

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features, _, _ = self._extract_features(obs, lstm_states, episode_starts)
        return self.value_head(features)

    # SB3 calls extract_features in a few internal paths; guard it explicitly.
    def extract_features(
        self,
        obs: torch.Tensor,
        features_extractor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "RecurrentPointerPolicy.extract_features() cannot be called without "
            "lstm_states and episode_starts. Use _extract_features() internally."
        )
