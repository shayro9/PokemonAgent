"""
Attention-Pointer Policy for Pokemon RL
========================================
Architecture overview:
  - The flat observation is split into:
      * context vector  : everything EXCEPT my_moves (hp, stats, boosts, status,
                          effects, opp info, type multipliers, weight, protect belief)
      * move embeddings : 4 × MOVE_EMBED_LEN tensors, one per move slot

  - A context encoder MLP encodes the context vector → context_hidden
  - A move encoder MLP encodes each move embedding independently → move_hidden_i
  - Cross-attention: context_hidden (query) attends over move_hidden_i (keys/values)
      → attended_context  (order-invariant summary of all moves)
  - A shared trunk MLP takes [attended_context || context_hidden] → shared_features
  - Actor head  : pointer scores = dot(shared_features, move_hidden_i) for i in 0..3
                  → 4 logits that are directly used as the move-action logits
                  (remaining action slots stay as a small MLP head)
  - Critic head : MLP(shared_features) → scalar value

This makes the policy fully permutation-equivariant over moves:
if you shuffle the 4 moves in the observation, the selected move stays the same.

Integration with your existing code
-------------------------------------
1.  `BattleState.to_array()` is **unchanged** — the flat vector is still the obs.
2.  The policy slices out move embeddings internally using the constants below.
3.  In train.py, replace `"MlpPolicy"` with `AttentionPointerPolicy`.

Usage
------
    from attention_policy import AttentionPointerPolicy

    model = MaskablePPO(
        AttentionPointerPolicy,
        env=train_env,
        policy_kwargs=dict(
            context_hidden=128,
            move_hidden=64,
            trunk_hidden=128,
            n_attention_heads=4,
        ),
        ...
    )
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from env.embed import MAX_MOVES, MOVE_EMBED_LEN
from env.battle_state import CONTEXT_BEFORE_MY_MOVES, CONTEXT_AFTER_OPP_MOVES

MY_MOVES_LEN = MAX_MOVES * MOVE_EMBED_LEN          # 156
OPP_MOVES_START = CONTEXT_BEFORE_MY_MOVES + MY_MOVES_LEN
OPP_MOVES_LEN = MAX_MOVES * MOVE_EMBED_LEN          # 156

# Total context (everything except my_moves):
CONTEXT_DIM = CONTEXT_BEFORE_MY_MOVES + OPP_MOVES_LEN + CONTEXT_AFTER_OPP_MOVES

# Action space layout (matches your existing wrapper):
#   0-5   : non-move actions (switches, etc.)  — kept as MLP head
#   6-9   : move 0-3  ← pointer scores replace these 4 logits
MOVE_ACTION_START = 6
N_MOVE_ACTIONS = 4
TOTAL_ACTIONS = 26


# ──────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Simple MLP with LayerNorm and ReLU activations."""
    mods: List[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class CrossAttention(nn.Module):
    """
    Single-query multi-head cross-attention.

    Query  : context_hidden  (B, context_hidden)
    Keys   : move_hidden     (B, 4, move_hidden)
    Values : move_hidden     (B, 4, move_hidden)
    Output : attended        (B, move_hidden)
    Also returns raw per-head scores (B, n_heads, 4) for the pointer head.
    """

    def __init__(self, query_dim: int, kv_dim: int, n_heads: int):
        super().__init__()
        assert kv_dim % n_heads == 0, "kv_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = kv_dim // n_heads

        self.q_proj = nn.Linear(query_dim, kv_dim)
        self.k_proj = nn.Linear(kv_dim, kv_dim)
        self.v_proj = nn.Linear(kv_dim, kv_dim)
        self.out_proj = nn.Linear(kv_dim, kv_dim)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,          # (B, query_dim)
        kv: torch.Tensor,             # (B, 4, kv_dim)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, 4) bool, True=ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = kv.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, H, D)           # (B, H, D)
        k = self.k_proj(kv).view(B, N, H, D)           # (B, N, H, D)
        v = self.v_proj(kv).view(B, N, H, D)           # (B, N, H, D)

        # Scores: (B, H, N)
        scores = torch.einsum("bhd,bnhd->bhn", q, k) * self.scale

        if key_padding_mask is not None:
            # mask shape (B, N) → (B, 1, N)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)            # (B, H, N)

        # Attended value: (B, H, D) → (B, H*D)
        out = torch.einsum("bhn,bnhd->bhd", attn, v).reshape(B, H * D)
        out = self.out_proj(out)

        return out, scores   # (B, kv_dim),  (B, H, N)


# ──────────────────────────────────────────────────────────────────────────
# Main feature extractor
# ──────────────────────────────────────────────────────────────────────────

class AttentionPointerExtractor(nn.Module):
    """
    Splits the flat obs into context + move embeddings, then:
      1. Encodes context    → context_hidden  (B, context_hidden)
      2. Encodes each move  → move_hidden_i   (B, 4, move_hidden)
      3. Cross-attention(context_hidden, move_hidden) → attended  (B, move_hidden)
      4. trunk([context_hidden || attended]) → features  (B, trunk_hidden)

    Also exposes `move_hidden` and `attn_scores` as attributes so the actor
    head can build pointer logits without a second forward pass.
    """

    def __init__(
        self,
        obs_dim: int,
        context_hidden: int = 128,
        move_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
    ):
        super().__init__()

        self.context_encoder = _mlp(CONTEXT_DIM, context_hidden, context_hidden)
        self.move_encoder = _mlp(MOVE_EMBED_LEN, move_hidden, move_hidden)
        self.attn = CrossAttention(context_hidden, move_hidden, n_attention_heads)
        self.trunk = _mlp(context_hidden + move_hidden, trunk_hidden, trunk_hidden)

        # Stored for the actor head
        self.move_hidden: Optional[torch.Tensor] = None
        self.attn_scores: Optional[torch.Tensor] = None

        self.features_dim = trunk_hidden
        # Required by SB3's MaskableActorCriticPolicy._build()
        self.latent_dim_pi = trunk_hidden
        self.latent_dim_vf = trunk_hidden

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 calls this to get actor latent. Just return features (trunk output)."""
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """SB3 calls this to get critic latent. Just return features (trunk output)."""
        return features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, OBS_SIZE)
        returns: (B, trunk_hidden)
        """
        # ── slice observation ───────────────────────────────────────────
        ctx_before = obs[:, :CONTEXT_BEFORE_MY_MOVES]                 # (B, 49)
        my_moves_flat = obs[:, CONTEXT_BEFORE_MY_MOVES: OPP_MOVES_START]  # (B, 144)
        opp_moves_flat = obs[:, OPP_MOVES_START: OPP_MOVES_START + OPP_MOVES_LEN]  # (B,144)
        ctx_after = obs[:, OPP_MOVES_START + OPP_MOVES_LEN:]          # (B, 11)

        # Context = everything except my_moves
        context = torch.cat([ctx_before, opp_moves_flat, ctx_after], dim=-1)  # (B, 204)

        # ── encode ─────────────────────────────────────────────────────
        ctx_h = self.context_encoder(context)                          # (B, ctx_hidden)

        # Reshape my moves: (B, 4, MOVE_EMBED_LEN)
        my_moves = my_moves_flat.reshape(-1, MAX_MOVES, MOVE_EMBED_LEN)
        # Encode each move independently (same weights → permutation equivariant)
        B = my_moves.shape[0]
        move_h = self.move_encoder(
            my_moves.reshape(B * MAX_MOVES, MOVE_EMBED_LEN)
        ).reshape(B, MAX_MOVES, -1)                                    # (B, 4, move_hidden)

        # Detect padding (all-zero move slots = unavailable move)
        is_padding = (my_moves.abs().sum(dim=-1) == 0)                 # (B, 4)

        # ── cross-attention ─────────────────────────────────────────────
        attended, attn_scores = self.attn(ctx_h, move_h, key_padding_mask=is_padding)

        # Store for actor head
        self.move_hidden = move_h          # (B, 4, move_hidden)
        self.attn_scores = attn_scores     # (B, n_heads, 4)

        # ── trunk ───────────────────────────────────────────────────────
        features = self.trunk(torch.cat([ctx_h, attended], dim=-1))    # (B, trunk_hidden)
        return features


# ──────────────────────────────────────────────────────────────────────────
# Custom Policy
# ──────────────────────────────────────────────────────────────────────────

class AttentionPointerPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO policy that uses an attention-pointer head for move selection.

    Move actions (indices 6-9) are scored via:
        logit_i = dot(trunk_features, move_hidden_i)
    making the policy order-invariant over moves.

    Non-move actions (indices 0-5, 10-25) are scored via a standard linear head.
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
        **kwargs,
    ):
        self._attn_kwargs = dict(
            context_hidden=context_hidden,
            move_hidden=move_hidden,
            trunk_hidden=trunk_hidden,
            n_attention_heads=n_attention_heads,
        )
        # Disable SB3's default mlp_extractor; we build our own
        kwargs.setdefault("net_arch", [])
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ── override mlp_extractor building ────────────────────────────────

    def _build_mlp_extractor(self) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        self.mlp_extractor = AttentionPointerExtractor(obs_dim, **self._attn_kwargs)

    # ── override actor / critic heads ──────────────────────────────────

    def _build(self, lr_schedule: Schedule) -> None:
        """Called by __init__ after _build_mlp_extractor."""
        super()._build(lr_schedule)

        trunk_hidden = self._attn_kwargs["trunk_hidden"]
        move_hidden = self._attn_kwargs["move_hidden"]
        n_non_move = TOTAL_ACTIONS - N_MOVE_ACTIONS   # 22

        # Pointer projection: trunk → move_hidden  (for dot product with move_hidden_i)
        self.pointer_proj = nn.Linear(trunk_hidden, move_hidden, bias=False)

        # Head for non-move actions
        self.non_move_head = nn.Linear(trunk_hidden, n_non_move)

        # Critic
        self.value_head = nn.Linear(trunk_hidden, 1)

        # Replace default action_net and value_net so SB3 doesn't interfere
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)       # (B, trunk_hidden)
        return self._forward_from_features(features, deterministic, action_masks)

    def _forward_from_features(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._build_logits(features)       # (B, 26)
        values = self.value_head(features)          # (B, 1)

        distribution = self._get_action_dist_from_logits(logits, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _build_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Build the full (B, 26) logit tensor."""
        B = features.shape[0]

        # Pointer logits for moves 0-3
        move_h = self.mlp_extractor.move_hidden          # (B, 4, move_hidden)
        ptr_query = self.pointer_proj(features)           # (B, move_hidden)
        move_logits = torch.einsum(
            "bd,bnd->bn", ptr_query, move_h
        )                                                  # (B, 4)

        # Non-move logits
        non_move_logits = self.non_move_head(features)    # (B, 22)

        # Reassemble: [actions 0-5 | moves 6-9 | actions 10-25]
        logits = torch.cat([
            non_move_logits[:, :MOVE_ACTION_START],        # actions 0-5
            move_logits,                                    # actions 6-9
            non_move_logits[:, MOVE_ACTION_START:],        # actions 10-25
        ], dim=-1)                                         # (B, 26)

        return logits

    def _get_action_dist_from_logits(
        self,
        logits: torch.Tensor,
        action_masks: Optional[np.ndarray],
    ):
        """Create a (masked) categorical distribution from raw logits."""
        from sb3_contrib.common.maskable.distributions import (
            MaskableCategoricalDistribution,
            MaskableCategorical,
        )

        n_actions = int(self.action_space.n)  # type: ignore[union-attr]
        dist = MaskableCategoricalDistribution(n_actions)
        # MaskableCategorical is the correct subtype expected by the distribution
        dist.distribution = MaskableCategorical(logits=logits)  # type: ignore[assignment]
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)
        logits = self._build_logits(features)
        values = self.value_head(features)

        distribution = self._get_action_dist_from_logits(logits, action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs, action_masks: Optional[np.ndarray] = None):
        """Override to use pointer logits instead of SB3's default action_net path."""
        features = self.extract_features(obs)
        logits = self._build_logits(features)
        return self._get_action_dist_from_logits(logits, action_masks)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        dist = self.get_distribution(observation, action_masks)
        return dist.get_actions(deterministic=deterministic)

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.value_head(features)

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Run the attention extractor and return trunk features.

        The ``features_extractor`` argument is accepted for signature
        compatibility with ``MaskableActorCriticPolicy`` but is unused —
        ``AttentionPointerExtractor`` acts as both extractor and trunk.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device)
        return self.mlp_extractor(obs.float())
