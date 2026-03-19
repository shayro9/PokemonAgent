"""
AttentionPointerPolicy — MaskablePPO policy with a pointer head for moves.

Move actions (indices 6-9) are scored via:
    logit_i = dot( pointer_proj(trunk_features), move_hidden_i )

Non-move actions (indices 0-5, 10-25) are scored by a standard linear head.
This makes move selection fully order-invariant over the 4 move slots.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .constants import MOVE_ACTION_START, N_MOVE_ACTIONS, TOTAL_ACTIONS
from .extractor import AttentionPointerExtractor


class AttentionPointerPolicy(MaskableActorCriticPolicy):
    """
    Parameters
    ----------
    context_hidden    : hidden size of the context encoder MLP
    move_hidden       : hidden size of the move encoder MLP
    trunk_hidden      : hidden size of the shared trunk MLP
    n_attention_heads : number of heads in CrossAttention
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
    ) -> None:
        self._attn_kwargs = dict(
            context_hidden=context_hidden,
            move_hidden=move_hidden,
            trunk_hidden=trunk_hidden,
            n_attention_heads=n_attention_heads,
        )
        kwargs.setdefault("net_arch", [])   # disable SB3's default MLP extractor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ── extractor ──────────────────────────────────────────────────────────

    def _build_mlp_extractor(self) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        self.mlp_extractor = AttentionPointerExtractor(obs_dim, **self._attn_kwargs)

    # ── heads ──────────────────────────────────────────────────────────────

    def _build(self, lr_schedule: Schedule) -> None:
        """Called by __init__ after _build_mlp_extractor."""
        super()._build(lr_schedule)

        trunk_hidden = self._attn_kwargs["trunk_hidden"]
        move_hidden  = self._attn_kwargs["move_hidden"]
        n_non_move   = TOTAL_ACTIONS - N_MOVE_ACTIONS    # 22

        # Pointer projection: trunk → move_hidden space (dot product with move_hidden_i)
        self.pointer_proj  = nn.Linear(trunk_hidden, move_hidden, bias=False)
        self.non_move_head = nn.Linear(trunk_hidden, n_non_move)
        self.value_head    = nn.Linear(trunk_hidden, 1)

        # Prevent SB3's default heads from interfering
        self.action_net = nn.Identity()
        self.value_net  = nn.Identity()

    # ── core forward ───────────────────────────────────────────────────────

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        return self._forward_from_features(features, deterministic, action_masks)

    def _forward_from_features(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._build_logits(features)       # (B, 26)
        values = self.value_head(features)          # (B, 1)

        dist    = self._distribution(logits, action_masks)
        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    # ── logit construction ─────────────────────────────────────────────────

    def _build_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Build the full (B, 26) logit vector."""
        # Pointer logits for move slots 0-3
        move_h    = self.mlp_extractor.move_hidden          # (B, 4, move_hidden)
        ptr_query = self.pointer_proj(features)              # (B, move_hidden)
        move_logits = torch.einsum("bd,bnd->bn", ptr_query, move_h)  # (B, 4)

        # Standard logits for all non-move actions
        non_move_logits = self.non_move_head(features)       # (B, 22)

        # Reassemble in action-space order: [0-5 | moves 6-9 | 10-25]
        return torch.cat([
            non_move_logits[:, :MOVE_ACTION_START],
            move_logits,
            non_move_logits[:, MOVE_ACTION_START:],
        ], dim=-1)                                           # (B, 26)

    # ── distribution helpers ───────────────────────────────────────────────

    def _distribution(
        self,
        logits: torch.Tensor,
        action_masks: Optional[np.ndarray],
    ) -> MaskableCategoricalDistribution:
        n_actions = int(self.action_space.n)  # type: ignore[union-attr]
        dist = MaskableCategoricalDistribution(n_actions)
        dist.distribution = MaskableCategorical(logits=logits)  # type: ignore[assignment]
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    # ── SB3 interface methods ──────────────────────────────────────────────

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)
        logits   = self._build_logits(features)
        values   = self.value_head(features)

        dist     = self._distribution(logits, action_masks)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: Optional[np.ndarray] = None,
    ) -> MaskableCategoricalDistribution:
        features = self.extract_features(obs)
        logits   = self._build_logits(features)
        return self._distribution(logits, action_masks)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(
            deterministic=deterministic
        )

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        return self.value_head(self.extract_features(obs))

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Accepts `features_extractor` only for SB3 signature compatibility."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device)
        return self.mlp_extractor(obs.float())

