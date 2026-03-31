"""
AttentionPointerPolicy — MaskablePPO policy with dual pointer heads for moves and switches.

Move actions (indices 6-9) are scored via:
    logit_i = dot( move_ptr_proj(trunk_features), move_hidden_i )

Switch actions (indices 0-5) are scored via:
    logit_j = dot( switch_ptr_proj(trunk_features), bench_hidden_j )

Other actions (indices 10-25) are scored by a standard linear head.
This makes move and switch selection fully order-invariant over their respective slots.
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

from .constants import N_SWITCH_ACTIONS
from .extractor import AttentionPointerExtractor, ExtractorOutput


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
        team_hidden: int = 64,
        trunk_hidden: int = 128,
        n_attention_heads: int = 4,
        **kwargs,
    ) -> None:
        self._attn_kwargs = dict(
            context_hidden=context_hidden,
            move_hidden=move_hidden,
            team_hidden=team_hidden,
            trunk_hidden=trunk_hidden,
            n_attention_heads=n_attention_heads,
        )
        kwargs.setdefault("net_arch", [])   # disable SB3's default MLP extractor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ── extractor ──────────────────────────────────────────────────────────

    def _build_mlp_extractor(self) -> None:
        # Handle both Box and Dict observation spaces
        # Dict space occurs when ActionMasker wraps the environment
        if isinstance(self.observation_space, spaces.Dict):
            # Extract the actual observation space from the Dict
            obs_space = self.observation_space.spaces['observation']
            obs_dim = int(np.prod(obs_space.shape))
        else:
            # Direct Box observation space
            obs_dim = int(np.prod(self.observation_space.shape))
        
        self.mlp_extractor = AttentionPointerExtractor(obs_dim, **self._attn_kwargs)

    # ── heads ──────────────────────────────────────────────────────────────

    def _build(self, lr_schedule: Schedule) -> None:
        """Called by __init__ after _build_mlp_extractor."""
        super()._build(lr_schedule)

        trunk_hidden = self._attn_kwargs["trunk_hidden"]
        move_hidden  = self._attn_kwargs["move_hidden"]
        team_hidden  = self._attn_kwargs["team_hidden"]

        # Pointer projections: trunk → action_hidden space (dot product with encoded_i)
        self.move_ptr_proj  = nn.Linear(trunk_hidden, move_hidden, bias=False)
        self.switch_ptr_proj = nn.Linear(trunk_hidden, team_hidden, bias=False)
        self.value_head     = nn.Linear(trunk_hidden, 1)

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
        out = self._run_extractor(obs)
        return self._forward_from_features(out, deterministic, action_masks)

    def _forward_from_features(
        self,
        out: ExtractorOutput,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._build_logits(out.features, out.move_hidden, out.team_hidden)
        values = self.value_head(out.features)

        dist    = self._distribution(logits, action_masks)
        actions  = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    # ── logit construction ─────────────────────────────────────────────────

    def _build_logits(
        self,
        features: torch.Tensor,
        move_hidden: torch.Tensor,
        team_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Build the full (B, 26) logit vector.
        
        Action layout: [0-5 switches | 6-9 moves | 10-25 other]
        """
        # Pointer logits for move slots 0-3
        move_ptr_query = self.move_ptr_proj(features)        # (B, move_hidden)
        move_logits = torch.einsum("bd,bnd->bn", move_ptr_query, move_hidden)  # (B, 4)

        # Pointer logits for team slots 0-5 (team = active + bench)
        switch_ptr_query = self.switch_ptr_proj(features)   # (B, team_hidden)
        switch_logits = torch.einsum("bd,bnd->bn", switch_ptr_query, team_hidden)  # (B, 6)

        # Reassemble in action-space order: [0-5 switches | 6-9 moves ]
        return torch.cat([
            switch_logits,           # 0-5 (6 logits)
            move_logits,             # 6-9 (4 logits)
        ], dim=-1)                   # (B, 26)

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
        out      = self._run_extractor(obs)
        logits   = self._build_logits(out.features, out.move_hidden, out.team_hidden)
        values   = self.value_head(out.features)

        dist     = self._distribution(logits, action_masks)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: Optional[np.ndarray] = None,
    ) -> MaskableCategoricalDistribution:
        out    = self._run_extractor(obs)
        logits = self._build_logits(out.features, out.move_hidden, out.team_hidden)
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
        return self.value_head(self._run_extractor(obs).features)

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Accepts `features_extractor` only for SB3 signature compatibility."""
        return self._run_extractor(obs).features

    def _run_extractor(self, obs: PyTorchObs) -> ExtractorOutput:
        """Run the extractor and return the full output including hidden states."""
        if isinstance(obs, dict):
            obs_tensor = obs['observation']
        else:
            obs_tensor = obs

        if not isinstance(obs_tensor, torch.Tensor):
            obs_tensor = torch.as_tensor(obs_tensor, device=self.device)
        return self.mlp_extractor(obs_tensor.float())

