"""
Tests for policy/policy.py — AttentionPointerPolicy.

Covers:
  1. Policy construction (Box obs + Discrete action)
  2. forward() returns (actions, values, log_probs) with correct shapes/types
  3. Action masking — masked actions are never sampled (run many trials)
  4. evaluate_actions() returns (values, log_probs, entropy) with correct shapes
  5. get_distribution() returns a MaskableCategoricalDistribution
  6. predict_values() returns (B, 1) values tensor
  7. extract_features() returns (B, trunk_hidden) tensor
  8. _build_logits output shape is (B, ACTION_SPACE)
  9. _run_extractor returns ExtractorOutput with correct fields
"""

import numpy as np
import pytest
import torch
import gymnasium
from gymnasium import spaces

from policy.policy import AttentionPointerPolicy
from policy.extractor import ExtractorOutput
from env.action_mask_gen_1 import ActionMaskGen1

# ─── Constants ───────────────────────────────────────────────────────────────

OBS_DIM = 1279
ACTION_SPACE = ActionMaskGen1.ACTION_SPACE  # 10

# Small dims for fast CPU tests
CONTEXT_HIDDEN = 32
MOVE_HIDDEN = 16
TEAM_HIDDEN = 16
TRUNK_HIDDEN = 32
N_HEADS = 2


# ─── Fixture ─────────────────────────────────────────────────────────────────

def make_policy() -> AttentionPointerPolicy:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Discrete(ACTION_SPACE)
    lr_schedule = lambda _: 3e-4
    return AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lr_schedule,
        context_hidden=CONTEXT_HIDDEN,
        move_hidden=MOVE_HIDDEN,
        team_hidden=TEAM_HIDDEN,
        trunk_hidden=TRUNK_HIDDEN,
        n_attention_heads=N_HEADS,
    )


def rand_obs(B: int = 4) -> torch.Tensor:
    return torch.randn(B, OBS_DIM)


def all_valid_mask(B: int = 4) -> np.ndarray:
    """All actions enabled."""
    return np.ones((B, ACTION_SPACE), dtype=bool)


def moves_only_mask(B: int = 4) -> np.ndarray:
    """Only move actions 6-9 are valid."""
    mask = np.zeros((B, ACTION_SPACE), dtype=bool)
    mask[:, list(ActionMaskGen1.ACTION_MOVE_RANGE)] = True
    return mask


def switches_only_mask(B: int = 4) -> np.ndarray:
    """Only switch actions 0-5 are valid."""
    mask = np.zeros((B, ACTION_SPACE), dtype=bool)
    mask[:, list(ActionMaskGen1.ACTION_SWITCH_RANGE)] = True
    return mask


# ─── 1. Construction ─────────────────────────────────────────────────────────

class TestPolicyConstruction:

    def test_policy_creates_without_error(self):
        policy = make_policy()
        assert policy is not None

    def test_policy_has_mlp_extractor(self):
        policy = make_policy()
        assert hasattr(policy, "mlp_extractor")

    def test_policy_has_pointer_heads(self):
        policy = make_policy()
        assert hasattr(policy, "move_ptr_proj")
        assert hasattr(policy, "switch_ptr_proj")

    def test_policy_has_value_head(self):
        policy = make_policy()
        assert hasattr(policy, "value_head")

    def test_action_space_is_10(self):
        policy = make_policy()
        assert policy.action_space.n == ACTION_SPACE


# ─── 2. forward() ────────────────────────────────────────────────────────────

class TestPolicyForward:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_forward_returns_tuple_of_3(self):
        obs = rand_obs(4)
        with torch.no_grad():
            result = self.policy.forward(obs, action_masks=all_valid_mask(4))
        assert len(result) == 3

    def test_actions_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, values, log_probs = self.policy.forward(obs, action_masks=all_valid_mask(4))
        assert actions.shape == (4,), f"Expected (4,) got {actions.shape}"

    def test_values_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, values, log_probs = self.policy.forward(obs, action_masks=all_valid_mask(4))
        assert values.shape == (4, 1), f"Expected (4, 1) got {values.shape}"

    def test_log_probs_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, values, log_probs = self.policy.forward(obs, action_masks=all_valid_mask(4))
        assert log_probs.shape == (4,), f"Expected (4,) got {log_probs.shape}"

    def test_actions_in_valid_range(self):
        obs = rand_obs(8)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(8))
        assert (actions >= 0).all()
        assert (actions < ACTION_SPACE).all()

    def test_log_probs_are_negative(self):
        """Log probabilities must be ≤ 0."""
        obs = rand_obs(4)
        with torch.no_grad():
            _, _, log_probs = self.policy.forward(obs, action_masks=all_valid_mask(4))
        assert (log_probs <= 0).all()

    def test_batch_size_1(self):
        obs = rand_obs(1)
        with torch.no_grad():
            actions, values, log_probs = self.policy.forward(obs, action_masks=all_valid_mask(1))
        assert actions.shape == (1,)
        assert values.shape == (1, 1)
        assert log_probs.shape == (1,)


# ─── 3. Action masking ───────────────────────────────────────────────────────

class TestActionMasking:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def _sample_actions(self, mask_fn, n_samples: int = 200) -> set:
        sampled = set()
        for _ in range(n_samples):
            obs = rand_obs(1)
            mask = mask_fn(1)
            with torch.no_grad():
                actions, _, _ = self.policy.forward(obs, action_masks=mask)
            sampled.add(actions.item())
        return sampled

    def test_moves_only_mask_never_returns_switch(self):
        """With moves-only mask, no switch action (0-5) should appear."""
        sampled = self._sample_actions(moves_only_mask, n_samples=100)
        switch_set = set(ActionMaskGen1.ACTION_SWITCH_RANGE)
        assert sampled.isdisjoint(switch_set), \
            f"Got switch actions {sampled & switch_set} despite moves-only mask"

    def test_switches_only_mask_never_returns_move(self):
        """With switches-only mask, no move action (6-9) should appear."""
        sampled = self._sample_actions(switches_only_mask, n_samples=100)
        move_set = set(ActionMaskGen1.ACTION_MOVE_RANGE)
        assert sampled.isdisjoint(move_set), \
            f"Got move actions {sampled & move_set} despite switches-only mask"

    def test_moves_only_mask_all_samples_in_range(self):
        """All sampled actions should be in ACTION_MOVE_RANGE."""
        sampled = self._sample_actions(moves_only_mask, n_samples=100)
        move_set = set(ActionMaskGen1.ACTION_MOVE_RANGE)
        assert sampled.issubset(move_set)

    def test_single_action_always_selected(self):
        """If only one action is valid, it must always be selected."""
        single_mask = np.zeros((1, ACTION_SPACE), dtype=bool)
        single_mask[0, 7] = True  # only move slot 1
        for _ in range(20):
            obs = rand_obs(1)
            with torch.no_grad():
                actions, _, _ = self.policy.forward(obs, action_masks=single_mask)
            assert actions.item() == 7, f"Expected 7 but got {actions.item()}"


# ─── 4. evaluate_actions() ───────────────────────────────────────────────────

class TestEvaluateActions:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_evaluate_actions_returns_3_tuple(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            result = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert len(result) == 3

    def test_evaluate_values_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert values.shape == (4, 1)

    def test_evaluate_log_probs_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert log_probs.shape == (4,)

    def test_evaluate_entropy_not_none(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert entropy is not None

    def test_evaluate_entropy_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert entropy.shape == (4,)

    def test_evaluate_entropy_non_negative(self):
        obs = rand_obs(4)
        with torch.no_grad():
            actions, _, _ = self.policy.forward(obs, action_masks=all_valid_mask(4))
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=all_valid_mask(4))
        assert (entropy >= 0).all()

    def test_single_valid_action_entropy_near_zero(self):
        """When only one action is valid, entropy should be near zero."""
        single_mask = np.zeros((4, ACTION_SPACE), dtype=bool)
        single_mask[:, 6] = True
        obs = rand_obs(4)
        with torch.no_grad():
            actions = torch.full((4,), 6, dtype=torch.long)
            values, log_probs, entropy = self.policy.evaluate_actions(obs, actions, action_masks=single_mask)
        assert (entropy < 0.01).all(), f"Expected near-zero entropy, got {entropy}"


# ─── 5. get_distribution() ───────────────────────────────────────────────────

class TestGetDistribution:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_returns_distribution(self):
        from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
        obs = rand_obs(4)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs, action_masks=all_valid_mask(4))
        assert isinstance(dist, MaskableCategoricalDistribution)

    def test_distribution_can_sample(self):
        obs = rand_obs(4)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs, action_masks=all_valid_mask(4))
            actions = dist.get_actions(deterministic=False)
        assert actions.shape == (4,)


# ─── 6 & 7. predict_values / extract_features ────────────────────────────────

class TestPredictValues:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_predict_values_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            values = self.policy.predict_values(obs)
        assert values.shape == (4, 1)

    def test_predict_values_is_tensor(self):
        obs = rand_obs(4)
        with torch.no_grad():
            values = self.policy.predict_values(obs)
        assert isinstance(values, torch.Tensor)


class TestExtractFeatures:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_extract_features_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            features = self.policy.extract_features(obs)
        assert features.shape == (4, TRUNK_HIDDEN)

    def test_extract_features_is_tensor(self):
        obs = rand_obs(4)
        with torch.no_grad():
            features = self.policy.extract_features(obs)
        assert isinstance(features, torch.Tensor)


# ─── 8 & 9. _build_logits / _run_extractor ───────────────────────────────────

class TestInternalMethods:

    def setup_method(self):
        self.policy = make_policy()
        self.policy.eval()

    def test_build_logits_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
            logits = self.policy._build_logits(out.features, out.move_hidden, out.team_hidden)
        assert logits.shape == (4, ACTION_SPACE), f"Expected (4, {ACTION_SPACE}) got {logits.shape}"

    def test_run_extractor_returns_extractor_output(self):
        obs = rand_obs(4)
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
        assert isinstance(out, ExtractorOutput)

    def test_run_extractor_features_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
        assert out.features.shape == (4, TRUNK_HIDDEN)

    def test_run_extractor_move_hidden_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
        assert out.move_hidden.shape == (4, 4, MOVE_HIDDEN), \
            f"Expected (4, 4, {MOVE_HIDDEN}) got {out.move_hidden.shape}"

    def test_run_extractor_team_hidden_shape(self):
        obs = rand_obs(4)
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
        assert out.team_hidden.shape == (4, 6, TEAM_HIDDEN), \
            f"Expected (4, 6, {TEAM_HIDDEN}) got {out.team_hidden.shape}"

    def test_run_extractor_with_dict_obs(self):
        """_run_extractor should also handle dict observations (ActionMasker wrapper)."""
        obs = {"observation": rand_obs(2)}
        with torch.no_grad():
            out = self.policy._run_extractor(obs)
        assert out.features.shape == (2, TRUNK_HIDDEN)
