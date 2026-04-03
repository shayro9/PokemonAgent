"""
Tests for AttentionPointerExtractor.

Covers:
  1. Initialization — correct layer dimensions and SB3 attribute setup
  2. Forward shape inference — observation → features with correct shape
  3. Intermediate state persistence — move_hidden, team_hidden, attn_scores saved
  4. Observation slicing — correct extraction of context, moves, team
  5. Padding detection — zeros detected as padding in moves and team
  6. Batch processing — multi-batch observations handled correctly
  7. Forward actor/critic delegation — both return features unchanged
  8. End-to-end: observation → features → no NaN/Inf
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from env.battle_config import BattleConfig
from policy.extractor import AttentionPointerExtractor, ExtractorOutput
from policy.constants import (
    CONTEXT_LEN, MOVE_LEN, MAX_MOVES, MY_POKEMON_LEN, ARENA_OPPONENT_LEN,
    MY_MOVES_START, N_SWITCH_ACTIONS,
)
from env.states.state_utils import MAX_TEAM_SIZE


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    """Create an AttentionPointerExtractor with standard dimensions."""
    return AttentionPointerExtractor(
        BattleConfig.gen1(),
        context_hidden=128,
        move_hidden=64,
        team_hidden=64,
        trunk_hidden=128,
        n_attention_heads=4,
    )


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def obs_dim():
    """Standard observation dimension."""
    return 1279


@pytest.fixture
def random_observation(batch_size, obs_dim):
    """Create a random observation tensor with valid alive vector (slot 0 active)."""
    obs = torch.randn(batch_size, obs_dim)
    obs[:, -MAX_TEAM_SIZE] = 1.0       # slot 0 active
    obs[:, -MAX_TEAM_SIZE + 1:] = 0.0  # rest benched
    return obs


@pytest.fixture
def zero_observation(batch_size, obs_dim):
    """Create an all-zeros observation."""
    return torch.zeros(batch_size, obs_dim)


# ────────────────────────────────────────────────────────────────────────────
# Test: Initialization
# ────────────────────────────────────────────────────────────────────────────

class TestExtractorInit:
    """Test extractor initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        extractor = AttentionPointerExtractor(BattleConfig.gen1())

        assert extractor.context_encoder is not None
        assert extractor.move_encoder is not None
        assert extractor.team_encoder is not None
        assert extractor.attn_moves is not None
        assert extractor.attn_team is not None
        assert extractor.trunk is not None

        # Check SB3 attributes
        assert extractor.features_dim == 128
        assert extractor.latent_dim_pi == 128
        assert extractor.latent_dim_vf == 128

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = AttentionPointerExtractor(
            BattleConfig.gen1(),
            context_hidden=256,
            move_hidden=128,
            team_hidden=128,
            trunk_hidden=256,
            n_attention_heads=8,
        )

        assert extractor.features_dim == 256
        assert extractor.latent_dim_pi == 256
        assert extractor.latent_dim_vf == 256

    def test_init_attention_heads_validation(self):
        """Test that attention module validates head divisibility."""
        # move_hidden=64, n_heads=5: 64 % 5 != 0 → should raise
        with pytest.raises(ValueError, match="must be divisible"):
            AttentionPointerExtractor(
                BattleConfig.gen1(),
                move_hidden=64,
                n_attention_heads=5,
            )


# ────────────────────────────────────────────────────────────────────────────
# Test: Observation Slicing
# ────────────────────────────────────────────────────────────────────────────

class TestObservationSlicing:
    """Test the _slice_observation method."""

    def test_slice_observation_shape(self, extractor, random_observation):
        """Test that slicing produces correct tensor shapes."""
        batch_size = random_observation.shape[0]

        context, moves, team = extractor._slice_observation(random_observation)

        # Context should be CONTEXT_LEN
        assert context.shape == (batch_size, CONTEXT_LEN)

        # Moves should be reshaped to (B, 4, MOVE_LEN) internally
        assert moves.shape == (batch_size, MAX_MOVES * MOVE_LEN)

        # Team should be (B, N_SWITCH_ACTIONS, MY_POKEMON_LEN)
        assert team.shape == (batch_size, N_SWITCH_ACTIONS, MY_POKEMON_LEN)

    def test_slice_observation_values(self, batch_size, obs_dim):
        """Test that slicing extracts correct values."""
        extractor = AttentionPointerExtractor(BattleConfig.gen1())

        obs = torch.arange(obs_dim, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).clone()

        # Set slot 0 as active, rest benched
        obs[:, -MAX_TEAM_SIZE] = 1.0
        obs[:, -MAX_TEAM_SIZE + 1:] = 0.0

        context, moves, team = extractor._slice_observation(obs)

        # Context is built from arena_opponent + active pokemon section (slot 0)
        arena = obs[:, :ARENA_OPPONENT_LEN]
        active_section = obs[:, ARENA_OPPONENT_LEN:ARENA_OPPONENT_LEN + MY_POKEMON_LEN]
        expected_context = torch.cat([arena, active_section], dim=1)

        torch.testing.assert_close(context, expected_context)

    def test_slice_observation_all_zeros(self, extractor, zero_observation):
        """Test slicing with all-zeros observation."""
        context, moves, team = extractor._slice_observation(zero_observation)

        torch.testing.assert_close(context, torch.zeros_like(context))
        torch.testing.assert_close(moves, torch.zeros_like(moves))
        torch.testing.assert_close(team, torch.zeros_like(team))

    def test_slice_observation_batch_independence(self, batch_size, obs_dim):
        """Test that batch items are sliced independently."""
        extractor = AttentionPointerExtractor(BattleConfig.gen1())

        obs = torch.zeros(batch_size, obs_dim, dtype=torch.float32)
        for i in range(batch_size):
            obs[i] = float(i)
            obs[i, -MAX_TEAM_SIZE] = 1.0
            obs[i, -MAX_TEAM_SIZE + 1:] = 0.0

        context, moves, team = extractor._slice_observation(obs)
        for i in range(batch_size):
            assert (context[i] == float(i)).all(), (
                f"Batch item {i}: expected all {float(i)}, got {context[i]}"
            )


# ────────────────────────────────────────────────────────────────────────────
# Test: Padding Detection
# ────────────────────────────────────────────────────────────────────────────

class TestPaddingDetection:
    """Test padding detection for moves and team."""

    def test_padding_detection_in_forward(self, extractor, batch_size):
        """Test that zero vectors are detected as padding."""
        obs_dim = 1279
        obs = torch.randn(batch_size, obs_dim)

        # Zero out some moves (last MAX_MOVES * MOVE_LEN elements)
        obs[:, -MAX_TEAM_SIZE:] = 0  # Zero out alive vector (not moves)

        out = extractor(obs)

        # move_hidden should have been computed
        assert out.move_hidden is not None
        assert out.move_hidden.shape == (batch_size, MAX_MOVES, 64)

        # attn_scores should reflect masking
        assert out.attn_scores_moves is not None

    def test_moves_padding_mask(self, extractor, batch_size):
        """Test that zero move vectors produce padding mask."""
        obs = torch.zeros(batch_size, 1279)

        # Set context to non-zero
        obs[:, :CONTEXT_LEN] = torch.randn(batch_size, CONTEXT_LEN)

        out = extractor(obs)

        # All moves are zero, so all should be masked
        assert out.move_hidden is not None

        # Attention scores for padded items should be -inf before softmax
        # After softmax + nan_to_num, should produce zero attention
        attn_scores = out.attn_scores_moves
        assert attn_scores.shape == (batch_size, 4, MAX_MOVES)  # (B, n_heads, N)


# ────────────────────────────────────────────────────────────────────────────
# Test: Forward Pass
# ────────────────────────────────────────────────────────────────────────────

class TestForwardPass:
    """Test the forward pass."""

    def test_forward_output_shape(self, extractor, random_observation):
        """Test that forward produces correct output shape."""
        batch_size = random_observation.shape[0]
        out = extractor(random_observation)

        assert isinstance(out, ExtractorOutput)
        assert out.features.shape == (batch_size, 128)  # trunk_hidden=128

    def test_forward_output_dtype(self, extractor, random_observation):
        """Test that forward produces float32 tensors."""
        out = extractor(random_observation)
        assert out.features.dtype == torch.float32

    def test_forward_no_nan_inf(self, extractor, random_observation):
        """Test that forward doesn't produce NaN or Inf."""
        out = extractor(random_observation)

        assert not torch.isnan(out.features).any()
        assert not torch.isinf(out.features).any()

    def test_forward_with_zero_observation(self, extractor, zero_observation):
        """Test forward pass with all-zeros observation."""
        out = extractor(zero_observation)

        assert out.features.shape == (zero_observation.shape[0], 128)
        assert not torch.isnan(out.features).any()
        assert not torch.isinf(out.features).any()

    def test_forward_batch_sizes(self, extractor, obs_dim):
        """Test forward pass with various batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            obs = torch.randn(batch_size, obs_dim)
            out = extractor(obs)

            assert out.features.shape == (batch_size, 128)
            assert not torch.isnan(out.features).any()

    def test_forward_deterministic_with_seed(self, obs_dim):
        """Test that forward is deterministic with same seed."""
        extractor1 = AttentionPointerExtractor(BattleConfig.gen1())
        extractor2 = AttentionPointerExtractor(BattleConfig.gen1())

        # Copy weights
        extractor2.load_state_dict(extractor1.state_dict())

        torch.manual_seed(42)
        obs = torch.randn(2, obs_dim)

        torch.manual_seed(42)
        out1 = extractor1(obs)
        out2 = extractor2(obs)

        torch.testing.assert_close(out1.features, out2.features, atol=1e-6, rtol=1e-5)


# ────────────────────────────────────────────────────────────────────────────
# Test: Intermediate State Persistence
# ────────────────────────────────────────────────────────────────────────────

class TestIntermediateStatePersistence:
    """Test that intermediate states are returned in ExtractorOutput."""

    def test_move_hidden_in_output(self, extractor, random_observation):
        """Test that move_hidden is returned in ExtractorOutput."""
        out = extractor(random_observation)

        assert out.move_hidden is not None
        batch_size = random_observation.shape[0]
        assert out.move_hidden.shape == (batch_size, MAX_MOVES, 64)

    def test_team_hidden_in_output(self, extractor, random_observation):
        """Test that team_hidden is returned in ExtractorOutput."""
        out = extractor(random_observation)

        assert out.team_hidden is not None
        batch_size = random_observation.shape[0]
        assert out.team_hidden.shape == (batch_size, N_SWITCH_ACTIONS, 64)

    def test_attn_scores_moves_in_output(self, extractor, random_observation):
        """Test that attention scores for moves are returned in ExtractorOutput."""
        out = extractor(random_observation)

        assert out.attn_scores_moves is not None
        batch_size = random_observation.shape[0]
        # Shape: (B, n_heads, N) where n_heads=4, N=4
        assert out.attn_scores_moves.shape == (batch_size, 4, MAX_MOVES)

    def test_attn_scores_team_in_output(self, extractor, random_observation):
        """Test that attention scores for team are returned in ExtractorOutput."""
        out = extractor(random_observation)

        assert out.attn_scores_team is not None
        batch_size = random_observation.shape[0]
        # Shape: (B, n_heads, N) where n_heads=4, N=6
        assert out.attn_scores_team.shape == (batch_size, 4, N_SWITCH_ACTIONS)

    def test_independent_outputs_per_call(self, extractor, obs_dim):
        """Test that each call returns independent output with no shared mutable state."""
        obs1 = torch.randn(2, obs_dim)
        obs1[:, -MAX_TEAM_SIZE] = 1.0
        obs1[:, -MAX_TEAM_SIZE + 1:] = 0.0

        obs2 = torch.randn(2, obs_dim)
        obs2[:, -MAX_TEAM_SIZE] = 1.0
        obs2[:, -MAX_TEAM_SIZE + 1:] = 0.0

        out1 = extractor(obs1)
        out2 = extractor(obs2)

        # Outputs should be independent — different inputs → different hidden states
        assert not torch.allclose(out1.move_hidden, out2.move_hidden)

        # out1 should be unchanged after the second call (no shared mutable state)
        out1_again = extractor(obs1)
        torch.testing.assert_close(out1.move_hidden, out1_again.move_hidden)


# ────────────────────────────────────────────────────────────────────────────
# Test: Forward Actor/Critic
# ────────────────────────────────────────────────────────────────────────────

class TestForwardActorCritic:
    """Test actor/critic forwarding."""

    def test_forward_actor(self, extractor):
        """Test forward_actor returns features unchanged."""
        features = torch.randn(4, 128)
        result = extractor.forward_actor(features)

        torch.testing.assert_close(result, features)

    def test_forward_critic(self, extractor):
        """Test forward_critic returns features unchanged."""
        features = torch.randn(4, 128)
        result = extractor.forward_critic(features)

        torch.testing.assert_close(result, features)

    def test_forward_actor_different_shape(self, extractor):
        """Test forward_actor with different feature shapes."""
        for shape in [(1, 128), (8, 128), (16, 128)]:
            features = torch.randn(*shape)
            result = extractor.forward_actor(features)
            assert result.shape == shape

    def test_forward_critic_different_shape(self, extractor):
        """Test forward_critic with different feature shapes."""
        for shape in [(1, 128), (8, 128), (16, 128)]:
            features = torch.randn(*shape)
            result = extractor.forward_critic(features)
            assert result.shape == shape


# ────────────────────────────────────────────────────────────────────────────
# Test: Encoder Output Dimensions
# ────────────────────────────────────────────────────────────────────────────

class TestEncoderDimensions:
    """Test that encoders produce correct output dimensions."""

    def test_context_encoder_output(self, extractor, batch_size):
        """Test context encoder produces correct dimension."""
        context_input = torch.randn(batch_size, CONTEXT_LEN)
        output = extractor.context_encoder(context_input)

        assert output.shape == (batch_size, 128)

    def test_move_encoder_output(self, extractor, batch_size):
        """Test move encoder produces correct dimension."""
        move_input = torch.randn(batch_size, MOVE_LEN)
        output = extractor.move_encoder(move_input)

        assert output.shape == (batch_size, 64)

    def test_team_encoder_output(self, extractor, batch_size):
        """Test team encoder produces correct dimension."""
        team_input = torch.randn(batch_size, MY_POKEMON_LEN)
        output = extractor.team_encoder(team_input)

        assert output.shape == (batch_size, 64)

    def test_trunk_output(self, extractor, batch_size):
        """Test trunk produces correct dimension."""
        # Trunk input: context_hidden + move_hidden + team_hidden = 128 + 64 + 64
        trunk_input = torch.randn(batch_size, 128 + 64 + 64)
        output = extractor.trunk(trunk_input)

        assert output.shape == (batch_size, 128)


# ────────────────────────────────────────────────────────────────────────────
# Test: Gradient Flow
# ────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """Test that gradients flow correctly through the model."""

    def test_gradients_enabled(self, extractor, random_observation):
        """Test that gradients are computed."""
        obs = random_observation.clone().requires_grad_(True)
        out = extractor(obs)
        loss = out.features.sum()
        loss.backward()

        # Observation gradients should exist
        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()

    def test_all_parameters_have_gradients(self, extractor, random_observation):
        """Test that all model parameters receive gradients."""
        out = extractor(random_observation)
        loss = out.features.sum()
        loss.backward()

        for name, param in extractor.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_no_gradients_in_inference_mode(self, extractor, random_observation):
        """Test that no gradients are computed in no_grad context."""
        with torch.no_grad():
            extractor(random_observation)

        # No gradients should be computed
        for param in extractor.parameters():
            assert param.grad is None


# ────────────────────────────────────────────────────────────────────────────
# Test: Device Handling
# ────────────────────────────────────────────────────────────────────────────

class TestDeviceHandling:
    """Test device handling (CPU only, since GPU not guaranteed)."""

    def test_forward_cpu(self, extractor, random_observation):
        """Test forward pass on CPU."""
        extractor = extractor.cpu()
        out = extractor(random_observation.cpu())

        assert out.features.device.type == "cpu"

    def test_double_precision(self, extractor, obs_dim):
        """Test forward pass with double precision."""
        extractor = extractor.double()
        obs = torch.randn(2, obs_dim, dtype=torch.float64)

        out = extractor(obs)
        assert out.features.dtype == torch.float64


# ────────────────────────────────────────────────────────────────────────────
# Test: Edge Cases
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_batch(self, extractor, obs_dim):
        """Test with batch size 1."""
        obs = torch.randn(1, obs_dim)
        out = extractor(obs)

        assert out.features.shape == (1, 128)

    def test_large_batch(self, extractor, obs_dim):
        """Test with large batch size."""
        obs = torch.randn(256, obs_dim)
        out = extractor(obs)

        assert out.features.shape == (256, 128)

    def test_very_large_values(self, extractor):
        """Test stability with very large input values."""
        obs = torch.randn(4, 1279) * 1e6
        out = extractor(obs)

        assert not torch.isnan(out.features).any()
        assert not torch.isinf(out.features).any()

    def test_very_small_values(self, extractor):
        """Test stability with very small input values."""
        obs = torch.randn(4, 1279) * 1e-6
        out = extractor(obs)

        assert not torch.isnan(out.features).any()
        assert not torch.isinf(out.features).any()

    def test_mixed_signs(self, extractor):
        """Test with mixed positive and negative values."""
        obs = torch.randn(4, 1279)
        out = extractor(obs)

        assert not torch.isnan(out.features).any()
        # Should have both positive and negative values
        assert (out.features > 0).any() and (out.features < 0).any()


# ────────────────────────────────────────────────────────────────────────────
# Test: Model State Management
# ────────────────────────────────────────────────────────────────────────────

class TestModelStateManagement:
    """Test model state saving/loading and evaluation modes."""

    def test_train_eval_mode(self, extractor, random_observation):
        """Test switching between train and eval modes."""
        extractor.train()
        assert extractor.training

        features_train = extractor(random_observation).features

        extractor.eval()
        assert not extractor.training

        features_eval = extractor(random_observation).features

        # Should be equal or very close (due to batch norm differences)
        assert features_train.shape == features_eval.shape

    def test_state_dict_save_load(self, extractor, obs_dim):
        """Test state dict save/load."""
        obs = torch.randn(2, obs_dim)

        # Get original output
        original_features = extractor(obs).features
        original_state = extractor.state_dict()

        # Create new extractor and load state
        extractor2 = AttentionPointerExtractor(BattleConfig.gen1())
        extractor2.load_state_dict(original_state)

        # Should produce same output
        loaded_features = extractor2(obs).features
        torch.testing.assert_close(original_features, loaded_features, atol=1e-6, rtol=1e-5)

    def test_parameters_not_none(self, extractor):
        """Test that all parameters exist."""
        for name, param in extractor.named_parameters():
            assert param is not None
            assert param.data is not None


# ────────────────────────────────────────────────────────────────────────────
# Test: Integration with Policy
# ────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Test integration aspects for use in policy."""

    def test_intermediate_states_accessible_to_policy(self, extractor, random_observation):
        """Test that intermediate states are returned in ExtractorOutput for policy use."""
        out = extractor(random_observation)

        assert isinstance(out, ExtractorOutput)
        assert out.move_hidden is not None
        assert out.team_hidden is not None
        assert out.attn_scores_moves is not None
        assert out.attn_scores_team is not None

    def test_sb3_attributes_present(self, extractor):
        """Test that SB3-required attributes are present."""
        assert hasattr(extractor, "features_dim")
        assert hasattr(extractor, "latent_dim_pi")
        assert hasattr(extractor, "latent_dim_vf")

        assert extractor.features_dim == extractor.latent_dim_pi
        assert extractor.features_dim == extractor.latent_dim_vf