"""
Unit tests for encoder modules.

Tests shape consistency, gradient flow, edge cases (empty slots, masking),
and permutation equivariance properties.
"""

import pytest
import torch

from policy.encoders.pokemon_encoder import PokemonEncoder
from policy.encoders.move_encoder import MoveEncoder
from policy.encoders.bench_encoder import BenchEncoder
from policy.encoders.field_encoder import FieldEncoder
from env.states.move_state import MoveState


# ============================================================================
# PokemonEncoder Tests
# ============================================================================

class TestPokemonEncoder:
    """Test suite for PokemonEncoder."""

    @pytest.fixture
    def pokemon_encoder(self):
        """Create a PokemonEncoder instance."""
        return PokemonEncoder(pokemon_state_len=50, pokemon_hidden=64, layers=2)

    def test_single_pokemon_forward(self, pokemon_encoder):
        """Test forward pass with a single Pokémon state."""
        pokemon_state = torch.randn(50)
        output = pokemon_encoder(pokemon_state)
        assert output.shape == (64,), f"Expected shape (64,), got {output.shape}"

    def test_batch_pokemon_forward(self, pokemon_encoder):
        """Test forward pass with a batch of Pokémon states."""
        pokemon_states = torch.randn(4, 50)
        output = pokemon_encoder(pokemon_states)
        assert output.shape == (4, 64), f"Expected shape (4, 64), got {output.shape}"

    def test_batched_multiple_pokemon(self, pokemon_encoder):
        """Test forward pass with batch of multiple Pokémon (3D input)."""
        pokemon_states = torch.randn(2, 5, 50)  # batch_size=2, n_pokemon=5
        output = pokemon_encoder(pokemon_states)
        assert output.shape == (2, 5, 64), f"Expected shape (2, 5, 64), got {output.shape}"

    def test_gradient_flow(self, pokemon_encoder):
        """Test that gradients flow through the encoder."""
        pokemon_state = torch.randn(50, requires_grad=True)
        output = pokemon_encoder(pokemon_state)
        loss = output.sum()
        loss.backward()
        assert pokemon_state.grad is not None, "Gradients did not flow"
        assert pokemon_state.grad.shape == pokemon_state.shape

    def test_permutation_equivariance(self, pokemon_encoder):
        """Test that encoding individual Pokémon is independent of their order."""
        pokemon1 = torch.randn(50)
        pokemon2 = torch.randn(50)
        pokemon3 = torch.randn(50)

        # Encode individually
        enc1_solo = pokemon_encoder(pokemon1).detach()
        enc2_solo = pokemon_encoder(pokemon2).detach()
        enc3_solo = pokemon_encoder(pokemon3).detach()

        # Encode as batch in original order
        batch_original = torch.stack([pokemon1, pokemon2, pokemon3])
        enc_batch_original = pokemon_encoder(batch_original).detach()

        # Encode as batch in permuted order
        batch_permuted = torch.stack([pokemon3, pokemon1, pokemon2])
        enc_batch_permuted = pokemon_encoder(batch_permuted).detach()

        # Individual encodings should match batch encodings
        assert torch.allclose(enc1_solo, enc_batch_original[0], atol=1e-5)
        assert torch.allclose(enc2_solo, enc_batch_original[1], atol=1e-5)
        assert torch.allclose(enc3_solo, enc_batch_original[2], atol=1e-5)

        # Permuted batch should have permuted outputs (not the same global order)
        assert torch.allclose(enc_batch_permuted[0], enc3_solo, atol=1e-5)  # position 0 is pokemon3
        assert torch.allclose(enc_batch_permuted[1], enc1_solo, atol=1e-5)  # position 1 is pokemon1
        assert torch.allclose(enc_batch_permuted[2], enc2_solo, atol=1e-5)  # position 2 is pokemon2


# ============================================================================
# MoveEncoder Tests
# ============================================================================

class TestMoveEncoder:
    """Test suite for MoveEncoder."""

    @pytest.fixture
    def move_encoder(self):
        """Create a MoveEncoder instance."""
        return MoveEncoder(move_state_len=MoveState.array_len(), move_hidden=64, layers=2)

    def test_single_move_forward(self, move_encoder):
        """Test forward pass with a single move state."""
        move_state = torch.randn(MoveState.array_len())
        output = move_encoder(move_state)
        assert output.shape == (64,), f"Expected shape (64,), got {output.shape}"

    def test_batch_moves_forward(self, move_encoder):
        """Test forward pass with a batch of move states."""
        move_states = torch.randn(4, MoveState.array_len())
        output = move_encoder(move_states)
        assert output.shape == (4, 64), f"Expected shape (4, 64), got {output.shape}"

    def test_batched_multiple_moves(self, move_encoder):
        """Test forward pass with batch of multiple moves (3D input)."""
        move_states = torch.randn(2, 4, MoveState.array_len())  # batch_size=2, n_moves=4
        output = move_encoder(move_states)
        assert output.shape == (2, 4, 64), f"Expected shape (2, 4, 64), got {output.shape}"

    def test_gradient_flow(self, move_encoder):
        """Test that gradients flow through the encoder."""
        move_state = torch.randn(MoveState.array_len(), requires_grad=True)
        output = move_encoder(move_state)
        loss = output.sum()
        loss.backward()
        assert move_state.grad is not None, "Gradients did not flow"
        assert move_state.grad.shape == move_state.shape

    def test_zero_move_handling(self, move_encoder):
        """Test handling of zero/empty move states (padding)."""
        zero_move = torch.zeros(MoveState.array_len())
        output = move_encoder(zero_move)
        assert output.shape == (64,)
        assert not torch.isnan(output).any(), "Output contains NaN"


# ============================================================================
# BenchEncoder Tests
# ============================================================================

class TestBenchEncoder:
    """Test suite for BenchEncoder."""

    @pytest.fixture
    def bench_encoder_mean(self):
        """Create a BenchEncoder with mean pooling."""
        return BenchEncoder(pokemon_hidden=64, pooling="mean")

    @pytest.fixture
    def bench_encoder_max(self):
        """Create a BenchEncoder with max pooling."""
        return BenchEncoder(pokemon_hidden=64, pooling="max")

    @pytest.fixture
    def bench_encoder_attention(self):
        """Create a BenchEncoder with attention pooling."""
        return BenchEncoder(pokemon_hidden=64, pooling="attention")

    def test_mean_pooling_single(self, bench_encoder_mean):
        """Test mean pooling with unbatched input."""
        pokemon_encodings = torch.randn(5, 64)  # 5 Pokémon
        output = bench_encoder_mean(pokemon_encodings)
        assert output.shape == (64,), f"Expected shape (64,), got {output.shape}"
        # Mean pooling should give average of all encodings
        expected = pokemon_encodings.mean(dim=0)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_mean_pooling_batch(self, bench_encoder_mean):
        """Test mean pooling with batched input."""
        pokemon_encodings = torch.randn(2, 5, 64)  # batch_size=2, n_pokemon=5
        output = bench_encoder_mean(pokemon_encodings)
        assert output.shape == (2, 64), f"Expected shape (2, 64), got {output.shape}"
        # Verify mean for each batch
        expected = pokemon_encodings.mean(dim=1)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_max_pooling_batch(self, bench_encoder_max):
        """Test max pooling with batched input."""
        pokemon_encodings = torch.randn(2, 5, 64)  # batch_size=2, n_pokemon=5
        output = bench_encoder_max(pokemon_encodings)
        assert output.shape == (2, 64), f"Expected shape (2, 64), got {output.shape}"
        # Verify max for each batch
        expected, _ = pokemon_encodings.max(dim=1)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_attention_pooling_batch(self, bench_encoder_attention):
        """Test attention pooling with batched input."""
        pokemon_encodings = torch.randn(2, 5, 64)  # batch_size=2, n_pokemon=5
        output = bench_encoder_attention(pokemon_encodings)
        assert output.shape == (2, 64), f"Expected shape (2, 64), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_masking_mean_pooling(self, bench_encoder_mean):
        """Test masking with mean pooling."""
        pokemon_encodings = torch.randn(2, 5, 64)
        mask = torch.tensor([[True, True, False, False, False],
                            [True, True, True, False, False]])  # Only first N valid
        output = bench_encoder_mean(pokemon_encodings, mask=mask)
        assert output.shape == (2, 64)

        # Manual check: first batch should average only first 2
        expected_0 = pokemon_encodings[0, :2].mean(dim=0)
        assert torch.allclose(output[0], expected_0, atol=1e-5)

        # Second batch should average only first 3
        expected_1 = pokemon_encodings[1, :3].mean(dim=0)
        assert torch.allclose(output[1], expected_1, atol=1e-5)

    def test_permutation_invariance_mean(self, bench_encoder_mean):
        """Test that mean pooling is permutation-invariant."""
        pokemon_encodings = torch.randn(5, 64)
        output_original = bench_encoder_mean(pokemon_encodings).detach()

        # Permute and pool
        perm_indices = torch.tensor([3, 1, 4, 0, 2])
        pokemon_permuted = pokemon_encodings[perm_indices]
        output_permuted = bench_encoder_mean(pokemon_permuted).detach()

        # Mean pooling is permutation-invariant
        assert torch.allclose(output_original, output_permuted, atol=1e-5)

    def test_gradient_flow(self, bench_encoder_mean):
        """Test that gradients flow through the encoder."""
        pokemon_encodings = torch.randn(2, 5, 64, requires_grad=True)
        output = bench_encoder_mean(pokemon_encodings)
        loss = output.sum()
        loss.backward()
        assert pokemon_encodings.grad is not None, "Gradients did not flow"


# ============================================================================
# FieldEncoder Tests
# ============================================================================

class TestFieldEncoder:
    """Test suite for FieldEncoder."""

    @pytest.fixture
    def field_encoder(self):
        """Create a FieldEncoder instance."""
        return FieldEncoder(context_len=100, bench_hidden=64, field_hidden=128, layers=2)

    def test_single_forward(self, field_encoder):
        """Test forward pass with unbatched inputs."""
        context = torch.randn(100)
        bench_encoding = torch.randn(64)
        output = field_encoder(context, bench_encoding)
        assert output.shape == (128,), f"Expected shape (128,), got {output.shape}"

    def test_batch_forward(self, field_encoder):
        """Test forward pass with batched inputs."""
        context = torch.randn(4, 100)
        bench_encoding = torch.randn(4, 64)
        output = field_encoder(context, bench_encoding)
        assert output.shape == (4, 128), f"Expected shape (4, 128), got {output.shape}"

    def test_gradient_flow(self, field_encoder):
        """Test that gradients flow through the encoder."""
        context = torch.randn(100, requires_grad=True)
        bench_encoding = torch.randn(64, requires_grad=True)
        output = field_encoder(context, bench_encoding)
        loss = output.sum()
        loss.backward()
        assert context.grad is not None, "Gradients did not flow to context"
        assert bench_encoding.grad is not None, "Gradients did not flow to bench_encoding"

    def test_batch_mixed_with_unbatched(self, field_encoder):
        """Test that batched context works with unbatched bench and vice versa."""
        context = torch.randn(4, 100)
        bench_encoding = torch.randn(4, 64)
        output = field_encoder(context, bench_encoding)
        assert output.shape == (4, 128)

    def test_deterministic_forward(self, field_encoder):
        """Test that forward pass is deterministic (no randomness except dropout)."""
        context = torch.randn(100)
        bench_encoding = torch.randn(64)
        field_encoder.eval()  # Disable dropout
        with torch.no_grad():
            output1 = field_encoder(context, bench_encoding)
            output2 = field_encoder(context, bench_encoding)
        assert torch.allclose(output1, output2, atol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestEncoderIntegration:
    """Integration tests combining multiple encoders."""

    def test_full_pipeline_unbatched(self):
        """Test the full encoding pipeline without batching."""
        # Setup
        pokemon_encoder = PokemonEncoder(pokemon_state_len=50, pokemon_hidden=64, layers=2)
        bench_encoder = BenchEncoder(pokemon_hidden=64, pooling="mean")
        field_encoder = FieldEncoder(context_len=100, bench_hidden=64, field_hidden=128, layers=2)

        # Opponent bench encoding
        opp_pokemon_states = torch.randn(5, 50)  # 5 Pokémon on opponent bench
        opp_encodings = pokemon_encoder(opp_pokemon_states)  # (5, 64)
        opp_bench = bench_encoder(opp_encodings)  # (64,)

        # Field context
        context = torch.randn(100)
        field_ctx = field_encoder(context, opp_bench)  # (128,)

        assert opp_encodings.shape == (5, 64)
        assert opp_bench.shape == (64,)
        assert field_ctx.shape == (128,)

    def test_full_pipeline_batched(self):
        """Test the full encoding pipeline with batching."""
        # Setup
        pokemon_encoder = PokemonEncoder(pokemon_state_len=50, pokemon_hidden=64, layers=2)
        bench_encoder = BenchEncoder(pokemon_hidden=64, pooling="mean")
        field_encoder = FieldEncoder(context_len=100, bench_hidden=64, field_hidden=128, layers=2)

        batch_size = 4

        # Opponent bench encoding
        opp_pokemon_states = torch.randn(batch_size, 5, 50)  # batch_size x 5 Pokémon
        opp_encodings = pokemon_encoder(opp_pokemon_states)  # (batch_size, 5, 64)
        opp_bench = bench_encoder(opp_encodings)  # (batch_size, 64)

        # Field context
        context = torch.randn(batch_size, 100)
        field_ctx = field_encoder(context, opp_bench)  # (batch_size, 128)

        assert opp_encodings.shape == (batch_size, 5, 64)
        assert opp_bench.shape == (batch_size, 64)
        assert field_ctx.shape == (batch_size, 128)

    def test_no_nan_inf_values(self):
        """Test that no NaN or Inf values appear in forward passes."""
        pokemon_encoder = PokemonEncoder(pokemon_state_len=50, pokemon_hidden=64, layers=2)
        bench_encoder = BenchEncoder(pokemon_hidden=64, pooling="attention")
        field_encoder = FieldEncoder(context_len=100, bench_hidden=64, field_hidden=128, layers=2)

        opp_pokemon_states = torch.randn(2, 5, 50)
        opp_encodings = pokemon_encoder(opp_pokemon_states)
        assert not torch.isnan(opp_encodings).any(), "NaN in opp_encodings"
        assert not torch.isinf(opp_encodings).any(), "Inf in opp_encodings"

        opp_bench = bench_encoder(opp_encodings)
        assert not torch.isnan(opp_bench).any(), "NaN in opp_bench"
        assert not torch.isinf(opp_bench).any(), "Inf in opp_bench"

        context = torch.randn(2, 100)
        field_ctx = field_encoder(context, opp_bench)
        assert not torch.isnan(field_ctx).any(), "NaN in field_ctx"
        assert not torch.isinf(field_ctx).any(), "Inf in field_ctx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
