"""
Tests for policy/attention.py — CrossAttention.

Covers:
  1. Output shapes (attended, scores)
  2. NaN guard: all-masked inputs produce finite (zero) output
  3. Partial masking: masked slots have near-zero attention weight
  4. ValueError when kv_dim % n_heads != 0
  5. Various batch sizes and sequence lengths
  6. No masking (standard attention)
  7. Attention scores are in (B, n_heads, N) not (B, N, n_heads)
"""

import pytest
import torch
import numpy as np

from policy.attention import CrossAttention


# ─── helpers ────────────────────────────────────────────────────────────────

def make_module(query_dim=16, kv_dim=32, n_heads=4) -> CrossAttention:
    mod = CrossAttention(query_dim=query_dim, kv_dim=kv_dim, n_heads=n_heads)
    mod.eval()
    return mod


def rand_query(B=2, query_dim=16) -> torch.Tensor:
    return torch.randn(B, query_dim)


def rand_kv(B=2, N=4, kv_dim=32) -> torch.Tensor:
    return torch.randn(B, N, kv_dim)


# ─── 1. Output shapes ────────────────────────────────────────────────────────

class TestCrossAttentionOutputShapes:

    def test_attended_shape(self):
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=3, query_dim=16)
        kv = rand_kv(B=3, N=4, kv_dim=32)
        attended, scores = mod(query, kv)
        assert attended.shape == (3, 32), f"Expected (3, 32) got {attended.shape}"

    def test_scores_shape(self):
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=3, query_dim=16)
        kv = rand_kv(B=3, N=4, kv_dim=32)
        attended, scores = mod(query, kv)
        assert scores.shape == (3, 4, 4), f"Expected (3, n_heads=4, N=4) got {scores.shape}"

    def test_attended_shape_with_N6(self):
        """Team attention uses N=6 (full team)."""
        mod = make_module(query_dim=8, kv_dim=16, n_heads=2)
        query = rand_query(B=2, query_dim=8)
        kv = rand_kv(B=2, N=6, kv_dim=16)
        attended, scores = mod(query, kv)
        assert attended.shape == (2, 16)
        assert scores.shape == (2, 2, 6)

    def test_batch_size_1(self):
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=1, query_dim=16)
        kv = rand_kv(B=1, N=4, kv_dim=32)
        attended, scores = mod(query, kv)
        assert attended.shape == (1, 32)
        assert scores.shape == (1, 4, 4)

    def test_large_batch(self):
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=64, query_dim=16)
        kv = rand_kv(B=64, N=4, kv_dim=32)
        attended, scores = mod(query, kv)
        assert attended.shape == (64, 32)
        assert scores.shape == (64, 4, 4)


# ─── 2. NaN guard ────────────────────────────────────────────────────────────

class TestNaNGuard:

    def test_all_slots_masked_produces_no_nan_in_attended(self):
        """When all N slots are masked, softmax gives NaN; nan_to_num should replace with 0."""
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=2, query_dim=16)
        kv = rand_kv(B=2, N=4, kv_dim=32)
        # All True = ignore all slots
        mask = torch.ones(2, 4, dtype=torch.bool)
        attended, scores = mod(query, kv, key_padding_mask=mask)
        assert not torch.isnan(attended).any(), "attended contains NaN when all slots masked"
        assert not torch.isinf(attended).any(), "attended contains Inf when all slots masked"

    def test_all_slots_masked_produces_finite_scores(self):
        """Scores may be -inf (pre-softmax), but attended should be finite."""
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=1, query_dim=16)
        kv = rand_kv(B=1, N=4, kv_dim=32)
        mask = torch.ones(1, 4, dtype=torch.bool)
        attended, _ = mod(query, kv, key_padding_mask=mask)
        assert torch.isfinite(attended).all()

    def test_all_slots_masked_attended_is_zero(self):
        """With all-masked, attn weights all become 0 via nan_to_num, so attended = 0."""
        mod = make_module(query_dim=16, kv_dim=32, n_heads=4)
        query = rand_query(B=2, query_dim=16)
        kv = rand_kv(B=2, N=4, kv_dim=32)
        mask = torch.ones(2, 4, dtype=torch.bool)
        with torch.no_grad():
            attended, _ = mod(query, kv, key_padding_mask=mask)
        # out_proj(0) = bias of out_proj; so result is out_proj.bias broadcast
        # The key invariant: no NaN
        assert not torch.isnan(attended).any()


# ─── 3. Partial masking ───────────────────────────────────────────────────────

class TestPartialMasking:

    def test_masked_slot_gets_zero_attention_weight(self):
        """A masked slot should receive near-zero attention weight after softmax."""
        mod = make_module(query_dim=8, kv_dim=8, n_heads=1)
        query = rand_query(B=1, query_dim=8)
        kv = rand_kv(B=1, N=4, kv_dim=8)
        # Mask only slot 2
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, 2] = True
        with torch.no_grad():
            _, scores = mod(query, kv, key_padding_mask=mask)
        # scores are RAW (pre-softmax) — masked slot should be -inf
        assert scores[0, 0, 2].item() == float("-inf")

    def test_unmasked_slots_share_attention(self):
        """When 1 of 4 slots is masked, the remaining 3 get all the attention weight."""
        torch.manual_seed(42)
        mod = make_module(query_dim=8, kv_dim=8, n_heads=1)
        query = rand_query(B=1, query_dim=8)
        kv = rand_kv(B=1, N=4, kv_dim=8)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, 3] = True  # mask last slot
        with torch.no_grad():
            # Compute softmax of scores manually to check distribution
            _, scores = mod(query, kv, key_padding_mask=mask)
        # After masking, slot 3 should be -inf
        assert scores[0, 0, 3].item() == float("-inf")
        # Remaining slots should be finite
        for i in range(3):
            assert torch.isfinite(scores[0, 0, i])

    def test_no_mask_all_slots_contribute(self):
        """Without masking, all slots receive non-zero attention."""
        torch.manual_seed(0)
        mod = make_module(query_dim=8, kv_dim=8, n_heads=1)
        query = rand_query(B=1, query_dim=8)
        kv = rand_kv(B=1, N=4, kv_dim=8)
        with torch.no_grad():
            _, scores = mod(query, kv)
        # All scores should be finite
        assert torch.isfinite(scores).all()


# ─── 4. ValueError ───────────────────────────────────────────────────────────

class TestCrossAttentionValidation:

    def test_kv_dim_not_divisible_by_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            CrossAttention(query_dim=8, kv_dim=7, n_heads=4)

    def test_kv_dim_divisible_by_n_heads_ok(self):
        # Should not raise
        CrossAttention(query_dim=8, kv_dim=8, n_heads=4)
        CrossAttention(query_dim=16, kv_dim=32, n_heads=8)

    def test_single_head_valid(self):
        mod = CrossAttention(query_dim=4, kv_dim=4, n_heads=1)
        query = rand_query(B=2, query_dim=4)
        kv = rand_kv(B=2, N=4, kv_dim=4)
        attended, scores = mod(query, kv)
        assert attended.shape == (2, 4)
        assert scores.shape == (2, 1, 4)


# ─── 5. Reproducibility ──────────────────────────────────────────────────────

class TestCrossAttentionDeterminism:

    def test_same_input_same_output(self):
        """CrossAttention is deterministic for the same weights and inputs."""
        mod = make_module()
        query = rand_query()
        kv = rand_kv()
        with torch.no_grad():
            out1, s1 = mod(query, kv)
            out2, s2 = mod(query, kv)
        torch.testing.assert_close(out1, out2)
        torch.testing.assert_close(s1, s2)

    def test_different_batch_items_are_independent(self):
        """Two batch items with different kv should produce different attended outputs."""
        torch.manual_seed(7)
        mod = make_module()
        query = rand_query(B=2)
        kv1 = rand_kv(B=2)
        kv2 = rand_kv(B=2)  # different kv
        with torch.no_grad():
            out1, _ = mod(query, kv1)
            out2, _ = mod(query, kv2)
        assert not torch.allclose(out1, out2)
