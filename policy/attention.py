"""
Single-query multi-head cross-attention.

The battle context (query) attends over the 4 move embeddings (keys/values),
producing an order-invariant summary of the available moves.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Shapes
    ------
    query  : (B, query_dim)
    kv     : (B, N, kv_dim)       N is typically 4 (move slots)
    output : (B, kv_dim),  (B, n_heads, N)
                 attended        raw attention scores (reused by pointer head)
    """

    def __init__(self, query_dim: int, kv_dim: int, n_heads: int) -> None:
        super().__init__()
        if kv_dim % n_heads != 0:
            raise ValueError(
                f"kv_dim ({kv_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = kv_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, kv_dim)
        self.k_proj = nn.Linear(kv_dim, kv_dim)
        self.v_proj = nn.Linear(kv_dim, kv_dim)
        self.out_proj = nn.Linear(kv_dim, kv_dim)

    def forward(
        self,
        query: torch.Tensor,                               # (B, query_dim)
        kv: torch.Tensor,                                  # (B, N, kv_dim)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, N) bool — True=ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = kv.shape
        H, D = self.n_heads, self.head_dim

        # reshape instead of view — safe even if tensors are non-contiguous
        q = self.q_proj(query).reshape(B, H, D)            # (B, H, D)
        k = self.k_proj(kv).reshape(B, N, H, D)            # (B, N, H, D)
        v = self.v_proj(kv).reshape(B, N, H, D)            # (B, N, H, D)

        scores = torch.einsum("bhd,bnhd->bhn", q, k) * self.scale  # (B, H, N)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1), float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)                # (B, H, N)

        # Guard against NaN: if every slot is masked, softmax gives NaN → replace with 0
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.einsum("bhn,bnhd->bhd", attn, v).reshape(B, H * D)
        out = self.out_proj(out)                            # (B, kv_dim)

        return out, scores                                  # attended, raw scores

