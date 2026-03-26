"""
FieldEncoder — contextualizes field state with opponent bench information.

Combines: arena state, active Pokémon (both), opponent active moves, and
opponent bench encoding into a single query vector for downstream attention.

Output (field_ctx) is used by both move and switch pointer networks.

Shape: (ctx_components,) → (field_hidden,)
"""

import torch

from policy.encoders.base_encoder import BaseEncoder
from policy.mlp import build_mlp


class FieldEncoder(BaseEncoder):
    """
    Encodes the battle field context into a fixed-size query vector.

    The field context combines static battle information (arena, active Pokémon,
    available moves) with learned representations (opponent bench encoding).
    This produces a unified query that both move and switch pointer networks attend over.

    Parameters
    ----------
    context_len : int
        Total length of flattened context (arena + my_active + opp_active + opp_moves).
    bench_hidden : int
        Dimension of opponent bench encoding (from BenchEncoder output).
    field_hidden : int
        Output dimension of the encoder (becomes the query dimension for pointers).
    layers : int
        Number of MLP layers (default: 2).
    """

    def __init__(
        self,
        context_len: int,
        bench_hidden: int,
        field_hidden: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.bench_hidden = bench_hidden
        self.field_hidden = field_hidden

        # Total input: context features + bench encoding
        total_input = context_len + bench_hidden

        self.mlp = build_mlp(total_input, field_hidden, field_hidden, layers=layers)

    def forward(
        self,
        context: torch.Tensor,
        bench_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode field context with opponent bench information.

        Parameters
        ----------
        context : torch.Tensor
            Shape (context_len,) or (batch_size, context_len).
            Arena, active Pokémon (both), opponent active moves flattened.
        bench_encoding : torch.Tensor
            Shape (bench_hidden,) or (batch_size, bench_hidden).
            Pooled opponent bench encoding (from BenchEncoder).

        Returns
        -------
        torch.Tensor
            Field context query with shape (field_hidden,) or (batch_size, field_hidden).
        """
        # Handle unbatched inputs
        context, context_squeezed = self._ensure_batch_dim(context, target_dims=2)
        bench_encoding, bench_squeezed = self._ensure_batch_dim(bench_encoding, target_dims=2)

        # Concatenate context and bench encoding
        combined = torch.cat([context, bench_encoding], dim=-1)  # (batch_size, context_len + bench_hidden)

        # Encode through MLP
        field_ctx = self.mlp(combined)  # (batch_size, field_hidden)

        field_ctx = self._remove_batch_dim_if_needed(field_ctx, context_squeezed)

        return field_ctx

    def _get_output_shape(self, context_shape: tuple, bench_shape: tuple) -> tuple:
        """Return expected output shape for given input shapes."""
        if len(context_shape) == 1:
            return (self.field_hidden,)
        elif len(context_shape) == 2:
            return (context_shape[0], self.field_hidden)
        return (self.field_hidden,)

    def describe(self) -> str:
        """Describe the encoder configuration."""
        return (
            f"FieldEncoder(\n"
            f"  context_len={self.context_len},\n"
            f"  bench_hidden={self.bench_hidden},\n"
            f"  field_hidden={self.field_hidden},\n"
            f")"
        )
