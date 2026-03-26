"""
BenchEncoder — aggregates opponent's bench Pokémon into a single pooled vector.

Input: list of encoded Pokémon from the opponent's team (excluding active)
Output: single pooled vector (enc_bnch) representing opponent bench composition

Supports multiple pooling strategies: mean, max, attention-based weighted average.
"""

from typing import Optional

import torch
import torch.nn as nn

from env.encoders.base_encoder import BaseEncoder


class BenchEncoder(BaseEncoder):
    """
    Aggregates multiple encoded Pokémon into a single representation.

    Uses permutation-invariant pooling (mean or max) to summarize the opponent's
    bench composition. This ensures that team ordering doesn't affect the output,
    only the set of available Pokémon.

    Parameters
    ----------
    pokemon_hidden : int
        Expected input dimension (output of PokemonEncoder).
    pooling : str
        Aggregation strategy: 'mean', 'max', or 'attention' (default: 'mean').
    attention_dim : int
        If pooling='attention', dimension of attention score MLP.
        If None, uses pokemon_hidden as attention_dim.
    """

    def __init__(
        self,
        pokemon_hidden: int,
        pooling: str = "mean",
        attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pokemon_hidden = pokemon_hidden
        self.pooling = pooling

        if pooling == "attention":
            if attention_dim is None:
                attention_dim = pokemon_hidden
            # Simple attention: learned projection to scalar scores
            self.attention_mlp = nn.Sequential(
                nn.Linear(pokemon_hidden, attention_dim),
                nn.ReLU(),
                nn.Linear(attention_dim, 1),
            )
        elif pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be 'mean', 'max', or 'attention', got {pooling}")

    def forward(
        self,
        encoded_pokemon: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool encoded Pokémon into a single representation.

        Parameters
        ----------
        encoded_pokemon : torch.Tensor
            Shape (batch_size, n_pokemon, pokemon_hidden) or (n_pokemon, pokemon_hidden).
            Encoded Pokémon vectors (output of PokemonEncoder applied to each team member).
        mask : torch.Tensor, optional
            Shape (batch_size, n_pokemon) or (n_pokemon,) with dtype bool.
            True indicates valid Pokémon, False indicates padding/empty slots.
            If None, all Pokémon are considered valid.

        Returns
        -------
        torch.Tensor
            Pooled representation with shape (batch_size, pokemon_hidden) or (pokemon_hidden,).
        """
        # Handle 2D input (unbatched)
        if encoded_pokemon.dim() == 2:
            encoded_pokemon = encoded_pokemon.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, n_pokemon, _ = encoded_pokemon.shape

        if mask is None:
            mask = torch.ones(batch_size, n_pokemon, dtype=torch.bool, device=encoded_pokemon.device)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

        # Apply pooling strategy
        if self.pooling == "mean":
            pooled = self._pool_mean(encoded_pokemon, mask)
        elif self.pooling == "max":
            pooled = self._pool_max(encoded_pokemon, mask)
        else:  # attention
            pooled = self._pool_attention(encoded_pokemon, mask)

        if squeeze_output:
            pooled = pooled.squeeze(0)

        return pooled

    def _pool_mean(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with masking."""
        # Shape: (batch_size, n_pokemon, pokemon_hidden)
        # Mask: (batch_size, n_pokemon)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, n_pokemon, 1)
        masked_sum = (encoded * mask_expanded).sum(dim=1)  # (batch_size, pokemon_hidden)
        count = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch_size, 1)
        return masked_sum / count

    def _pool_max(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max pooling with masking."""
        # Replace masked-out positions with very negative values before max
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, n_pokemon, 1)
        masked = encoded.clone()
        masked[~mask.unsqueeze(-1).expand_as(encoded)] = float("-inf")
        pooled, _ = masked.max(dim=1)  # (batch_size, pokemon_hidden)
        # If all masked, replace -inf with 0
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        return pooled

    def _pool_attention(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-based weighted pooling."""
        # Compute attention scores for each Pokémon
        scores = self.attention_mlp(encoded)  # (batch_size, n_pokemon, 1)
        scores = scores.squeeze(-1)  # (batch_size, n_pokemon)

        # Mask out invalid Pokémon
        scores[~mask] = float("-inf")

        # Softmax with NaN guard (if all masked)
        attn_weights = torch.softmax(scores, dim=1)  # (batch_size, n_pokemon)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum
        pooled = torch.einsum("bn,bnd->bd", attn_weights, encoded)  # (batch_size, pokemon_hidden)
        return pooled

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """Return expected output shape for given input shape."""
        if len(input_shape) == 2:
            return (input_shape[0],)  # Pool over dimension 1
        elif len(input_shape) == 3:
            return (input_shape[0], input_shape[2])  # Pool over dimension 1
        return (self.pokemon_hidden,)

    def describe(self) -> str:
        """Describe the encoder configuration."""
        return (
            f"BenchEncoder(\n"
            f"  pokemon_hidden={self.pokemon_hidden},\n"
            f"  pooling={self.pooling},\n"
            f")"
        )
