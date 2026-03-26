"""
PokemonEncoder — encodes individual Pokémon state into a fixed-size vector.

Takes a PokemonState array as input and outputs a dense embedding via MLP.
Uses shared weights across all team members for permutation equivariance.

Shape: (POKEMON_STATE_LEN,) → (pokemon_hidden,)
"""

import torch

from env.encoders.base_encoder import BaseEncoder
from policy.mlp import build_mlp


class PokemonEncoder(BaseEncoder):
    """
    Encodes a single Pokémon state vector into a dense hidden representation.

    The encoder is applied independently to each team member with shared weights,
    ensuring permutation-equivariance: reordering team members doesn't change
    individual encodings, only their collection ordering.

    Parameters
    ----------
    pokemon_state_len : int
        Length of flattened PokemonState vector (from PokemonState.array_len()).
    pokemon_hidden : int
        Output dimension of the encoder.
    layers : int
        Number of MLP layers (default: 2).
    """

    def __init__(
        self,
        pokemon_state_len: int,
        pokemon_hidden: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.pokemon_state_len = pokemon_state_len
        self.pokemon_hidden = pokemon_hidden

        self.mlp = build_mlp(pokemon_state_len, pokemon_hidden, pokemon_hidden, layers=layers)

    def forward(self, pokemon_states: torch.Tensor) -> torch.Tensor:
        """
        Encode Pokémon state(s).

        Parameters
        ----------
        pokemon_states : torch.Tensor
            Shape (pokemon_state_len,) for single Pokémon or
            (batch_size, n_pokemon, pokemon_state_len) for batched/multiple Pokémon.
            Can also be (batch_size * n_pokemon, pokemon_state_len) if flattened.

        Returns
        -------
        torch.Tensor
            Encoded representation(s) with shape:
            - (pokemon_hidden,) for single input
            - (batch_size, n_pokemon, pokemon_hidden) for batched
            - (batch_size * n_pokemon, pokemon_hidden) for flattened batched
        """
        # Handle 1D input (single Pokémon)
        pokemon_states, squeeze_output = self._ensure_batch_dim(pokemon_states, target_dims=2)

        # Handle 3D input: reshape to 2D for MLP, then reshape back
        if pokemon_states.dim() == 3:
            flattened, original_shape = self._reshape_for_mlp(pokemon_states)
            output = self.mlp(flattened)
            output = self._reshape_from_mlp(output, original_shape)
            return output

        # Handle 2D input: process directly
        output = self.mlp(pokemon_states)
        output = self._remove_batch_dim_if_needed(output, squeeze_output)

        return output

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """Return expected output shape for given input shape."""
        if len(input_shape) == 1:
            return (self.pokemon_hidden,)
        elif len(input_shape) == 2:
            return (input_shape[0], self.pokemon_hidden)
        elif len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.pokemon_hidden)
        return (self.pokemon_hidden,)

    def describe(self) -> str:
        """Describe the encoder configuration."""
        return (
            f"PokemonEncoder(\n"
            f"  input_len={self.pokemon_state_len},\n"
            f"  output_dim={self.pokemon_hidden},\n"
            f")"
        )
