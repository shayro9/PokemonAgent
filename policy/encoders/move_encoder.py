"""
MoveEncoder — encodes individual move state into a fixed-size vector.

Takes a MoveState array as input and outputs a dense embedding via MLP.
Uses shared weights across all move slots for permutation equivariance.

Shape: (MOVE_STATE_LEN,) → (move_hidden,)
"""

import torch

from policy.encoders.base_encoder import BaseEncoder
from policy.mlp import build_mlp


class MoveEncoder(BaseEncoder):
    """
    Encodes a single Move state vector into a dense hidden representation.

    The encoder is applied independently to each available move with shared weights,
    ensuring permutation-equivariance: reordering moves doesn't change individual
    encodings, only their collection ordering.

    Parameters
    ----------
    move_state_len : int
        Length of flattened MoveState vector (from MoveState.array_len()).
    move_hidden : int
        Output dimension of the encoder.
    layers : int
        Number of MLP layers (default: 2).
    """

    def __init__(
        self,
        move_state_len: int,
        move_hidden: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.move_state_len = move_state_len
        self.move_hidden = move_hidden

        self.mlp = build_mlp(move_state_len, move_hidden, move_hidden, layers=layers)

    def forward(self, move_states: torch.Tensor) -> torch.Tensor:
        """
        Encode move state(s).

        Parameters
        ----------
        move_states : torch.Tensor
            Shape (move_state_len,) for single move or
            (batch_size, n_moves, move_state_len) for batched/multiple moves.
            Can also be (batch_size * n_moves, move_state_len) if flattened.

        Returns
        -------
        torch.Tensor
            Encoded representation(s) with shape:
            - (move_hidden,) for single input
            - (batch_size, n_moves, move_hidden) for batched
            - (batch_size * n_moves, move_hidden) for flattened batched
        """
        # Handle 1D input (single move)
        move_states, squeeze_output = self._ensure_batch_dim(move_states, target_dims=2)

        # Handle 3D input: reshape to 2D for MLP, then reshape back
        if move_states.dim() == 3:
            flattened, original_shape = self._reshape_for_mlp(move_states)
            output = self.mlp(flattened)
            output = self._reshape_from_mlp(output, original_shape)
            return output

        # Handle 2D input: process directly
        output = self.mlp(move_states)
        output = self._remove_batch_dim_if_needed(output, squeeze_output)

        return output

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """Return expected output shape for given input shape."""
        if len(input_shape) == 1:
            return (self.move_hidden,)
        elif len(input_shape) == 2:
            return (input_shape[0], self.move_hidden)
        elif len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.move_hidden)
        return (self.move_hidden,)

    def describe(self) -> str:
        """Describe the encoder configuration."""
        return (
            f"MoveEncoder(\n"
            f"  input_len={self.move_state_len},\n"
            f"  output_dim={self.move_hidden},\n"
            f")"
        )
