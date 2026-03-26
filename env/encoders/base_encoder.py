"""
BaseEncoder — abstract base class for all encoder modules.

Defines the common interface and shared functionality for feature encoders.
All encoders should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoder modules.

    Provides:
    - Consistent interface for all encoders (forward, describe, etc.)
    - Shared utility methods for shape handling
    - Input validation and type checking

    Subclasses must implement:
    - forward(): Main forward pass logic
    - _get_output_shape(): Return expected output shape(s)
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass logic.

        Must be implemented by subclasses.

        Returns
        -------
        torch.Tensor
            Encoded representation.
        """
        pass

    @abstractmethod
    def _get_output_shape(self, *input_shapes) -> tuple:
        """
        Return expected output shape(s) for given input(s).

        Used for documentation and shape validation.

        Returns
        -------
        tuple
            Expected output shape (can be nested tuple for multiple outputs).
        """
        pass

    def describe(self) -> str:
        """
        Human-readable description of the encoder configuration.

        Default implementation; override in subclasses for custom descriptions.
        """
        return f"{self.__class__.__name__}()"

    # ── Shared Shape Handling ──────────────────────────────────────────

    @staticmethod
    def _ensure_batch_dim(tensor: torch.Tensor, target_dims: int = 2) -> tuple[torch.Tensor, bool]:
        """
        Ensure tensor has batch dimension, tracking if unsqueezed.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.
        target_dims : int
            Number of dimensions we're targeting (e.g., 2 for (batch, features)).

        Returns
        -------
        tuple[torch.Tensor, bool]
            (tensor_with_batch, was_squeezed) - tensor reshaped to have batch dim,
            and flag indicating if we added a batch dimension.
        """
        if tensor.dim() == target_dims - 1:
            # Add batch dimension
            return tensor.unsqueeze(0), True
        return tensor, False

    @staticmethod
    def _remove_batch_dim_if_needed(tensor: torch.Tensor, was_squeezed: bool) -> torch.Tensor:
        """
        Remove batch dimension if it was added by _ensure_batch_dim.

        Parameters
        ----------
        tensor : torch.Tensor
            Output tensor.
        was_squeezed : bool
            Flag from _ensure_batch_dim indicating if we added a dimension.

        Returns
        -------
        torch.Tensor
            Tensor with batch dimension removed (if was_squeezed=True).
        """
        if was_squeezed:
            return tensor.squeeze(0)
        return tensor

    @staticmethod
    def _reshape_for_mlp(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """
        Reshape 3D tensor to 2D for MLP processing, returning original shape.

        For processing collections (e.g., multiple Pokémon) through an MLP,
        we need to flatten the batch and collection dimensions, then reshape back.

        Parameters
        ----------
        tensor : torch.Tensor
            3D tensor of shape (batch, n_items, features).

        Returns
        -------
        tuple[torch.Tensor, tuple[int, ...]]
            (flattened_2d_tensor, original_shape) - flattened to (batch*n_items, features),
            and the original shape for reconstruction.
        """
        original_shape = tensor.shape
        batch_size, n_items, features = original_shape
        flattened = tensor.reshape(batch_size * n_items, features)
        return flattened, original_shape

    @staticmethod
    def _reshape_from_mlp(tensor: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
        """
        Reshape 2D MLP output back to 3D.

        Parameters
        ----------
        tensor : torch.Tensor
            2D output from MLP of shape (batch*n_items, features).
        original_shape : tuple[int, ...]
            Original 3D shape (batch, n_items, original_features) - we use batch and n_items.

        Returns
        -------
        torch.Tensor
            Reshaped to 3D: (batch, n_items, features).
        """
        batch_size, n_items = original_shape[0], original_shape[1]
        output_features = tensor.shape[-1]
        return tensor.reshape(batch_size, n_items, output_features)

    # ── Validation ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_no_nan_inf(tensor: torch.Tensor, name: str = "output") -> None:
        """
        Validate that tensor contains no NaN or Inf values.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to check.
        name : str
            Name for error message context.

        Raises
        ------
        RuntimeError
            If NaN or Inf values are detected.
        """
        if torch.isnan(tensor).any():
            raise RuntimeError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise RuntimeError(f"{name} contains Inf values")

    @staticmethod
    def _validate_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor") -> None:
        """
        Validate tensor shape matches expected shape.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to check.
        expected_shape : tuple
            Expected shape (can use -1 for variable dimensions).
        name : str
            Name for error message context.

        Raises
        ------
        AssertionError
            If shape doesn't match (ignoring -1 wildcards).
        """
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise AssertionError(
                    f"{name} dimension {i}: expected {expected}, got {actual} "
                    f"(full shape: {tensor.shape})"
                )

    # ── Device Management ──────────────────────────────────────────────

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to same device as first encoder parameter."""
        if len(list(self.parameters())) > 0:
            device = next(self.parameters()).device
            return tensor.to(device)
        return tensor
