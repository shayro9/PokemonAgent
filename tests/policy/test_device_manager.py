"""
Tests for policy/device_manager.py — get_device().

Covers:
  1. "cpu"  → always returns "cpu"
  2. "auto" when CUDA unavailable → "cpu"
  3. "cuda" when CUDA unavailable → raises ValueError
  4. "auto" when CUDA available  → "cuda"
  5. "cuda" when CUDA available  → "cuda"
  6. Default argument is "auto"
"""

import pytest
from unittest.mock import patch

from policy.device_manager import get_device


class TestGetDeviceCpu:

    def test_cpu_always_returns_cpu(self):
        assert get_device("cpu") == "cpu"

    def test_cpu_ignores_cuda_availability(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert get_device("cpu") == "cpu"


class TestGetDeviceCudaRequested:

    def test_cuda_requested_unavailable_raises_value_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA"):
                get_device("cuda")

    def test_cuda_requested_available_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            result = get_device("cuda")
        assert result == "cuda"


class TestGetDeviceAuto:

    def test_auto_cuda_unavailable_returns_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            result = get_device("auto")
        assert result == "cpu"

    def test_auto_cuda_available_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="Tesla T4"):
            result = get_device("auto")
        assert result == "cuda"

    def test_default_argument_is_auto(self):
        """Calling with no argument should behave identically to 'auto'."""
        with patch("torch.cuda.is_available", return_value=False):
            assert get_device() == "cpu"
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", return_value="GPU"):
            assert get_device() == "cuda"
