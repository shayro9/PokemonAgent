"""
Device configuration for training.
Simple device selection and validation.
"""

from policy.device_manager import get_device, print_device_info


class DeviceConfig:
    """Container for device settings."""
    
    def __init__(self, device: str = "auto"):
        """
        Args:
            device: "auto" (detect GPU), "cuda", or "cpu"
        """
        self.device_str = get_device(device)

    def __str__(self) -> str:
        return self.device_str
    
    def print_info(self) -> None:
        """Print device information."""
        print_device_info(self.device_str)
