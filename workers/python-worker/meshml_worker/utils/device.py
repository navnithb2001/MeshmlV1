"""
Device detection and management

Automatically detects and selects the best available device:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["auto", "cuda", "cpu", "mps"]


def get_device(device_type: DeviceType = "auto") -> str:
    """Get the training device

    Args:
        device_type: Device type (auto, cuda, cpu, mps)

    Returns:
        Device string (e.g., "cuda:0", "mps", "cpu")
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, falling back to CPU")
        return "cpu"

    if device_type == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Auto-detected Apple Silicon MPS device")
        else:
            device = "cpu"
            logger.info("Auto-detected CPU device")
    elif device_type == "cuda":
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
    elif device_type == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon MPS device")
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        logger.info("Using CPU device")

    return device


def get_device_info() -> dict:
    """Get information about available devices

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
        "cpu_count": 1,
    }

    try:
        import multiprocessing

        import torch

        # CUDA info
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i),
                }
                for i in range(info["cuda_device_count"])
            ]

        # MPS info
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # CPU info
        info["cpu_count"] = multiprocessing.cpu_count()

    except ImportError:
        pass

    return info


def set_device_memory_fraction(device: str, fraction: float = 0.9) -> None:
    """Set maximum memory fraction for CUDA device

    Args:
        device: Device string (e.g., "cuda:0")
        fraction: Memory fraction to use (0.0 to 1.0)
    """
    if not device.startswith("cuda"):
        return

    try:
        import torch

        device_id = int(device.split(":")[-1])
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        logger.info(f"Set CUDA memory fraction to {fraction * 100:.0f}%")
    except Exception as e:
        logger.warning(f"Failed to set CUDA memory fraction: {e}")


def clear_device_cache(device: str) -> None:
    """Clear device cache

    Args:
        device: Device string
    """
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
    elif device == "mps":
        try:
            import torch

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")
