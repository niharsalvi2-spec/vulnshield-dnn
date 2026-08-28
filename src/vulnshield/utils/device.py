"""Device Management and GPU Acceleration Utilities for VulnShield-DNN."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Select the optimal compute device (CUDA GPU or CPU).

    Args:
        preferred: Optional device name string ('cuda', 'cuda:0', 'cpu', 'auto').

    Returns:
        torch.device instance.
    """
    if preferred in [None, "auto"]:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    dev_str = preferred.lower().strip()
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(dev_str)
    elif dev_str == "cpu":
        return torch.device("cpu")
    
    return torch.device(dev_str)


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Retrieve detailed hardware and memory metrics for the target device."""
    if device is None:
        device = get_device()

    info: Dict[str, Any] = {
        "device_type": device.type,
        "device_index": device.index,
        "is_cuda": device.type == "cuda"
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "allocated_memory_gb": round(torch.cuda.memory_allocated(idx) / (1024 ** 3), 4),
            "reserved_memory_gb": round(torch.cuda.memory_reserved(idx) / (1024 ** 3), 4),
            "cuda_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count
        })
    else:
        info.update({
            "name": "CPU",
            "total_memory_gb": "System RAM",
        })

    return info


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors, dicts, lists, or tuples to the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data
