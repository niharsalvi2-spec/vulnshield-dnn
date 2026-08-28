"""Reproducibility and Random Seed Management for VulnShield-DNN.

Guarantees full deterministic execution across Python, NumPy, PyTorch, and CUDA.
"""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Integer seed value.
        deterministic: If True, configures PyTorch/cuDNN for strict determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Create a seeded PyTorch Generator for DataLoader shuffling."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialization function for reproducible multi-process loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
