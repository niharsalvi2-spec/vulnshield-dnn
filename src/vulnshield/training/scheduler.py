"""Learning Rate Scheduler Factory and Builder."""

from __future__ import annotations

from typing import List, Optional
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from vulnshield.core.exceptions import ConfigurationError


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    epochs: int = 100,
    eta_min: float = 1e-5,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    **kwargs
) -> lr_scheduler._LRScheduler:
    """Instantiate a learning rate scheduler for the training loop.

    Args:
        optimizer: PyTorch optimizer instance.
        name: Name of scheduler ('cosine', 'multistep', 'step', 'none').
        epochs: Total number of training epochs (T_max for cosine annealing).
        eta_min: Minimum learning rate for cosine annealing.
        milestones: Epoch milestones for MultiStepLR.
        gamma: Multiplicative factor of learning rate decay.

    Returns:
        torch.optim.lr_scheduler instance.
    """
    sched_name = name.lower().strip()

    if sched_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min
        )
    elif sched_name == "multistep":
        ms = milestones if milestones is not None else [int(epochs * 0.5), int(epochs * 0.75)]
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=ms,
            gamma=gamma
        )
    elif sched_name == "step":
        step_size = kwargs.get("step_size", 30)
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif sched_name in ["none", "constant"]:
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ConfigurationError(f"Unsupported scheduler: '{name}'. Supported: ['cosine', 'multistep', 'step', 'none']")
