"""Decoupled Adversarial Attack Robustness Evaluator (FGSM & PGD-20).

Provides standardized white-box gradient-based adversarial attack evaluation:
  - FGSM: Fast Gradient Sign Method (Goodfellow et al., 2014)
  - PGD-20: Projected Gradient Descent with 20 steps (Madry et al., 2017)
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.training.losses import calculate_accuracy


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    criterion: Optional[nn.Module] = None
) -> torch.Tensor:
    """Generate FGSM adversarial perturbations: x_adv = x + eps * sign(grad_x L)."""
    crit = criterion or nn.CrossEntropyLoss()
    images_adv = images.clone().detach().requires_grad_(True)

    logits = model(images_adv)
    loss = crit(logits, labels)
    loss.backward()

    with torch.no_grad():
        grad_sign = images_adv.grad.data.sign()
        images_adv = images + epsilon * grad_sign

    return images_adv.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 20,
    random_start: bool = True,
    criterion: Optional[nn.Module] = None
) -> torch.Tensor:
    """Generate PGD-k adversarial perturbations with projected gradient descent."""
    crit = criterion or nn.CrossEntropyLoss()
    images_orig = images.clone().detach()

    if random_start:
        images_adv = images_orig + torch.FloatTensor(*images.shape).uniform_(-epsilon, epsilon).to(images.device)
    else:
        images_adv = images_orig.clone()

    for _ in range(steps):
        images_adv.requires_grad_(True)
        logits = model(images_adv)
        loss = crit(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad_sign = images_adv.grad.data.sign()
            images_adv = images_adv + alpha * grad_sign
            # Project back onto epsilon L_inf ball
            delta = torch.clamp(images_adv - images_orig, min=-epsilon, max=epsilon)
            images_adv = images_orig + delta

    return images_adv.detach()


def evaluate_adversarial_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    fgsm_epsilon: float = 8.0 / 255.0,
    pgd_epsilon: float = 8.0 / 255.0,
    pgd_alpha: float = 2.0 / 255.0,
    pgd_steps: int = 20,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """Evaluate model accuracy under both FGSM and PGD-20 adversarial attacks.

    Returns:
        Tuple of (fgsm_accuracy, pgd_accuracy).
    """
    dev = device or torch.device("cpu")
    model.eval()
    model.to(dev)

    fgsm_correct, pgd_correct, total_samples = 0, 0, 0

    for images, targets in tqdm(dataloader, desc="Adversarial Robustness Evaluation", leave=False):
        images = images.to(dev)
        targets = targets.to(dev)
        batch_size = targets.size(0)
        total_samples += batch_size

        # 1. FGSM
        adv_fgsm = fgsm_attack(model, images, targets, epsilon=fgsm_epsilon)
        with torch.no_grad():
            preds_fgsm = model(adv_fgsm).argmax(dim=1)
            fgsm_correct += (preds_fgsm == targets).sum().item()

        # 2. PGD-20
        adv_pgd = pgd_attack(model, images, targets, epsilon=pgd_epsilon, alpha=pgd_alpha, steps=pgd_steps)
        with torch.no_grad():
            preds_pgd = model(adv_pgd).argmax(dim=1)
            pgd_correct += (preds_pgd == targets).sum().item()

    n = max(total_samples, 1)
    return (fgsm_correct / n) * 100.0, (pgd_correct / n) * 100.0
