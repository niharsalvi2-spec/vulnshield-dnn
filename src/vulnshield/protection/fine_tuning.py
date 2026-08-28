"""Fault-Aware Fine-Tuning Trainer for Robust Model Hardening."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.fault_injection.fault_injector import FaultInjector, FaultSpec
from vulnshield.models.model_factory import save_checkpoint
from vulnshield.training.losses import calculate_accuracy
from vulnshield.training.evaluator import evaluate_model, EvaluationResult
from vulnshield.training.optimizer import build_optimizer
from vulnshield.training.scheduler import build_scheduler
from vulnshield.protection.losses import FaultAwareLoss
from vulnshield.protection.regularizer import WeightDriftRegularizer
from vulnshield.utils.device import get_device


@dataclass
class ProtectionTrainingConfig:
    """Hyperparameters for fault-aware fine-tuning."""
    epochs: int = 30
    learning_rate: float = 0.01
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler_name: str = "cosine"
    eta_min: float = 1e-5
    alpha: float = 0.5
    beta: float = 0.5
    lambda_drift: float = 1e-4
    faults_per_step: int = 1
    checkpoint_dir: str = "checkpoints/protected"


class FaultAwareTrainer:
    """Hardens a pre-trained model against channel faults via fault-aware fine-tuning."""

    def __init__(
        self,
        model: nn.Module,
        protected_channels: Sequence[FaultSpec],
        config: Optional[ProtectionTrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        if not protected_channels:
            raise ValueError("protected_channels list must be non-empty.")

        self.device = device or get_device()
        self.model = model.to(self.device)
        self.protected_channels = list(protected_channels)
        self.config = config or ProtectionTrainingConfig()

        self.injector = FaultInjector(self.model)
        self.criterion = FaultAwareLoss(alpha=self.config.alpha, beta=self.config.beta)
        self.regularizer = WeightDriftRegularizer(self.model, lambda_drift=self.config.lambda_drift)

        self.optimizer = build_optimizer(
            self.model,
            name=self.config.optimizer_name,
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = build_scheduler(
            self.optimizer,
            name=self.config.scheduler_name,
            epochs=self.config.epochs,
            eta_min=self.config.eta_min
        )

        self.history: Dict[str, List[float]] = {
            "clean_loss": [],
            "fault_loss": [],
            "total_loss": [],
            "val_clean_acc": [],
            "val_fault_acc": [],
            "lr": []
        }
        self.best_score = 0.0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """Train one epoch with alternating clean and faulted forward passes."""
        self.model.train()
        tot_clean_loss, tot_fault_loss, tot_loss = 0.0, 0.0, 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs} [Fault-Aware]", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 1. Clean forward pass
            clean_logits = self.model(images)

            # 2. Sample random protected channel(s)
            sampled_faults = random.sample(
                self.protected_channels,
                k=min(self.config.faults_per_step, len(self.protected_channels))
            )

            # 3. Faulted forward pass
            with self.injector.inject(sampled_faults):
                fault_logits = self.model(images)

            # 4. Composite loss + drift penalty
            loss_total, l_clean, l_fault = self.criterion(clean_logits, fault_logits, targets)
            loss_total = loss_total + self.regularizer.compute_penalty(self.model)

            loss_total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            batch_size = targets.size(0)
            tot_clean_loss += l_clean.item() * batch_size
            tot_fault_loss += l_fault.item() * batch_size
            tot_loss += loss_total.item() * batch_size
            total_samples += batch_size

            pbar.set_postfix({
                "clean_l": f"{l_clean.item():.3f}",
                "fault_l": f"{l_fault.item():.3f}"
            })

        n = max(total_samples, 1)
        return tot_clean_loss / n, tot_fault_loss / n, tot_loss / n

    def evaluate_protected_fault_accuracy(
        self,
        dataloader: DataLoader,
        num_eval_channels: int = 10
    ) -> float:
        """Evaluate average model accuracy under faults across protected channels."""
        self.model.eval()
        sample_channels = self.protected_channels[:num_eval_channels]
        accuracies = []

        for ch in sample_channels:
            with self.injector.inject([ch]):
                res = evaluate_model(self.model, dataloader, device=self.device)
                accuracies.append(res.accuracy)

        return sum(accuracies) / max(len(accuracies), 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eval_fault_loader: DataLoader,
        checkpoint_dir: Union[str, Path],
        checkpoint_name: str = "protected_model"
    ) -> Dict[str, Any]:
        """Execute complete fault-aware fine-tuning loop."""
        save_dir = Path(checkpoint_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = save_dir / f"{checkpoint_name}_best.pt"
        last_ckpt = save_dir / f"{checkpoint_name}_last.pt"

        print(f"[*] Starting Fault-Aware Fine-Tuning ({self.config.epochs} epochs, {len(self.protected_channels)} protected channels)...")
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            clean_l, fault_l, tot_l = self.train_epoch(train_loader, epoch)
            val_clean = evaluate_model(self.model, val_loader, device=self.device)
            val_fault_acc = self.evaluate_protected_fault_accuracy(eval_fault_loader)

            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Track
            self.history["clean_loss"].append(clean_l)
            self.history["fault_loss"].append(fault_l)
            self.history["total_loss"].append(tot_l)
            self.history["val_clean_acc"].append(val_clean.accuracy)
            self.history["val_fault_acc"].append(val_fault_acc)
            self.history["lr"].append(curr_lr)

            # Combined score: clean accuracy + fault accuracy
            combined_score = 0.5 * val_clean.accuracy + 0.5 * val_fault_acc
            is_best = combined_score > self.best_score
            status = "NEW BEST" if is_best else ""

            print(f"Epoch {epoch:2d}/{self.config.epochs} | Clean Acc: {val_clean.accuracy:.2f}% | Fault Acc: {val_fault_acc:.2f}% | LR: {curr_lr:.5f} {status}")

            if is_best:
                self.best_score = combined_score
                save_checkpoint(
                    model=self.model,
                    path=best_ckpt,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={
                        "val_clean_acc": val_clean.accuracy,
                        "val_fault_acc": val_fault_acc,
                        "combined_score": combined_score
                    }
                )

        save_checkpoint(
            model=self.model,
            path=last_ckpt,
            optimizer=self.optimizer,
            epoch=self.config.epochs,
            metrics={
                "val_clean_acc": val_clean.accuracy,
                "val_fault_acc": val_fault_acc
            }
        )

        total_time = time.time() - start_time
        print(f"[PASS] Fine-tuning finished in {total_time:.2f}s.")
        print(f"  - Best Checkpoint: {best_ckpt}")
        print(f"  - Last Checkpoint: {last_ckpt}")

        return {
            "best_combined_score": self.best_score,
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
            "total_duration_sec": total_time,
            "history": self.history
        }
