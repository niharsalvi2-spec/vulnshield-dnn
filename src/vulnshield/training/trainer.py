"""Base Model Training Engine with Metrics Tracking and Checkpoint Management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.models.model_factory import save_checkpoint
from vulnshield.training.losses import calculate_accuracy
from vulnshield.training.evaluator import evaluate_model, EvaluationResult
from vulnshield.training.optimizer import build_optimizer
from vulnshield.training.scheduler import build_scheduler
from vulnshield.utils.device import get_device


@dataclass
class TrainerConfig:
    """Configuration hyperparameters for base model training."""
    epochs: int = 100
    learning_rate: float = 0.1
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler_name: str = "cosine"
    eta_min: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    checkpoint_dir: str = "checkpoints/base_models"
    save_best_only: bool = False
    log_interval: int = 50


class BaseTrainer:
    """Orchestrates standard clean model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.config = config or TrainerConfig()
        self.criterion = criterion or nn.CrossEntropyLoss()

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
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }
        self.best_val_acc = 0.0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Execute one complete training epoch over the DataLoader."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]", leave=False)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            acc = calculate_accuracy(logits, targets)
            total_correct += (acc / 100.0) * batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        return avg_loss, avg_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Union[str, Path],
        checkpoint_name: str = "base_model"
    ) -> Dict[str, Any]:
        """Run the complete training loop across all configured epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            checkpoint_dir: Directory where checkpoints will be saved.
            checkpoint_name: Base filename prefix for checkpoints.

        Returns:
            Dictionary containing best validation accuracy and complete metrics history.
        """
        save_dir = Path(checkpoint_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = save_dir / f"{checkpoint_name}_best.pt"
        last_ckpt_path = save_dir / f"{checkpoint_name}_last.pt"

        print(f"[*] Starting base model training for {self.config.epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_result = evaluate_model(self.model, val_loader, self.criterion, self.device)
            val_loss, val_acc = val_result.loss, val_result.accuracy

            curr_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(curr_lr)

            is_best = val_acc > self.best_val_acc
            status = "NEW BEST" if is_best else ""
            print(f"Epoch {epoch:3d}/{self.config.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {curr_lr:.5f} {status}")

            if is_best:
                self.best_val_acc = val_acc
                save_checkpoint(
                    model=self.model,
                    path=best_ckpt_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={"val_loss": val_loss, "val_acc": val_acc, "train_acc": train_acc}
                )

        # Always save final checkpoint
        save_checkpoint(
            model=self.model,
            path=last_ckpt_path,
            optimizer=self.optimizer,
            epoch=self.config.epochs,
            metrics={"val_loss": val_loss, "val_acc": val_acc, "train_acc": train_acc}
        )

        total_time = time.time() - start_time
        print(f"[PASS] Training completed in {total_time:.2f}s. Best Val Accuracy = {self.best_val_acc:.2f}%")
        print(f"  - Best Checkpoint: {best_ckpt_path}")
        print(f"  - Last Checkpoint: {last_ckpt_path}")

        return {
            "best_val_acc": self.best_val_acc,
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "total_duration_sec": total_time,
            "history": self.history
        }
