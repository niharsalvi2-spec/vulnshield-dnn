"""Unit Tests for VulnShield-DNN Training Layer."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.training.losses import (
    ClassificationLoss,
    calculate_accuracy,
    calculate_topk_accuracy
)
from vulnshield.training.optimizer import build_optimizer
from vulnshield.training.scheduler import build_scheduler
from vulnshield.training.evaluator import evaluate_model, EvaluationResult
from vulnshield.training.trainer import BaseTrainer, TrainerConfig
from vulnshield.core.exceptions import ConfigurationError


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_tiny_model() -> nn.Module:
    """3-layer tiny net for fast testing (no GPU needed)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )


def _make_dataloader(n_samples: int = 64, batch_size: int = 16) -> DataLoader:
    images = torch.randn(n_samples, 3, 8, 8)
    labels = torch.randint(0, 10, (n_samples,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ── Loss & Accuracy ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLossAndAccuracy:

    def test_classification_loss_forward(self):
        criterion = ClassificationLoss()
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        loss = criterion(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_calculate_accuracy_perfect(self):
        logits = torch.eye(5)          # argmax(i) == i
        targets = torch.arange(5)
        acc = calculate_accuracy(logits, targets)
        assert abs(acc - 100.0) < 1e-5

    def test_calculate_accuracy_zero(self):
        logits = torch.zeros(4, 10)
        logits[:, 0] = 1.0             # always predict class 0
        targets = torch.ones(4, dtype=torch.long) * 9   # all class 9
        acc = calculate_accuracy(logits, targets)
        assert acc == 0.0

    def test_calculate_topk_accuracy(self):
        logits = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        top1, top5 = calculate_topk_accuracy(logits, targets, topk=(1, 5))
        assert 0.0 <= top1 <= 100.0
        assert 0.0 <= top5 <= 100.0
        assert top5 >= top1


# ── Optimizer ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestOptimizerBuilder:

    def test_sgd_optimizer(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="sgd", lr=0.1)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["lr"] == 0.1

    def test_adam_optimizer(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw_optimizer(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="adamw", lr=3e-4)
        assert isinstance(opt, torch.optim.AdamW)

    def test_unsupported_optimizer_raises(self):
        model = _make_tiny_model()
        with pytest.raises(ConfigurationError):
            build_optimizer(model, name="rmsprop")


# ── Scheduler ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSchedulerBuilder:

    def test_cosine_scheduler(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="sgd", lr=0.1)
        sched = build_scheduler(opt, name="cosine", epochs=100)
        sched.step()
        new_lr = opt.param_groups[0]["lr"]
        # After 1 cosine step LR should have decremented slightly
        assert new_lr <= 0.1

    def test_multistep_scheduler(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="sgd", lr=0.1)
        sched = build_scheduler(opt, name="multistep", epochs=100, milestones=[5, 10], gamma=0.1)
        assert sched is not None

    def test_unsupported_scheduler_raises(self):
        model = _make_tiny_model()
        opt = build_optimizer(model, name="sgd", lr=0.1)
        with pytest.raises(ConfigurationError):
            build_scheduler(opt, name="cyclical_unknown")


# ── Evaluator ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestEvaluator:

    def test_evaluate_model_returns_result(self):
        model = _make_tiny_model()
        loader = _make_dataloader(n_samples=32, batch_size=16)
        result = evaluate_model(model, loader, device=torch.device("cpu"))
        assert isinstance(result, EvaluationResult)
        assert result.num_samples == 32
        assert 0.0 <= result.accuracy <= 100.0
        assert torch.isfinite(torch.tensor(result.loss))

    def test_evaluation_result_to_dict(self):
        model = _make_tiny_model()
        loader = _make_dataloader(n_samples=16, batch_size=16)
        result = evaluate_model(model, loader, device=torch.device("cpu"))
        d = result.to_dict()
        assert "loss" in d and "accuracy" in d and "top5_accuracy" in d
        assert d["num_samples"] == 16


# ── Trainer ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestBaseTrainer:

    def test_trainer_runs_single_epoch(self, tmp_path):
        model = _make_tiny_model()
        train_loader = _make_dataloader(n_samples=32, batch_size=16)
        val_loader = _make_dataloader(n_samples=16, batch_size=16)

        cfg = TrainerConfig(
            epochs=2,
            learning_rate=0.01,
            optimizer_name="adam",
            scheduler_name="cosine"
        )
        trainer = BaseTrainer(model=model, config=cfg, device=torch.device("cpu"))
        results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=tmp_path / "ckpts",
            checkpoint_name="test_model"
        )

        assert "best_val_acc" in results
        assert 0.0 <= results["best_val_acc"] <= 100.0
        assert len(trainer.history["train_loss"]) == 2
        assert len(trainer.history["val_acc"]) == 2
