"""Unit Tests for Phase 12 — Master Pipeline Orchestrator."""

from pathlib import Path
import pytest
import torch

from vulnshield.pipeline import PipelineConfig, VulnShieldMasterPipeline


@pytest.mark.unit
class TestMasterPipeline:

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.model_name == "resnet18"
        assert cfg.seed == 42
        assert cfg.train_epochs == 100
        assert cfg.protection_budgets == [0.01, 0.03, 0.05, 0.10]

    def test_pipeline_initialization(self, tmp_path):
        cfg = PipelineConfig(model_name="resnet18", seed=123)
        repo_root = Path(__file__).resolve().parent.parent.parent
        pipeline = VulnShieldMasterPipeline(config=cfg, project_root=repo_root)

        assert pipeline.config.model_name == "resnet18"
        assert pipeline.config.seed == 123
        assert pipeline.repo_root == repo_root
        assert pipeline.device is not None
