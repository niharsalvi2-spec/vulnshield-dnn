"""Unit tests for VulnShield-DNN configuration, reproducibility, and device utilities."""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from vulnshield.core.exceptions import ConfigurationError
from vulnshield.utils.config import (
    ConfigDict,
    load_yaml,
    save_yaml,
    merge_configs,
    resolve_project_paths
)
from vulnshield.utils.reproducibility import set_seed, get_generator
from vulnshield.utils.device import get_device, get_device_info, to_device


@pytest.mark.unit
class TestConfigDict:
    """Test ConfigDict dot notation and nested retrieval."""

    def test_attribute_access(self):
        cfg = ConfigDict({"a": 1, "b": {"c": 2}})
        assert cfg.a == 1
        assert cfg.b.c == 2
        assert cfg.get_nested("b.c") == 2
        assert cfg.get_nested("b.d", default=99) == 99

    def test_missing_attribute_raises_error(self):
        cfg = ConfigDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = cfg.non_existent_key

    def test_attribute_assignment(self):
        cfg = ConfigDict({"a": 1})
        cfg.b = 20
        assert cfg.b == 20
        assert cfg["b"] == 20


@pytest.mark.unit
class TestYamlOperations:
    """Test YAML loading, saving, and merging."""

    def test_load_and_save_yaml(self, tmp_path):
        test_file = tmp_path / "test.yaml"
        data = {"model": {"name": "resnet18", "num_classes": 10}, "lr": 0.001}
        save_yaml(data, test_file)

        loaded = load_yaml(test_file)
        assert loaded.model.name == "resnet18"
        assert loaded.model.num_classes == 10
        assert loaded.lr == 0.001

    def test_load_missing_yaml_raises_error(self):
        with pytest.raises(ConfigurationError):
            load_yaml("non_existent_file_12345.yaml")

    def test_merge_configs(self):
        base = ConfigDict({"a": 1, "nested": {"x": 10, "y": 20}})
        override = ConfigDict({"b": 2, "nested": {"y": 99, "z": 100}})
        merged = merge_configs(base, override)

        assert merged.a == 1
        assert merged.b == 2
        assert merged.nested.x == 10
        assert merged.nested.y == 99
        assert merged.nested.z == 100

    def test_resolve_project_paths(self):
        cfg = ConfigDict({
            "path1": "{project_root}/data/raw",
            "nested": {"path2": "{project_root}/checkpoints"}
        })
        resolved = resolve_project_paths(cfg, project_root="/mock/root")
        assert resolved.path1 == "/mock/root/data/raw"
        assert resolved.nested.path2 == "/mock/root/checkpoints"

    def test_load_actual_project_configs(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        proj_cfg = load_yaml(repo_root / "configs/project/project.yaml")
        assert proj_cfg.project.name == "VulnShield-DNN"
        assert "resnet18" in proj_cfg.framework.target_architectures

        paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
        resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
        assert Path(resolved_paths.paths.data.raw).is_dir()
        assert Path(resolved_paths.paths.checkpoints.base_models).is_dir()

        reprod_cfg = load_yaml(repo_root / "configs/project/reproducibility.yaml")
        assert reprod_cfg.reproducibility.master_seed == 42
        assert len(reprod_cfg.reproducibility.evaluation_seeds) == 5


@pytest.mark.unit
class TestReproducibility:
    """Test seed management and determinism."""

    def test_set_seed_determinism(self):
        set_seed(42)
        r1 = np.random.rand(5)
        t1 = torch.rand(5)

        set_seed(42)
        r2 = np.random.rand(5)
        t2 = torch.rand(5)

        np.testing.assert_allclose(r1, r2)
        assert torch.allclose(t1, t2)

    def test_get_generator(self):
        gen1 = get_generator(123)
        t1 = torch.randint(0, 100, (5,), generator=gen1)

        gen2 = get_generator(123)
        t2 = torch.randint(0, 100, (5,), generator=gen2)

        assert torch.equal(t1, t2)


@pytest.mark.unit
class TestDevice:
    """Test device management utilities."""

    def test_get_device(self):
        cpu_dev = get_device("cpu")
        assert cpu_dev.type == "cpu"

        auto_dev = get_device("auto")
        assert auto_dev.type in ["cuda", "cpu"]

    def test_get_device_info(self):
        info = get_device_info(torch.device("cpu"))
        assert info["device_type"] == "cpu"
        assert info["is_cuda"] is False

    def test_to_device(self):
        dev = torch.device("cpu")
        tensor = torch.tensor([1, 2, 3])
        nested = {"a": tensor, "b": [tensor, tensor]}
        moved = to_device(nested, dev)

        assert moved["a"].device == dev
        assert moved["b"][0].device == dev
