"""Unit Tests for VulnShield-DNN Model Layer (ResNet-18 & VGG-16)."""

import tempfile
from pathlib import Path
import pytest
import torch

from vulnshield.core.exceptions import ModelNotFoundError
from vulnshield.models.common import count_parameters, get_named_conv_layers
from vulnshield.models.resnet import CIFARResNet18, resnet18
from vulnshield.models.vgg import CIFARVGG16, vgg16
from vulnshield.models.model_registry import list_available_models, get_model_constructor
from vulnshield.models.model_factory import create_model, save_checkpoint, load_model_weights


@pytest.mark.unit
class TestResNet18:
    """Test CIFAR-10 ResNet-18 architecture, parameters, and gradient flow."""

    def test_resnet18_forward_pass(self):
        model = resnet18(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)

        assert out.shape == (4, 10)
        assert torch.isfinite(out).all()

    def test_resnet18_parameter_count(self):
        model = resnet18(num_classes=10)
        total, trainable = count_parameters(model)
        # CIFAR-10 ResNet-18 is ~11.17M parameters
        assert 11_000_000 < total < 11_500_000
        assert total == trainable

    def test_resnet18_backward_pass(self):
        model = resnet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        target = torch.tensor([1, 4])

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_resnet18_conv_layers_count(self):
        model = resnet18(num_classes=10)
        convs = get_named_conv_layers(model)
        # 1 initial + 8 standard convs + 8 standard convs + 3 downsample shortcut convs = 20 Conv2d layers
        assert len(convs) == 20
        assert convs[0][0] == "conv1"


@pytest.mark.unit
class TestVGG16:
    """Test CIFAR-10 VGG-16 architecture, parameters, and gradient flow."""

    def test_vgg16_forward_pass(self):
        model = vgg16(num_classes=10)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)

        assert out.shape == (4, 10)
        assert torch.isfinite(out).all()

    def test_vgg16_parameter_count(self):
        model = vgg16(num_classes=10)
        total, trainable = count_parameters(model)
        # CIFAR-10 VGG16-BN is ~14.7M parameters
        assert 14_000_000 < total < 15_500_000
        assert total == trainable

    def test_vgg16_conv_layers_count(self):
        model = vgg16(num_classes=10)
        convs = get_named_conv_layers(model)
        # VGG-16 has exactly 13 Conv2d layers
        assert len(convs) == 13


@pytest.mark.unit
class TestModelFactoryAndRegistry:
    """Test model registry lookup, creation, and checkpoint saving/loading."""

    def test_registry_lookup(self):
        available = list_available_models()
        assert "resnet18" in available
        assert "vgg16" in available

        constructor = get_model_constructor("resnet18")
        assert callable(constructor)

        with pytest.raises(ModelNotFoundError):
            get_model_constructor("non_existent_architecture_xyz")

    def test_create_model(self):
        m = create_model("resnet18", num_classes=10, device="cpu")
        assert isinstance(m, CIFARResNet18)

    def test_checkpoint_save_and_load(self, tmp_path):
        model1 = create_model("resnet18", num_classes=10, device="cpu")
        ckpt_file = tmp_path / "model_ckpt.pt"

        save_checkpoint(
            model=model1,
            path=ckpt_file,
            epoch=5,
            metrics={"acc": 92.5}
        )

        assert ckpt_file.exists()

        model2 = create_model("resnet18", num_classes=10, device="cpu")
        meta = load_model_weights(model2, ckpt_file, device=torch.device("cpu"))

        assert meta["epoch"] == 5
        assert meta["metrics"]["acc"] == 92.5

        # Verify weights match identically
        x = torch.randn(2, 3, 32, 32)
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        torch.testing.assert_close(out1, out2)
