"""VulnShield-DNN Model Layer Package."""

from vulnshield.models.common import ConvBNReLU, count_parameters, get_named_conv_layers
from vulnshield.models.resnet import BasicBlock, CIFARResNet18, resnet18
from vulnshield.models.vgg import CIFARVGG16, vgg16, VGG16_BN_CFG
from vulnshield.models.model_registry import (
    register_model,
    get_model_constructor,
    list_available_models
)
from vulnshield.models.model_factory import (
    create_model,
    save_checkpoint,
    load_model_weights
)

__all__ = [
    "ConvBNReLU",
    "count_parameters",
    "get_named_conv_layers",
    "BasicBlock",
    "CIFARResNet18",
    "resnet18",
    "CIFARVGG16",
    "vgg16",
    "VGG16_BN_CFG",
    "register_model",
    "get_model_constructor",
    "list_available_models",
    "create_model",
    "save_checkpoint",
    "load_model_weights"
]
