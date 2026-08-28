"""VGG-16 Architecture with Batch Normalization Adapted for CIFAR-10."""

from __future__ import annotations

from typing import List, Union
import torch
import torch.nn as nn

VGG16_BN_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M"
]


class CIFARVGG16(nn.Module):
    """VGG-16 architecture with Batch Normalization tailored for CIFAR-10."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.features = self._make_layers(VGG16_BN_CFG, in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _make_layers(self, cfg: List[Union[int, str]], in_channels: int = 3) -> nn.Sequential:
        layers: List[nn.Module] = []
        curr_channels = in_channels
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(curr_channels, int(v), kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
                curr_channels = int(v)
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def vgg16(num_classes: int = 10, **kwargs) -> CIFARVGG16:
    """Factory helper to instantiate CIFAR-10 VGG-16."""
    return CIFARVGG16(num_classes=num_classes, **kwargs)
