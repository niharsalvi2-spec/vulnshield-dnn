"""CLI Script to Inspect Model Architectures and Layer Topologies."""

import sys
import argparse
from pathlib import Path
import torch

from vulnshield.models.model_factory import create_model
from vulnshield.models.common import count_parameters, get_named_conv_layers


def inspect_architecture(model_name: str):
    print("=" * 65)
    print(f"      VulnShield-DNN Model Inspection: {model_name.upper()}")
    print("=" * 65)

    model = create_model(model_name, num_classes=10, device="cpu")
    total_params, trainable_params = count_parameters(model)
    conv_layers = get_named_conv_layers(model)

    print(f"[*] Architecture Name  : {model_name}")
    print(f"[*] Total Parameters   : {total_params:,} ({total_params * 4 / (1024**2):.2f} MB in FP32)")
    print(f"[*] Trainable Params   : {trainable_params:,}")
    print(f"[*] Conv2d Layers Count: {len(conv_layers)}")
    
    total_channels = 0
    print("\n[*] Conv2d Layer Topology Breakdown:")
    print(f"  {'#':<3} {'Layer Name':<28} {'In C':<6} {'Out C (Channels)':<18} {'Kernel':<8}")
    print("  " + "-" * 62)
    for idx, (name, layer) in enumerate(conv_layers):
        total_channels += layer.out_channels
        k_str = f"{layer.kernel_size[0]}x{layer.kernel_size[1]}"
        print(f"  {idx+1:<3} {name:<28} {layer.in_channels:<6} {layer.out_channels:<18} {k_str:<8}")

    print("  " + "-" * 62)
    print(f"[*] Total Convolutional Channels in Network: {total_channels}")

    # Forward pass check
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"[*] Dummy Forward Pass Output Shape: {out.shape} (Finite={torch.isfinite(out).all()})")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect VulnShield-DNN model architectures.")
    parser.add_argument("--model", type=str, default="all", choices=["resnet18", "vgg16", "all"], help="Model to inspect")
    args = parser.parse_args()

    if args.model in ["resnet18", "all"]:
        inspect_architecture("resnet18")
    if args.model in ["vgg16", "all"]:
        inspect_architecture("vgg16")


if __name__ == "__main__":
    main()
