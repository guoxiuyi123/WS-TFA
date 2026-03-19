"""
Backbone module for WS-TFA (Weakly Supervised Tiny Feature Aggregation).
Extracts multi-scale features (C1 to C5) using a modified ResNet50.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict


class ResNet50Backbone(nn.Module):
    """
    ResNet50 Backbone for extracting multi-stage features.
    The fully connected layer and global average pooling are stripped.
    Outputs C1, C2, C3, C4, C5 stages for FPN and subsequent modules.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initializes the ResNet50 backbone.

        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # C1: Shallowest feature with high geometric details (stride 2)
        # Consists of conv1, bn1, relu
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        # C2: Output of layer1 (stride 4), preceded by maxpool
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        # C3: Output of layer2 (stride 8)
        self.layer2 = resnet.layer2

        # C4: Output of layer3 (stride 16)
        self.layer3 = resnet.layer3

        # C5: Output of layer4 (stride 32)
        self.layer4 = resnet.layer4

        # Output channels for each stage
        self.out_channels = {
            "C1": 64,
            "C2": 256,
            "C3": 512,
            "C4": 1024,
            "C5": 2048,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract multi-scale features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature maps corresponding to C1-C5.
        """
        # Stage C1 (Stride 2)
        c1 = self.relu(self.bn1(self.conv1(x)))

        # Stage C2 (Stride 4)
        c2 = self.layer1(self.maxpool(c1))

        # Stage C3 (Stride 8)
        c3 = self.layer2(c2)

        # Stage C4 (Stride 16)
        c4 = self.layer3(c3)

        # Stage C5 (Stride 32)
        c5 = self.layer4(c4)

        return {
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "C5": c5
        }
