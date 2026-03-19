"""
Feature Supplement Module (FSM) for WS-TFA.
Designed to supplement high-resolution geometric details from shallow layers
into semantically rich feature maps.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeatureSupplementModule(nn.Module):
    """
    Feature Supplement Module (FSM).
    Extracts receptive fields from C1 via dilated convolution, adjusts channels,
    and fuses with P2 (which is derived from C2) to output a detailed P2_prime.
    """

    def __init__(
        self,
        c1_channels: int = 64,
        p2_channels: int = 256,
        dilation: int = 2
    ) -> None:
        """
        Initializes the Feature Supplement Module.

        Args:
            c1_channels (int): Channel size of the C1 feature map (default: 64 for ResNet).
            p2_channels (int): Channel size of the P2 feature map.
            dilation (int): Dilation rate for the receptive field expansion.
        """
        super().__init__()

        # 1. Dilated Convolution for receptive field extraction on C1
        self.dilated_conv = nn.Conv2d(
            in_channels=c1_channels,
            out_channels=c1_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(c1_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. 1x1 Convolution to adjust channels and spatially downsample to match P2
        # C1 has stride 2, P2 has stride 4, so we need to downsample by factor of 2.
        # We achieve this by setting stride=2 in the 1x1 convolution.
        self.channel_adjust = nn.Conv2d(
            in_channels=c1_channels,
            out_channels=p2_channels,
            kernel_size=1,
            stride=2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(p2_channels)

    def forward(self, c1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FSM.

        Args:
            c1 (torch.Tensor): C1 feature map from backbone (B, C1_dim, H/2, W/2).
            p2 (torch.Tensor): P2 feature map from FPN (B, P2_dim, H/4, W/4).

        Returns:
            torch.Tensor: The fused feature map P2_prime (B, P2_dim, H/4, W/4).
        """
        # Step 1: Dilated Convolution
        x = self.dilated_conv(c1)
        x = self.bn1(x)
        x = self.relu(x)

        # Step 2: Channel Adjustment and Spatial Downsampling (stride 2)
        x = self.channel_adjust(x)
        x = self.bn2(x)

        # Ensure spatial dimensions exactly match (in case of odd input sizes)
        if x.shape[2:] != p2.shape[2:]:
            import torch.nn.functional as F
            x = F.interpolate(x, size=p2.shape[2:], mode='bilinear', align_corners=False)

        # Step 3: Fusion (Element-wise Addition)
        p2_prime = self.relu(p2 + x)

        return p2_prime


if __name__ == '__main__':
    # Simple test block to verify dimension alignment
    import sys
    import os
    # Ensure backbone can be imported
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backbone import ResNet50Backbone

    print("=== Testing WS-TFA Modules ===")
    
    # Instantiate models
    backbone = ResNet50Backbone(pretrained=False)
    # Assume FPN outputs 256 channels for P2
    p2_channels = 256
    fsm = FeatureSupplementModule(c1_channels=64, p2_channels=p2_channels, dilation=2)

    # Dummy tensor: Batch Size 2, 3 Channels, 512x512 Image
    dummy_input = torch.randn(2, 3, 512, 512)
    print(f"Input Tensor Shape: {dummy_input.shape}")

    # Forward pass through backbone
    features = backbone(dummy_input)
    c1 = features['C1']
    c2 = features['C2']
    print(f"Extracted C1 Shape: {c1.shape}")
    print(f"Extracted C2 Shape: {c2.shape}")

    # Mock P2 feature from C2 (FPN typically reduces C2 to 256 channels via 1x1 conv)
    mock_fpn_conv = nn.Conv2d(256, p2_channels, kernel_size=1)
    p2 = mock_fpn_conv(c2)
    print(f"Mocked P2 Shape: {p2.shape}")

    # Forward pass through FSM
    p2_prime = fsm(c1, p2)
    print(f"FSM Output P2_prime Shape: {p2_prime.shape}")

    # Validation
    assert p2_prime.shape == p2.shape, f"Dimension mismatch! Expected {p2.shape}, got {p2_prime.shape}"
    print("\n✅ Test Passed: C1 to P2_prime dimension alignment is correct without errors.")
