"""
Full Model Integration for WS-TFA (Weakly Supervised Tiny Feature Aggregation).
Combines Backbone, Dynamic Attention FPN, and Sparse MIL Head into a unified architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

import os
import sys

# Ensure models module can import its siblings when loaded directly or externally
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone import ResNet50Backbone
from dynamic_fpn import DynamicAttentionFPN
from sparse_mil_head import SparseMILHead


class WS_TFA_Net(nn.Module):
    """
    WS-TFA Network.
    Integrates all modules:
    1. Backbone (ResNet50) -> C1 to C5
    2. Dynamic Attention FPN -> P2_prime, P3, P4, P5
    3. Sparse MIL Head -> BBoxes, Objectness, MIL Probs, Final Joint Probs
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained_backbone: bool = True,
        fpn_out_channels: int = 256,
        num_queries: int = 300
    ):
        """
        Args:
            num_classes (int): Number of object categories.
            pretrained_backbone (bool): Whether to use ImageNet pretrained weights.
            fpn_out_channels (int): Channels for FPN output and Transformer hidden dim.
            num_queries (int): Number of learnable object queries for DETR.
        """
        super().__init__()
        
        # 1. Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        
        # ResNet50 output channels
        in_channels_list = [256, 512, 1024, 2048]
        c1_channels = 64
        
        # 2. Dynamic Attention FPN (includes FSM)
        self.fpn = DynamicAttentionFPN(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            c1_channels=c1_channels
        )
        
        # 3. Sparse MIL Head
        self.mil_head = SparseMILHead(
            num_classes=num_classes,
            hidden_dim=fpn_out_channels,
            num_queries=num_queries
        )
        
        # Lightweight module to flatten and concatenate FPN features for the Transformer
        self.input_proj = nn.ModuleList([
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=1)
            for _ in range(4) # For P2_prime, P3, P4, P5
        ])
        
        # Spatial Reduction to prevent OOM
        # P2_prime is stride 4 (e.g. 800x800 -> 200x200 = 40,000 tokens)
        # P3 is stride 8 (100x100 = 10,000 tokens)
        # We apply stride=2 convolution to P2_prime and P3 to reduce sequence length
        # P2_prime becomes effectively stride 8, P3 becomes stride 16
        self.spatial_reduction_p2 = nn.Sequential(
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_reduction_p3 = nn.Sequential(
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU(inplace=True)
        )

    def _flatten_and_concat_features(self, fpn_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Flattens multi-scale spatial features into a 1D sequence for the Transformer.
        Returns the concatenated sequence and a list of spatial shapes for positional encoding.
        """
        keys = ['P2_prime', 'P3', 'P4', 'P5']
        
        srcs = []
        spatial_shapes = []
        for i, key in enumerate(keys):
            feat = fpn_features[key]
            
            # Apply Spatial Reduction for P2_prime and P3
            if key == 'P2_prime':
                feat = self.spatial_reduction_p2(feat)
            elif key == 'P3':
                feat = self.spatial_reduction_p3(feat)
                
            # Optional 1x1 projection
            feat = self.input_proj[i](feat)
            
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            
            # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            feat_flat = feat.view(B, C, -1).permute(0, 2, 1)
            srcs.append(feat_flat)
            
        # Concatenate along the sequence length dimension
        concat_srcs = torch.cat(srcs, dim=1)
        return concat_srcs, spatial_shapes

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full Forward Pass.
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W].
            return_features (bool): If True, returns spatial FPN features for visualization.
            
        Returns:
            Dict containing final predictions, intermediate alphas, and optionally FPN features.
        """
        # 1. Extract Backbone Features
        c_features = self.backbone(x)
        
        # 2. Dynamic Attention FPN
        p_features = self.fpn(c_features)
        alphas = p_features.pop('alphas') # Extract alphas for regularization
        
        # Keep a copy of p_features if requested for visualization
        spatial_features = None
        if return_features:
            spatial_features = {k: v.clone() for k, v in p_features.items()}
        
        # 3. Prepare Transformer Input
        seq_features, spatial_shapes = self._flatten_and_concat_features(p_features)
        
        # 4. Sparse MIL Head Predictions
        # Pass spatial_shapes for positional encoding
        head_outputs = self.mil_head(seq_features, spatial_shapes)
        
        # Merge outputs
        head_outputs['alphas'] = alphas
        
        if return_features:
            head_outputs['spatial_features'] = spatial_features
            
        return head_outputs


if __name__ == '__main__':
    print("=== Testing WS-TFA Full Network Integration ===")
    
    # Instantiate Model
    model = WS_TFA_Net(num_classes=20, pretrained_backbone=False)
    
    # Dummy Image [Batch, Channels, Height, Width]
    # Keep spatial size small to reduce memory/compute in dummy test
    dummy_img = torch.randn(2, 3, 256, 256)
    print(f"Input Image Shape: {dummy_img.shape}")
    
    # Forward Pass
    outputs = model(dummy_img)
    
    print("\n--- Network Outputs ---")
    print(f"Final Joint Probabilities: {outputs['final_prob'].shape}")
    print(f"Bounding Boxes: {outputs['bboxes'].shape}")
    print(f"Objectness Scores: {outputs['objectness_scores'].shape}")
    
    print("\n--- Learned Dynamic Alphas ---")
    for k, v in outputs['alphas'].items():
        print(f"{k}: {v.shape}")
        
    print("\n✅ Test Passed: WS-TFA Full Network Forward Pass completed.")
