import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops import MobileMambaBlock, DynamicSnakeConv

class HybridEncoderPRM(nn.Module):
    """
    HybridEncoderPRM
    
    这是本框架的核心 Neck 模块，结合了多种机制来处理多尺度特征：
    1. 接收 Backbone 的 4 个尺度的特征 [P2, P3, P4, P5]（P2 用于高分辨率小目标）。
    2. 使用 MobileMambaBlock 处理 P2 层的全局特征，避免高分辨率带来的显存爆炸问题。
    3. 使用 Transformer Encoder 处理 P5 层的深层语义特征。
    4. 使用基于 DynamicSnakeConv 的特征金字塔融合层，在点监督下动态寻找小目标的边界。
    """
    def __init__(self, in_channels_list=[64, 128, 256, 512], hidden_dim=256):
        super().__init__()
        
        # 统一各层的通道数为 hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1) 
            for in_channels in in_channels_list
        ])
        
        # 1. 针对 P2 层高分辨率特征，使用 MobileMambaBlock
        # MobileMambaModule 通常需要 dim 参数
        self.p2_mamba = MobileMambaBlock(dim=hidden_dim)
        
        # 2. 针对 P5 层深层语义特征，使用 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.p5_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. 基于 DynamicSnakeConv 的特征金字塔融合（FPN 自顶向下融合）
        # DynamicSnakeConv(inc, ouc, k=3)
        self.snake_convs = nn.ModuleList([
            DynamicSnakeConv(hidden_dim, hidden_dim) for _ in range(4)
        ])

    def forward(self, features):
        """
        Args:
            features: 包含 [P2, P3, P4, P5] 特征图的列表。
        Returns:
            enhanced_features: 融合增强后的 [P2, P3, P4, P5] 特征图列表。
        """
        assert len(features) == 4, "Expected 4 feature maps: [P2, P3, P4, P5]"
        
        # 1. 统一通道维度
        proj_features = [proj(feat) for proj, feat in zip(self.input_proj, features)]
        p2, p3, p4, p5 = proj_features
        
        # 2. 对 P2 层应用 MobileMambaBlock
        p2 = self.p2_mamba(p2)
        
        # 3. 对 P5 层应用 Transformer Encoder
        B, C, H, W = p5.shape
        # 将空间维度展平供 Transformer 处理：[B, C, H, W] -> [B, H*W, C]
        p5_flat = p5.flatten(2).permute(0, 2, 1)
        p5_enc = self.p5_transformer(p5_flat)
        # 还原维度：[B, H*W, C] -> [B, C, H, W]
        p5 = p5_enc.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 4. 自顶向下的特征金字塔融合，并使用 DynamicSnakeConv 进行特征增强
        # P5 -> P4
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        # P4 -> P3
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        # P3 -> P2
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        
        # 5. 利用 DynamicSnakeConv 进行最终的特征精炼
        enhanced_features = [
            self.snake_convs[0](p2),
            self.snake_convs[1](p3),
            self.snake_convs[2](p4),
            self.snake_convs[3](p5)
        ]
        
        return enhanced_features
