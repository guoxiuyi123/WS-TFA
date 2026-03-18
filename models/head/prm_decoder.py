import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops import KALNConv

class PRMDecoder(nn.Module):
    """
    PRMDecoder: Point-to-Region Mamba Decoder
    
    该解码器基于 Transformer 架构，但在分类预测头（Classification Head）中引入了 
    KALNConv (Kolmogorov-Arnold Legendre Network Convolution)。
    
    KAN (Kolmogorov-Arnold Network) 利用高阶多项式（如勒让德多项式）作为激活函数，
    具有比传统 MLP 更强的非线性拟合能力。在点监督下，小目标的特征往往非常微弱且难以区分，
    引入 KAN 可以大幅提升分类分支对微弱特征的判别能力。
    """
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, nheads=8, num_decoder_layers=6):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=1024, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Learnable Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Classification Head using KAN (KALNConv)
        # 替代传统的 nn.Linear(hidden_dim, num_classes)
        # KALNConv2DLayer 需要 2D 输入，这里我们将 (B, N, C) 视为 (B, C, N, 1) 处理
        self.class_embed = KALNConv(
            input_dim=hidden_dim, 
            output_dim=num_classes, 
            kernel_size=1, 
            degree=3,  # 勒让德多项式阶数
            groups=1
        )
        
        # Point Regression Head (predicting normalized x, y)
        self.point_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid() # 归一化坐标 [0, 1]
        )

    def forward(self, features, targets=None):
        """
        Args:
            features: 来自 Neck 的多尺度特征列表 (目前仅使用最后一层 P5 或融合后的特征)
                      为了简化，这里假设 features 是一个 Tensor 或者我们取最后一个特征图
            targets: 训练时的标签
            
        Returns:
            outputs: 包含 pred_logits 和 pred_points 的字典
        """
        # 假设 features 是一个列表，我们取最高层特征 P5 进行解码，或者如果是融合后的单一特征
        # 在 DETR 中通常会将特征展平。这里为了适配 PRM 架构，我们假设 Neck 输出的最后一层包含了丰富语义
        
        # 取最后一层特征 [B, C, H, W]
        x = features[-1] 
        B, C, H, W = x.shape
        
        # 展平特征作为 Memory: [B, H*W, C]
        memory = x.flatten(2).permute(0, 2, 1)
        
        # 准备 Query: [B, num_queries, C]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Transformer Decoder
        # tgt: [B, num_queries, C], memory: [B, H*W, C]
        hs = self.transformer_decoder(tgt=query_embed, memory=memory)
        # hs: [B, num_queries, C]
        
        # Classification Head (KAN)
        # KALNConv 期望输入 (B, C, H, W)
        # 将 hs 转换为 (B, C, num_queries, 1)
        hs_kan = hs.permute(0, 2, 1).unsqueeze(-1)
        outputs_class = self.class_embed(hs_kan)
        # outputs_class: (B, num_classes, num_queries, 1) -> (B, num_queries, num_classes)
        outputs_class = outputs_class.squeeze(-1).permute(0, 2, 1)
        
        # Point Regression Head
        outputs_point = self.point_embed(hs)
        
        return {
            "pred_logits": outputs_class,
            "pred_points": outputs_point
        }
