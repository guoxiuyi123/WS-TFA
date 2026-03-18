import torch
import torch.nn as nn

class PRMDETR(nn.Module):
    """
    PRM-DETR (Point-to-Region Mamba-DETR)
    
    这是一个专为点监督下的小目标检测设计的解耦架构框架。
    由于小目标在图像中占据极小的像素，且点监督仅仅提供中心点信息（无边界框宽高），
    因此我们需要一个能够高效处理高分辨率特征（捕捉小目标细节）、
    动态寻找目标边界，并且兼顾全局上下文信息的网络。
    
    本框架由三大核心模块解耦组成：
    1. Backbone: 负责提取多尺度特征（保留高分辨率的 P2 层用于小目标）。
    2. Neck (特征增强与融合): 利用 Mamba 处理高分辨率特征的全局依赖（避免显存爆炸），
       结合 Transformer 和动态蛇形卷积（DynamicSnakeConv）实现多尺度特征的高效融合，
       以便在点监督下更好地动态定位小目标的边界区域。
    3. Head: 接收融合后的多尺度特征进行分类和点位置回归。
    """
    def __init__(self, backbone, neck, head):
        super(PRMDETR, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, images, targets=None):
        """
        前向传播逻辑：
        
        Args:
            images: 输入的图像张量，通常形状为 [B, C, H, W]
            targets: 点监督的目标标签，包含类别和点坐标信息（在推理时为 None）
            
        Returns:
            outputs: 包含分类预测和点位置回归结果的字典
        """
        # 1. Backbone: 提取多尺度特征
        # 通常返回一个特征列表，例如 [P2, P3, P4, P5]
        features = self.backbone(images)
        
        # 2. Neck: 特征增强与多尺度融合
        # 接收 Backbone 特征，进行自顶向下的融合，返回增强后的特征列表
        enhanced_features = self.neck(features)
        
        # 3. Head: 输出检测结果
        # 接收增强特征，输出分类和点预测，如果是训练阶段可以结合 targets 计算 Loss
        # 这里解耦了 Loss 计算逻辑，具体可以由 Engine 或者 Head 内部来处理
        outputs = self.head(enhanced_features, targets)
        
        return outputs
