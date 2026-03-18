import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.ops import sigmoid_focal_loss 

class PointCriterion(nn.Module): 
    """ 
    专为点监督小目标检测设计的损失函数 (Criterion)。 
    
    核心创新/重写点： 
    1. 彻底抛弃了传统 DETR 的 Bounding Box 回归损失 (GIoU Loss 和 L1 Box Loss)。 
    2. 引入了针对极微弱小目标的 Point Distance Loss (Smooth L1 距离)。 
    3. 结合 Focal Loss 解决小目标与巨大背景（负样本）之间极度不平衡的分类问题。 
    """ 
    def __init__(self, num_classes, matcher, weight_dict, alpha=0.25, gamma=2.0): 
        super().__init__() 
        self.num_classes = num_classes 
        self.matcher = matcher 
        self.weight_dict = weight_dict 
        self.alpha = alpha 
        self.gamma = gamma 

    def loss_labels(self, outputs, targets, indices, num_points): 
        """计算分类损失：使用 Focal Loss""" 
        src_logits = outputs['pred_logits'] 
        idx = self._get_src_permutation_idx(indices) 
        
        # 提取匹配上的真实类别 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        # 初始化一个 full background (全背景) 的 target tensor，背景类默认为 self.num_classes 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device) 
        # 将匹配上的位置填入真实类别 
        target_classes[idx] = target_classes_o 

        # 转换为 one-hot 编码 (处理 Focal Loss 需要的格式) 
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], 
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) 
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) 
        # 移除最后一列（背景类），让网络只预测这 num_classes 个类 
        target_classes_onehot = target_classes_onehot[:, :, :-1] 

        # 计算 Focal Loss 
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=self.alpha, gamma=self.gamma, reduction="sum") 
        
        return {"loss_ce": loss_ce / num_points} 

    def loss_points(self, outputs, targets, indices, num_points): 
        """计算点回归损失：计算预测点和真实点之间的 Smooth L1 距离""" 
        idx = self._get_src_permutation_idx(indices) 
        
        # 提取匹配上的预测点和真实点 
        src_points = outputs['pred_points'][idx] 
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0) 

        # 点监督的核心：只有中心点 (x, y)，没有宽高 (w, h)，直接计算 Smooth L1 距离 
        loss_point = F.smooth_l1_loss(src_points, target_points, reduction='sum') 

        return {"loss_point": loss_point / num_points} 

    def _get_src_permutation_idx(self, indices): 
        """辅助函数：将匹配的局部索引转换为 batch 中的绝对张量索引""" 
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) 
        src_idx = torch.cat([src for (src, _) in indices]) 
        return batch_idx, src_idx 

    def forward(self, outputs, targets): 
        """ 
        计算总损失。 
        :param outputs: 包含 'pred_logits' [B, num_queries, num_classes] 和 'pred_points' [B, num_queries, 2] 
        :param targets: 包含 'labels' 和 'points' 的字典列表 
        """ 
        # 1. 调用 PointMatcher 进行二分图匹配 
        indices = self.matcher(outputs, targets) 

        # 2. 计算当前 batch 中真实目标的总数，用于均值化 Loss 
        num_points = sum(len(t["labels"]) for t in targets) 
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device) 
        # (注意：如果是多卡 DDP 训练，这里需要增加 torch.distributed.all_reduce(num_points) 来同步跨卡目标总数) 
        num_points = torch.clamp(num_points, min=1).item() 

        # 3. 计算分类和回归 Loss 
        losses = {} 
        losses.update(self.loss_labels(outputs, targets, indices, num_points)) 
        losses.update(self.loss_points(outputs, targets, indices, num_points)) 

        # 4. 乘以权重 
        for k in losses.keys(): 
            if k in self.weight_dict: 
                losses[k] *= self.weight_dict[k] 

        return losses