import torch 
import torch.nn as nn 
from scipy.optimize import linear_sum_assignment 

class PointMatcher(nn.Module): 
    """ 
    专为点监督和小目标检测设计的匈牙利匹配器。 
    传统 DETR 使用 Bounding Box 的 L1 和 GIoU 距离作为 Cost。 
    但在点监督下，我们没有物体的宽高 (w, h) 信息。 
    因此，此匹配器被彻底重写，完全基于： 
    1. 分类 Focal Cost 
    2. 预测中心点与真实点之间的欧式距离 (L2 Distance Cost) 
    """ 
    def __init__(self, cost_class: float = 2.0, cost_point: float = 5.0, alpha: float = 0.25, gamma: float = 2.0): 
        super().__init__() 
        self.cost_class = cost_class 
        self.cost_point = cost_point 
        self.alpha = alpha 
        self.gamma = gamma 

    @torch.no_grad() 
    def forward(self, outputs, targets): 
        """ 
        执行二分图匹配。 
        :param outputs: 字典，包含 "pred_logits" [B, num_queries, num_classes] 和 "pred_points" [B, num_queries, 2] 
        :param targets: 列表，每个元素是字典，包含 "labels" 和 "points" [num_targets, 2] 
        """ 
        bs, num_queries = outputs["pred_logits"].shape[:2] 

        # 1. 提取并展平网络输出 (Flatten batch dimension for parallel computation) 
        # [batch_size * num_queries, num_classes] 
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() 
        # [batch_size * num_queries, 2] (预测的归一化 x, y 坐标) 
        out_point = outputs["pred_points"].flatten(0, 1) 

        # 2. 提取并展平真实标签 
        # [total_target_points_in_batch] 
        tgt_ids = torch.cat([v["labels"] for v in targets]) 
        # [total_target_points_in_batch, 2] 
        tgt_point = torch.cat([v["points"] for v in targets]) 

        # 处理极端情况：如果整个 Batch 没有一个目标，直接返回空匹配 
        if len(tgt_ids) == 0: 
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)] 

        # 3. 计算分类成本 (Focal Loss Cost) 
        # 采用 Deformable DETR 的标准 Focal Cost 计算方式，使得正负样本匹配更加稳定 
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log()) 
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log()) 
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] 

        # 4. 计算点距离成本 (L2 Distance Cost) -> 【核心创新点】 
        # 弃用传统的 L1 和 GIoU Box 损失，直接使用 L2 范数 (p=2) 计算预测点与真实点的欧式距离 
        cost_point = torch.cdist(out_point, tgt_point, p=2) 

        # 5. 构建总代价矩阵 (Cost Matrix) 
        C = self.cost_point * cost_point + self.cost_class * cost_class 
        
        # 将代价矩阵重塑为每张图独立的矩阵 [batch_size, num_queries, total_target_points] 
        C = C.view(bs, num_queries, -1).cpu() 

        # 6. 使用匈牙利算法 (Hungarian Algorithm) 进行最优一对一匹配 
        sizes = [len(v["points"]) for v in targets] 
        # 按每张图的真实目标数量拆分矩阵，并分别计算 
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] 

        # 返回格式：[(query_indices, target_indices), ...] 
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]