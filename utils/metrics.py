import torch
import numpy as np

class PointEvaluator:
    """
    专门为点监督小目标检测设计的评估体系 (Point-based AP)
    
    由于点监督数据缺乏 Bounding Box 宽高，传统的基于 IoU 的 COCO AP 无法使用。
    该类使用预测点与真实点之间的欧式距离 (L2 Distance) 来判定 True Positive (TP)。
    定义了一组距离阈值 (distance_thresholds, 单位: 像素)，类似 COCO 中的 IoU 阈值。
    """
    def __init__(self, distance_thresholds=[4.0, 8.0, 16.0]):
        self.distance_thresholds = distance_thresholds
        self.reset()
        
    def reset(self):
        """清空缓存的预测和真实标签记录"""
        # 为每个阈值维护一个列表，保存所有的预测匹配状态：(score, is_tp)
        self.predictions = {thresh: [] for thresh in self.distance_thresholds}
        self.total_gts = {thresh: 0 for thresh in self.distance_thresholds}
        
    @torch.no_grad()
    def update(self, pred_points, pred_scores, gt_points):
        """
        处理单张图片的预测和真实点，进行基于距离的贪心匹配。
        
        Args:
            pred_points: Tensor [N, 2], 预测点坐标 (x, y)
            pred_scores: Tensor [N], 预测得分
            gt_points: Tensor [M, 2], 真实点坐标 (x, y)
        """
        # 转为 CPU numpy 方便处理
        pred_points = pred_points.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        gt_points = gt_points.cpu().numpy()
        
        num_gts = len(gt_points)
        for thresh in self.distance_thresholds:
            self.total_gts[thresh] += num_gts
            
        if len(pred_points) == 0:
            return
            
        # 按得分降序排列预测点
        sort_idx = np.argsort(-pred_scores)
        pred_points = pred_points[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        if num_gts == 0:
            # 如果没有真实目标，所有的预测都是 FP
            for thresh in self.distance_thresholds:
                for score in pred_scores:
                    self.predictions[thresh].append((score, 0)) # 0 表示 FP
            return
            
        # 计算距离矩阵 [N, M]
        # dist_matrix[i, j] 表示第 i 个预测点和第 j 个真实点的欧式距离
        diff = pred_points[:, np.newaxis, :] - gt_points[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        
        # 对每个阈值进行贪心匹配
        for thresh in self.distance_thresholds:
            matched_gt = np.zeros(num_gts, dtype=bool)
            
            for i, score in enumerate(pred_scores):
                # 寻找距离当前预测点最近且未被匹配的 GT
                min_dist = float('inf')
                best_gt_idx = -1
                
                for j in range(num_gts):
                    if not matched_gt[j] and dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        best_gt_idx = j
                        
                if best_gt_idx != -1 and min_dist <= thresh:
                    # 匹配成功，记为 TP
                    matched_gt[best_gt_idx] = True
                    self.predictions[thresh].append((score, 1))
                else:
                    # 匹配失败，记为 FP
                    self.predictions[thresh].append((score, 0))

    def accumulate(self):
        """
        计算并打印所有距离阈值下的 Point AP。
        
        Returns:
            aps: 包含各阈值下 AP 的字典
        """
        aps = {}
        for thresh in self.distance_thresholds:
            preds = self.predictions[thresh]
            total_gt = self.total_gts[thresh]
            
            if total_gt == 0 or len(preds) == 0:
                aps[f"AP@{thresh}"] = 0.0
                continue
                
            # 按得分降序排序 (在单图处理时已经排过，但在整个数据集中合并后需重新排序)
            preds.sort(key=lambda x: x[0], reverse=True)
            
            # 提取 TP 序列
            tps = np.array([p[1] for p in preds])
            fps = 1 - tps
            
            # 计算累积 TP 和 FP
            acc_tps = np.cumsum(tps)
            acc_fps = np.cumsum(fps)
            
            # 计算 Precision 和 Recall
            recalls = acc_tps / total_gt
            precisions = acc_tps / (acc_tps + acc_fps + 1e-8)
            
            # 使用 11点插值法 或 所有点积分 计算 AP (这里使用类似 VOC 2010+ 的所有点面积积分)
            ap = self._compute_ap(recalls, precisions)
            aps[f"AP@{thresh}"] = ap
            
        # 打印结果
        print("====== Point-based Evaluation ======")
        mean_ap = 0
        for k, v in aps.items():
            print(f"  {k}px : {v:.4f}")
            mean_ap += v
        if len(aps) > 0:
            print(f"  mAP     : {mean_ap / len(aps):.4f}")
        print("====================================")
        
        return aps

    def _compute_ap(self, rec, prec):
        """计算 PR 曲线下的面积"""
        # 在两端加上边界点
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        
        # 使 precision 单调递减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        # 寻找 recall 变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 计算面积
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
