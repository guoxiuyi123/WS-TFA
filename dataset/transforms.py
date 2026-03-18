import cv2
import numpy as np
import random
import torch

class PointCopyPaste:
    """
    Point-Level Contextual Copy-Paste 增强
    
    这是为了解决点监督下小目标数据量不足的创新增强策略。
    由于小目标在图像中占比极小，且只有中心点标注，我们可以以标注点为中心，
    裁剪出一个固定大小的上下文 Patch，并将其随机粘贴到图像的背景区域（没有其他目标的区域），
    从而人为地增加图像中小目标的绝对数量，提升模型的泛化能力。
    """
    def __init__(self, patch_size=32, prob=0.5, max_paste_objs=3):
        """
        Args:
            patch_size: 裁剪的正方形 Patch 的边长
            prob: 执行此数据增强的概率
            max_paste_objs: 每张图最多复制粘贴多少个目标
        """
        self.patch_size = patch_size
        self.prob = prob
        self.max_paste_objs = max_paste_objs

    def __call__(self, image, points):
        """
        Args:
            image: numpy array 形状为 (H, W, C)
            points: numpy array 形状为 (N, 3)，每行为 [x, y, class_id]
        Returns:
            image: 增强后的图像
            points: 增强后的点集合
        """
        if random.random() > self.prob or len(points) == 0:
            return image, points

        h, w, _ = image.shape
        half_p = self.patch_size // 2
        
        # 将点转换为列表以便动态添加
        new_points = points.tolist() if isinstance(points, np.ndarray) else list(points)
        
        # 决定要粘贴的数量
        num_paste = min(self.max_paste_objs, len(new_points))
        paste_indices = random.sample(range(len(new_points)), num_paste)
        
        for idx in paste_indices:
            px, py, cls_id = new_points[idx]
            px, py = int(px), int(py)
            
            # 计算裁剪边界并处理越界情况
            x1 = max(0, px - half_p)
            y1 = max(0, py - half_p)
            x2 = min(w, px + half_p)
            y2 = min(h, py + half_p)
            
            # 提取 Patch
            patch = image[y1:y2, x1:x2].copy()
            patch_h, patch_w = patch.shape[:2]
            
            if patch_h == 0 or patch_w == 0:
                continue
                
            # 寻找一个没有目标的空白区域来粘贴
            max_attempts = 10
            for _ in range(max_attempts):
                # 随机生成新的中心点
                new_px = random.randint(half_p, w - half_p - 1)
                new_py = random.randint(half_p, h - half_p - 1)
                
                # 检查新位置是否与其他目标太近 (简单距离检查，避免覆盖)
                is_valid = True
                for existing_pt in new_points:
                    ex, ey, _ = existing_pt
                    if (new_px - ex)**2 + (new_py - ey)**2 < self.patch_size**2:
                        is_valid = False
                        break
                        
                if is_valid:
                    # 计算粘贴边界
                    nx1 = new_px - patch_w // 2
                    ny1 = new_py - patch_h // 2
                    nx2 = nx1 + patch_w
                    ny2 = ny1 + patch_h
                    
                    # 确保不越界
                    if nx1 >= 0 and ny1 >= 0 and nx2 <= w and ny2 <= h:
                        # 粘贴 Patch
                        # 可以使用泊松融合(cv2.seamlessClone)或者高斯边缘融合来减少伪影，这里简单直接覆盖
                        image[ny1:ny2, nx1:nx2] = patch
                        
                        # 添加新的标注点
                        new_points.append([new_px, new_py, cls_id])
                        break
                        
        return image, np.array(new_points)

class PointMosaic:
    """
    针对点监督的 4-Mosaic 增强
    
    Mosaic 能够将 4 张图片缩放并拼接成一张大图，这对于小目标检测极为重要：
    1. 它极大地丰富了图像的上下文和背景变化。
    2. 它有效增加了单幅图中小目标的绝对数量（合并了 4 张图的目标）。
    3. 对于 DETR 类算法，图中目标数量的增加能极大地稳定早期的匈牙利二分图匹配过程。
    """
    def __init__(self, output_size=(800, 800)):
        """
        Args:
            output_size: 拼接后输出图像的大小 (H, W)
        """
        self.output_size = output_size
        
    def __call__(self, images, points_list):
        """
        Args:
            images: 包含 4 张图片的列表，每张图为 (H, W, C) 的 numpy array
            points_list: 包含 4 个点集合的列表，每个集合形状为 (N, 3) [x, y, class_id]
        Returns:
            mosaic_img: 拼接后的大图
            mosaic_points: 坐标转换后的点集合
        """
        assert len(images) == 4 and len(points_list) == 4, "Mosaic needs exactly 4 images and point lists"
        
        out_h, out_w = self.output_size
        
        # 随机决定交接十字的中心点坐标 (xc, yc)
        # 限制在中心区域，避免某张图过小
        xc = int(random.uniform(out_w * 0.25, out_w * 0.75))
        yc = int(random.uniform(out_h * 0.25, out_h * 0.75))
        
        # 初始化输出图像和点列表
        mosaic_img = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
        mosaic_points = []
        
        for i, (img, points) in enumerate(zip(images, points_list)):
            h, w, _ = img.shape
            
            # 计算每张图在 Mosaic 大图中的放置区域
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, out_w), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(out_h, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, out_w), min(out_h, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
            # 将图片区域放入大图
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # 转换点坐标
            # dx, dy 是从原图坐标系到大图坐标系的平移量
            dx = x1a - x1b
            dy = y1a - y1b
            
            for pt in points:
                px, py, cls_id = pt
                n_px = px + dx
                n_py = py + dy
                
                # 检查点是否在当前图块的有效边界内（没有被裁剪掉）
                if x1a <= n_px < x2a and y1a <= n_py < y2a:
                    mosaic_points.append([n_px, n_py, cls_id])
                    
        return mosaic_img, np.array(mosaic_points) if mosaic_points else np.empty((0, 3))
