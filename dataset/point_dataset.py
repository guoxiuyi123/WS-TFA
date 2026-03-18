import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class PointSupervisedDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # 只读取图片文件
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. 读取图像并统一缩放到 800x800
        image = Image.open(img_path).convert("RGB")
        image = image.resize((640, 640))
        image = TF.to_tensor(image) # 转换为 [3, 800, 800] 的 Tensor

        # 2. 读取对应的点标注
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        labels = []
        points = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        labels.append(int(parts[0]))
                        # txt 里已经是归一化的中心点 (cx, cy)
                        points.append([float(parts[1]), float(parts[2])])

        # 3. 构造 target 字典
        targets = {}
        if len(labels) > 0:
            targets['labels'] = torch.tensor(labels, dtype=torch.int64)
            targets['points'] = torch.tensor(points, dtype=torch.float32)
        else:
            # 处理没有任何目标的负样本图像
            targets['labels'] = torch.zeros((0,), dtype=torch.int64)
            targets['points'] = torch.zeros((0, 2), dtype=torch.float32)

        return image, targets