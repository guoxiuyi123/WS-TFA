"""
Test script to verify the WSOD Data Pipeline and Augmentations.
Loads dummy/real images, applies augmentations, and visualizes the Cutout effect.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import WSODDataset, get_wsod_transforms, VOC_CLASSES
from PIL import Image
import numpy as np

def create_dummy_data(root_dir: str):
    """Creates a small dummy dataset structure for testing."""
    img_dir = os.path.join(root_dir, 'JPEGImages')
    os.makedirs(img_dir, exist_ok=True)
    
    # Create 4 dummy images with color patterns
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors):
        img_np = np.full((500, 600, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_np)
        img.save(os.path.join(img_dir, f"dummy_{i:04d}.jpg"))
        
    # Create dummy split file
    split_dir = os.path.join(root_dir, 'ImageSets', 'Main')
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        for i in range(4):
            f.write(f"dummy_{i:04d}\n")
            
    print(f"Created dummy dataset at {root_dir}")

def main():
    print("=== Testing WSOD Data Pipeline ===")
    
    # 1. Setup Dummy Data Directory
    dummy_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dummy_voc')
    create_dummy_data(dummy_root)
    
    # 2. Get Transforms (Training mode to enable Cutout/CoarseDropout)
    # Using a high probability (p=1.0) for testing to guarantee Cutout appears
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    target_size = 800
    test_transforms = A.Compose([
        A.Resize(height=target_size, width=target_size),
        A.HorizontalFlip(p=0.5),
        # Guarantee Cutout for visualization
        A.CoarseDropout(
            num_holes_range=(5, 10), 
            hole_height_range=(target_size // 10, target_size // 10), 
            hole_width_range=(target_size // 10, target_size // 10), 
            p=1.0 
        ),
        # Standard ImageNet Normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 3. Instantiate Dataset and DataLoader
    dataset = WSODDataset(
        root_dir=dummy_root,
        image_set='train',
        transforms=test_transforms,
        classes=VOC_CLASSES
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # 4. Fetch a Batch
    print("\nFetching a batch from DataLoader...")
    images, labels = next(iter(dataloader))
    
    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")
    
    # Print present classes for the first image
    present_classes = [VOC_CLASSES[i] for i, val in enumerate(labels[0]) if val == 1.0]
    print(f"Random Image-Level Labels for Image 0: {present_classes}")
    
    # 5. Denormalize and Save Image to check Cutout
    # Un-normalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    vis_images = images * std + mean
    vis_images = torch.clamp(vis_images, 0, 1)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_augmentation.jpg')
    save_image(vis_images, save_path, nrow=2)
    
    print(f"\n✅ Data Pipeline Test Passed!")
    print(f"Saved augmented images (showing Cutout/CoarseDropout) to: {save_path}")

if __name__ == '__main__':
    main()
