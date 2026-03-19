"""
Academic Visualization Tools for WS-TFA.
Used to generate high-quality figures for papers, demonstrating the effectiveness
of overcoming Local Dominance and extracting rich spatial details via Attention Heatmaps.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os

class WSODVisualizer:
    """
    Visualizer class for Weakly Supervised Object Detection.
    Provides methods to draw bounding boxes and generate attention heatmaps.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names (List[str]): List of class names for label mapping.
        """
        self.class_names = class_names
        # Standard colors for bounding boxes
        self.colors = plt.cm.get_cmap('hsv', len(class_names) if class_names else 20)

    def denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a normalized PyTorch image tensor back to a standard RGB numpy array.
        
        Args:
            image_tensor (torch.Tensor): [3, H, W] normalized image.
            
        Returns:
            np.ndarray: [H, W, 3] image in range [0, 255] uint8.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
        
        img = image_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy [H, W, 3]
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return img_np

    def draw_bounding_boxes(
        self, 
        image_np: np.ndarray, 
        boxes: torch.Tensor, 
        labels: torch.Tensor, 
        scores: torch.Tensor
    ) -> np.ndarray:
        """
        Draws precise bounding boxes on the image.
        
        Args:
            image_np (np.ndarray): [H, W, 3] RGB image.
            boxes (torch.Tensor): [N, 4] absolute coordinates (x1, y1, x2, y2).
            labels (torch.Tensor): [N] class indices.
            scores (torch.Tensor): [N] confidence scores.
            
        Returns:
            np.ndarray: Image with bounding boxes drawn.
        """
        # Create a copy to draw on
        img_draw = image_np.copy()
        
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            label_idx = int(labels[i])
            score = float(scores[i])
            
            # Get color for this class
            color_rgba = self.colors(label_idx)
            color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255)) # OpenCV uses BGR
            color_rgb = (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
            
            # Draw Rectangle
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Prepare Label Text
            class_name = self.class_names[label_idx] if self.class_names else str(label_idx)
            text = f"{class_name}: {score:.2f}"
            
            # Calculate text size and draw background for text readability
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color_rgb, -1)
            
            # Put Text
            cv2.putText(img_draw, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return img_draw

    def generate_attention_heatmap(self, image_np: np.ndarray, feature_map: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
        """
        Generates an attention heatmap by averaging channels of a feature map and overlaying it on the image.
        
        Args:
            image_np (np.ndarray): [H, W, 3] original RGB image.
            feature_map (torch.Tensor): [C, H_f, W_f] feature map (e.g., P2_prime).
            alpha (float): Overlay transparency.
            
        Returns:
            np.ndarray: [H, W, 3] Heatmap overlaid image.
        """
        # Average across channels to get spatial activation [H_f, W_f]
        activation = torch.mean(feature_map, dim=0).cpu().numpy()
        
        # Normalize activation to [0, 1]
        activation = np.maximum(activation, 0) # ReLU-like
        activation = activation / (np.max(activation) + 1e-8)
        
        # Resize to original image size
        H, W = image_np.shape[:2]
        activation_resized = cv2.resize(activation, (W, H))
        
        # Convert to pseudo-color heatmap (cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_resized), cv2.COLORMAP_JET)
        
        # OpenCV colormap is BGR, convert to RGB for matplotlib consistency
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay with original image
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        
        return overlay

    def save_academic_figure(
        self, 
        image_np: np.ndarray, 
        bbox_img: np.ndarray, 
        heatmap_img: np.ndarray, 
        save_path: str = "paper_figure_1.png",
        dpi: int = 300
    ):
        """
        Creates a side-by-side subplot for academic publication.
        
        Args:
            image_np (np.ndarray): Original image (optional use).
            bbox_img (np.ndarray): Image with drawn bounding boxes.
            heatmap_img (np.ndarray): Image with overlaid attention heatmap.
            save_path (str): Output file path.
            dpi (int): Resolution of the saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
        
        axes[0].imshow(bbox_img)
        axes[0].set_title("Detection Results", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_img)
        axes[1].set_title("P2_prime Attention Heatmap", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
        print(f"✅ Academic figure saved to: {save_path}")


if __name__ == '__main__':
    print("=== Testing Academic Visualization ===")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.ws_tfa import WS_TFA_Net
    from inference import predict
    
    # Dummy VOC classes
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    device = torch.device('cpu')
    model = WS_TFA_Net(num_classes=20, pretrained_backbone=False).to(device)
    model.eval()
    
    # 1. Create a dummy test image [1, 3, 800, 800]
    # We create a simple gradient image instead of pure random noise so the heatmap looks slightly structural
    y_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, 800), torch.linspace(0, 1, 800), indexing='ij')
    dummy_img = torch.stack([x_grid, y_grid, (x_grid+y_grid)/2]).unsqueeze(0).to(device)
    # Normalize it roughly to ImageNet stats
    dummy_img = (dummy_img - 0.45) / 0.22
    
    print("Running forward pass with return_features=True...")
    with torch.no_grad():
        outputs = model(dummy_img, return_features=True)
        
    # Get P2_prime feature map for the first image
    p2_prime = outputs['spatial_features']['P2_prime'][0] # [C, H_f, W_f]
    print(f"Extracted P2_prime Shape: {p2_prime.shape}")
    
    # 2. Run standard inference to get boxes
    # Using a negative threshold just to get some boxes for the visualization test
    predictions = predict(model, dummy_img, conf_threshold=-1.0, nms_iou_threshold=0.4)
    pred = predictions[0]
    
    # 3. Visualization
    visualizer = WSODVisualizer(class_names=VOC_CLASSES)
    
    # Denormalize original image
    orig_img_np = visualizer.denormalize_image(dummy_img[0])
    
    # Draw boxes
    bbox_img = visualizer.draw_bounding_boxes(
        orig_img_np, 
        pred['boxes'][:3], # Just take top 3 to avoid clutter in random test
        pred['labels'][:3], 
        pred['scores'][:3]
    )
    
    # Generate Heatmap
    heatmap_img = visualizer.generate_attention_heatmap(orig_img_np, p2_prime, alpha=0.6)
    
    # Save Subplots
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figure_1.png')
    visualizer.save_academic_figure(orig_img_np, bbox_img, heatmap_img, save_path=save_path, dpi=300)
