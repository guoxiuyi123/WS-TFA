# 🎯 PRM-DETR: Point-to-Region Mamba-DETR for Weakly Supervised Small Object Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success.svg?style=for-the-badge)

This is the official PyTorch implementation of **PRM-DETR**, a cutting-edge framework designed specifically for **Weakly Supervised Small Object Detection (WSSOD)** using only **Point Annotations**.

## 📖 Abstract / Introduction
Detecting extremely small objects is notoriously challenging, and the problem is further exacerbated when only point annotations (without bounding box width and height) are available. Traditional bounding box-based regressors and IoU metrics fail in this weakly supervised scenario.

**PRM-DETR** introduces a fully decoupled and innovative architecture to tackle these challenges:
1. **Mamba-driven High-Resolution Processing**: We utilize a Mamba encoder to efficiently model global dependencies on the high-resolution P2 feature map, preventing the typical memory explosion while capturing crucial fine-grained details of small objects.
2. **Dynamic Boundary Perception**: By integrating `DynamicSnakeConv`, the network dynamically perceives irregular boundaries of targets directly from point priors.
3. **KAN-enhanced Classification**: We replace the traditional MLP classification head with a Kolmogorov-Arnold Legendre Network (`KALNConv`), leveraging its superior high-order non-linear fitting capabilities to distinguish extremely weak small object features from massive backgrounds.
4. **Point-based Hungarian Matching**: We completely rewrite the DETR matching mechanism, abandoning IoU and L1 box costs in favor of a pure **L2 Distance Cost** combined with Focal Loss, ensuring stable bipartite matching solely from point supervision.

## 🚀 Highlights (Contributions)
- 🐍 **Hybrid Mamba-Snake Encoder**: A novel neck design combining `MobileMamba` for global context and `DynamicSnakeConv` for dynamic region expansion.
- 🧠 **KAN Decoder Head**: Integrating high-order Legendre polynomial networks (`KALNConv`) into the transformer decoder for robust weak-feature classification.
- 🎯 **Pure Point-Matching Engine**: A customized Hungarian Matcher and Criterion tailored for point annotations, discarding all box-dependent constraints.
- 🧩 **Point-Level Data Augmentation**: Innovative `PointCopyPaste` and `PointMosaic` techniques designed to enrich small object samples and stabilize early-stage matching.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PRM-DETR.git
cd PRM-DETR

# Create a virtual environment (optional but recommended)
conda create -n prm_detr python=3.10
conda activate prm_detr

# Install dependencies
pip install torch torchvision torchaudio
pip install scipy numpy opencv-python einops PyWavelets
```
*(Note: Mamba and Triton dependencies are required for optimal performance on GPU. The code currently falls back to native PyTorch implementations if Triton is not available.)*

## ⚡ Quick Start

You can instantly experience the architecture and verify the entire training pipeline (Forward, Point Matching, Loss Calculation, and Point AP Evaluation) using our mock data script.

```bash
python tools/train.py
```

Expected output:
```text
Using device: cuda (or cpu)
Starting training loop...
Epoch [1/5] | loss_ce: 3.4620 | loss_point: 0.2123 | Total Loss: 3.6743
Running validation...
====== Point-based Evaluation ======
  AP@0.01px : 0.0000
  AP@0.05px : 0.0018
  AP@0.1px : 0.0562
  mAP     : 0.0193
====================================
...
Training loop finished successfully!
```

## 📁 Project Structure

PRM-DETR features an extremely clean, highly decoupled architecture:

```text
PRM-DETR/
├── configs/                 # YAML configuration files
├── dataset/                 # Dataset loaders and Point-Level Augmentations
│   ├── point_dataset.py     # PointSupervisedDataset
│   └── transforms.py        # PointCopyPaste & PointMosaic
├── engine/                  # Core training mechanisms
│   ├── criterion.py         # PointCriterion (Focal + Smooth L1 Distance)
│   └── matcher.py           # PointMatcher (L2 Distance Bipartite Matching)
├── models/                  # Network architecture
│   ├── backbone/            # Multi-scale feature extractors
│   ├── head/                # PRMDecoder with KAN Classification Head
│   ├── neck/                # HybridEncoderPRM
│   ├── ops/                 # Decoupled Core Operators
│   │   ├── kan/             # KALNConv (Kolmogorov-Arnold Networks)
│   │   ├── mamba/           # MobileMambaBlock
│   │   └── snake_conv/      # DynamicSnakeConv
│   └── prm_detr.py          # Top-level PRMDETR entry point
├── tools/                   # Execution scripts
│   └── train.py             # Main training loop
└── utils/                   # Utilities
    └── metrics.py           # PointEvaluator (Point AP / Location Recall)
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
