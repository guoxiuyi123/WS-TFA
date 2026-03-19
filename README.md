# WS-TFA: Weakly Supervised Transformer-Fusion Attention for Tiny Object Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> Official PyTorch implementation of the paper **WS-TFA: Weakly Supervised Transformer-Fusion Attention for Tiny Object Detection**.

## Abstract
Weakly Supervised Object Detection (WSOD) relies solely on image-level labels to train object detectors. Traditional approaches suffer severely from the **"Local Dominance"** problem—where networks focus only on the most discriminative parts of an object (e.g., a dog's head) rather than its full extent—and **"Gradient Submergence"**, which causes tiny objects to be ignored during feature propagation.

To address these challenges, we introduce **WS-TFA** (Weakly Supervised Transformer-Fusion Attention). Our framework seamlessly integrates a dynamic attention-based feature pyramid with a sparse Multiple Instance Learning (MIL) head, achieving state-of-the-art performance, especially on tiny objects.

## Architecture

![Architecture](docs/architecture.png)

The WS-TFA architecture consists of three core components:

1. **Feature Supplement Module (FSM)**:
   Extracts high-resolution geometric details from shallow layers (C1) using dilated convolutions and adaptively supplements them into semantically rich deep layers to prevent tiny object feature loss.

2. **Dynamic Attention FPN**:
   Replaces the traditional 1:1 addition in FPNs. It learns a dynamic spatial weight factor $\alpha$ to adaptively fuse deep semantic features with shallow detailed features.

3. **Sparse MIL Head**:
   Combines a Class-Agnostic DETR-like proposal generator with a novel `Sparsemax` MIL classifier. By enforcing sparsity across object queries, it aggressively suppresses low-confidence noisy proposals and effectively mitigates the local dominance issue.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/WS-TFA.git
cd WS-TFA

# Create a virtual environment (optional but recommended)
conda create -n wstfa python=3.9 -y
conda activate wstfa

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Model Connectivity (Dummy Training)
We provide a dummy training script to verify the forward pass, backward pass, and the Pseudo-Label Mining box regression loss mechanism.

```bash
python train_dummy.py
```
*Expected Output:* You should see the loss decreasing and a confirmation that the computation graph is fully connected (no zero gradients in the Box Head after the Warm-up epochs).

### 2. Generate Academic Visualizations
To see how WS-TFA overcomes local dominance, you can extract the `P2_prime` attention heatmaps overlaid on the original image.

```bash
python visualize.py
```
*Expected Output:* A high-resolution figure (`paper_figure_1.png`) will be saved in the root directory, showcasing both the final bounding boxes and the extracted spatial attention heatmap.

### 3. Data Pipeline Test
To verify the WSOD data augmentations (specifically the `CoarseDropout`/Cutout mechanism that forces the network to learn global representations):

```bash
python test_dataloader.py
```

## TODO / Roadmap

- [x] Core Architecture (Backbone, FSM, Dynamic FPN)
- [x] Sparse MIL Detection Head
- [x] Pseudo-Label Mining & Box Regression Loss
- [x] Academic Visualization Pipeline
- [ ] Integrate PASCAL VOC 2007 dataset loader
- [ ] Add mAP evaluation script
- [ ] Release pre-trained weights

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{wstfa2026,
  title={WS-TFA: Weakly Supervised Transformer-Fusion Attention for Tiny Object Detection},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
