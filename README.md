# CV-Based Food Waste Estimation

**⚠️ PATENT PENDING - PROPRIETARY PROJECT**

## Overview
Computer vision system for food detection and classification using deep learning.

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- Detectron2

## Installation

1. **Create virtual environment:**
```bash
python -m venv foodenv
foodenv\Scripts\activate
```

2. **Install PyTorch with CUDA:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Install Detectron2:**
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

4. **Install other dependencies:**
```bash
pip install opencv-python matplotlib numpy==1.23.5
```

## Project Structure
```
├── run_detection.py          # Main detection script
├── test.ipynb                # Testing notebook
├── food_type_model/          # Food classification model (not included)
│   ├── config.yaml
│   └── metadata.json
└── shape_model/              # Shape detection model (not included)
    ├── config.yaml
    └── metadata.json
```

## Usage

**Note:** Model weights and datasets are not included in this repository due to proprietary nature.

```bash
python run_detection.py
```

## Hardware Requirements
- NVIDIA RTX 4070 or equivalent
- 12GB+ VRAM
- CUDA 11.8+

## License
**PROPRIETARY - All Rights Reserved**

This project is patent pending. No part of this code may be reproduced, distributed, or transmitted without explicit permission.

## Contact
For inquiries: [Your Contact Info]
