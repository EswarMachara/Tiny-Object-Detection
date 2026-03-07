# AI-TOD Tiny Object Detection Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **fully reproducible** baseline training pipeline for the AI-TOD (Aerial Images for Tiny Object Detection) dataset using YOLOv11.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate deterministic splits
python scripts/generate_splits.py

# Train YOLOv11M baseline
python scripts/train_baseline.py --model yolo11m.pt --epochs 120

# Evaluate on test set
python scripts/evaluate_baseline.py --weights results/baseline/<run_name>/best.pt
```

## Reproducibility

This pipeline ensures **exact reproducibility** across different machines:

| Component | Seed | Guarantee |
|-----------|------|-----------|
| Data splits | 42 | SHA-256 hash verification |
| PyTorch ops | 42 | `torch.backends.cudnn.deterministic=True` |
| Python random | 42 | `random.seed(42)` |
| NumPy | 42 | `np.random.seed(42)` |

### Verifying Reproducibility

```bash
# Generate splits and check hash
python scripts/generate_splits.py --verify

# Expected hash (first 64 chars):
# The hash will be consistent across all machines with the same dataset
```

## Dataset Overview

**AI-TOD** contains aerial images with extremely small objects:

| Statistic | Value |
|-----------|-------|
| Total images | 27,955 |
| Training folder | 19,462 images |
| Test folder | 8,493 images |
| Total objects | 697,586 |
| Tiny objects (<16×16) | 85.25% |
| Objects <32×32 | 98.05% |
| Classes | 8 |

### Class Distribution

| Class | Percentage | Count |
|-------|------------|-------|
| vehicle | 87.82% | 612,678 |
| ship | 5.01% | 34,964 |
| person | 4.67% | 32,545 |
| storage-tank | 1.92% | 13,414 |
| airplane | 0.22% | 1,530 |
| bridge | 0.19% | 1,355 |
| swimming-pool | 0.09% | 618 |
| wind-mill | 0.08% | 534 |

## Project Structure

```
TOD_Mini_Project/
├── AI_TOD/                      # Dataset (not in repo)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── configs/
│   ├── ai_tod.yaml              # YOLO dataset config
│   └── data_splits.json         # Deterministic splits (generated)
├── scripts/
│   ├── generate_splits.py       # Create reproducible splits
│   ├── train_baseline.py        # Training script
│   ├── evaluate_baseline.py     # Test evaluation script
│   └── utils.py                 # Shared utilities
├── notebooks/
│   └── kaggle_baseline_training.ipynb  # Complete Kaggle notebook
├── results/
│   └── baseline/                # Training outputs
├── docs/
│   ├── dataset_audit.md         # Dataset analysis report
│   └── baseline_analysis.md     # Baseline experiment analysis
├── requirements.txt
└── README.md
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `imgsz` | 800 | Preserve original resolution |
| `epochs` | 120 | Allow sufficient convergence |
| `batch` | 16 | Balance memory/throughput |
| `optimizer` | AdamW | Better for fine-tuning |
| `lr0` | 0.001 | Conservative for pretrained |
| `patience` | 30 | Prevent premature stopping |

### Augmentation Settings

Optimized for tiny objects:
- **Mosaic**: Enabled (1.0)
- **Mixup**: Disabled (can harm tiny objects)
- **Rotation**: Disabled (preserve small shapes)
- **Scale**: Moderate (0.5)
- **Copy-paste**: Disabled

## Usage

### 1. Generate Splits

```bash
python scripts/generate_splits.py
```

This creates `configs/data_splits.json` with:
- 90% of train folder → training set
- 10% of train folder → validation set  
- 100% of test folder → test set (untouched)

### 2. Train Models

```bash
# YOLOv11M (recommended baseline)
python scripts/train_baseline.py --model yolo11m.pt --epochs 120

# YOLOv11L (larger model)
python scripts/train_baseline.py --model yolo11l.pt --epochs 120 --batch 8

# YOLOv11S (faster training)
python scripts/train_baseline.py --model yolo11s.pt --epochs 120 --batch 32

# Custom configuration
python scripts/train_baseline.py \
    --model yolo11m.pt \
    --epochs 150 \
    --batch 8 \
    --imgsz 1280 \
    --lr0 0.0005
```

### 3. Evaluate on Test Set

```bash
python scripts/evaluate_baseline.py --weights results/baseline/<run>/best.pt
```

Output includes:
- `test_results.json` - Detailed metrics
- `test_metrics.csv` - Per-class breakdown
- Confusion matrix and PR curves

### 4. Kaggle Notebook

The notebook `notebooks/kaggle_baseline_training.ipynb` provides:
- Self-contained, single "Run All" execution
- Automatic dataset path detection
- Complete train → evaluate → visualize pipeline

## Expected Results

Based on previous experiments:

| Model | Val mAP50 | Val mAP50-95 | Parameters |
|-------|-----------|--------------|------------|
| YOLOv11M | ~53% | ~24% | 22.4M |
| YOLOv11L | ~53% | ~24% | 27.6M |

**Note**: These are validation metrics. Test set performance may differ.

## Key Findings

### Tiny Object Detection Challenges

1. **Feature map resolution**: At stride 8, average object is only 1.4 pixels
2. **Class imbalance**: 1149× ratio between largest and smallest class
3. **Scale distribution**: 85% of objects are <16×16 pixels

### Recommendations for Improvement

1. Add P2 detection head (stride 4) for finer detection
2. Use higher input resolution (1280×1280)
3. Apply class-balanced sampling or focal loss
4. Consider specialized tiny object architectures

## Documentation

- [Dataset Audit Report](docs/dataset_audit.md) - Comprehensive dataset analysis
- [Baseline Analysis Report](docs/baseline_analysis.md) - Previous experiment insights

## Citation

If you use this pipeline, please cite the AI-TOD dataset:

```bibtex
@inproceedings{wang2021aitod,
  title={Tiny Object Detection in Aerial Images},
  author={Wang, Jinwang and Yang, Wen and Guo, Haowen and Zhang, Ruixiang and Xia, Gui-Song},
  booktitle={ICPR},
  year={2021}
}
```

## License

This project is licensed under the MIT License.
