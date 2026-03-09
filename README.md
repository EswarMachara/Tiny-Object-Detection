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

# Split train into train/val physically
python scripts/split_train_val.py

# Train baseline
python scripts/train_baseline.py --model yolo11m.pt --epochs 50

# Train TAAM+NWD
python scripts/train_taam.py --epochs 50

# Evaluate on test set
python scripts/evaluate_baseline.py --weights results/taam/<run_name>/best.pt

# SAHI evaluation
python scripts/evaluate_sahi.py --weights results/taam/<run_name>/best.pt --slice-size 400 --compare-standard
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
Tiny-Object-Detection/
├── AI_TOD/                        # Dataset
│   ├── train/images/ & labels/    # Training set (17,516 images)
│   ├── val/images/ & labels/      # Validation set (1,946 images)
│   ├── test/images/ & labels/     # Test set (8,493 images)
│   └── data.yaml                  # Ultralytics dataset config
├── configs/
│   ├── ai_tod.yaml                # YOLO dataset config (documented)
│   └── data_splits.json           # Deterministic train/val/test splits
├── models/
│   ├── yolo11m_taam.yaml          # TAAM architecture definition
│   ├── taam.py                    # TAAM module implementation
│   └── nwd_loss.py                # NWD loss implementation
├── scripts/
│   ├── train_baseline.py          # Baseline YOLOv11m training
│   ├── train_taam.py              # TAAM+NWD training with weight transfer
│   ├── evaluate_baseline.py       # Test evaluation (model.val())
│   ├── evaluate_sahi.py           # SAHI sliced inference evaluation
│   ├── generate_splits.py         # Create reproducible splits
│   ├── split_train_val.py         # Physical train/val split
│   ├── convert_seg_to_det.py      # Segmentation → detection label conversion
│   └── utils.py                   # Shared utilities
├── results/
│   ├── baseline/                  # Baseline training + test eval
│   ├── taam/                      # TAAM+NWD training + test eval
│   └── sahi/                      # SAHI evaluation results
├── yolo11m.pt                     # Pretrained COCO weights
├── EXPERIMENT_REPORT.md           # Detailed experiment report
├── requirements.txt
└── README.md
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `imgsz` | 800 | Preserve original resolution |
| `epochs` | 50 | Sufficient for convergence |
| `batch` | 8 | RTX A5000 24GB (nbs=64 gradient accumulation) |
| `optimizer` | AdamW | Better for fine-tuning |
| `lr0` | 0.0005 | Conservative for pretrained |
| `patience` | 10 | Early stopping |

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
# Baseline YOLOv11M
python scripts/train_baseline.py --model yolo11m.pt --epochs 50

# TAAM+NWD (Tiny-Aware Attention + Normalized Wasserstein Distance)
python scripts/train_taam.py --epochs 50
```

### 3. Evaluate on Test Set

```bash
# Standard evaluation
python scripts/evaluate_baseline.py --weights results/taam/<run>/best.pt

# SAHI sliced inference evaluation
python scripts/evaluate_sahi.py \
    --weights results/taam/<run>/best.pt \
    --slice-size 400 --overlap 0.2 --compare-standard
```

Output includes:
- `test_results.json` — Detailed metrics
- `test_metrics.csv` — Per-class breakdown
- Confusion matrix and PR curves

## Results

### Test Set Performance

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| YOLOv11m Baseline | 60.07% | 27.33% | 65.03% | 58.95% |
| YOLOv11m + TAAM + NWD | 60.98% | 27.99% | 62.73% | 59.71% |

### With SAHI (400×400 slices)

| Model | mAP50 | Recall | Δ mAP50 |
|-------|-------|--------|----------|
| Baseline + SAHI | 59.02% | 89.01% | +0.18% |
| TAAM+NWD + SAHI | 60.22% | 89.07% | +0.98% |

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for detailed analysis.

## Key Findings

### Tiny Object Detection Challenges

1. **Feature map resolution**: At stride 8, average object is only 1.4 pixels
2. **Class imbalance**: 1149× ratio between largest and smallest class
3. **Scale distribution**: 85% of objects are <16×16 pixels

### Recommendations for Improvement

1. Use higher input resolution (1280×1280)
2. Apply class-balanced sampling or focal loss
3. Use SAHI for offline/batch processing to boost recall
4. Tune NWD weight (α) — only α=0.5 was tested

## Documentation

- [Experiment Report](EXPERIMENT_REPORT.md) — Detailed report covering all experiments, issues, and fixes

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
