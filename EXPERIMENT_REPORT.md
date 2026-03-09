# Experiment Report: YOLOv11m + TAAM + NWD for Tiny Object Detection

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Baseline Experiment](#2-baseline-experiment)
- [3. Full Model Attempt (P2 + BiFPN + TAAM + NWD)](#3-full-model-attempt-p2--bifpn--taam--nwd)
- [4. Issues Encountered (Chronological)](#4-issues-encountered-chronological)
  - [4.1 Dataset Cache Collision](#41-dataset-cache-collision)
  - [4.2 AMP Gradient Overflow](#42-amp-gradient-overflow)
  - [4.3 Custom Module Registration Failure](#43-custom-module-registration-failure)
  - [4.4 NWD Loss Signature Mismatch](#44-nwd-loss-signature-mismatch)
  - [4.5 Out of Memory (OOM)](#45-out-of-memory-oom)
  - [4.6 Extremely High cls_loss (~17–25)](#46-extremely-high-cls_loss-1725)
  - [4.7 Root Cause: P2 Channel Width Destroys Weight Transfer](#47-root-cause-p2-channel-width-destroys-weight-transfer)
  - [4.8 Architecture Pivot — Remove P2, Keep Only TAAM + NWD](#48-architecture-pivot--remove-p2-keep-only-taam--nwd)
  - [4.9 Ultralytics Rebuilding Model from Scratch (cls_loss = 57.99)](#49-ultralytics-rebuilding-model-from-scratch-cls_loss--5799)
  - [4.10 Optimizer Showing "3 Trainable Parameters"](#410-optimizer-showing-3-trainable-parameters)
- [5. Final Fixes Applied](#5-final-fixes-applied)
- [6. Final Model Architecture](#6-final-model-architecture)
  - [6.1 Architecture Overview](#61-architecture-overview)
  - [6.2 TAAM — Tiny-Aware Attention Module](#62-taam--tiny-aware-attention-module)
  - [6.3 NWD — Normalized Wasserstein Distance Loss](#63-nwd--normalized-wasserstein-distance-loss)
  - [6.4 Pretrained Weight Transfer Strategy](#64-pretrained-weight-transfer-strategy)
- [7. Validation Results (2-Epoch Test)](#7-validation-results-2-epoch-test)
- [8. Final Training Configuration](#8-final-training-configuration)
- [9. File Inventory](#9-file-inventory)
- [10. SAHI (Slicing Aided Hyper Inference) Evaluation](#10-sahi-slicing-aided-hyper-inference-evaluation)

---

## 1. Project Overview

**Goal:** Improve tiny object detection on the AI-TOD aerial dataset using YOLOv11m as the backbone, enhanced with two novel components:

1. **TAAM (Tiny-Aware Attention Module)** — A scale-adaptive attention module designed specifically for tiny objects
2. **NWD (Normalized Wasserstein Distance)** — A loss function that replaces IoU-based loss with Gaussian distribution overlap, providing smoother gradients for tiny objects

**Dataset:** AI-TOD (Aerial Images for Tiny Object Detection)
- **8 classes:** airplane, bridge, person, ship, storage-tank, swimming-pool, vehicle, wind-mill
- **85% of objects are smaller than 16×16 pixels**
- **Training split:** 17,516 images | **Validation:** 1,946 images | **Test:** 8,493 images

**Hardware:** NVIDIA RTX A5000 (24 GB VRAM), Python 3.10.12, PyTorch 2.9.1+cu128, Ultralytics 8.4.2

---

## 2. Baseline Experiment

Before implementing the novel components, a baseline was established using stock YOLOv11m:

| Metric | Value |
|--------|-------|
| Model | YOLOv11m (pretrained on COCO) |
| Epochs | 50 |
| Batch Size | 8 |
| Image Size | 800×800 |
| **mAP50 (test)** | **60.07%** |
| **mAP50-95 (test)** | **27.33%** |
| cls_loss (epoch 1) | ~2.0 |

The baseline's cls_loss of ~2.0 at epoch 1 became the critical reference point for validating weight transfer in all subsequent experiments.

---

## 3. Full Model Attempt (P2 + BiFPN + TAAM + NWD)

The original plan was an ambitious architecture with **four** novel components:

1. **P2 Detection Head** — Add a stride-4 detection level for higher-resolution features (4× resolution vs P3)
2. **BiFPN** — Bidirectional Feature Pyramid Network for better multi-scale fusion
3. **TAAM** — Tiny-Aware Attention Module
4. **NWD** — Normalized Wasserstein Distance Loss

**Architecture files created:**
- `models/yolo11m_p2.yaml` — P2 head variant
- `models/yolo11m_full.yaml` — Complete P2 + BiFPN + TAAM architecture
- `scripts/train_full.py` — Training script for the full model

**This approach was ultimately abandoned** due to fundamental weight transfer incompatibilities caused by the P2 head (detailed below in Issues #6 and #7).

---

## 4. Issues Encountered (Chronological)

### 4.1 Dataset Cache Collision

**Symptom:** Training crashed at epoch 2 with all losses becoming 0/NaN.

**Root Cause:** The original AI-TOD dataset had `train` and `val` using the same image folder. Ultralytics caches label data per-folder, so the train cache was used for validation (and vice versa), causing data leakage and eventually corrupted training.

**Fix:** Created `scripts/split_train_val.py` to physically separate train and validation images into distinct directories:
```
AI_TOD/
├── train/images/   (17,516 images)
├── train/labels/
├── val/images/     (1,946 images)
├── val/labels/
└── test/images/    (8,493 images)
```

---

### 4.2 AMP Gradient Overflow

**Symptom:** Training crashed with NaN gradients during early epochs when using custom optimizer settings (low learning rate, custom warmup).

**Root Cause:** Custom optimizer parameters (very low lr0, long warmup) interacted badly with PyTorch AMP's gradient scaling, causing overflow in the custom architecture's randomly-initialized layers.

**Fix:** Reverted to Ultralytics' default optimizer settings (`optimizer='auto'`), which automatically selects `MuSGD` with appropriate learning rate and momentum. Let the framework handle AMP scaling without interference.

---

### 4.3 Custom Module Registration Failure

**Symptom:** `KeyError: 'TAAM'` when Ultralytics tried to parse the custom YAML model definition.

**Root Cause:** Initially tried registering modules using `modules.__all__.append('TAAM')`, but Ultralytics' `parse_model()` looks up module classes by attribute access on the `modules` and `tasks` namespaces, not through `__all__`.

**Fix:** Used `setattr()` to inject TAAM classes directly into the Ultralytics module namespaces:
```python
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks_module
modules.TAAM = TAAM
modules.TAAMBlock = TAAMBlock
setattr(tasks_module, 'TAAM', TAAM)
setattr(tasks_module, 'TAAMBlock', TAAMBlock)
```

---

### 4.4 NWD Loss Signature Mismatch

**Symptom:** `TypeError` when the patched `BboxLoss.forward()` was called during training — wrong number of arguments.

**Root Cause:** Ultralytics' `BboxLoss.forward()` takes **10 positional arguments** (including `self`), including `imgsz` and `stride` parameters that we initially missed in our patched function signature.

**Fix:** Updated the `hybrid_forward()` monkey-patch to match the exact Ultralytics signature:
```python
def hybrid_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                   target_scores, target_scores_sum, fg_mask, imgsz, stride):
```

---

### 4.5 Out of Memory (OOM)

**Symptom:** CUDA OOM at batch=8 with the full P2 + BiFPN + TAAM model.

**Root Cause:** The P2 detection head operates at stride 4, producing feature maps 4× larger than P3. Combined with BiFPN's two rounds of feature fusion, GPU memory exceeded 24 GB at batch=8.

**Fix:** Reduced batch size to 4 for the full model. (This issue became moot when the P2 architecture was abandoned.)

---

### 4.6 Extremely High cls_loss (~17–25)

**Symptom:** With the full P2 + BiFPN + TAAM model, cls_loss started at ~17–25 instead of the expected ~2. This indicated the classification head was essentially randomly initialized — learning from scratch rather than leveraging pretrained features.

**Attempted fixes that DID NOT work:**
1. **Bias initialization** — Checked cls head bias; it was already correctly initialized at −10.6 (matching Ultralytics default of `−log((1−p)/p)` where `p=1/nc`)
2. **Backbone freezing + longer warmup** — Froze backbone for first 5 epochs with extended warmup. cls_loss still ~20+
3. **Custom `initialize_detection_head()` function** — Multiple versions tried to manually set the Detect head bias. No improvement because the problem wasn't the bias — it was the **weights**.

**Diagnostic that revealed the truth:** A detailed parameter-by-parameter comparison showed only **31.7% of pretrained weights transferred** (427 out of 1,345 parameters matched by name and shape).

---

### 4.7 Root Cause: P2 Channel Width Destroys Weight Transfer

**This was the critical breakthrough in the debugging process.**

**The mechanism:**

YOLOv11's `Detect` head has an internal intermediate channel width for the classification branch:
```python
c3 = max(ch[0], min(self.nc, 100))  # cls intermediate channels
```

Where `ch[0]` is the channel count of the **first input level** (P2 in our case, P3 in standard YOLO).

- **Standard YOLOv11m:** First input to Detect is P3 (256 channels) → `c3 = max(256, min(80, 100)) = 256`
- **Our P2 model:** First input to Detect is P2 (128 channels) → `c3 = max(128, min(8, 100)) = 128`

This halved the intermediate width from 256 to 128, making **ALL `cv3` (classification) weights incompatible** in shape, even when names were correctly remapped. Since cv3 has 3 sub-modules per detection level × 4 detection levels = 12 sets of weights, this blocked the transfer of the entire classification head.

Additionally, BiFPN Round 2 layers were entirely new (no pretrained equivalent), meaning random features flowed into the Detect head even for layers that did transfer.

**First attempted fix:** Changed P2 output from 128 → 256 channels in the YAML. This improved transfer to 47.8% (643/1,345 params), but cls_loss remained ~23 because BiFPN Round 2 still injected random features.

**Conclusion:** The P2 + BiFPN architecture was fundamentally incompatible with effective pretrained weight transfer. No amount of weight remapping could fix the problem — the architecture itself had to change.

---

### 4.8 Architecture Pivot — Remove P2, Keep Only TAAM + NWD

**Decision:** Abandon P2 and BiFPN entirely. Keep only the two truly novel components (TAAM + NWD) on top of a **completely standard YOLOv11m architecture**.

**Rationale:**
- P2 was the bottleneck preventing weight transfer — it modified the Detect head's internal dimensions
- BiFPN introduced entirely new layers with no pretrained equivalent
- TAAM can be inserted between the existing FPN/PAN and Detect head **without modifying any existing layers**
- NWD is a loss-level change that doesn't affect architecture at all

**New architecture:** `models/yolo11m_taam.yaml`
- Layers 0–22: **Byte-for-byte identical** to standard YOLOv11m
- Layers 23–25: TAAM modules (new, but initialize as identity via `gamma=0`)
- Layer 26: Standard Detect head (same structure as pretrained layer 23, just renumbered)

---

### 4.9 Ultralytics Rebuilding Model from Scratch (cls_loss = 57.99)

**Symptom:** After creating the TAAM-only architecture and implementing semantic weight remapping (layer 23→26), the first 2-epoch test showed cls_loss = **57.99** — even worse than before. The optimizer log showed only "3 trainable parameters" in certain parameter groups.

**Root Cause:** Deep inside Ultralytics' `YOLO.train()` method (in `model.py`, line ~771):
```python
weights = self.model if self.ckpt else None
```

When loading a model from YAML (not from a `.pt` file), `self.ckpt` is `None`. This means `weights=None`, so the trainer **completely rebuilds the model from the YAML definition**, discarding all our carefully transferred pretrained weights. The model trained from random initialization.

**Fix:** After transferring weights, set `model.ckpt` to a truthy value so Ultralytics passes the existing (weight-loaded) model to the trainer:
```python
transfer_pretrained_weights(model, str(pretrained_weights))
model.ckpt = {"model": model.model}  # Prevents model rebuild
```

---

### 4.10 Optimizer Showing "3 Trainable Parameters"

**Symptom:** Optimizer log showed `3 weight(decay=0.0), 0 weight(decay=0.0005), 0 bias(decay=0.0)` — implying only 3 parameters were trainable.

**Root Cause:** This was a **consequence of Issue 4.9** — when the model was rebuilt from scratch inside the trainer, the optimizer was initialized on the rebuilt model that hadn't been properly set up. The "3 parameters" were likely only the DFL conv weights.

**Fix:** Same as Issue 4.9. Once `model.ckpt` was set correctly, the optimizer properly reported all parameter groups with the full model's parameters.

---

## 5. Final Fixes Applied

A summary of all fixes that made it into the final working implementation:

| # | Issue | Fix | File |
|---|-------|-----|------|
| 1 | Cache collision (train/val same folder) | Physical train/val split via `split_train_val.py` | `scripts/split_train_val.py` |
| 2 | AMP gradient overflow | Use Ultralytics default `optimizer='auto'` | `scripts/train_taam.py` |
| 3 | Custom module registration | `setattr()` injection into `ultralytics.nn.tasks` | `scripts/train_taam.py` → `register_custom_modules()` |
| 4 | NWD loss signature mismatch | Match exact 10-arg `BboxLoss.forward()` signature | `scripts/train_taam.py` → `patch_bbox_loss_with_nwd()` |
| 5 | P2 destroys weight transfer | **Removed P2 and BiFPN entirely** | `models/yolo11m_taam.yaml` (new architecture) |
| 6 | Detect head layer renumbering | Semantic weight remapping: pretrained `model.23.*` → custom `model.26.*` | `scripts/train_taam.py` → `transfer_pretrained_weights()` |
| 7 | Ultralytics rebuilds model from YAML | Set `model.ckpt = {"model": model.model}` after weight transfer | `scripts/train_taam.py` (line ~339) |

---

## 6. Final Model Architecture

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              YOLOv11m + TAAM + NWD              │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  Backbone (layers 0-10)                 │    │
│  │  100% IDENTICAL to standard YOLOv11m    │    │
│  │  100% pretrained weights from COCO      │    │
│  │                                         │    │
│  │  Conv → Conv → C3k2 → Conv → C3k2 →    │    │
│  │  Conv → C3k2 → Conv → C3k2 → SPPF →    │    │
│  │  C2PSA                                  │    │
│  └────────────┬──────────┬─────────────────┘    │
│               │P4        │P5                    │
│  ┌────────────▼──────────▼─────────────────┐    │
│  │  Top-down FPN (layers 11-16)            │    │
│  │  100% IDENTICAL to standard YOLOv11m    │    │
│  │  100% pretrained weights from COCO      │    │
│  │                                         │    │
│  │  Upsample → Concat → C3k2 (P4')        │    │
│  │  Upsample → Concat → C3k2 (P3')        │    │
│  └──────┬────────────────┬─────────────────┘    │
│         │P3'             │P4'                   │
│  ┌──────▼────────────────▼─────────────────┐    │
│  │  Bottom-up PAN (layers 17-22)           │    │
│  │  100% IDENTICAL to standard YOLOv11m    │    │
│  │  100% pretrained weights from COCO      │    │
│  │                                         │    │
│  │  Conv → Concat → C3k2 (P4 out)         │    │
│  │  Conv → Concat → C3k2 (P5 out)         │    │
│  └──┬────────────┬────────────┬────────────┘    │
│     │P3 (256ch)  │P4 (512ch)  │P5 (512ch)      │
│  ┌──▼────────────▼────────────▼────────────┐    │
│  │  ★ TAAM Attention (layers 23-25) [NEW]  │    │
│  │  Identity init (gamma=0)                │    │
│  │  No disruption at training start        │    │
│  │                                         │    │
│  │  TAAM(P3) → P3' (256ch)                │    │
│  │  TAAM(P4) → P4' (512ch)                │    │
│  │  TAAM(P5) → P5' (512ch)                │    │
│  └──┬────────────┬────────────┬────────────┘    │
│     │P3'         │P4'         │P5'              │
│  ┌──▼────────────▼────────────▼────────────┐    │
│  │  Detect Head (layer 26)                 │    │
│  │  Same structure as pretrained layer 23  │    │
│  │  Weights remapped: model.23→model.26    │    │
│  │  Only nc-dependent final layers random  │    │
│  │  (COCO 80 classes → AI-TOD 8 classes)   │    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  ★ NWD Loss [APPLIED DURING TRAINING ONLY]      │
│  Hybrid: 50% NWD + 50% CIoU                    │
│  Monkey-patched into BboxLoss.forward()         │
└─────────────────────────────────────────────────┘
```

**Model Statistics:**
- **Parameters:** 21,532,375
- **GFLOPs:** 72.8
- **Layers:** 325
- **Pretrained coverage:** 643/799 parameters = **80.5%**
  - 528 direct transfer (backbone + FPN + PAN, layers 0-22)
  - 115 remapped transfer (Detect head, 23→26)
  - 6 shape mismatches (nc=80→8, expected)
  - 150 TAAM parameters (new, identity initialized)

### 6.2 TAAM — Tiny-Aware Attention Module

TAAM is a scale-adaptive attention module specifically designed for tiny object detection. It is applied independently to each of the three detection feature levels (P3, P4, P5) before they enter the Detect head.

**Design Principles:**
- Tiny objects have low feature signal-to-noise ratio (SNR)
- Different object sizes need different attention strategies
- Tiny objects are detected by contrast with background, not absolute feature values
- The module must not disrupt pretrained features at initialization (identity init)

**Components (applied sequentially):**

| Component | Purpose | Mechanism |
|-----------|---------|-----------|
| **ScaleEstimator** | Predicts local object scale σ ∈ (0,1) at each spatial position | Depthwise-separable conv → pointwise → sigmoid. Low σ = tiny, high σ = large |
| **ContrastEnhancer** | Enhances object-background differences (inspired by Weber's Law) | `F_enhanced = F + α·(F - F_local_mean)` where α is inversely proportional to predicted scale |
| **GaussianSpatialAttention** | Focuses spatially on object regions | Depthwise conv → channel reduction → scale-refined attention map → sigmoid |
| **ChannelAttention** | Recalibrates channel importance (ECA-style) | Avg+Max pool → shared FC → sigmoid. Certain channels (edge, texture) are more important for tiny objects |

**Critical design choice — Identity initialization:**
```python
self.gamma = nn.Parameter(torch.zeros(1))
# output = identity + gamma * taam_output
# At init: gamma=0, so output = identity (zero disruption)
```

This means at the start of training, TAAM passes features through unchanged, allowing the pretrained Detect head to operate on familiar features. TAAM's contribution is gradually learned.

**Parameter counts per TAAM instance:**
- P3 TAAM (256 channels): 166,981 parameters
- P4 TAAM (512 channels): 653,109 parameters
- P5 TAAM (512 channels): 653,109 parameters
- **Total TAAM overhead: 1,473,199 parameters** (~7.3% of full model)

### 6.3 NWD — Normalized Wasserstein Distance Loss

NWD replaces standard IoU-based loss for bounding box regression, specifically targeting the problem of IoU degradation for tiny objects.

**The Problem with IoU for Tiny Objects:**

A 2-pixel positional error causes dramatically different IoU values depending on object size:
- **8×8 box** → IoU drops by >50%
- **64×64 box** → IoU drops by ~5%

This makes IoU-based gradients extremely noisy and unstable for tiny objects, causing training difficulty.

**NWD Solution:**

Model each bounding box as a 2D Gaussian distribution:
- Mean μ = (cx, cy) — box center
- Covariance Σ = diag((w/4)², (h/4)²) — box extent

Then compute the Wasserstein-2 distance between predicted and GT Gaussians:

$$W_2^2 = \|\mu_1 - \mu_2\|^2 + \|\sigma_1 - \sigma_2\|^2$$

Normalize to (0, 1]:

$$\text{NWD} = \exp\left(-\frac{W_2}{C}\right)$$

where C = 12.0 is a normalizing constant.

**Hybrid Loss (as implemented):**

$$\mathcal{L}_{box} = (1 - \alpha) \cdot \mathcal{L}_{CIoU} + \alpha \cdot \mathcal{L}_{NWD}$$

where α = 0.5 (equal weighting of NWD and CIoU).

**Benefits:**
- Smooth, continuous gradients even for very small boxes
- Less sensitive to minor positional errors
- Provides meaningful signal even when IoU ≈ 0 for tiny objects
- CIoU component retains well-established regression behavior for larger objects

**Implementation:** NWD is applied via monkey-patching `BboxLoss.forward()` at runtime, avoiding any modification to Ultralytics source code. The original CIoU loss is computed first, then NWD is computed on foreground predictions and blended.

### 6.4 Pretrained Weight Transfer Strategy

The weight transfer uses a **two-phase semantic remapping** approach:

**Phase 1 — Direct Name-Based Transfer (layers 0–22):**
```
pretrained model.{0-22}.* → custom model.{0-22}.*
```
These layers are byte-identical between standard YOLOv11m and our TAAM model. All 528 parameters transfer directly by matching names and shapes.

**Phase 2 — Detect Head Remapping (layer 23 → 26):**
```
pretrained model.23.cv2.0.0.* → custom model.26.cv2.0.0.*
pretrained model.23.cv2.0.1.* → custom model.26.cv2.0.1.*
pretrained model.23.cv3.0.0.* → custom model.26.cv3.0.0.*
...
```
The Detect head simply shifted from layer index 23 to 26 (because TAAM occupies 23-25). A string replacement `model.23.` → `model.26.` handles all 115 transferable parameters.

**6 parameters that don't transfer (expected):**
- `model.26.cv3.{0,1,2}.2.weight` — Final 1×1 conv producing class logits: shape `[80, 256, 1, 1]` vs `[8, 256, 1, 1]` (COCO 80 classes → AI-TOD 8 classes)
- `model.26.cv3.{0,1,2}.2.bias` — Corresponding biases: shape `[80]` vs `[8]`

These are initialized by Ultralytics' default (bias = `−log((1−p)/p)` where `p = 5/(nc·640²)`), which is appropriate.

**Critical `model.ckpt` Fix:**

After weight transfer, we set:
```python
model.ckpt = {"model": model.model}
```
This prevents Ultralytics from discarding our weight-loaded model and rebuilding from scratch inside `YOLO.train()`.

---

## 7. Validation Results (2-Epoch Test)

A 2-epoch test run confirmed the experiment is properly set up:

| Metric | TAAM+NWD (Epoch 1) | Baseline (Epoch 1) | Status |
|--------|-------------------|--------------------|----|
| **cls_loss** | **2.37** | ~2.0 | ✅ Matches baseline — weights transferred correctly |
| **box_loss** | **1.31** | ~1.3 | ✅ Normal |
| **dfl_loss** | **1.00** | ~1.0 | ✅ Normal |
| mAP50 (val) | 32.2% | — | After just 1 epoch |
| Precision | 73.2% | — | Reasonable for epoch 1 |
| Recall | 31.3% | — | Will improve with training |
| GPU Memory | 15.7 GB | — | Well within 24 GB budget |

**Key takeaway:** cls_loss = 2.37 confirms the pretrained classification weights are properly loaded and functioning. Compare this to the failed P2 model (cls_loss ~24) or the model.ckpt bug (cls_loss = 57.99).

---

## 8. Final Training Configuration

```bash
python scripts/train_taam.py --epochs 50 --batch 8 --data-path "./AI_TOD"
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model YAML | `models/yolo11m_taam.yaml` | Standard YOLOv11m + TAAM |
| Pretrained | `yolo11m.pt` (COCO) | 80.5% weight transfer |
| Epochs | 50 | Same as baseline |
| Batch size | 8 | Same as baseline |
| Image size | 800×800 | Same as baseline |
| Optimizer | `auto` (MuSGD) | Ultralytics default |
| LR | Auto-determined | MuSGD default: 0.000833 |
| Patience | 10 | Early stopping |
| NWD weight | 0.5 | Equal CIoU + NWD blend |
| Close mosaic | 10 | Last 10 epochs without mosaic |
| Seed | 42 | Reproducibility |
| Deterministic | True | Reproducibility |

---

## 9. File Inventory

### Models & Architecture

| File | Purpose |
|------|---------|
| `models/yolo11m_taam.yaml` | Model architecture — Standard YOLOv11m + TAAM |
| `models/taam.py` | TAAM module (ScaleEstimator, ContrastEnhancer, GaussianSpatialAttention, ChannelAttention) |
| `models/nwd_loss.py` | NWD loss (compute_nwd, NWDLoss, HybridNWDIoULoss) |
| `yolo11m.pt` | Pretrained COCO weights (source for transfer) |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/train_taam.py` | TAAM+NWD training with weight transfer, NWD patching, and all fixes |
| `scripts/train_baseline.py` | Baseline YOLOv11m training script |
| `scripts/evaluate_baseline.py` | Test set evaluation (Ultralytics model.val()) |
| `scripts/evaluate_sahi.py` | SAHI sliced inference evaluation |
| `scripts/split_train_val.py` | Dataset preparation — physical train/val split |
| `scripts/generate_splits.py` | Generate deterministic train/val/test splits JSON |
| `scripts/convert_seg_to_det.py` | Convert segmentation labels to detection format |
| `scripts/utils.py` | Shared utilities (seed setting, path management, label validation) |

### Configs

| File | Purpose |
|------|---------|
| `configs/ai_tod.yaml` | Ultralytics-format dataset config |
| `configs/data_splits.json` | Deterministic train/val/test split definitions |
| `AI_TOD/data.yaml` | Dataset root-level config for Ultralytics |

### Results

| Path | Description |
|------|-------------|
| `results/baseline/yolo11m_2026-03-08_09-29-22/` | Baseline training (mAP50=60.07%) |
| `results/baseline/eval_2026-03-08_14-42-52/` | Baseline test evaluation |
| `results/taam/taam_2026-03-08_18-18-58/` | TAAM+NWD training (mAP50=60.98%) |
| `results/taam/eval_2026-03-09_00-58-31/` | TAAM+NWD test evaluation |
| `results/sahi/taam_slice400/` | SAHI evaluation — TAAM+NWD |
| `results/sahi/baseline_slice400/` | SAHI evaluation — Baseline |

---

## 10. SAHI (Slicing Aided Hyper Inference) Evaluation

### 10.1 Motivation

Standard YOLO inference resizes images to 800×800. At P3 (stride 8), an 8×8 px object maps to a single 1×1 feature pixel — insufficient for reliable detection. SAHI addresses this by slicing images into overlapping patches, running inference on each slice, and merging predictions with NMS.

**Effective resolution gain**: With 400×400 slices, the same 8×8 px object becomes 2×2 feature pixels — 4× more information.

### 10.2 Configuration

| Parameter | Value |
|-----------|-------|
| Slice size | 400×400 px |
| Overlap | 20% |
| Confidence threshold | 0.001 (for mAP computation) |
| Post-process | NMS (IOU ≥ 0.5) |
| SAHI version | 0.11.36 |
| Evaluation | Custom COCO-style all-point AP interpolation |

### 10.3 Results — TAAM+NWD Model

| Metric | Standard | SAHI | Δ |
|--------|----------|------|---|
| **mAP@50** | 59.24% | **60.22%** | **+0.98%** |
| Precision | 18.01% | 9.76% | -8.24% |
| Recall | 80.69% | **89.07%** | **+8.38%** |
| Detections | 946,854 | 1,927,497 | +103% |
| Speed | 60.2 img/s | 4.7 img/s | 12.8× slower |

**Per-class AP@50 (TAAM+NWD):**

| Class | Standard | SAHI | Δ |
|-------|----------|------|---|
| airplane | 80.33% | 82.97% | +2.65% ★ |
| bridge | 51.43% | 53.56% | +2.13% ★ |
| person | 41.30% | 44.36% | +3.07% ★ |
| ship | 77.95% | 74.60% | -3.35% |
| storage-tank | 81.29% | 81.59% | +0.30% |
| swimming-pool | 44.92% | 43.13% | -1.80% |
| vehicle | 72.59% | 77.77% | +5.18% ★ |
| wind-mill | 24.13% | 23.78% | -0.35% |

### 10.4 Results — Baseline Model

| Metric | Standard | SAHI | Δ |
|--------|----------|------|---|
| **mAP@50** | 58.83% | **59.02%** | **+0.18%** |
| Precision | 17.47% | 9.26% | -8.20% |
| Recall | 80.61% | **89.01%** | **+8.40%** |
| Detections | 975,047 | 2,030,015 | +108% |
| Speed | 70.0 img/s | 5.2 img/s | 13.4× slower |

**Per-class AP@50 (Baseline):**

| Class | Standard | SAHI | Δ |
|-------|----------|------|---|
| airplane | 76.83% | 77.02% | +0.19% |
| bridge | 52.82% | 51.80% | -1.02% |
| person | 41.65% | 44.46% | +2.81% ★ |
| ship | 78.04% | 74.33% | -3.71% |
| storage-tank | 81.37% | 80.56% | -0.81% |
| swimming-pool | 41.49% | 40.93% | -0.57% |
| vehicle | 72.40% | 77.47% | +5.07% ★ |
| wind-mill | 26.08% | 25.56% | -0.51% |

### 10.5 Cross-Model Comparison

| Configuration | mAP@50 | Δ from Baseline Standard |
|---------------|--------|--------------------------|
| Baseline (standard) | 58.83% | — |
| Baseline + SAHI | 59.02% | +0.18% |
| TAAM+NWD (standard) | 59.24% | +0.41% |
| **TAAM+NWD + SAHI** | **60.22%** | **+1.39%** |

### 10.6 Key Findings

1. **SAHI boosts recall dramatically** (+8.4%) for both models, confirming tiny objects are missed at full-image resolution.

2. **TAAM+NWD benefits 5× more from SAHI** than baseline (+0.98% vs +0.18% mAP), suggesting the TAAM attention features complement SAHI's higher-resolution slices — the attention-enhanced features are more useful when the detector actually has enough spatial information.

3. **Vehicle and person see largest gains** across both models (+5% vehicle, +3% person), confirming these small, crowded categories benefit most from sliced inference.

4. **Ship degrades with SAHI** (-3.5% for both models) — likely because ships span multiple slices and get fragmented during slicing.

5. **Precision drops with SAHI** (18% → 10%) because the model produces 2× more detections. However, mAP still improves because the *additional* detections include many true positives ranked highly by confidence.

6. **Speed cost is 13×** — SAHI is an inference-time technique, not suitable for real-time applications but valuable for offline analysis of aerial imagery.

---

*Report updated: 8 March 2026*
*Project: Tiny Object Detection on AI-TOD using YOLOv11m + TAAM + NWD*
