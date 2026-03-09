"""
Training Script: YOLOv11m + TAAM + NWD
=======================================

Architecture: Standard YOLOv11m backbone/neck + TAAM attention + NWD loss
- Backbone (layers 0-10): IDENTICAL to standard YOLOv11m → 100% pretrained
- Top-down FPN (layers 11-16): IDENTICAL → 100% pretrained
- Bottom-up PAN (layers 17-22): IDENTICAL → 100% pretrained
- TAAM (layers 23-25): NEW, but initializes as identity (gamma=0)
- Detect (layer 26): Same architecture as pretrained layer 23 → remap transfer
- NWD Loss: Patched into BboxLoss during training

Expected ~95%+ pretrained weight transfer. Only TAAM params and cls output
layers (nc=80→8) are randomly initialized. cls_loss should start at ~2
(same as baseline).

Usage:
    python scripts/train_taam.py --epochs 50 --batch 8 --data-path "path/to/AI_TOD"
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import utilities
from scripts.utils import (
    set_all_seeds,
    get_repo_root,
    get_dataset_path,
    print_section,
    format_time,
    get_timestamp,
    ensure_dir,
    validate_label_format,
    clear_yolo_cache
)

# Import custom modules
from models.taam import TAAM, TAAMBlock
from models.nwd_loss import compute_nwd, NWDLoss, HybridNWDIoULoss


def register_custom_modules():
    """Register TAAM modules with Ultralytics so parse_model can find them."""
    import ultralytics.nn.modules as modules

    if not hasattr(modules, 'TAAM'):
        modules.TAAM = TAAM
    if not hasattr(modules, 'TAAMBlock'):
        modules.TAAMBlock = TAAMBlock

    try:
        from ultralytics.nn import tasks
        if not hasattr(tasks, 'TAAM'):
            tasks.TAAM = TAAM
        if not hasattr(tasks, 'TAAMBlock'):
            tasks.TAAMBlock = TAAMBlock
    except ImportError:
        pass

    try:
        import ultralytics.nn.tasks as tasks_module
        setattr(tasks_module, 'TAAM', TAAM)
        setattr(tasks_module, 'TAAMBlock', TAAMBlock)
    except ImportError:
        pass

    print("✓ Registered custom modules: TAAM, TAAMBlock")


def transfer_pretrained_weights(custom_model, pretrained_path="yolo11m.pt"):
    """
    Transfer pretrained weights from standard YOLOv11m to TAAM model.

    The TAAM model has IDENTICAL layers 0-22 as standard YOLOv11m.
    Only change: Detect shifts from layer 23 → 26 (TAAM occupies 23-25).

    Mapping:
      pretrained model.{0-22}.* → custom model.{0-22}.*  (direct, same names)
      pretrained model.23.*      → custom model.26.*       (Detect head remap)
      custom model.{23-25}.*     → TAAM layers (no pretrained, identity init)

    Args:
        custom_model: YOLO model loaded from yolo11m_taam.yaml
        pretrained_path: Path to yolo11m.pt

    Returns:
        Number of parameters successfully transferred
    """
    import torch
    from ultralytics import YOLO

    print(f"\nTransferring pretrained weights from: {pretrained_path}")

    pretrained = YOLO(pretrained_path)
    pretrained_state = pretrained.model.state_dict()
    custom_state = custom_model.model.state_dict()

    # --- Phase 1: Direct name-based transfer (layers 0-22) ---
    direct_transferred = 0
    for name, param in pretrained_state.items():
        if name in custom_state and param.shape == custom_state[name].shape:
            custom_state[name].copy_(param)
            direct_transferred += 1

    # --- Phase 2: Remap Detect head (23 → 26) ---
    remap_transferred = 0
    remap_skipped = 0

    for name, param in pretrained_state.items():
        # Skip already transferred
        if name in custom_state:
            continue

        # Remap model.23.* → model.26.*
        if name.startswith('model.23.'):
            remapped = name.replace('model.23.', 'model.26.', 1)
            if remapped in custom_state:
                if param.shape == custom_state[remapped].shape:
                    custom_state[remapped].copy_(param)
                    remap_transferred += 1
                else:
                    remap_skipped += 1
                    # Expected: cv3 final layer (80 classes → 8 classes)
                    print(f"    Shape mismatch (expected): {name} {list(param.shape)} → {remapped} {list(custom_state[remapped].shape)}")

    # Load the updated state dict
    custom_model.model.load_state_dict(custom_state)

    total = direct_transferred + remap_transferred
    total_params = len(custom_state)
    taam_params = sum(1 for k in custom_state if k.startswith('model.23.') or
                      k.startswith('model.24.') or k.startswith('model.25.'))

    print(f"  Direct transfer:   {direct_transferred} params (backbone + FPN + PAN)")
    print(f"  Remap transfer:    {remap_transferred} params (Detect head 23→26)")
    print(f"  Shape mismatch:    {remap_skipped} params (nc=80→8, expected)")
    print(f"  TAAM params:       {taam_params} params (new, identity init)")
    print(f"  Total: {total}/{total_params} ({100*total/total_params:.1f}%)")
    print("✓ Pretrained weights loaded")

    return total


def patch_bbox_loss_with_nwd(model, nwd_weight=0.5):
    """
    Patch BboxLoss to use hybrid NWD + CIoU loss.

    NWD (Normalized Wasserstein Distance) is more suitable for tiny objects
    because it uses Gaussian distribution overlap instead of IoU, which
    degrades rapidly for small boxes.
    """
    import torch
    from ultralytics.utils.loss import BboxLoss

    original_forward = BboxLoss.forward

    def hybrid_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                       target_scores, target_scores_sum, fg_mask, imgsz, stride):
        """Hybrid NWD + CIoU loss for bounding box regression."""
        loss_iou, iou = original_forward(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride
        )

        if fg_mask.sum() > 0:
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]

            # Convert xyxy → cxcywh for NWD
            pred_cx = (pred_fg[:, 0] + pred_fg[:, 2]) / 2
            pred_cy = (pred_fg[:, 1] + pred_fg[:, 3]) / 2
            pred_w = pred_fg[:, 2] - pred_fg[:, 0]
            pred_h = pred_fg[:, 3] - pred_fg[:, 1]
            pred_cxcywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)

            target_cx = (target_fg[:, 0] + target_fg[:, 2]) / 2
            target_cy = (target_fg[:, 1] + target_fg[:, 3]) / 2
            target_w = target_fg[:, 2] - target_fg[:, 0]
            target_h = target_fg[:, 3] - target_fg[:, 1]
            target_cxcywh = torch.stack([target_cx, target_cy, target_w, target_h], dim=1)

            nwd = compute_nwd(pred_cxcywh, target_cxcywh, C=12.0)
            nwd_loss = 1.0 - nwd

            weight = target_scores[fg_mask].sum(-1)
            nwd_loss = (nwd_loss * weight).sum() / target_scores_sum

            loss_hybrid = (1 - nwd_weight) * loss_iou + nwd_weight * nwd_loss
            return loss_hybrid, iou

        return loss_iou, iou

    BboxLoss.forward = hybrid_forward
    print(f"✓ Patched BboxLoss with NWD (weight={nwd_weight})")


def create_data_config(dataset_path: Path, output_dir: Path) -> Path:
    """Create YOLO data configuration file."""
    import yaml

    class_names = [
        'airplane', 'bridge', 'person', 'ship',
        'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
    ]

    config = {
        'path': str(dataset_path.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    config_path = output_dir / "data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def train_taam_model(args):
    """Train YOLOv11m + TAAM + NWD model."""

    print_section("TAAM + NWD Training (Standard YOLOv11m + TAAM Attention + NWD Loss)")

    # Set seeds
    set_all_seeds(args.seed)

    # Get paths
    repo_root = get_repo_root()

    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = get_dataset_path()

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    # Validate labels
    print("\nValidating label format...")
    label_check = validate_label_format(dataset_path)
    if not label_check['valid']:
        print(f"ERROR: {label_check['error_message']}")
        sys.exit(1)
    print(f"✓ Labels validated ({label_check['detection_count']} samples)")

    # Clear cache
    clear_yolo_cache(dataset_path)

    # Verify train/val split
    train_images_dir = dataset_path / "train" / "images"
    val_images_dir = dataset_path / "val" / "images"

    if not train_images_dir.exists() or not val_images_dir.exists():
        print("ERROR: Train/val directories not found.")
        print("Run 'python scripts/split_train_val.py' first.")
        sys.exit(1)

    train_count = len(list(train_images_dir.glob("*")))
    val_count = len(list(val_images_dir.glob("*")))

    print(f"\nDataset:")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")

    # Setup output
    timestamp = get_timestamp()
    output_dir = repo_root / "results" / "taam" / f"taam_{timestamp}"
    ensure_dir(output_dir)
    print(f"\nOutput: {output_dir}")

    # Data config
    data_config = create_data_config(dataset_path, output_dir)

    # Save experiment config
    exp_config = {
        'experiment': 'taam_nwd',
        'components': ['TAAM', 'NWD'],
        'description': 'Standard YOLOv11m + TAAM attention + NWD loss',
        'timestamp': timestamp,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'nwd_weight': args.nwd_weight,
        'model_yaml': 'models/yolo11m_taam.yaml',
        'dataset_path': str(dataset_path),
        'output_dir': str(output_dir)
    }

    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)

    # Register TAAM modules
    register_custom_modules()

    # Load model from TAAM YAML
    from ultralytics import YOLO

    model_yaml = repo_root / "models" / "yolo11m_taam.yaml"
    if not model_yaml.exists():
        print(f"Error: Model YAML not found: {model_yaml}")
        sys.exit(1)

    print(f"\nLoading model: {model_yaml}")
    model = YOLO(str(model_yaml))

    # Transfer pretrained weights
    pretrained_weights = repo_root / "yolo11m.pt"
    if pretrained_weights.exists():
        transfer_pretrained_weights(model, str(pretrained_weights))
        # CRITICAL: Set ckpt to truthy so YOLO.train() passes our modified model
        # to the trainer instead of building a fresh model from YAML (line 771 in model.py:
        # weights=self.model if self.ckpt else None)
        model.ckpt = {"model": model.model}
    else:
        print("Warning: Pretrained weights not found. Training from scratch.")

    # Patch NWD loss
    if args.nwd_weight > 0:
        patch_bbox_loss_with_nwd(model, nwd_weight=args.nwd_weight)

    # Training config — YOLO defaults (pretrained backbone is fully transferred)
    train_args = {
        'data': str(data_config),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'seed': args.seed,
        'device': args.device,
        'project': str(output_dir),
        'name': 'train',
        'exist_ok': True,
        'verbose': True,
        'deterministic': True,
        'close_mosaic': 10,
    }

    print(f"\nTraining Configuration:")
    print(f"  Model:      yolo11m_taam.yaml")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  NWD weight: {args.nwd_weight}")
    print(f"  Patience:   {args.patience}")

    # Train
    print_section("Starting Training")
    import time
    start_time = time.time()

    try:
        results = model.train(**train_args)
        elapsed = time.time() - start_time

        print_section(f"Training Completed in {format_time(elapsed)}")

        # Copy best weights
        train_output = output_dir / "train"
        best_weights = train_output / "weights" / "best.pt"

        if best_weights.exists():
            shutil.copy(best_weights, output_dir / "best.pt")
            print(f"Best weights: {output_dir / 'best.pt'}")

        # Copy results CSV
        results_csv = train_output / "results.csv"
        if results_csv.exists():
            shutil.copy(results_csv, output_dir / "results.csv")

        # Save summary
        summary = {
            'status': 'completed',
            'elapsed_time': elapsed,
            'elapsed_formatted': format_time(elapsed),
            'best_weights': str(output_dir / 'best.pt'),
        }

        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"TAAM + NWD TRAINING COMPLETE")
        print(f"Results: {output_dir}")
        print(f"{'='*60}")

        print(f"\nTo evaluate on test set:")
        print(f"  python scripts/evaluate_baseline.py --weights \"{output_dir / 'best.pt'}\" --data-path \"{dataset_path}\"")

        return output_dir

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nTraining failed after {format_time(elapsed)}")
        print(f"Error: {e}")

        import traceback
        traceback.print_exc()

        with open(output_dir / "error.log", 'w') as f:
            f.write(f"Experiment: taam_nwd\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())

        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train YOLOv11m + TAAM + NWD for tiny object detection'
    )

    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--imgsz', type=int, default=800,
                        help='Image size (default: 800)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (default: 0)')
    parser.add_argument('--data-path', type=str, required=False,
                        help='Path to AI-TOD dataset')
    parser.add_argument('--nwd-weight', type=float, default=0.5,
                        help='NWD loss weight (0-1, default: 0.5)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_taam_model(args)
