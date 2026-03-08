"""
Training Script for Full Model with NWD Loss
=============================================

This script trains the full tiny object detection model with:
- P2 Head (stride 4)
- BiFPN Neck
- TAAM (Tiny-Aware Attention Module)
- NWD Loss (Normalized Wasserstein Distance)

The key challenge is integrating NWD loss with YOLO's training pipeline.
This is done by patching the bbox loss computation during training.

Usage:
    python scripts/train_full.py --epochs 50 --batch 8 --data-path "path/to/AI_TOD"
    
Author: Research Implementation
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

# Import custom modules - these will be registered with Ultralytics
from models.taam import TAAM, TAAMBlock
from models.nwd_loss import compute_nwd, NWDLoss, HybridNWDIoULoss


def register_custom_modules():
    """Register custom modules with Ultralytics."""
    import ultralytics.nn.modules as modules
    
    # Register TAAM
    if not hasattr(modules, 'TAAM'):
        modules.TAAM = TAAM
        modules.__all__.append('TAAM')
    
    if not hasattr(modules, 'TAAMBlock'):
        modules.TAAMBlock = TAAMBlock
        modules.__all__.append('TAAMBlock')
    
    # Also add to tasks module for model loading
    try:
        from ultralytics.nn import tasks
        if not hasattr(tasks, 'TAAM'):
            tasks.TAAM = TAAM
        if not hasattr(tasks, 'TAAMBlock'):
            tasks.TAAMBlock = TAAMBlock
    except ImportError:
        pass
    
    print("✓ Registered custom modules: TAAM, TAAMBlock")


def patch_bbox_loss_with_nwd(model, nwd_weight=0.5):
    """
    Patch the model's bbox loss to use hybrid NWD + CIoU loss.
    
    This patches the loss computation during training to incorporate
    NWD loss for better tiny object detection.
    
    Args:
        model: YOLO model instance
        nwd_weight: Weight for NWD loss (0-1), rest goes to CIoU
    """
    import torch
    from ultralytics.utils.loss import BboxLoss
    
    # Store original forward method
    original_forward = BboxLoss.forward
    
    def hybrid_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, 
                       target_scores, target_scores_sum, fg_mask):
        """Hybrid NWD + CIoU loss for bounding box regression."""
        # Get original CIoU loss
        loss_iou, iou = original_forward(
            self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask
        )
        
        # Compute NWD loss on foreground boxes
        if fg_mask.sum() > 0:
            # Get foreground predictions and targets
            pred_fg = pred_bboxes[fg_mask]
            target_fg = target_bboxes[fg_mask]
            
            # Convert to cxcywh format if needed (YOLO uses xyxy internally)
            # pred_bboxes are already decoded to xyxy
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
            
            # Compute NWD
            nwd = compute_nwd(pred_cxcywh, target_cxcywh, C=12.0)
            nwd_loss = 1.0 - nwd
            
            # Weight by target scores (quality)
            weight = target_scores[fg_mask].sum(-1)
            nwd_loss = (nwd_loss * weight).sum() / target_scores_sum
            
            # Hybrid loss
            loss_hybrid = (1 - nwd_weight) * loss_iou + nwd_weight * nwd_loss
            
            return loss_hybrid, iou
        
        return loss_iou, iou
    
    # Apply patch
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


def train_full_model(args):
    """Train the full model with all components."""
    
    print_section("Full Model Training: P2 + BiFPN + TAAM + NWD")
    
    # Set seeds
    set_all_seeds(args.seed)
    
    # Get paths
    repo_root = get_repo_root()
    
    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = get_dataset_path()
    
    # Validate dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Validate label format
    print("\nValidating label format...")
    label_check = validate_label_format(dataset_path)
    
    if not label_check['valid']:
        print("\n" + "=" * 60)
        print("ERROR: LABEL FORMAT VALIDATION FAILED!")
        print("=" * 60)
        print(label_check['error_message'])
        sys.exit(1)
    else:
        print(f"✓ Labels validated ({label_check['detection_count']} samples)")
    
    # Clear cache
    print("\nClearing YOLO cache files...")
    clear_yolo_cache(dataset_path)
    
    # Verify train/val directories
    train_images_dir = dataset_path / "train" / "images"
    val_images_dir = dataset_path / "val" / "images"
    
    if not train_images_dir.exists() or not val_images_dir.exists():
        print("ERROR: Train/val directories not found.")
        print("Run 'python scripts/split_train_val.py' first.")
        sys.exit(1)
    
    train_count = len(list(train_images_dir.glob("*")))
    val_count = len(list(val_images_dir.glob("*")))
    
    print(f"\nDataset:")
    print(f"  Train images: {train_count}")
    print(f"  Val images:   {val_count}")
    
    # Setup output directory
    timestamp = get_timestamp()
    output_dir = repo_root / "results" / "full" / f"full_{timestamp}"
    ensure_dir(output_dir)
    
    print(f"\nOutput: {output_dir}")
    
    # Create data config
    data_config = create_data_config(dataset_path, output_dir)
    
    # Save experiment config
    exp_config = {
        'experiment': 'full',
        'components': ['P2', 'BiFPN', 'TAAM', 'NWD'],
        'timestamp': timestamp,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'nwd_weight': args.nwd_weight,
        'model_yaml': 'models/yolo11m_full.yaml',
        'dataset_path': str(dataset_path),
        'output_dir': str(output_dir)
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Register custom modules BEFORE loading YOLO
    register_custom_modules()
    
    # Import and load YOLO
    from ultralytics import YOLO
    
    # Load custom model
    model_yaml = repo_root / "models" / "yolo11m_full.yaml"
    if not model_yaml.exists():
        print(f"Error: Model YAML not found: {model_yaml}")
        sys.exit(1)
    
    print(f"\nLoading model: {model_yaml}")
    model = YOLO(str(model_yaml))
    
    # Patch bbox loss with NWD
    if args.nwd_weight > 0:
        patch_bbox_loss_with_nwd(model, nwd_weight=args.nwd_weight)
    
    # Training arguments - USE YOLO DEFAULTS
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
        'pretrained': True,
        'close_mosaic': 10,
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  NWD weight: {args.nwd_weight}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    
    # Start training
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
        
        # Copy results
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
        print(f"FULL MODEL TRAINING COMPLETE")
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
            f.write(f"Experiment: full\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train full tiny object detection model'
    )
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
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
                        help='Weight for NWD loss (0-1, default: 0.5)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_full_model(args)
