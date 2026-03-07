#!/usr/bin/env python3
"""
YOLOv11 Baseline Training Script for AI-TOD Dataset

This script trains YOLOv11 models for tiny object detection using
deterministic splits for reproducibility.

Features:
- Deterministic training with seed=42
- Uses pre-defined splits from configs/data_splits.json
- Supports both detection and segmentation modes
- Configurable hyperparameters via command line
- Automatic logging and checkpointing

Usage:
    python scripts/train_baseline.py --model yolo11m.pt --epochs 120
    python scripts/train_baseline.py --model yolo11l.pt --epochs 120 --batch 8
    python scripts/train_baseline.py --model yolo11m-seg.pt --mode segment

Environment Variables:
    YOLO_TOD_SEED: Override default seed (default: 42)
    YOLO_TOD_DATA: Override dataset path
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_all_seeds, 
    get_repo_root, 
    get_dataset_path, 
    load_splits,
    create_subset_txt,
    print_section,
    format_time,
    get_timestamp,
    ensure_dir
)


def create_training_config(
    dataset_path: Path,
    splits: dict,
    output_dir: Path,
    class_names: list
) -> Path:
    """
    Create a temporary YOLO config with correct paths.
    
    Args:
        dataset_path: Path to AI-TOD dataset
        splits: Dictionary with train/val/test splits
        output_dir: Output directory for temp files
        class_names: List of class names
        
    Returns:
        Path to created YAML config
    """
    import yaml
    
    # Create subset text files for train and val
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    
    create_subset_txt(splits['train'], dataset_path, train_txt)
    create_subset_txt(splits['val'], dataset_path, val_txt)
    
    # Create YOLO config
    config = {
        'path': str(dataset_path.resolve()),
        'train': str(train_txt.resolve()),
        'val': str(val_txt.resolve()),
        'test': str(dataset_path / "test" / "images"),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    config_path = output_dir / "train_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created training config: {config_path}")
    return config_path


def train_model(args):
    """Main training function."""
    
    print_section("AI-TOD YOLOv11 Baseline Training")
    
    # Set seeds for reproducibility
    seed = int(os.environ.get('YOLO_TOD_SEED', args.seed))
    set_all_seeds(seed)
    
    # Get paths
    repo_root = get_repo_root()
    
    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = get_dataset_path()
    
    print(f"Repository root: {repo_root}")
    print(f"Dataset path: {dataset_path}")
    
    # Load splits
    splits_path = repo_root / "configs" / "data_splits.json"
    splits = load_splits(splits_path)
    
    # Verify split integrity
    with open(splits_path, 'r') as f:
        split_data = json.load(f)
    
    print(f"\nSplit configuration:")
    print(f"  Seed: {split_data['metadata']['seed']}")
    print(f"  Created: {split_data['metadata']['created']}")
    print(f"  Hash: {split_data['metadata']['hash'][:16]}...")
    
    # Class names from AI-TOD
    class_names = [
        'airplane', 'bridge', 'person', 'ship', 
        'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
    ]
    
    # Setup output directory
    timestamp = get_timestamp()
    model_name = Path(args.model).stem
    run_name = f"{model_name}_{timestamp}"
    output_dir = repo_root / "results" / "baseline" / run_name
    ensure_dir(output_dir)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create training config
    config_path = create_training_config(
        dataset_path, splits, output_dir, class_names
    )
    
    # Save training parameters
    train_params = {
        'model': args.model,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'patience': args.patience,
        'seed': seed,
        'mode': args.mode,
        'device': args.device,
        'timestamp': timestamp,
        'dataset_path': str(dataset_path),
        'splits_file': str(splits_path),
        'train_images': len(splits['train']),
        'val_images': len(splits['val'])
    }
    
    params_path = output_dir / "train_params.json"
    with open(params_path, 'w') as f:
        json.dump(train_params, f, indent=2)
    
    print(f"\nTraining Parameters:")
    for k, v in train_params.items():
        if k not in ['dataset_path', 'splits_file']:
            print(f"  {k}: {v}")
    
    # Import ultralytics
    print("\nLoading YOLO model...")
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(args.model)
    
    # Training arguments
    train_args = {
        'data': str(config_path),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'patience': args.patience,
        'seed': seed,
        'device': args.device,
        'project': str(output_dir),
        'name': 'train',
        'exist_ok': True,
        'verbose': True,
        'deterministic': True,
        'pretrained': True,
        # Additional augmentation settings for tiny objects
        'mosaic': 1.0,  # Enable mosaic augmentation
        'mixup': 0.0,   # Disable mixup (can harm tiny objects)
        'copy_paste': 0.0,  # Disable copy-paste
        'degrees': 0.0,  # No rotation (preserve tiny object shapes)
        'scale': 0.5,    # Moderate scaling
        'fliplr': 0.5,   # Horizontal flip
        'flipud': 0.0,   # No vertical flip
        'hsv_h': 0.015,  # Hue augmentation
        'hsv_s': 0.7,    # Saturation augmentation
        'hsv_v': 0.4,    # Value augmentation
    }
    
    # Add warmup settings
    train_args['warmup_epochs'] = 3.0
    train_args['warmup_bias_lr'] = 0.1
    train_args['warmup_momentum'] = 0.8
    
    # Add weight decay
    train_args['weight_decay'] = 0.0005
    
    # Start training
    print_section("Starting Training")
    import time
    start_time = time.time()
    
    try:
        results = model.train(**train_args)
        
        elapsed = time.time() - start_time
        print_section(f"Training Completed in {format_time(elapsed)}")
        
        # Copy best weights to main output directory
        train_output = output_dir / "train"
        best_weights = train_output / "weights" / "best.pt"
        last_weights = train_output / "weights" / "last.pt"
        
        if best_weights.exists():
            shutil.copy(best_weights, output_dir / "best.pt")
            print(f"Best weights saved to: {output_dir / 'best.pt'}")
        
        if last_weights.exists():
            shutil.copy(last_weights, output_dir / "last.pt")
        
        # Copy results CSV
        results_csv = train_output / "results.csv"
        if results_csv.exists():
            shutil.copy(results_csv, output_dir / "results.csv")
            print(f"Results saved to: {output_dir / 'results.csv'}")
        
        # Save final summary
        summary = {
            'status': 'completed',
            'elapsed_time': elapsed,
            'elapsed_formatted': format_time(elapsed),
            'best_weights': str(output_dir / 'best.pt'),
            'results_csv': str(output_dir / 'results.csv'),
        }
        
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {output_dir / 'training_summary.json'}")
        print(f"\n{'='*60}")
        print(f"To evaluate on test set, run:")
        print(f"  python scripts/evaluate_baseline.py --weights {output_dir / 'best.pt'}")
        print(f"{'='*60}")
        
        return output_dir
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nTraining failed after {format_time(elapsed)}")
        print(f"Error: {e}")
        
        # Save error log
        with open(output_dir / "error.log", 'w') as f:
            f.write(f"Training failed at {datetime.now()}\n")
            f.write(f"Error: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv11 baseline on AI-TOD dataset'
    )
    
    # Model configuration
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolo11m.pt',
        help='YOLO model to train (default: yolo11m.pt)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='detect',
        choices=['detect', 'segment'],
        help='Training mode: detect or segment (default: detect)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=120,
        help='Number of training epochs (default: 120)'
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=800,
        help='Image size (default: 800)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
        help='Optimizer (default: AdamW)'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early stopping patience (default: 30)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (default: 0 for GPU 0, use cpu for CPU)'
    )
    
    # Paths
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Override auto-detected dataset path'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Adjust model name for segmentation mode
    if args.mode == 'segment' and '-seg' not in args.model:
        print(f"Note: Switching to segmentation model variant")
        args.model = args.model.replace('.pt', '-seg.pt')
    
    train_model(args)
