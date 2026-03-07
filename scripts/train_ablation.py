#!/usr/bin/env python3
"""
Ablation Study Training Script for AI-TOD Tiny Object Detection

This script runs controlled ablation experiments with:
- Independent output directories (no overwriting)
- Consistent hyperparameters across all experiments
- Automatic logging and result tracking

Experiments:
    1. baseline    - Standard YOLOv11M
    2. p2          - YOLOv11M + P2 head (stride 4)
    3. bifpn       - YOLOv11M + P2 + BiFPN
    4. taam        - YOLOv11M + P2 + BiFPN + TAAM
    5. anchorfree  - YOLOv11M + P2 + BiFPN + TAAM + Anchor-Free
    6. full        - YOLOv11M + P2 + BiFPN + TAAM + Anchor-Free + NWD

Usage:
    python scripts/train_ablation.py --experiment p2 --epochs 50
    python scripts/train_ablation.py --experiment bifpn --epochs 50
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    set_all_seeds,
    get_repo_root,
    load_splits,
    create_subset_txt,
    print_section,
    format_time,
    get_timestamp,
    ensure_dir
)


# Experiment configurations
EXPERIMENTS = {
    'baseline': {
        'name': 'YOLOv11M Baseline',
        'model': 'yolo11m.pt',
        'custom_model': None,
        'description': 'Standard YOLOv11M without modifications'
    },
    'p2': {
        'name': 'YOLOv11M + P2',
        'model': 'yolo11m.pt',  # Pretrained weights
        'custom_model': 'models/yolo11m_p2.yaml',  # Custom architecture
        'description': 'YOLOv11M with P2 detection head (stride 4)'
    },
    'bifpn': {
        'name': 'YOLOv11M + P2 + BiFPN',
        'model': None,  # Will use previous best
        'custom_model': 'models/yolo11m_p2_bifpn.yaml',
        'description': 'Adding BiFPN neck for multi-scale fusion'
    },
    'taam': {
        'name': 'YOLOv11M + P2 + BiFPN + TAAM',
        'model': None,
        'custom_model': 'models/yolo11m_p2_bifpn_taam.yaml',
        'description': 'Adding Tiny-Aware Attention Module'
    },
    'anchorfree': {
        'name': 'YOLOv11M + P2 + BiFPN + TAAM + AnchorFree',
        'model': None,
        'custom_model': 'models/yolo11m_p2_bifpn_taam_af.yaml',
        'description': 'Switching to anchor-free detection head'
    },
    'full': {
        'name': 'Full Proposed Method',
        'model': None,
        'custom_model': 'models/yolo11m_full.yaml',
        'description': 'Complete method with NWD loss'
    }
}


def create_data_config(dataset_path: Path, splits: dict, output_dir: Path) -> Path:
    """Create YOLO data configuration file."""
    import yaml
    
    # Create image list files
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    
    create_subset_txt(splits['train'], dataset_path, train_txt)
    create_subset_txt(splits['val'], dataset_path, val_txt)
    
    class_names = [
        'airplane', 'bridge', 'person', 'ship',
        'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
    ]
    
    config = {
        'path': str(dataset_path.resolve()),
        'train': str(train_txt.resolve()),
        'val': str(val_txt.resolve()),
        'test': str(dataset_path / "test" / "images"),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    config_path = output_dir / "data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_experiment(args):
    """Run a single ablation experiment."""
    
    # Validate experiment name
    if args.experiment not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        sys.exit(1)
    
    exp_config = EXPERIMENTS[args.experiment]
    
    print_section(f"Ablation Experiment: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    
    # Set seeds
    set_all_seeds(args.seed)
    
    # Get paths
    repo_root = get_repo_root()
    dataset_path = Path(args.data_path)
    
    # Validate dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Load splits
    splits_path = repo_root / "configs" / "data_splits.json"
    splits = load_splits(splits_path)
    
    # Create unique output directory for this experiment
    timestamp = get_timestamp()
    exp_name = f"{args.experiment}_{timestamp}"
    output_dir = repo_root / "results" / "ablation" / exp_name
    ensure_dir(output_dir)
    
    print(f"\nExperiment: {args.experiment}")
    print(f"Output directory: {output_dir}")
    print(f"This experiment is INDEPENDENT and will NOT overwrite others.")
    
    # Create data config
    data_config = create_data_config(dataset_path, splits, output_dir)
    
    # Save experiment metadata
    metadata = {
        'experiment': args.experiment,
        'experiment_name': exp_config['name'],
        'description': exp_config['description'],
        'timestamp': timestamp,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch': args.batch,
        'patience': args.patience,
        'imgsz': args.imgsz,
        'model': exp_config['model'],
        'custom_model': exp_config['custom_model'],
        'dataset_path': str(dataset_path),
        'output_dir': str(output_dir)
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Import YOLO
    from ultralytics import YOLO
    
    # Determine which model to use
    if exp_config['custom_model']:
        custom_model_path = repo_root / exp_config['custom_model']
        if not custom_model_path.exists():
            print(f"Error: Custom model not found: {custom_model_path}")
            print("This experiment's model architecture hasn't been implemented yet.")
            sys.exit(1)
        
        print(f"\nLoading custom architecture: {custom_model_path}")
        model = YOLO(str(custom_model_path))
        
        # Load pretrained weights if specified
        if exp_config['model']:
            print(f"Initializing with pretrained: {exp_config['model']}")
            # Note: YOLO will handle weight loading automatically
    else:
        print(f"\nLoading model: {exp_config['model']}")
        model = YOLO(exp_config['model'])
    
    # Training arguments (CONSISTENT across all experiments)
    train_args = {
        'data': str(data_config),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'patience': args.patience,
        'seed': args.seed,
        'device': args.device,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'project': str(output_dir),
        'name': 'train',
        'exist_ok': True,
        'verbose': True,
        'deterministic': True,
        'pretrained': True if exp_config['model'] else False,
        # Augmentation (same for all experiments)
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 0.0,
        'scale': 0.5,
        'fliplr': 0.5,
        'flipud': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Warmup
        'warmup_epochs': 3.0,
        'weight_decay': 0.0005,
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
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
        print(f"EXPERIMENT COMPLETE: {args.experiment}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        # Print next steps
        print(f"\nTo evaluate on test set:")
        print(f"  python scripts/evaluate_baseline.py --weights {output_dir / 'best.pt'} --data-path \"{dataset_path}\"")
        
        return output_dir
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nTraining failed after {format_time(elapsed)}")
        print(f"Error: {e}")
        
        with open(output_dir / "error.log", 'w') as f:
            f.write(f"Experiment: {args.experiment}\n")
            f.write(f"Error: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        
        raise


def list_experiments():
    """List all available experiments."""
    print("\nAvailable Ablation Experiments:")
    print("=" * 60)
    for key, config in EXPERIMENTS.items():
        status = "✓ Ready" if config['custom_model'] is None or Path(config['custom_model']).stem == 'yolo11m_p2' else "○ Not implemented"
        print(f"\n  {key:12s} - {config['name']}")
        print(f"               {config['description']}")
        print(f"               Status: {status}")
    print("\n" + "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ablation experiments for AI-TOD'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help='Experiment to run (baseline, p2, bifpn, taam, anchorfree, full)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=800,
        help='Image size (default: 800)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device (default: 0)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=False,
        help='Path to AI-TOD dataset'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.list:
        list_experiments()
        sys.exit(0)
    
    if not args.experiment:
        print("Error: --experiment is required")
        print("Use --list to see available experiments")
        sys.exit(1)
    
    if not args.data_path:
        print("Error: --data-path is required")
        sys.exit(1)
    
    run_experiment(args)
