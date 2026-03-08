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

# Add scripts directory and project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from utils import (
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


def register_custom_modules():
    """Register custom modules (TAAM) with Ultralytics."""
    try:
        from models.taam import TAAM, TAAMBlock
        import ultralytics.nn.modules as modules
        
        # Register TAAM
        if not hasattr(modules, 'TAAM'):
            modules.TAAM = TAAM
            modules.__all__.append('TAAM')
        
        if not hasattr(modules, 'TAAMBlock'):
            modules.TAAMBlock = TAAMBlock
            modules.__all__.append('TAAMBlock')
        
        # Also add to tasks module
        try:
            from ultralytics.nn import tasks
            if not hasattr(tasks, 'TAAM'):
                tasks.TAAM = TAAM
            if not hasattr(tasks, 'TAAMBlock'):
                tasks.TAAMBlock = TAAMBlock
        except ImportError:
            pass
        
        print("✓ Registered custom modules: TAAM")
        return True
    except ImportError as e:
        print(f"Warning: Could not register TAAM: {e}")
        return False


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
        'model': 'yolo11m.pt',  # Pretrained weights for fair comparison
        'custom_model': 'models/yolo11m_p2_bifpn.yaml',
        'description': 'Adding BiFPN neck for multi-scale fusion'
    },
    'taam': {
        'name': 'YOLOv11M + P2 + BiFPN + TAAM',
        'model': 'yolo11m.pt',  # Pretrained weights for fair comparison
        'custom_model': 'models/yolo11m_p2_bifpn_taam.yaml',
        'description': 'Adding Tiny-Aware Attention Module'
    },
    'anchorfree': {
        'name': 'YOLOv11M + P2 + BiFPN + TAAM + AnchorFree',
        'model': 'yolo11m.pt',  # Pretrained weights for fair comparison
        'custom_model': 'models/yolo11m_p2_bifpn_taam_af.yaml',
        'description': 'Switching to anchor-free detection head'
    },
    'full': {
        'name': 'Full Proposed Method',
        'model': 'yolo11m.pt',  # Pretrained weights for fair comparison
        'custom_model': 'models/yolo11m_full.yaml',
        'description': 'Complete method with NWD loss'
    }
}


def create_data_config(dataset_path: Path, output_dir: Path) -> Path:
    """Create YOLO data configuration file using physically split directories."""
    import yaml
    
    # Use physically split train/val directories to avoid cache collision
    # (both would otherwise use same cache file since images come from same folder)
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
    
    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = get_dataset_path()
    
    # Validate dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # CRITICAL: Validate label format before training
    print("\nValidating label format...")
    label_check = validate_label_format(dataset_path)
    
    if not label_check['valid']:
        print("\n" + "=" * 60)
        print("ERROR: LABEL FORMAT VALIDATION FAILED!")
        print("=" * 60)
        print(label_check['error_message'])
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"✓ Labels are in correct detection format ({label_check['detection_count']} samples checked)")
    
    # ALWAYS clear cache files to prevent train/val cache collision
    print("\nClearing YOLO cache files...")
    clear_yolo_cache(dataset_path)
    
    # Verify train/val directories exist
    train_images_dir = dataset_path / "train" / "images"
    val_images_dir = dataset_path / "val" / "images"
    
    if not train_images_dir.exists():
        print(f"ERROR: Train images directory not found: {train_images_dir}")
        print("Run 'python scripts/split_train_val.py' first to create val split.")
        sys.exit(1)
    
    if not val_images_dir.exists():
        print(f"ERROR: Val images directory not found: {val_images_dir}")
        print("Run 'python scripts/split_train_val.py' first to create val split.")
        sys.exit(1)
    
    # Count images
    train_count = len(list(train_images_dir.glob("*")))
    val_count = len(list(val_images_dir.glob("*")))
    
    print(f"\nDataset split:")
    print(f"  Train images: {train_count}")
    print(f"  Val images:   {val_count}")
    
    # Create unique output directory for this experiment
    timestamp = get_timestamp()
    exp_name = f"{args.experiment}_{timestamp}"
    output_dir = repo_root / "results" / "ablation" / exp_name
    ensure_dir(output_dir)
    
    print(f"\nExperiment: {args.experiment}")
    print(f"Output directory: {output_dir}")
    print(f"This experiment is INDEPENDENT and will NOT overwrite others.")
    
    # Create data config
    data_config = create_data_config(dataset_path, output_dir)
    
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
    
    # Handle 'full' experiment separately (requires NWD loss patching)
    if args.experiment == 'full':
        print("\n" + "=" * 60)
        print("NOTE: For the 'full' experiment with NWD loss, use:")
        print("  python scripts/train_full.py --epochs 50 --batch 8")
        print("=" * 60)
        print("\nContinuing with TAAM model (without NWD loss)...")
        print("To get full benefits, run train_full.py instead.\n")
    
    # Register custom modules (TAAM) BEFORE importing YOLO
    if args.experiment in ['taam', 'anchorfree', 'full']:
        register_custom_modules()
    
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
    
    # Training arguments - USE YOLO DEFAULTS to prevent training collapse!
    # Previous version caused NaN/0 losses due to:
    # 1. Custom optimizer/lr settings conflicting with YOLO internals
    # 2. AMP issues with high initial cls_loss
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
        # Disable mosaic for last 10 epochs (helps tiny objects)
        'close_mosaic': 10,
    }
    # Let YOLO handle: optimizer, lr0, warmup, weight_decay, AMP
    # These defaults are battle-tested and prevent training collapse
    
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
    repo_root = get_repo_root()
    
    print("\nAvailable Ablation Experiments:")
    print("=" * 60)
    for key, config in EXPERIMENTS.items():
        # Check if model file exists
        if config['custom_model'] is None:
            status = "✓ Ready (pretrained)"
        else:
            model_path = repo_root / config['custom_model']
            if model_path.exists():
                status = "✓ Ready"
            else:
                status = "○ Not implemented"
        
        print(f"\n  {key:12s} - {config['name']}")
        print(f"               {config['description']}")
        if config['custom_model']:
            print(f"               Model: {config['custom_model']}")
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
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear YOLO label cache files before training'
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
    
    run_experiment(args)
