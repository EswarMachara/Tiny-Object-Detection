#!/usr/bin/env python3
"""
YOLOv11 Baseline Evaluation Script for AI-TOD Dataset

This script evaluates trained YOLOv11 models on the test set
to obtain final performance metrics.

Features:
- Evaluates on held-out test set (never seen during training)
- Computes mAP50, mAP50-95, precision, recall
- Per-class performance breakdown
- Saves detailed results to JSON and CSV
- Optional visualization of predictions

Usage:
    python scripts/evaluate_baseline.py --weights results/baseline/yolo11m_xxx/best.pt
    python scripts/evaluate_baseline.py --weights best.pt --conf 0.001 --iou 0.65

Environment Variables:
    YOLO_TOD_DATA: Override dataset path
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root and scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_repo_root, 
    get_dataset_path, 
    load_splits,
    print_section,
    format_time,
    get_timestamp,
    ensure_dir
)


def evaluate_model(args):
    """Main evaluation function."""
    
    print_section("AI-TOD YOLOv11 Test Set Evaluation")
    
    # Get paths
    repo_root = get_repo_root()
    
    if args.data_path:
        dataset_path = Path(args.data_path)
    else:
        dataset_path = get_dataset_path()
    
    print(f"Repository root: {repo_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Weights: {args.weights}")
    
    # Load splits for verification
    splits_path = repo_root / "configs" / "data_splits.json"
    splits = load_splits(splits_path)
    
    # Setup output directory
    weights_path = Path(args.weights)
    if weights_path.parent.name in ['weights', 'train']:
        # Weights inside training output directory
        output_dir = weights_path.parent.parent.parent / "eval_test"
    else:
        # Standalone weights file
        output_dir = repo_root / "results" / "baseline" / f"eval_{get_timestamp()}"
    
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    # Class names
    class_names = [
        'airplane', 'bridge', 'person', 'ship', 
        'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
    ]
    
    # Save evaluation parameters
    eval_params = {
        'weights': str(args.weights),
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'device': args.device,
        'timestamp': get_timestamp(),
        'dataset_path': str(dataset_path),
        'test_images': len(splits['test'])
    }
    
    params_path = output_dir / "eval_params.json"
    with open(params_path, 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    print(f"\nEvaluation Parameters:")
    for k, v in eval_params.items():
        print(f"  {k}: {v}")
    
    # Import ultralytics
    print("\nLoading YOLO model...")
    from ultralytics import YOLO

    # Register custom modules (needed for TAAM model weights)
    try:
        from models.taam import TAAM, TAAMBlock
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks_module
        for cls in [TAAM, TAAMBlock]:
            name = cls.__name__
            if not hasattr(modules, name):
                setattr(modules, name, cls)
            if not hasattr(tasks_module, name):
                setattr(tasks_module, name, cls)
        print("✓ Registered custom modules: TAAM, TAAMBlock")
    except ImportError:
        pass  # Not a TAAM model, skip

    # Load model
    model = YOLO(args.weights)
    
    # Create temporary config for test evaluation
    import yaml
    test_config = {
        'path': str(dataset_path.resolve()),
        'test': 'test/images',
        'val': 'test/images',  # Use test as val for evaluation
        'train': 'train/images',  # Required but not used
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    config_path = output_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    # Run evaluation
    print_section("Running Test Set Evaluation")
    import time
    start_time = time.time()
    
    try:
        # Validation mode evaluates on the 'val' split which we set to test
        results = model.val(
            data=str(config_path),
            split='val',  # This will use our test set
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=True,
            save_json=True,
            save=True,  # Save visualizations
            project=str(output_dir),
            name='test_results',
            exist_ok=True
        )
        
        elapsed = time.time() - start_time
        print_section(f"Evaluation Completed in {format_time(elapsed)}")
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10)),
        }
        
        # Per-class metrics
        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0,
                'mAP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
                'mAP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            }
        
        # Print results
        print_section("Test Set Results")
        print(f"Overall Metrics:")
        print(f"  mAP@50:     {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
        print(f"  mAP@50-95:  {metrics['mAP50-95']:.4f} ({metrics['mAP50-95']*100:.2f}%)")
        print(f"  Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:   {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        
        print(f"\nPer-Class mAP@50:")
        for name, class_metrics in per_class.items():
            print(f"  {name:15s}: {class_metrics['mAP50']:.4f}")
        
        # Save detailed results
        full_results = {
            'status': 'completed',
            'elapsed_time': elapsed,
            'elapsed_formatted': format_time(elapsed),
            'eval_params': eval_params,
            'overall_metrics': metrics,
            'per_class_metrics': per_class,
            'test_images': len(splits['test']),
        }
        
        results_path = output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_path}")
        
        # Save CSV summary
        import csv
        csv_path = output_dir / "test_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in metrics.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(['class', 'precision', 'recall', 'mAP50', 'mAP50-95'])
            for name, class_metrics in per_class.items():
                writer.writerow([
                    name,
                    class_metrics['precision'],
                    class_metrics['recall'],
                    class_metrics['mAP50'],
                    class_metrics['mAP50-95']
                ])
        
        print(f"CSV metrics saved to: {csv_path}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"Test mAP@50:    {metrics['mAP50']*100:.2f}%")
        print(f"Test mAP@50-95: {metrics['mAP50-95']*100:.2f}%")
        print(f"{'='*60}")
        
        return full_results
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nEvaluation failed after {format_time(elapsed)}")
        print(f"Error: {e}")
        
        # Save error log
        with open(output_dir / "error.log", 'w') as f:
            f.write(f"Evaluation failed at {datetime.now()}\n")
            f.write(f"Error: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv11 baseline on AI-TOD test set'
    )
    
    # Required
    parser.add_argument(
        '--weights', 
        type=str, 
        required=True,
        help='Path to trained model weights (e.g., best.pt)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold (default: 0.001 for evaluation)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.65,
        help='IoU threshold for NMS (default: 0.65)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=800,
        help='Image size (default: 800)'
    )
    
    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (default: 0 for GPU 0)'
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
    evaluate_model(args)
