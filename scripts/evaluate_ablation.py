#!/usr/bin/env python3
"""
Evaluate Ablation Experiments on Test Set

This script evaluates trained ablation models on the held-out test set
and generates a comparison table.

Usage:
    python scripts/evaluate_ablation.py --weights results/ablation/p2_xxx/best.pt
    python scripts/evaluate_ablation.py --compare  # Compare all completed experiments
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_repo_root,
    load_splits,
    print_section,
    format_time,
    ensure_dir
)


def evaluate_single(weights_path: Path, data_path: Path, output_dir: Path = None):
    """Evaluate a single model on test set."""
    
    from ultralytics import YOLO
    import yaml
    
    print_section(f"Evaluating: {weights_path.name}")
    
    # Setup output
    if output_dir is None:
        output_dir = weights_path.parent / "eval_test"
    ensure_dir(output_dir)
    
    # Load model
    model = YOLO(str(weights_path))
    
    # Create test config
    class_names = [
        'airplane', 'bridge', 'person', 'ship',
        'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
    ]
    
    test_config = {
        'path': str(data_path),
        'val': str(data_path / 'test' / 'images'),
        'train': str(data_path / 'train' / 'images'),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    config_path = output_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    # Run evaluation
    results = model.val(
        data=str(config_path),
        split='val',
        conf=0.001,
        iou=0.65,
        imgsz=800,
        verbose=True,
        save_json=True,
        project=str(output_dir),
        name='results',
        exist_ok=True
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10))
    }
    
    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'mAP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            'mAP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
        }
    
    # Save results
    eval_results = {
        'weights': str(weights_path),
        'metrics': metrics,
        'per_class': per_class,
        'evaluated_at': datetime.now().isoformat()
    }
    
    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Print results
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"mAP@50:     {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
    print(f"mAP@50-95:  {metrics['mAP50-95']:.4f} ({metrics['mAP50-95']*100:.2f}%)")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    
    return eval_results


def compare_experiments(repo_root: Path):
    """Compare all completed ablation experiments."""
    
    ablation_dir = repo_root / "results" / "ablation"
    
    if not ablation_dir.exists():
        print("No ablation experiments found.")
        return
    
    # Find all experiments with test metrics
    experiments = []
    
    for exp_dir in ablation_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check for test metrics
        test_metrics = exp_dir / "eval_test" / "test_metrics.json"
        if not test_metrics.exists():
            # Try alternative location
            test_metrics = exp_dir / "test_metrics.json"
        
        if test_metrics.exists():
            with open(test_metrics, 'r') as f:
                metrics = json.load(f)
            
            # Get experiment config
            config_file = exp_dir / "experiment_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                exp_name = config.get('experiment', exp_dir.name)
            else:
                exp_name = exp_dir.name.split('_')[0]
            
            experiments.append({
                'name': exp_name,
                'dir': exp_dir.name,
                'mAP50': metrics['metrics']['mAP50'],
                'mAP50-95': metrics['metrics']['mAP50-95'],
                'precision': metrics['metrics']['precision'],
                'recall': metrics['metrics']['recall'],
                'f1': metrics['metrics']['f1']
            })
    
    if not experiments:
        print("No evaluated experiments found.")
        print("Run evaluation first: python scripts/evaluate_ablation.py --weights <path>")
        return
    
    # Sort by experiment order
    exp_order = ['baseline', 'p2', 'bifpn', 'taam', 'anchorfree', 'full']
    experiments.sort(key=lambda x: exp_order.index(x['name']) if x['name'] in exp_order else 999)
    
    # Print comparison table
    print_section("ABLATION STUDY RESULTS")
    
    print(f"{'Experiment':<15} {'mAP50':>10} {'mAP50-95':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    baseline_map = None
    for exp in experiments:
        if exp['name'] == 'baseline':
            baseline_map = exp['mAP50']
        
        delta = ""
        if baseline_map and exp['name'] != 'baseline':
            delta = f" (+{(exp['mAP50'] - baseline_map)*100:.1f}%)"
        
        print(f"{exp['name']:<15} {exp['mAP50']*100:>9.2f}% {exp['mAP50-95']*100:>9.2f}% {exp['precision']*100:>9.2f}% {exp['recall']*100:>9.2f}% {exp['f1']*100:>9.2f}%{delta}")
    
    print("-" * 70)
    
    # Save comparison to file
    comparison_file = ablation_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")
    
    # Generate markdown table
    md_table = "| Model | P2 | BiFPN | TAAM | AnchorFree | NWD | mAP50 | mAP50-95 |\n"
    md_table += "|-------|:--:|:-----:|:----:|:----------:|:---:|-------|----------|\n"
    
    component_map = {
        'baseline': {'p2': '✗', 'bifpn': '✗', 'taam': '✗', 'af': '✗', 'nwd': '✗'},
        'p2': {'p2': '✓', 'bifpn': '✗', 'taam': '✗', 'af': '✗', 'nwd': '✗'},
        'bifpn': {'p2': '✓', 'bifpn': '✓', 'taam': '✗', 'af': '✗', 'nwd': '✗'},
        'taam': {'p2': '✓', 'bifpn': '✓', 'taam': '✓', 'af': '✗', 'nwd': '✗'},
        'anchorfree': {'p2': '✓', 'bifpn': '✓', 'taam': '✓', 'af': '✓', 'nwd': '✗'},
        'full': {'p2': '✓', 'bifpn': '✓', 'taam': '✓', 'af': '✓', 'nwd': '✓'},
    }
    
    for exp in experiments:
        comp = component_map.get(exp['name'], {})
        md_table += f"| {exp['name']} | {comp.get('p2', '?')} | {comp.get('bifpn', '?')} | {comp.get('taam', '?')} | {comp.get('af', '?')} | {comp.get('nwd', '?')} | {exp['mAP50']*100:.2f}% | {exp['mAP50-95']*100:.2f}% |\n"
    
    md_file = ablation_dir / "ablation_table.md"
    with open(md_file, 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(md_table)
    
    print(f"Markdown table saved to: {md_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ablation experiments')
    
    parser.add_argument('--weights', type=str, help='Path to model weights')
    parser.add_argument('--data-path', type=str, help='Path to AI-TOD dataset')
    parser.add_argument('--compare', action='store_true', help='Compare all experiments')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    repo_root = get_repo_root()
    
    if args.compare:
        compare_experiments(repo_root)
    elif args.weights:
        if not args.data_path:
            print("Error: --data-path required for evaluation")
            sys.exit(1)
        evaluate_single(Path(args.weights), Path(args.data_path))
    else:
        print("Usage:")
        print("  Evaluate single: python scripts/evaluate_ablation.py --weights <path> --data-path <path>")
        print("  Compare all:     python scripts/evaluate_ablation.py --compare")
