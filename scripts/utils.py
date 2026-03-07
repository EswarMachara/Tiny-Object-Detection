#!/usr/bin/env python3
"""
Utility functions for AI-TOD Tiny Object Detection Pipeline

Provides common utilities for:
- Seed setting for reproducibility
- Dataset split loading
- Path handling
- Metrics computation
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds for all random number generators for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"All seeds set to {seed}")


def get_repo_root() -> Path:
    """
    Get the repository root directory.
    
    Returns:
        Path to repository root
    """
    # Try to find repo root by looking for key files
    current = Path(__file__).parent.absolute()
    
    for _ in range(5):  # Go up at most 5 levels
        if (current / "configs").exists() or (current / "scripts").exists():
            return current
        current = current.parent
    
    # Fallback to parent of scripts folder
    return Path(__file__).parent.parent.absolute()


def get_dataset_path() -> Path:
    """
    Auto-detect the dataset path.
    
    Returns:
        Path to AI-TOD dataset root
    """
    repo_root = get_repo_root()
    
    # Try local paths FIRST, then Kaggle
    possible_paths = [
        # Local paths (prioritized)
        repo_root / "AI_TOD",  # Local repo
        Path.cwd() / "AI_TOD",  # Current working directory
        Path("AI_TOD"),  # Relative to cwd
        Path("../AI_TOD"),  # Parent directory
        repo_root.parent / "AI_TOD",  # Sibling to repo
        # Kaggle paths (fallback)
        Path("/kaggle/input/ai-tod-dataset/AI_TOD"),
        Path("/kaggle/input/ai-tod-yolo-8"),
    ]
    
    for p in possible_paths:
        try:
            if p.exists() and (p / "train").exists():
                print(f"Auto-detected dataset at: {p.resolve()}")
                return p.resolve()
        except Exception:
            continue
    
    raise FileNotFoundError(
        "Could not find AI-TOD dataset.\n"
        "Please specify --data-path explicitly.\n"
        "Example: python scripts/train_baseline.py --data-path /path/to/AI_TOD"
    )


def load_splits(splits_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load dataset splits from JSON file.
    
    Args:
        splits_path: Path to data_splits.json (auto-detected if None)
        
    Returns:
        Dictionary with 'train', 'val', 'test' image lists
    """
    if splits_path is None:
        repo_root = get_repo_root()
        splits_path = repo_root / "configs" / "data_splits.json"
    
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}\n"
            "Run 'python scripts/generate_splits.py' first."
        )
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    print(f"Loaded splits from: {splits_path}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    
    return splits


def get_label_path(image_path: str) -> str:
    """
    Get the label file path corresponding to an image.
    
    Args:
        image_path: Relative path to image (e.g., 'train/images/xxx.jpg')
        
    Returns:
        Relative path to label file (e.g., 'train/labels/xxx.txt')
    """
    # Replace 'images' with 'labels' and change extension to .txt
    label_path = image_path.replace('/images/', '/labels/')
    label_path = str(Path(label_path).with_suffix('.txt'))
    return label_path


def create_subset_txt(
    image_list: List[str],
    dataset_path: Path,
    output_path: Path
) -> None:
    """
    Create a text file listing absolute paths for a subset of images.
    Useful for YOLO training with custom splits.
    
    Args:
        image_list: List of relative image paths
        dataset_path: Root path to dataset
        output_path: Output .txt file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for img_path in image_list:
            full_path = dataset_path / img_path
            f.write(str(full_path) + '\n')
    
    print(f"Created subset file: {output_path} ({len(image_list)} images)")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., '1h 23m 45s')
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_timestamp() -> str:
    """Get current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def count_parameters(model) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_yolo_results(results_csv_path: Path) -> Dict[str, Any]:
    """
    Parse YOLO training results CSV.
    
    Args:
        results_csv_path: Path to results.csv
        
    Returns:
        Dictionary with parsed metrics
    """
    import pandas as pd
    
    df = pd.read_csv(results_csv_path)
    
    # Clean column names (remove leading/trailing whitespace)
    df.columns = df.columns.str.strip()
    
    final_row = df.iloc[-1]
    
    # Find best epoch
    if 'metrics/mAP50(B)' in df.columns:
        best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
        best_mAP50 = df['metrics/mAP50(B)'].max()
    else:
        best_epoch = len(df)
        best_mAP50 = None
    
    return {
        'epochs_trained': len(df),
        'best_epoch': best_epoch,
        'best_mAP50': best_mAP50,
        'final_metrics': final_row.to_dict()
    }
