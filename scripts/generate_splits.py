#!/usr/bin/env python3
"""
Generate Deterministic Dataset Splits for AI-TOD

This script creates reproducible train/val/test splits stored in a JSON file.
The splits are deterministic based on seed=42, ensuring reproducibility across
different machines and hardware systems.

Usage:
    python scripts/generate_splits.py [--dataset_path PATH] [--output PATH] [--force]
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def get_image_files(directory: Path) -> List[str]:
    """
    Get sorted list of image files from a directory.
    
    Args:
        directory: Path to image directory
        
    Returns:
        Sorted list of image filenames
    """
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    
    # Common image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            image_files.append(f.name)
    
    # Sort alphabetically for determinism
    return sorted(image_files)


def split_list(items: List[str], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Split a list into train and validation sets.
    
    Args:
        items: List of items to split
        train_ratio: Fraction of items for training (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_items, val_items)
    """
    # Set seed for this operation
    set_seed(seed)
    
    # Create a copy and shuffle
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    split_idx = int(len(shuffled) * train_ratio)
    
    train_items = sorted(shuffled[:split_idx])
    val_items = sorted(shuffled[split_idx:])
    
    return train_items, val_items


def generate_splits(dataset_path: Path, output_path: Path, 
                   train_ratio: float = 0.9, seed: int = 42,
                   force: bool = False) -> Dict:
    """
    Generate deterministic dataset splits.
    
    Args:
        dataset_path: Path to AI-TOD dataset root
        output_path: Path to output JSON file
        train_ratio: Fraction of training data for training (rest goes to val)
        seed: Random seed for reproducibility
        force: If True, regenerate even if file exists
        
    Returns:
        Dictionary containing splits
    """
    # Check if splits already exist
    if output_path.exists() and not force:
        print(f"Splits file already exists: {output_path}")
        print("Use --force to regenerate splits.")
        with open(output_path, 'r') as f:
            return json.load(f)
    
    print(f"Generating deterministic splits with seed={seed}...")
    print(f"Dataset path: {dataset_path}")
    
    # Get training images
    train_images_dir = dataset_path / "train" / "images"
    train_images = get_image_files(train_images_dir)
    print(f"Found {len(train_images)} training images")
    
    # Get test images
    test_images_dir = dataset_path / "test" / "images"
    test_images = get_image_files(test_images_dir)
    print(f"Found {len(test_images)} test images")
    
    if len(train_images) == 0:
        raise ValueError(f"No training images found in {train_images_dir}")
    
    # Split training set into train and val
    train_split, val_split = split_list(train_images, train_ratio, seed)
    
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_split)} images ({len(train_split)/len(train_images)*100:.1f}%)")
    print(f"  Val:   {len(val_split)} images ({len(val_split)/len(train_images)*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images")
    
    # Create splits dictionary with relative paths
    splits = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "total_train_images": len(train_images),
            "total_test_images": len(test_images),
            "generated_by": "generate_splits.py"
        },
        "train": [f"train/images/{img}" for img in train_split],
        "val": [f"train/images/{img}" for img in val_split],
        "test": [f"test/images/{img}" for img in test_images]
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {output_path}")
    
    return splits


def verify_splits(splits: Dict, dataset_path: Path) -> bool:
    """
    Verify that all files in splits exist.
    
    Args:
        splits: Splits dictionary
        dataset_path: Path to dataset root
        
    Returns:
        True if all files exist
    """
    print("\nVerifying splits...")
    
    all_valid = True
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
            
        missing = 0
        for img_path in splits[split_name]:
            full_path = dataset_path / img_path
            if not full_path.exists():
                missing += 1
                if missing <= 3:
                    print(f"  Missing: {img_path}")
        
        if missing > 0:
            print(f"  {split_name}: {missing} files missing!")
            all_valid = False
        else:
            print(f"  {split_name}: All {len(splits[split_name])} files verified ✓")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic dataset splits for AI-TOD"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=None,
        help="Path to AI-TOD dataset root (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="configs/data_splits.json",
        help="Output path for splits JSON file"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.9,
        help="Fraction of training data for training set (default: 0.9)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force regeneration even if splits file exists"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify all files in splits exist"
    )
    
    args = parser.parse_args()
    
    # Determine script location for relative paths
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent
    
    # Auto-detect dataset path
    if args.dataset_path is None:
        # Try common locations
        possible_paths = [
            Path("/kaggle/input/ai-tod-dataset/AI_TOD"),  # Kaggle
            Path("/kaggle/input/ai-tod-yolo-8"),  # Kaggle alternative
            repo_root / "AI_TOD",  # Local
            Path("AI_TOD"),  # Current directory
        ]
        
        for p in possible_paths:
            if p.exists():
                args.dataset_path = str(p)
                print(f"Auto-detected dataset path: {p}")
                break
        
        if args.dataset_path is None:
            print("Error: Could not auto-detect dataset path.")
            print("Please specify --dataset_path")
            sys.exit(1)
    
    dataset_path = Path(args.dataset_path)
    
    # Handle relative output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    
    # Generate splits
    splits = generate_splits(
        dataset_path=dataset_path,
        output_path=output_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        force=args.force
    )
    
    # Verify if requested
    if args.verify:
        verify_splits(splits, dataset_path)
    
    print("\n" + "="*50)
    print("Split generation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
