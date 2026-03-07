#!/usr/bin/env python3
"""
Convert YOLO Segmentation Labels to Detection Labels

This script converts polygon-based segmentation annotations to bounding box
detection annotations by finding the enclosing box of each polygon.

Input format (segmentation):
    class_id x1 y1 x2 y2 x3 y3 x4 y4 ...

Output format (detection):
    class_id center_x center_y width height

All coordinates are normalized (0-1).

Usage:
    python scripts/convert_seg_to_det.py --dataset_path /path/to/AI_TOD
    python scripts/convert_seg_to_det.py --dataset_path /path/to/AI_TOD --backup
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_segmentation_line(line: str) -> tuple:
    """
    Parse a segmentation annotation line.
    
    Args:
        line: Line from label file (class_id x1 y1 x2 y2 ...)
        
    Returns:
        Tuple of (class_id, list of (x, y) points)
    """
    parts = line.strip().split()
    if len(parts) < 5:  # Need at least class + 2 points
        return None, None
    
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    # Group into (x, y) pairs
    points = []
    for i in range(0, len(coords) - 1, 2):
        points.append((coords[i], coords[i + 1]))
    
    return class_id, points


def polygon_to_bbox(points: list) -> tuple:
    """
    Convert polygon points to bounding box.
    
    Args:
        points: List of (x, y) tuples
        
    Returns:
        Tuple of (center_x, center_y, width, height) normalized
    """
    if not points:
        return None
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Calculate center and dimensions
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Clamp to valid range
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height


def is_detection_format(line: str) -> bool:
    """
    Check if a line is already in detection format.
    
    Detection format has exactly 5 values: class cx cy w h
    Segmentation format has more values: class x1 y1 x2 y2 x3 y3 ...
    """
    parts = line.strip().split()
    return len(parts) == 5


def convert_label_file(input_path: Path, output_path: Path) -> dict:
    """
    Convert a single label file from segmentation to detection format.
    
    Args:
        input_path: Path to input segmentation label file
        output_path: Path to output detection label file
        
    Returns:
        Dictionary with conversion statistics
    """
    stats = {
        'total_lines': 0,
        'converted': 0,
        'skipped': 0,
        'already_detection': 0,
        'errors': 0
    }
    
    output_lines = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    stats['total_lines'] = len(lines)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if already detection format
        if is_detection_format(line):
            output_lines.append(line)
            stats['already_detection'] += 1
            continue
        
        # Parse segmentation format
        class_id, points = parse_segmentation_line(line)
        
        if class_id is None or not points:
            stats['errors'] += 1
            continue
        
        # Convert to bounding box
        bbox = polygon_to_bbox(points)
        
        if bbox is None:
            stats['errors'] += 1
            continue
        
        cx, cy, w, h = bbox
        
        # Skip if bbox is essentially zero size
        if w < 1e-6 or h < 1e-6:
            stats['skipped'] += 1
            continue
        
        # Write detection format
        output_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stats['converted'] += 1
    
    # Write output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
        if output_lines:
            f.write('\n')
    
    return stats


def convert_dataset(dataset_path: Path, backup: bool = True, in_place: bool = True) -> dict:
    """
    Convert all label files in the dataset.
    
    Args:
        dataset_path: Path to AI-TOD dataset root
        backup: Whether to backup original labels
        in_place: Whether to overwrite original files
        
    Returns:
        Dictionary with overall statistics
    """
    total_stats = {
        'files_processed': 0,
        'total_annotations': 0,
        'converted': 0,
        'already_detection': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Process both train and test splits
    for split in ['train', 'test']:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping.")
            continue
        
        # Backup original labels
        if backup:
            backup_dir = dataset_path / split / 'labels_segmentation_backup'
            if not backup_dir.exists():
                print(f"Creating backup: {backup_dir}")
                shutil.copytree(labels_dir, backup_dir)
            else:
                print(f"Backup already exists: {backup_dir}")
        
        # Get all label files
        label_files = list(labels_dir.glob('*.txt'))
        print(f"\nProcessing {split}: {len(label_files)} label files")
        
        for label_file in tqdm(label_files, desc=f"Converting {split}"):
            if in_place:
                output_file = label_file
            else:
                output_dir = dataset_path / split / 'labels_detection'
                output_file = output_dir / label_file.name
            
            stats = convert_label_file(label_file, output_file)
            
            total_stats['files_processed'] += 1
            total_stats['total_annotations'] += stats['total_lines']
            total_stats['converted'] += stats['converted']
            total_stats['already_detection'] += stats['already_detection']
            total_stats['skipped'] += stats['skipped']
            total_stats['errors'] += stats['errors']
    
    return total_stats


def verify_conversion(dataset_path: Path) -> bool:
    """
    Verify that conversion was successful by checking sample files.
    """
    print("\nVerifying conversion...")
    
    for split in ['train', 'test']:
        labels_dir = dataset_path / split / 'labels'
        label_files = list(labels_dir.glob('*.txt'))[:5]
        
        print(f"\n{split} samples:")
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()[:3]
            
            print(f"  {label_file.name}:")
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    print(f"    ✓ Detection format: {line.strip()[:60]}...")
                else:
                    print(f"    ✗ Not detection ({len(parts)} values): {line.strip()[:40]}...")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation labels to detection format"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to AI-TOD dataset root"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup original labels before conversion (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup of original labels"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion after completion"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    backup = not args.no_backup
    
    print("=" * 60)
    print("YOLO Segmentation to Detection Label Converter")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Backup:  {'Yes' if backup else 'No'}")
    print()
    
    # Check current format
    sample_label = None
    for split in ['train', 'test']:
        labels_dir = dataset_path / split / 'labels'
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            if label_files:
                sample_label = label_files[0]
                break
    
    if sample_label:
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip()
        parts = first_line.split()
        print(f"Sample label ({sample_label.name}):")
        print(f"  {first_line[:80]}...")
        print(f"  Values per line: {len(parts)}")
        
        if len(parts) == 5:
            print("\n⚠ Labels appear to already be in detection format!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        else:
            print(f"  Format: Segmentation (polygon with {(len(parts)-1)//2} vertices)")
    
    print("\nStarting conversion...")
    
    # Convert
    stats = convert_dataset(dataset_path, backup=backup, in_place=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Files processed:     {stats['files_processed']:,}")
    print(f"Total annotations:   {stats['total_annotations']:,}")
    print(f"Converted:           {stats['converted']:,}")
    print(f"Already detection:   {stats['already_detection']:,}")
    print(f"Skipped (zero-size): {stats['skipped']:,}")
    print(f"Errors:              {stats['errors']:,}")
    
    if backup:
        print(f"\nOriginal labels backed up to:")
        print(f"  {dataset_path}/train/labels_segmentation_backup/")
        print(f"  {dataset_path}/test/labels_segmentation_backup/")
    
    # Verify
    if args.verify:
        verify_conversion(dataset_path)
    
    print("\n✓ Labels are now in detection format.")
    print("You can now train with detection models (yolo11m.pt, yolo11l.pt, etc.)")


if __name__ == '__main__':
    main()
