#!/usr/bin/env python3
"""
Split train folder into train and val based on data_splits.json

This script physically moves validation images and labels from train/ to val/
based on the pre-defined splits in configs/data_splits.json.
"""

import os
import sys
import json
import shutil
from pathlib import Path


def main():
    # Get paths
    repo_root = Path(__file__).parent.parent.resolve()
    dataset_path = repo_root / "AI_TOD"
    splits_path = repo_root / "configs" / "data_splits.json"
    
    print(f"Repository root: {repo_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Splits file: {splits_path}")
    
    # Verify paths exist
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    if not splits_path.exists():
        print(f"ERROR: Splits file not found: {splits_path}")
        sys.exit(1)
    
    # Load splits
    print("\nLoading splits...")
    with open(splits_path, 'r') as f:
        split_data = json.load(f)
    
    val_images = split_data['val']
    print(f"Validation images to move: {len(val_images)}")
    
    # Create val directories
    val_images_dir = dataset_path / "val" / "images"
    val_labels_dir = dataset_path / "val" / "labels"
    
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated directories:")
    print(f"  {val_images_dir}")
    print(f"  {val_labels_dir}")
    
    # Move validation files
    print("\nMoving validation files...")
    moved_images = 0
    moved_labels = 0
    missing_images = []
    missing_labels = []
    
    for img_rel_path in val_images:
        # img_rel_path is like "train/images/filename.jpg"
        # Extract just the filename
        img_filename = Path(img_rel_path).name
        label_filename = Path(img_filename).stem + ".txt"
        
        # Source paths (in train folder)
        src_image = dataset_path / "train" / "images" / img_filename
        src_label = dataset_path / "train" / "labels" / label_filename
        
        # Destination paths (in val folder)
        dst_image = val_images_dir / img_filename
        dst_label = val_labels_dir / label_filename
        
        # Move image
        if src_image.exists():
            shutil.move(str(src_image), str(dst_image))
            moved_images += 1
        else:
            missing_images.append(img_filename)
        
        # Move label
        if src_label.exists():
            shutil.move(str(src_label), str(dst_label))
            moved_labels += 1
        else:
            missing_labels.append(label_filename)
        
        # Progress indicator
        if (moved_images + len(missing_images)) % 500 == 0:
            print(f"  Processed {moved_images + len(missing_images)}/{len(val_images)} images...")
    
    print(f"\nCompleted!")
    print(f"  Moved images: {moved_images}")
    print(f"  Moved labels: {moved_labels}")
    
    if missing_images:
        print(f"\n  WARNING: {len(missing_images)} images not found")
        if len(missing_images) <= 5:
            for m in missing_images:
                print(f"    - {m}")
    
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} labels not found")
        if len(missing_labels) <= 5:
            for m in missing_labels:
                print(f"    - {m}")
    
    # Verify final counts
    print("\nFinal directory counts:")
    train_images_count = len(list((dataset_path / "train" / "images").glob("*")))
    train_labels_count = len(list((dataset_path / "train" / "labels").glob("*.txt")))
    val_images_count = len(list(val_images_dir.glob("*")))
    val_labels_count = len(list(val_labels_dir.glob("*.txt")))
    
    print(f"  train/images: {train_images_count}")
    print(f"  train/labels: {train_labels_count}")
    print(f"  val/images:   {val_images_count}")
    print(f"  val/labels:   {val_labels_count}")
    
    # Update data.yaml
    print("\nUpdating data.yaml...")
    data_yaml_path = dataset_path / "data.yaml"
    
    data_yaml_content = f"""# AI-TOD Dataset Configuration
# Split from original train folder based on configs/data_splits.json

train: train/images
val: val/images
test: test/images

nc: 8
names: ['airplane', 'bridge', 'person', 'ship', 'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill']
"""
    
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"Updated: {data_yaml_path}")
    print("\nDone! Dataset is now properly split into train/val/test folders.")


if __name__ == "__main__":
    main()
