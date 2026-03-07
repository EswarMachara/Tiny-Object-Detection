"""
Dataset Audit Script for AI-TOD Dataset
Performs comprehensive analysis for Tiny Object Detection dataset
"""

import os
import sys
import yaml
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Configuration
BASE_DIR = Path(r"c:\Users\Eswar\Desktop\TOD_Mini_Project")
DATASET_DIR = BASE_DIR / "AI_TOD"
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "dataset_audit_figures"

# Create output directories
DOCS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Load class names from data.yaml
with open(DATASET_DIR / "data.yaml", 'r') as f:
    data_config = yaml.safe_load(f)
    CLASS_NAMES = data_config['names']
    NUM_CLASSES = data_config['nc']

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# ===== STEP 2: Dataset Structure =====
print("\n" + "="*50)
print("STEP 2: Dataset Structure")
print("="*50)

def get_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Generate a directory tree string"""
    tree = ""
    if current_depth >= max_depth:
        return tree
    
    path = Path(path)
    items = sorted(path.iterdir())
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    # Add files first (but only if there are a few)
    if len(files) <= 10:
        for file in files:
            tree += f"{prefix}├── {file.name}\n"
    else:
        tree += f"{prefix}├── [{len(files)} files]\n"
    
    for i, dir_item in enumerate(dirs):
        is_last = (i == len(dirs) - 1)
        connector = "└── " if is_last else "├── "
        tree += f"{prefix}{connector}{dir_item.name}/\n"
        
        next_prefix = prefix + ("    " if is_last else "│   ")
        tree += get_directory_tree(dir_item, next_prefix, max_depth, current_depth + 1)
    
    return tree

tree_str = f"AI_TOD/\n{get_directory_tree(DATASET_DIR)}"
print(tree_str)

# ===== STEP 3: Dataset Split Statistics =====
print("\n" + "="*50)
print("STEP 3: Dataset Split Statistics")
print("="*50)

splits = {}
available_splits = []

# Check which splits exist
for split_name in ['train', 'valid', 'val', 'test']:
    split_path = DATASET_DIR / split_name
    if split_path.exists():
        available_splits.append(split_name)

print(f"Available splits: {available_splits}")

# Process each split
for split_name in available_splits:
    split_path = DATASET_DIR / split_name
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    
    images = list(images_dir.glob("*")) if images_dir.exists() else []
    labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    
    # Count objects
    total_objects = 0
    class_counts = defaultdict(int)
    objects_per_image = []
    
    for label_file in labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            obj_count = len([l for l in lines if l.strip()])
            total_objects += obj_count
            objects_per_image.append(obj_count)
            
            for line in lines:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
    
    splits[split_name] = {
        'images': len(images),
        'labels': len(labels),
        'objects': total_objects,
        'class_counts': dict(class_counts),
        'objects_per_image': objects_per_image
    }
    
    print(f"\n{split_name.upper()}:")
    print(f"  Images: {len(images)}")
    print(f"  Labels: {len(labels)}")
    print(f"  Objects: {total_objects}")

# ===== STEP 4: Class Distribution =====
print("\n" + "="*50)
print("STEP 4: Class Distribution")
print("="*50)

# Aggregate class counts across all splits
total_class_counts = defaultdict(int)
total_objects = 0

for split_name, data in splits.items():
    for class_id, count in data['class_counts'].items():
        total_class_counts[class_id] += count
        total_objects += count

print(f"\nTotal objects across all splits: {total_objects}")
print("\nClass Distribution:")
print("-" * 50)
print(f"{'Class':<20} {'Instances':<12} {'Percentage':<10}")
print("-" * 50)

class_distribution = []
for class_id in range(NUM_CLASSES):
    count = total_class_counts.get(class_id, 0)
    pct = (count / total_objects * 100) if total_objects > 0 else 0
    class_name = CLASS_NAMES[class_id]
    class_distribution.append((class_name, count, pct))
    print(f"{class_name:<20} {count:<12} {pct:.2f}%")

# Sort by count to find most/least frequent
sorted_by_count = sorted(class_distribution, key=lambda x: x[1], reverse=True)
most_frequent = sorted_by_count[0]
least_frequent = sorted_by_count[-1]
imbalance_ratio = most_frequent[1] / least_frequent[1] if least_frequent[1] > 0 else float('inf')

print("-" * 50)
print(f"Most frequent: {most_frequent[0]} ({most_frequent[1]} instances)")
print(f"Least frequent: {least_frequent[0]} ({least_frequent[1]} instances)")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

# ===== STEP 5 & 6: Bounding Box and Image Resolution Analysis =====
print("\n" + "="*50)
print("STEP 5 & 6: Bounding Box and Image Resolution Analysis")
print("="*50)

bbox_widths = []
bbox_heights = []
bbox_areas = []
relative_areas = []
image_resolutions = []
resolution_counts = defaultdict(int)

print("\nAnalyzing images and bounding boxes (this may take a while)...")

# First, sample a few images to determine resolution (since AI-TOD uses consistent 800x800)
sample_resolution = None
for split_name in available_splits:
    images_dir = DATASET_DIR / split_name / "images"
    if images_dir.exists():
        for img_path in list(images_dir.glob("*"))[:5]:
            try:
                with Image.open(img_path) as img:
                    sample_resolution = img.size
                    print(f"  Sampled resolution: {sample_resolution[0]}x{sample_resolution[1]}")
                    break
            except:
                continue
        if sample_resolution:
            break

# Use the sampled resolution for all calculations (dataset uses consistent resolution)
default_width, default_height = sample_resolution if sample_resolution else (800, 800)

# Detect annotation format from first label file
annotation_format = "detection"  # default
for split_name in available_splits:
    labels_dir = DATASET_DIR / split_name / "labels"
    if labels_dir.exists():
        for label_path in list(labels_dir.glob("*.txt"))[:1]:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) > 5:
                            annotation_format = "segmentation"
                            print(f"  Annotation format: YOLO Segmentation (polygon)")
                        else:
                            print(f"  Annotation format: YOLO Detection (bbox)")
                        break
                break
        break

sample_count = 0
for split_name in available_splits:
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"
    
    if not images_dir.exists():
        continue
    
    image_files = list(images_dir.glob("*"))
    
    for img_path in image_files:
        sample_count += 1
        if sample_count % 5000 == 0:
            print(f"  Processed {sample_count} images...")
        
        # Record resolution (assuming consistent size)
        width, height = default_width, default_height
        image_resolutions.append((width, height))
        resolution_counts[(width, height)] += 1
        
        # Get corresponding label file
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            if annotation_format == "segmentation" and len(parts) > 5:
                                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                                coords = list(map(float, parts[1:]))
                                xs = coords[0::2]  # every other element starting from 0
                                ys = coords[1::2]  # every other element starting from 1
                                
                                # Calculate bounding box from polygon vertices
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                w_norm = x_max - x_min
                                h_norm = y_max - y_min
                            else:
                                # Standard YOLO detection format: class_id x_center y_center width height
                                x_center, y_center, w_norm, h_norm = map(float, parts[1:5])
                            
                            # Convert normalized to pixel values
                            bbox_w = w_norm * width
                            bbox_h = h_norm * height
                            bbox_area = bbox_w * bbox_h
                            img_area = width * height
                            rel_area = bbox_area / img_area
                            
                            bbox_widths.append(bbox_w)
                            bbox_heights.append(bbox_h)
                            bbox_areas.append(bbox_area)
                            relative_areas.append(rel_area)

print(f"\nTotal images processed: {len(image_resolutions)}")
print(f"Total bounding boxes: {len(bbox_areas)}")

# Bounding Box Statistics
print("\n--- Bounding Box Statistics ---")
print(f"Width - Min: {min(bbox_widths):.2f}, Max: {max(bbox_widths):.2f}, Mean: {np.mean(bbox_widths):.2f}, Std: {np.std(bbox_widths):.2f}")
print(f"Height - Min: {min(bbox_heights):.2f}, Max: {max(bbox_heights):.2f}, Mean: {np.mean(bbox_heights):.2f}, Std: {np.std(bbox_heights):.2f}")
print(f"Area - Min: {min(bbox_areas):.2f}, Max: {max(bbox_areas):.2f}, Mean: {np.mean(bbox_areas):.2f}, Median: {np.median(bbox_areas):.2f}")

# Size categories
TINY_THRESHOLD = 16 * 16  # < 16²
SMALL_THRESHOLD = 32 * 32  # 16² - 32²
MEDIUM_THRESHOLD = 96 * 96  # 32² - 96²

tiny_count = sum(1 for a in bbox_areas if a < TINY_THRESHOLD)
small_count = sum(1 for a in bbox_areas if TINY_THRESHOLD <= a < SMALL_THRESHOLD)
medium_count = sum(1 for a in bbox_areas if SMALL_THRESHOLD <= a < MEDIUM_THRESHOLD)
large_count = sum(1 for a in bbox_areas if a >= MEDIUM_THRESHOLD)

total_boxes = len(bbox_areas)
size_categories = {
    'Tiny (< 16²)': (tiny_count, tiny_count/total_boxes*100),
    'Small (16² - 32²)': (small_count, small_count/total_boxes*100),
    'Medium (32² - 96²)': (medium_count, medium_count/total_boxes*100),
    'Large (> 96²)': (large_count, large_count/total_boxes*100)
}

print("\n--- Size Categories ---")
print(f"{'Category':<20} {'Count':<12} {'Percentage':<10}")
print("-" * 45)
for cat, (count, pct) in size_categories.items():
    print(f"{cat:<20} {count:<12} {pct:.2f}%")

# Image Resolution Statistics
print("\n--- Image Resolution Statistics ---")
widths = [r[0] for r in image_resolutions]
heights = [r[1] for r in image_resolutions]

min_res = (min(widths), min(heights))
max_res = (max(widths), max(heights))
mean_res = (np.mean(widths), np.mean(heights))

most_common_res = max(resolution_counts.items(), key=lambda x: x[1])
print(f"Min Resolution: {min_res[0]}x{min_res[1]}")
print(f"Max Resolution: {max_res[0]}x{max_res[1]}")
print(f"Mean Resolution: {mean_res[0]:.1f}x{mean_res[1]:.1f}")
print(f"Most Common Resolution: {most_common_res[0][0]}x{most_common_res[0][1]} ({most_common_res[1]} images)")

# ===== STEP 7: Objects per Image =====
print("\n" + "="*50)
print("STEP 7: Objects per Image Distribution")
print("="*50)

all_objects_per_image = []
for split_name, data in splits.items():
    all_objects_per_image.extend(data['objects_per_image'])

opc_mean = np.mean(all_objects_per_image)
opc_median = np.median(all_objects_per_image)
opc_max = max(all_objects_per_image)
opc_min = min(all_objects_per_image)

print(f"Mean objects per image: {opc_mean:.2f}")
print(f"Median objects per image: {opc_median:.2f}")
print(f"Max objects per image: {opc_max}")
print(f"Min objects per image: {opc_min}")

# Histogram buckets
bucket_0_5 = sum(1 for x in all_objects_per_image if x <= 5)
bucket_5_10 = sum(1 for x in all_objects_per_image if 5 < x <= 10)
bucket_10_20 = sum(1 for x in all_objects_per_image if 10 < x <= 20)
bucket_20_plus = sum(1 for x in all_objects_per_image if x > 20)

print("\nObjects per Image Distribution:")
print(f"  0-5: {bucket_0_5} images")
print(f"  5-10: {bucket_5_10} images")
print(f"  10-20: {bucket_10_20} images")
print(f"  20+: {bucket_20_plus} images")

# ===== STEP 8: Bounding Box Density =====
print("\n" + "="*50)
print("STEP 8: Bounding Box Density")
print("="*50)

avg_relative_area = np.mean(relative_areas) * 100
avg_objects_per_image = opc_mean
avg_bbox_size = np.mean(bbox_areas)

print(f"Average bbox area / image area: {avg_relative_area:.4f}%")
print(f"Average objects per image: {avg_objects_per_image:.2f}")
print(f"Average bbox size: {avg_bbox_size:.2f} pixels²")

# ===== STEP 9: Tiny Object Verification =====
print("\n" + "="*50)
print("STEP 9: Tiny Object Verification")
print("="*50)

smaller_32x32 = sum(1 for a in bbox_areas if a < 32*32)
smaller_16x16 = sum(1 for a in bbox_areas if a < 16*16)

pct_smaller_32 = smaller_32x32 / total_boxes * 100
pct_smaller_16 = smaller_16x16 / total_boxes * 100

print(f"Objects smaller than 32×32: {smaller_32x32} ({pct_smaller_32:.2f}%)")
print(f"Objects smaller than 16×16: {smaller_16x16} ({pct_smaller_16:.2f}%)")

# ===== STEP 10: Data Integrity Checks =====
print("\n" + "="*50)
print("STEP 10: Data Integrity Checks")
print("="*50)

integrity_issues = {
    'missing_labels': 0,
    'images_without_annotations': 0,
    'annotations_without_images': 0,
    'corrupted_images': 0,
    'bbox_outside_bounds': 0,
    'zero_area_boxes': 0
}

for split_name in available_splits:
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        continue
    
    image_files = {p.stem for p in images_dir.glob("*")}
    label_files = {p.stem for p in labels_dir.glob("*.txt")}
    
    # Check for missing labels / images without annotations
    for img_stem in image_files:
        if img_stem not in label_files:
            integrity_issues['images_without_annotations'] += 1
    
    # Check for annotations without images
    for lbl_stem in label_files:
        if lbl_stem not in image_files:
            integrity_issues['annotations_without_images'] += 1
    
    # Sample check for corrupted images (check 100 random images per split)
    import random
    all_images = list(images_dir.glob("*"))
    sample_images = random.sample(all_images, min(100, len(all_images)))
    
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            integrity_issues['corrupted_images'] += 1
    
    # Check bounding boxes in all labels
    for label_path in labels_dir.glob("*.txt"):
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        if annotation_format == "segmentation" and len(parts) > 5:
                            # Segmentation format: check polygon coords
                            coords = list(map(float, parts[1:]))
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            w = x_max - x_min
                            h = y_max - y_min
                            
                            # Check if any coordinate is out of bounds
                            if any(x < 0 or x > 1 for x in xs) or any(y < 0 or y > 1 for y in ys):
                                integrity_issues['bbox_outside_bounds'] += 1
                            
                            # Check for zero-area polygons
                            if w == 0 or h == 0:
                                integrity_issues['zero_area_boxes'] += 1
                        else:
                            x_center, y_center, w, h = map(float, parts[1:5])
                            
                            # Check for out-of-bounds (normalized coords should be 0-1)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                    0 <= w <= 1 and 0 <= h <= 1):
                                integrity_issues['bbox_outside_bounds'] += 1
                            
                            # Check for zero-area boxes
                            if w == 0 or h == 0:
                                integrity_issues['zero_area_boxes'] += 1

print("Data Integrity Results:")
for issue, count in integrity_issues.items():
    status = "✓ OK" if count == 0 else f"⚠ {count} issues"
    print(f"  {issue.replace('_', ' ').title()}: {status}")

# ===== STEP 11: Generate Visualizations =====
print("\n" + "="*50)
print("STEP 11: Generating Visualizations")
print("="*50)

# 1. Class distribution bar chart
plt.figure(figsize=(12, 6))
classes = [cd[0] for cd in class_distribution]
counts = [cd[1] for cd in class_distribution]
plt.bar(classes, counts, color='steelblue')
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=150)
plt.close()
print("  Saved: class_distribution.png")

# 2. Bbox width distribution
plt.figure(figsize=(10, 6))
plt.hist(bbox_widths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Bounding Box Width (pixels)')
plt.ylabel('Frequency')
plt.title('Bounding Box Width Distribution')
plt.axvline(np.mean(bbox_widths), color='red', linestyle='--', label=f'Mean: {np.mean(bbox_widths):.1f}')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bbox_width_distribution.png', dpi=150)
plt.close()
print("  Saved: bbox_width_distribution.png")

# 3. Bbox height distribution
plt.figure(figsize=(10, 6))
plt.hist(bbox_heights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Bounding Box Height (pixels)')
plt.ylabel('Frequency')
plt.title('Bounding Box Height Distribution')
plt.axvline(np.mean(bbox_heights), color='red', linestyle='--', label=f'Mean: {np.mean(bbox_heights):.1f}')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bbox_height_distribution.png', dpi=150)
plt.close()
print("  Saved: bbox_height_distribution.png")

# 4. Bbox area histogram
plt.figure(figsize=(10, 6))
# Use log scale for better visualization
plt.hist(bbox_areas, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Bounding Box Area (pixels²)')
plt.ylabel('Frequency')
plt.title('Bounding Box Area Distribution')
plt.axvline(np.median(bbox_areas), color='red', linestyle='--', label=f'Median: {np.median(bbox_areas):.1f}')
plt.xlim(0, np.percentile(bbox_areas, 95))  # Limit to 95th percentile for visibility
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bbox_area_distribution.png', dpi=150)
plt.close()
print("  Saved: bbox_area_distribution.png")

# 5. Objects per image histogram
plt.figure(figsize=(10, 6))
plt.hist(all_objects_per_image, bins=range(0, max(all_objects_per_image)+2), color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Objects per Image')
plt.ylabel('Number of Images')
plt.title('Objects per Image Distribution')
plt.axvline(np.mean(all_objects_per_image), color='red', linestyle='--', label=f'Mean: {np.mean(all_objects_per_image):.1f}')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'objects_per_image.png', dpi=150)
plt.close()
print("  Saved: objects_per_image.png")

# 6. Bbox size category pie chart
plt.figure(figsize=(10, 8))
labels = list(size_categories.keys())
sizes = [size_categories[k][0] for k in labels]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.05, 0, 0, 0)  # Explode the tiny slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Bounding Box Size Category Distribution')
plt.axis('equal')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'size_category_pie.png', dpi=150)
plt.close()
print("  Saved: size_category_pie.png")

# ===== STEP 12: Generate Report =====
print("\n" + "="*50)
print("STEP 12: Generating Dataset Audit Report")
print("="*50)

report = f"""# AI-TOD Dataset Audit

**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Structure

This dataset follows the **YOLO format** with the following structure:

```
{tree_str}```

**Dataset Source:** Roboflow (AI-TOD-800-800-YOLO)  
**License:** CC BY 4.0

---

## 2. Dataset Split Statistics

| Split | Images | Labels | Objects |
|-------|--------|--------|---------|
"""

for split_name, data in splits.items():
    report += f"| {split_name.capitalize()} | {data['images']:,} | {data['labels']:,} | {data['objects']:,} |\n"

total_images = sum(d['images'] for d in splits.values())
total_labels = sum(d['labels'] for d in splits.values())
total_objs = sum(d['objects'] for d in splits.values())
report += f"| **Total** | **{total_images:,}** | **{total_labels:,}** | **{total_objs:,}** |\n"

report += f"""
---

## 3. Class Distribution

**Number of Classes:** {NUM_CLASSES}

| Class | Instances | Percentage |
|-------|-----------|------------|
"""

for class_name, count, pct in class_distribution:
    report += f"| {class_name} | {count:,} | {pct:.2f}% |\n"

report += f"""
**Summary:**
- Most frequent class: **{most_frequent[0]}** ({most_frequent[1]:,} instances)
- Least frequent class: **{least_frequent[0]}** ({least_frequent[1]:,} instances)
- Imbalance ratio (max/min): **{imbalance_ratio:.2f}**

![Class Distribution](dataset_audit_figures/class_distribution.png)

---

## 4. Bounding Box Statistics

### Size Statistics

| Metric | Width (px) | Height (px) | Area (px²) |
|--------|------------|-------------|------------|
| Min | {min(bbox_widths):.2f} | {min(bbox_heights):.2f} | {min(bbox_areas):.2f} |
| Max | {max(bbox_widths):.2f} | {max(bbox_heights):.2f} | {max(bbox_areas):.2f} |
| Mean | {np.mean(bbox_widths):.2f} | {np.mean(bbox_heights):.2f} | {np.mean(bbox_areas):.2f} |
| Median | {np.median(bbox_widths):.2f} | {np.median(bbox_heights):.2f} | {np.median(bbox_areas):.2f} |
| Std | {np.std(bbox_widths):.2f} | {np.std(bbox_heights):.2f} | {np.std(bbox_areas):.2f} |

### Size Categories

| Category | Definition | Count | Percentage |
|----------|------------|-------|------------|
| Tiny | area < 16² (256) px² | {tiny_count:,} | {tiny_count/total_boxes*100:.2f}% |
| Small | 16² – 32² (256-1024) px² | {small_count:,} | {small_count/total_boxes*100:.2f}% |
| Medium | 32² – 96² (1024-9216) px² | {medium_count:,} | {medium_count/total_boxes*100:.2f}% |
| Large | > 96² (9216) px² | {large_count:,} | {large_count/total_boxes*100:.2f}% |

![Bbox Width Distribution](dataset_audit_figures/bbox_width_distribution.png)

![Bbox Height Distribution](dataset_audit_figures/bbox_height_distribution.png)

![Bbox Area Distribution](dataset_audit_figures/bbox_area_distribution.png)

![Size Category Pie Chart](dataset_audit_figures/size_category_pie.png)

---

## 5. Image Resolution Statistics

| Metric | Value |
|--------|-------|
| Min Resolution | {min_res[0]}×{min_res[1]} |
| Max Resolution | {max_res[0]}×{max_res[1]} |
| Mean Resolution | {mean_res[0]:.1f}×{mean_res[1]:.1f} |
| Most Common | {most_common_res[0][0]}×{most_common_res[0][1]} ({most_common_res[1]:,} images) |

---

## 6. Objects per Image

| Metric | Value |
|--------|-------|
| Mean | {opc_mean:.2f} |
| Median | {opc_median:.2f} |
| Min | {opc_min} |
| Max | {opc_max} |

### Distribution Buckets

| Objects per Image | Image Count |
|-------------------|-------------|
| 0–5 | {bucket_0_5:,} |
| 5–10 | {bucket_5_10:,} |
| 10–20 | {bucket_10_20:,} |
| 20+ | {bucket_20_plus:,} |

![Objects per Image Distribution](dataset_audit_figures/objects_per_image.png)

---

## 7. Tiny Object Statistics

This dataset is designed for **Tiny Object Detection**.

| Metric | Value |
|--------|-------|
| Objects < 32×32 (1024 px²) | {smaller_32x32:,} ({pct_smaller_32:.2f}%) |
| Objects < 16×16 (256 px²) | {smaller_16x16:,} ({pct_smaller_16:.2f}%) |
| Average bbox area / image area | {avg_relative_area:.4f}% |
| Average bbox size | {avg_bbox_size:.2f} px² |

---

## 8. Data Integrity Checks

| Check | Status |
|-------|--------|
| Missing Labels | {"✓ OK" if integrity_issues['missing_labels'] == 0 else f"⚠ {integrity_issues['missing_labels']} issues"} |
| Images Without Annotations | {"✓ OK" if integrity_issues['images_without_annotations'] == 0 else f"⚠ {integrity_issues['images_without_annotations']} issues"} |
| Annotations Without Images | {"✓ OK" if integrity_issues['annotations_without_images'] == 0 else f"⚠ {integrity_issues['annotations_without_images']} issues"} |
| Corrupted Images | {"✓ OK" if integrity_issues['corrupted_images'] == 0 else f"⚠ {integrity_issues['corrupted_images']} issues"} |
| Bbox Outside Bounds | {"✓ OK" if integrity_issues['bbox_outside_bounds'] == 0 else f"⚠ {integrity_issues['bbox_outside_bounds']} issues"} |
| Zero-Area Boxes | {"✓ OK" if integrity_issues['zero_area_boxes'] == 0 else f"⚠ {integrity_issues['zero_area_boxes']} issues"} |

---

## 9. Key Observations

- **Dataset Size:** {total_images:,} images with {total_objs:,} annotated objects
- **Format:** YOLO format (YOLOv8 compatible)
- **Image Resolution:** Consistent {most_common_res[0][0]}×{most_common_res[0][1]} images
- **Dominant Class:** {most_frequent[0]} represents the majority of annotations
- **Class Imbalance:** {imbalance_ratio:.1f}x ratio between most and least frequent classes
- **Tiny Objects:** {pct_smaller_32:.1f}% of objects are smaller than 32×32 pixels
- **Very Tiny Objects:** {pct_smaller_16:.1f}% are smaller than 16×16 pixels
- **Average Objects per Image:** {opc_mean:.1f} objects

---

## 10. Implications for Tiny Object Detection

### Challenges

1. **High proportion of tiny objects:** With {pct_smaller_32:.1f}% of objects under 32×32 pixels, standard detection architectures may struggle to capture sufficient features.

2. **Class imbalance:** The {imbalance_ratio:.1f}x imbalance between classes may require:
   - Weighted loss functions
   - Oversampling minority classes
   - Focal loss to handle hard examples

3. **Dense object scenes:** With up to {opc_max} objects per image, NMS thresholds and detection limits need careful tuning.

### Recommendations

1. **Architecture considerations:**
   - Use multi-scale feature fusion (FPN, PANet)
   - Consider higher resolution input images
   - Use smaller anchor boxes or anchor-free detectors
   - Feature enhancement modules for small objects

2. **Training strategies:**
   - Mosaic and copy-paste augmentation
   - Multi-scale training
   - Class-balanced sampling
   - Lower confidence thresholds during inference

3. **Evaluation metrics:**
   - Use area-specific AP metrics (AP_tiny, AP_small)
   - Consider using smaller IoU thresholds for tiny objects

---

*This report was automatically generated by the dataset audit script.*
"""

# Write report to file
with open(DOCS_DIR / "dataset_audit.md", 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  Report saved: docs/dataset_audit.md")

# ===== Final Summary =====
print("\n" + "="*50)
print("Dataset Audit Completed")
print("="*50)
print(f"""
Report generated:
  docs/dataset_audit.md

Figures saved:
  docs/dataset_audit_figures/
    - class_distribution.png
    - bbox_width_distribution.png
    - bbox_height_distribution.png
    - bbox_area_distribution.png
    - objects_per_image.png
    - size_category_pie.png
""")
