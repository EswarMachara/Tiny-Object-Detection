#!/usr/bin/env python3
"""
SAHI (Slicing Aided Hyper Inference) Evaluation for Tiny Object Detection

SAHI improves tiny object detection by:
1. Slicing images into overlapping patches (higher effective resolution)
2. Running detection on each patch independently
3. Merging predictions with NMS

For an 8×8 px object on 800×800 image at P3 (stride 8): 1×1 feature pixel.
With 400×400 slices, the same object becomes 2×2 feature pixels — 4× more
information for the detector.

Usage:
    python scripts/evaluate_sahi.py \\
        --weights results/taam/taam_xxx/best.pt \\
        --slice-size 400 --overlap 0.2 --compare-standard

Requirements:
    pip install sahi
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root and scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils import (
        get_repo_root,
        print_section,
        get_timestamp,
        ensure_dir
    )
except ImportError:
    def get_repo_root():
        return Path(__file__).parent.parent
    def print_section(title):
        print(f"\n{'='*60}\n {title}\n{'='*60}")
    def get_timestamp():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)


# ── Class names ──────────────────────────────────────────────
CLASS_NAMES = [
    'airplane', 'bridge', 'person', 'ship',
    'storage-tank', 'swimming-pool', 'vehicle', 'wind-mill'
]
NUM_CLASSES = len(CLASS_NAMES)


# ═══════════════════════════════════════════════════════════════
#  Module Registration
# ═══════════════════════════════════════════════════════════════

def register_taam_modules():
    """Register TAAM modules so YOLO can unpickle model weights."""
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
        pass  # Not a TAAM model


# ═══════════════════════════════════════════════════════════════
#  Ground Truth Loading
# ═══════════════════════════════════════════════════════════════

def get_image_dimensions(image_path):
    """Get image width and height without loading full image."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def load_ground_truth(labels_dir, images_dir):
    """
    Load GT annotations from YOLO-format label files.

    Returns:
        dict: {image_stem: [(class_id, x1, y1, x2, y2), ...]}
              in absolute pixel coordinates.
    """
    gt = {}
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)

    if not labels_path.exists():
        print(f"ERROR: Labels directory not found: {labels_path}")
        sys.exit(1)

    # Build a map of image stems → image files for dimension lookup
    image_files = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for f in images_path.glob(ext):
            image_files[f.stem] = f

    for label_file in sorted(labels_path.glob("*.txt")):
        stem = label_file.stem
        img_file = image_files.get(stem)
        if img_file is None:
            continue

        img_w, img_h = get_image_dimensions(img_file)
        boxes = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalised xywh → absolute xyxy
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                boxes.append((cls_id, x1, y1, x2, y2))
        gt[stem] = boxes

    return gt


# ═══════════════════════════════════════════════════════════════
#  mAP Computation (COCO-style, all-point interpolation)
# ═══════════════════════════════════════════════════════════════

def compute_iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_all_points(precision, recall):
    """
    Compute AP using all-point interpolation (COCO style).
    Monotonically decreasing precision, integrated over recall.
    """
    if len(precision) == 0:
        return 0.0
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 1e-3]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return float(ap)


def evaluate_detections(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth at a given IoU threshold.

    Args:
        predictions: dict {image_stem: [(cls_id, conf, x1, y1, x2, y2), ...]}
        ground_truth: dict {image_stem: [(cls_id, x1, y1, x2, y2), ...]}
        iou_threshold: IoU threshold for a true positive match

    Returns:
        dict with mAP50, per-class AP, precision, recall
    """
    # Collect all detections and GT counts per class
    all_dets = defaultdict(list)   # class_id → [(conf, is_tp, image)]
    num_gt = defaultdict(int)      # class_id → count

    all_images = set(ground_truth.keys()) | set(predictions.keys())

    for img_stem in all_images:
        gt_boxes = ground_truth.get(img_stem, [])
        pr_boxes = predictions.get(img_stem, [])

        # Count GT per class
        for g in gt_boxes:
            num_gt[g[0]] += 1

        # Sort preds by confidence (high → low)
        pr_boxes = sorted(pr_boxes, key=lambda x: x[1], reverse=True)
        matched_gt = [False] * len(gt_boxes)

        for pred in pr_boxes:
            cls_id, conf = pred[0], pred[1]
            pred_box = pred[2:6]

            best_iou = 0.0
            best_idx = -1
            for gi, g in enumerate(gt_boxes):
                if g[0] != cls_id or matched_gt[gi]:
                    continue
                iou = compute_iou(pred_box, g[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt[best_idx] = True
                all_dets[cls_id].append((conf, True))
            else:
                all_dets[cls_id].append((conf, False))

    # Per-class AP
    per_class_ap = {}
    per_class_prec = {}
    per_class_rec = {}

    for cls_id in range(NUM_CLASSES):
        ngt = num_gt.get(cls_id, 0)
        dets = sorted(all_dets.get(cls_id, []), key=lambda x: x[0], reverse=True)

        if ngt == 0:
            per_class_ap[cls_id] = 0.0
            per_class_prec[cls_id] = 0.0
            per_class_rec[cls_id] = 0.0
            continue

        tp_cum = 0
        fp_cum = 0
        precs = []
        recs = []
        for _, is_tp in dets:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precs.append(tp_cum / (tp_cum + fp_cum))
            recs.append(tp_cum / ngt)

        per_class_ap[cls_id] = compute_ap_all_points(
            np.array(precs), np.array(recs)
        )
        per_class_prec[cls_id] = precs[-1] if precs else 0.0
        per_class_rec[cls_id] = recs[-1] if recs else 0.0

    mAP50 = np.mean(list(per_class_ap.values()))

    total_tp = sum(1 for dets_list in all_dets.values() for _, tp in dets_list if tp)
    total_fp = sum(1 for dets_list in all_dets.values() for _, tp in dets_list if not tp)
    total_gt = sum(num_gt.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0

    return {
        'mAP50': float(mAP50),
        'precision': float(precision),
        'recall': float(recall),
        'per_class_ap': {CLASS_NAMES[i]: float(per_class_ap[i]) for i in range(NUM_CLASSES)},
        'per_class_prec': {CLASS_NAMES[i]: float(per_class_prec[i]) for i in range(NUM_CLASSES)},
        'per_class_rec': {CLASS_NAMES[i]: float(per_class_rec[i]) for i in range(NUM_CLASSES)},
        'total_gt': int(total_gt),
        'total_detections': int(total_tp + total_fp),
        'true_positives': int(total_tp),
    }


# ═══════════════════════════════════════════════════════════════
#  SAHI Inference
# ═══════════════════════════════════════════════════════════════

def run_sahi_inference(model_path, images_dir, slice_h, slice_w,
                       overlap, conf, device, verbose=True):
    """
    Run SAHI sliced inference.

    Returns:
        dict {image_stem: [(cls_id, conf, x1, y1, x2, y2), ...]}
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
    )

    images_path = Path(images_dir)
    image_files = sorted(
        list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    )

    if verbose:
        print(f"\nSAHI inference on {len(image_files)} images")
        print(f"  Slice: {slice_w}×{slice_h}, overlap: {overlap*100:.0f}%")

    predictions = {}
    t0 = time.time()

    for idx, img_path in enumerate(image_files):
        if verbose and (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (len(image_files) - idx - 1) / speed
            print(f"  [{idx+1}/{len(image_files)}]  {speed:.1f} img/s  ETA {eta:.0f}s")

        result = get_sliced_prediction(
            str(img_path),
            det_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        preds = []
        for pred_obj in result.object_prediction_list:
            cls_id = pred_obj.category.id
            score = pred_obj.score.value
            bb = pred_obj.bbox
            preds.append((cls_id, score, bb.minx, bb.miny, bb.maxx, bb.maxy))

        predictions[img_path.stem] = preds

    elapsed = time.time() - t0
    total_preds = sum(len(v) for v in predictions.values())
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s  ({len(image_files)/elapsed:.1f} img/s)")
        print(f"  Total detections: {total_preds}")
        print(f"  Avg per image: {total_preds/len(image_files):.1f}")

    return predictions, elapsed


# ═══════════════════════════════════════════════════════════════
#  Standard (full-image) Inference
# ═══════════════════════════════════════════════════════════════

def run_standard_inference(model_path, images_dir, conf, imgsz,
                           device, verbose=True):
    """
    Run standard YOLO inference (no slicing).

    Returns:
        dict {image_stem: [(cls_id, conf, x1, y1, x2, y2), ...]}
    """
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    images_path = Path(images_dir)
    image_files = sorted(
        list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    )

    if verbose:
        print(f"\nStandard inference on {len(image_files)} images (imgsz={imgsz})")

    predictions = {}
    t0 = time.time()

    results_gen = model.predict(
        source=str(images_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        stream=True,
    )

    for idx, result in enumerate(results_gen):
        stem = Path(result.path).stem
        preds = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append((cls_id, score, x1, y1, x2, y2))
        predictions[stem] = preds

        if verbose and (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(image_files)}]")

    elapsed = time.time() - t0
    total_preds = sum(len(v) for v in predictions.values())
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s  ({len(image_files)/elapsed:.1f} img/s)")
        print(f"  Total detections: {total_preds}")

    return predictions, elapsed


# ═══════════════════════════════════════════════════════════════
#  Printing helpers
# ═══════════════════════════════════════════════════════════════

def print_metrics(label, metrics):
    """Pretty-print an evaluation result block."""
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  mAP@50:     {metrics['mAP50']*100:.2f}%")
    print(f"  Precision:  {metrics['precision']*100:.2f}%")
    print(f"  Recall:     {metrics['recall']*100:.2f}%")
    print(f"  Detections: {metrics['total_detections']}  (TP {metrics['true_positives']}, "
          f"GT {metrics['total_gt']})")
    print(f"\n  Per-class mAP@50:")
    for name in CLASS_NAMES:
        print(f"    {name:<15s} {metrics['per_class_ap'][name]*100:6.2f}%")


def print_comparison(std_m, sahi_m):
    """Side-by-side comparison table."""
    print(f"\n{'Metric':<16} {'Standard':>10} {'SAHI':>10} {'Δ':>10}")
    print("─" * 50)
    for key, label in [('mAP50', 'mAP50'), ('precision', 'Precision'),
                       ('recall', 'Recall')]:
        s = std_m[key] * 100
        h = sahi_m[key] * 100
        print(f"  {label:<14} {s:>9.2f}% {h:>9.2f}% {h-s:>+9.2f}%")

    print(f"\n  {'Class':<14} {'Standard':>10} {'SAHI':>10} {'Δ':>10}")
    print("  " + "─" * 46)
    for name in CLASS_NAMES:
        s = std_m['per_class_ap'][name] * 100
        h = sahi_m['per_class_ap'][name] * 100
        marker = ' ★' if (h - s) > 2 else ''
        print(f"  {name:<14} {s:>9.2f}% {h:>9.2f}% {h-s:>+9.2f}%{marker}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SAHI Evaluation for Tiny Object Detection')

    parser.add_argument('--weights', required=True,
                        help='Path to model weights (.pt)')
    parser.add_argument('--slice-size', type=int, default=400,
                        help='Slice width & height (default: 400)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap ratio 0-1 (default: 0.2)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for mAP (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=800,
                        help='Image size for standard inference (default: 800)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to AI_TOD dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--compare-standard', action='store_true',
                        help='Also run standard inference for comparison')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Alias for --compare-standard')

    args = parser.parse_args()
    if args.compare_baseline:
        args.compare_standard = True

    # ── Setup ──────────────────────────────────────────────────
    print_section("SAHI Evaluation for Tiny Object Detection")

    # Check SAHI
    try:
        import sahi
        print(f"✓ SAHI {sahi.__version__}")
    except ImportError:
        print("✗ SAHI not installed.  Run:  pip install sahi")
        sys.exit(1)

    # Register TAAM modules (needed to unpickle TAAM model weights)
    register_taam_modules()

    repo_root = get_repo_root()
    dataset_path = Path(args.data_path) if args.data_path else repo_root / "AI_TOD"
    test_images = dataset_path / "test" / "images"
    test_labels = dataset_path / "test" / "labels"

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: Weights not found: {weights}")
        sys.exit(1)
    if not test_images.exists():
        print(f"ERROR: Test images not found: {test_images}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else \
        repo_root / "results" / "sahi" / f"eval_{get_timestamp()}"
    ensure_dir(out_dir)

    print(f"  Weights:     {weights}")
    print(f"  Dataset:     {dataset_path}")
    print(f"  Slice:       {args.slice_size}×{args.slice_size}")
    print(f"  Overlap:     {args.overlap*100:.0f}%")
    print(f"  Conf:        {args.conf}")
    print(f"  Output:      {out_dir}")

    # ── Load ground truth ─────────────────────────────────────
    print("\nLoading ground truth...")
    gt = load_ground_truth(test_labels, test_images)
    total_objects = sum(len(v) for v in gt.values())
    print(f"  {len(gt)} images, {total_objects} objects")

    results = {}

    # ── SAHI inference ────────────────────────────────────────
    print_section("SAHI Sliced Inference")
    sahi_preds, sahi_time = run_sahi_inference(
        model_path=weights,
        images_dir=test_images,
        slice_h=args.slice_size,
        slice_w=args.slice_size,
        overlap=args.overlap,
        conf=args.conf,
        device=args.device,
    )

    print("\nEvaluating SAHI predictions...")
    sahi_metrics = evaluate_detections(sahi_preds, gt, args.iou_threshold)
    sahi_metrics['inference_time'] = sahi_time
    results['sahi'] = {
        'params': {
            'slice_size': args.slice_size,
            'overlap': args.overlap,
            'conf': args.conf,
        },
        'metrics': sahi_metrics,
    }
    print_metrics("SAHI Results", sahi_metrics)

    # ── Standard inference (optional) ─────────────────────────
    if args.compare_standard:
        print_section("Standard Full-Image Inference")
        std_preds, std_time = run_standard_inference(
            model_path=weights,
            images_dir=test_images,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )

        print("\nEvaluating standard predictions...")
        std_metrics = evaluate_detections(std_preds, gt, args.iou_threshold)
        std_metrics['inference_time'] = std_time
        results['standard'] = {
            'params': {'imgsz': args.imgsz, 'conf': args.conf},
            'metrics': std_metrics,
        }
        print_metrics("Standard Results", std_metrics)

        # ── Comparison ────────────────────────────────────────
        print_section("SAHI vs Standard — Comparison")
        print_comparison(std_metrics, sahi_metrics)

        improvement = (sahi_metrics['mAP50'] - std_metrics['mAP50']) * 100
        results['comparison'] = {
            'mAP50_delta': round(improvement, 2),
            'precision_delta': round((sahi_metrics['precision'] - std_metrics['precision']) * 100, 2),
            'recall_delta': round((sahi_metrics['recall'] - std_metrics['recall']) * 100, 2),
            'speed_ratio': round(sahi_time / std_time, 1) if std_time > 0 else None,
        }

    # ── Save ──────────────────────────────────────────────────
    out_file = out_dir / "sahi_evaluation_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save eval config
    config = {
        'weights': str(weights),
        'dataset': str(dataset_path),
        'slice_size': args.slice_size,
        'overlap': args.overlap,
        'conf': args.conf,
        'iou_threshold': args.iou_threshold,
        'imgsz': args.imgsz,
        'timestamp': get_timestamp(),
    }
    with open(out_dir / "eval_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    print_section("Summary")
    print(f"  SAHI mAP@50: {sahi_metrics['mAP50']*100:.2f}%")
    if 'comparison' in results:
        print(f"  Improvement: {results['comparison']['mAP50_delta']:+.2f}% mAP50")
        print(f"  Speed ratio: {results['comparison']['speed_ratio']}× slower")
    print(f"\n  Results saved to: {out_file}")
    print(f"  Output dir:       {out_dir}")

    return results


if __name__ == "__main__":
    main()
