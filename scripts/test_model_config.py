#!/usr/bin/env python3
"""
Quick test to verify model configurations work before full training.

Usage:
    python scripts/test_model_config.py --model yolo11m.pt
    python scripts/test_model_config.py --model models/yolo11m_p2.yaml
"""

import argparse
import torch
from ultralytics import YOLO


def test_model(model_path: str):
    """Test model loading and forward pass."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)
    
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Get model info
    try:
        model.info()
    except Exception as e:
        print(f"Warning: Could not get model info: {e}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        # Create dummy input (batch=1, channels=3, height=800, width=800)
        dummy_input = torch.randn(1, 3, 800, 800)
        
        # Run inference
        results = model.predict(dummy_input, verbose=False)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {results[0].boxes.shape if results else 'No detections'}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test training mode (brief)
    print("\nTesting training initialization...")
    try:
        # Just check if we can set up training
        model.model.train()
        print(f"✓ Training mode works")
    except Exception as e:
        print(f"✗ Training mode failed: {e}")
        return False
    
    print(f"\n✓ All tests passed for {model_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test model configuration')
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                       help='Model path or config')
    args = parser.parse_args()
    
    # Test models
    models_to_test = [args.model]
    
    # Also test P2 if testing baseline
    if args.model == 'yolo11m.pt':
        models_to_test.append('models/yolo11m_p2.yaml')
    
    all_passed = True
    for model_path in models_to_test:
        try:
            if not test_model(model_path):
                all_passed = False
        except Exception as e:
            print(f"✗ Test failed for {model_path}: {e}")
            all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("All model tests PASSED!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Some tests FAILED!")
        print("="*60)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
