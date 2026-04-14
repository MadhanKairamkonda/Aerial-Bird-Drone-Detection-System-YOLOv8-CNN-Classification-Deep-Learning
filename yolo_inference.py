"""
YOLOv8 Training - CPU Optimized
Fast training for slow CPU with complete workflow
"""

import sys
import os
import subprocess
from pathlib import Path
import json
from datetime import datetime
import time

# ================================
# INSTALL ULTRALYTICS
# ================================

def install_ultralytics():
    """Install ultralytics safely"""
    try:
        import ultralytics
        print("✅ ultralytics already installed")
        return True
    except ImportError:
        print("📦 Installing ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
            print("✅ ultralytics installed")
            return True
        except Exception as e:
            print(f"❌ Error installing ultralytics: {e}")
            return False

# ================================
# IMPORTS
# ================================

if not install_ultralytics():
    print("❌ Cannot proceed without ultralytics")
    sys.exit(1)

from ultralytics import YOLO
import cv2
import numpy as np

# ================================
# CONFIGURATION FOR SLOW CPU
# ================================

class YOLOConfigCPU:
    """Optimized configuration for slow CPU"""
    
    def __init__(self):
        # Paths
        self.DATA_YAML = "dataset_yolo/data.yaml"
        self.RESULTS_DIR = "yolo_results"
        self.MODELS_DIR = "models"
        
        # Model (nano = smallest & fastest)
        self.MODEL_NAME = "yolov8n"  # Nano - 3.2M parameters
        
        # Training (optimized for slow CPU)
        self.EPOCHS = 5  # Reduced from 100
        self.IMGSZ = 416  # Reduced from 640
        self.BATCH_SIZE = 4  # Very small batch
        self.DEVICE = "cpu"  # Force CPU
        self.PATIENCE = 2  # Early stopping
        
        # Workers
        self.WORKERS = 0  # Single thread for CPU stability
        
        # Create directories
        Path(self.RESULTS_DIR).mkdir(exist_ok=True)
        Path(self.MODELS_DIR).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("YOLOv8 CPU-OPTIMIZED CONFIGURATION")
        print("="*60)
        print(f"Model: {self.MODEL_NAME} (Nano - 3.2M params)")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Image Size: {self.IMGSZ}x{self.IMGSZ}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Device: {self.DEVICE}")
        print("="*60 + "\n")

# ================================
# STEP 1: VERIFY SETUP
# ================================

def verify_setup(config):
    """Verify data.yaml and dataset structure"""
    
    print("\n" + "="*60)
    print("STEP 1: VERIFY SETUP")
    print("="*60)
    
    # Check data.yaml
    if not os.path.exists(config.DATA_YAML):
        print(f"\n❌ data.yaml not found at: {config.DATA_YAML}")
        print("\n💡 Solution: Run prepare_yolo_dataset.py first!")
        return False
    
    print(f"✅ data.yaml found: {config.DATA_YAML}")
    
    # Check dataset structure
    dataset_dir = Path(config.DATA_YAML).parent
    
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if dir_path.exists():
            img_count = len(list(dir_path.glob('*')))
            print(f"✅ {dir_name:20}: {img_count} files")
        else:
            print(f"❌ {dir_name:20}: NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Dataset structure incomplete!")
        return False
    
    print("\n✅ Setup verified!")
    return True

# ================================
# STEP 2: LOAD MODEL
# ================================

def load_model(config):
    """Load pre-trained YOLOv8 model"""
    
    print("\n" + "="*60)
    print("STEP 2: LOADING MODEL")
    print("="*60)
    
    print(f"\nLoading YOLOv8{config.MODEL_NAME}...")
    
    try:
        model = YOLO(f'{config.MODEL_NAME}.pt')
        print(f"✅ YOLOv8{config.MODEL_NAME} loaded successfully")
        print(f"   Parameters: 3.2M")
        print(f"   Speed: ~50ms (CPU)")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ================================
# STEP 3: TRAIN MODEL
# ================================

def train_model(model, config):
    """Train YOLOv8 model with CPU optimization"""
    
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Image Size: {config.IMGSZ}x{config.IMGSZ}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Early Stopping Patience: {config.PATIENCE}")
    print(f"\n⏱️  Estimated time: {config.EPOCHS * 5}-{config.EPOCHS * 10} minutes on slow CPU")
    print("\nStarting training...\n")
    
    start_time = time.time()
    
    try:
        results = model.train(
            data=config.DATA_YAML,
            epochs=config.EPOCHS,
            imgsz=config.IMGSZ,
            batch=config.BATCH_SIZE,
            device=config.DEVICE,
            patience=config.PATIENCE,
            save=True,
            project=config.RESULTS_DIR,
            name="bird_drone_detector",
            exist_ok=True,
            # CPU Optimizations
            workers=config.WORKERS,
            amp=False,  # Disable mixed precision on CPU
            # Training settings
            optimizer='SGD',
            lr0=0.005,  # Reduced learning rate
            lrf=0.0005,
            momentum=0.9,
            weight_decay=0.0005,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            # Callbacks
            verbose=True,
            plots=True
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Results saved to: {config.RESULTS_DIR}/bird_drone_detector/")
        
        return results, training_time
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return None, None

# ================================
# STEP 4: VALIDATE MODEL
# ================================

def validate_model(model, config):
    """Validate trained model"""
    
    print("\n" + "="*60)
    print("STEP 4: VALIDATING MODEL")
    print("="*60)
    
    print("\nValidating on test set...")
    
    try:
        metrics = model.val()
        
        print("\n✅ Validation Complete!")
        print(f"\nKey Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  Precision: {metrics.box.p[0] if hasattr(metrics.box, 'p') else 'N/A'}")
        print(f"  Recall: {metrics.box.r[0] if hasattr(metrics.box, 'r') else 'N/A'}")
        
        return metrics
    except Exception as e:
        print(f"⚠️  Validation error: {e}")
        return None

# ================================
# STEP 5: TEST INFERENCE
# ================================

def test_inference(config):
    """Test detection on sample images"""
    
    print("\n" + "="*60)
    print("STEP 5: TEST INFERENCE")
    print("="*60)
    
    # Load best model
    best_model_path = Path(config.RESULTS_DIR) / "bird_drone_detector" / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"❌ Best model not found at: {best_model_path}")
        return False
    
    print(f"\nLoading best model: {best_model_path}")
    
    try:
        best_model = YOLO(str(best_model_path))
        print("✅ Best model loaded")
    except Exception as e:
        print(f"❌ Error loading best model: {e}")
        return False
    
    # Find test images
    test_dir = Path(config.DATA_YAML).parent / "images" / "test"
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print(f"⚠️  No test images found in {test_dir}")
        return True
    
    print(f"\nRunning inference on {len(test_images[:3])} test images...")
    
    for idx, img_path in enumerate(test_images[:3]):
        try:
            print(f"\n  {idx+1}. Testing: {img_path.name}")
            
            results = best_model.predict(str(img_path), conf=0.3, verbose=False)
            result = results[0]
            
            print(f"     Detections: {len(result.boxes)}")
            
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                print(f"       • {class_name}: {confidence:.2%}")
        
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
    
    print("\n✅ Inference test complete!")
    
    return True

# ================================
# STEP 6: SAVE RESULTS
# ================================

def save_results(config, training_time, metrics=None):
    """Save training summary"""
    
    print("\n" + "="*60)
    print("STEP 6: SAVING RESULTS")
    print("="*60)
    
    # Copy best model to models directory
    best_model_src = Path(config.RESULTS_DIR) / "bird_drone_detector" / "weights" / "best.pt"
    best_model_dst = Path(config.MODELS_DIR) / "best_yolov8.pt"
    
    if best_model_src.exists():
        try:
            import shutil
            shutil.copy2(best_model_src, best_model_dst)
            print(f"✅ Model saved: {best_model_dst}")
        except Exception as e:
            print(f"⚠️  Could not copy model: {e}")
    
    # Create summary
    summary = {
        'model': 'YOLOv8 Nano',
        'dataset': 'Bird vs Drone',
        'training_time_minutes': round(training_time / 60, 2) if training_time else None,
        'epochs': config.EPOCHS,
        'image_size': config.IMGSZ,
        'batch_size': config.BATCH_SIZE,
        'device': config.DEVICE,
        'timestamp': datetime.now().isoformat(),
        'results_dir': str(Path(config.RESULTS_DIR).absolute()),
        'model_path': str((Path(config.MODELS_DIR) / "best_yolov8.pt").absolute())
    }
    
    summary_path = Path(config.RESULTS_DIR) / "training_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved: {summary_path}")
    
    return summary

# ================================
# MAIN PIPELINE
# ================================

def main():
    """Complete YOLOv8 training pipeline"""
    
    print("\n" + "="*70)
    print("🛩️  YOLOv8 BIRD vs DRONE DETECTION - CPU OPTIMIZED TRAINING")
    print("="*70)
    
    # Initialize config
    config = YOLOConfigCPU()
    
    # Step 1: Verify setup
    if not verify_setup(config):
        print("\n❌ Setup verification failed!")
        print("💡 Run: python prepare_yolo_dataset.py")
        return False
    
    # Step 2: Load model
    model = load_model(config)
    if model is None:
        print("\n❌ Failed to load model!")
        return False
    
    # Step 3: Train
    results, training_time = train_model(model, config)
    if results is None:
        print("\n❌ Training failed!")
        return False
    
    # Step 4: Validate
    metrics = validate_model(model, config)
    
    # Step 5: Test inference
    test_inference(config)
    
    # Step 6: Save results
    summary = save_results(config, training_time, metrics)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ COMPLETE TRAINING SUMMARY")
    print("="*70)
    print(f"\nModel: YOLOv8 Nano")
    print(f"Training Time: {summary['training_time_minutes']} minutes")
    print(f"Epochs: {summary['epochs']}")
    print(f"Image Size: {summary['image_size']}x{summary['image_size']}")
    print(f"Batch Size: {summary['batch_size']}")
    print(f"\nResults Directory: {summary['results_dir']}")
    print(f"Model Path: {summary['model_path']}")
    
    print("\n✨ Next Steps:")
    print("  1. Review results: python enhanced_data_analysis.py")
    print("  2. Deploy app: streamlit run streamlit_app_final.py")
    print("  3. Make predictions: Use the Streamlit web interface")
    
    print("\n" + "="*70)
    
    return True

# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)