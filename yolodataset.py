"""
Convert Classification Dataset to YOLOv8 Format
Converts folder structure (train/bird, train/drone) to YOLO format with bounding boxes
"""

import os
import cv2
import random
import shutil
from pathlib import Path
import json

# ================================
# CONFIGURATION
# ================================

SOURCE_DATASET = "dataset"           # Your current dataset
YOLO_DATASET = "dataset_yolo"        # Output YOLO format dataset
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Class mapping
CLASS_MAP = {
    'bird': 0,
    'drone': 1
}

# ================================
# CREATE DIRECTORY STRUCTURE
# ================================

def create_yolo_structure():
    """Create YOLOv8 compatible directory structure"""
    print("\n📂 Creating YOLOv8 Directory Structure...\n")
    
    base = Path(YOLO_DATASET)
    
    # Create main directories
    for split in ['train', 'val', 'test']:
        (base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("✅ YOLOv8 directory structure created:")
    print(f"   {YOLO_DATASET}/")
    print(f"   ├── images/")
    print(f"   │   ├── train/")
    print(f"   │   ├── val/")
    print(f"   │   └── test/")
    print(f"   └── labels/")
    print(f"       ├── train/")
    print(f"       ├── val/")
    print(f"       └── test/")

# ================================
# GENERATE BOUNDING BOXES
# ================================

def get_yolo_bbox(img_height, img_width):
    """
    Generate YOLO format bounding box for full image
    For single object detection, box covers most of the image
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to 0-1
    """
    
    # Bounding box that covers ~90% of image (center)
    bbox_width = 0.85  # 85% of image width
    bbox_height = 0.85  # 85% of image height
    
    x_center = 0.5  # Center of image
    y_center = 0.5  # Center of image
    
    return x_center, y_center, bbox_width, bbox_height

def create_yolo_label(class_name, img_height, img_width):
    """Create YOLO format label file content"""
    
    class_id = CLASS_MAP.get(class_name, 0)
    x_center, y_center, width, height = get_yolo_bbox(img_height, img_width)
    
    # YOLO format: class_id x_center y_center width height
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    return label_line

# ================================
# COPY AND CONVERT DATASET
# ================================

def copy_to_yolo_format():
    """Copy dataset from classification to YOLO format"""
    
    print("\n" + "="*60)
    print("Converting Dataset to YOLOv8 Format")
    print("="*60)
    
    # Collect all image files
    all_files = {
        'bird': [],
        'drone': []
    }
    
    # Gather files from train/val/test
    for split in ['train', 'val', 'test']:
        source_path = Path(SOURCE_DATASET) / split
        
        if not source_path.exists():
            print(f"⚠️  Skipping {split} - folder not found")
            continue
        
        for class_name in CLASS_MAP.keys():
            class_path = source_path / class_name
            
            if class_path.exists():
                image_files = [f for f in class_path.glob('*') 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                all_files[class_name].extend(image_files)
    
    # Shuffle and split
    total_images = sum(len(v) for v in all_files.values())
    print(f"\n📊 Total images to convert: {total_images}")
    
    # Split into train/val/test
    yolo_split = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for class_name, files in all_files.items():
        random.shuffle(files)
        
        n_train = int(len(files) * TRAIN_SPLIT)
        n_val = int(len(files) * VAL_SPLIT)
        
        yolo_split['train'].extend(files[:n_train])
        yolo_split['val'].extend(files[n_train:n_train+n_val])
        yolo_split['test'].extend(files[n_train+n_val:])
    
    # Copy files to YOLO structure
    total_converted = 0
    
    for split, image_list in yolo_split.items():
        print(f"\n📋 Processing {split.upper()} ({len(image_list)} images)...")
        
        for idx, image_path in enumerate(image_list):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"  ⚠️  Could not read: {image_path.name}")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Get class name from parent folder
                class_name = image_path.parent.name
                
                # Copy image to YOLO images folder
                output_img_path = Path(YOLO_DATASET) / 'images' / split / image_path.name
                shutil.copy2(image_path, output_img_path)
                
                # Create YOLO label file
                label_content = create_yolo_label(class_name, img_height, img_width)
                
                label_filename = image_path.stem + '.txt'
                output_label_path = Path(YOLO_DATASET) / 'labels' / split / label_filename
                
                with open(output_label_path, 'w') as f:
                    f.write(label_content)
                
                total_converted += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  ✅ Converted {idx + 1}/{len(image_list)} images")
            
            except Exception as e:
                print(f"  ❌ Error processing {image_path.name}: {e}")
    
    print(f"\n✅ Total converted: {total_converted} images")
    
    return total_converted

# ================================
# CREATE DATA.YAML
# ================================

def create_data_yaml():
    """Create data.yaml configuration for YOLOv8"""
    
    print("\n📝 Creating data.yaml...")
    
    # Get absolute path
    dataset_path = str(Path(YOLO_DATASET).absolute())
    
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Aerial Bird vs Drone Detection

path: {dataset_path}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 2

# Class names
names:
  0: bird
  1: drone
"""
    
    yaml_path = Path(YOLO_DATASET) / 'data.yaml'
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ data.yaml created at: {yaml_path}")
    
    return str(yaml_path)

# ================================
# VERIFY CONVERSION
# ================================

def verify_conversion():
    """Verify that conversion was successful"""
    
    print("\n" + "="*60)
    print("Verifying Conversion")
    print("="*60)
    
    yolo_path = Path(YOLO_DATASET)
    
    for split in ['train', 'val', 'test']:
        images_dir = yolo_path / 'images' / split
        labels_dir = yolo_path / 'labels' / split
        
        img_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {img_count}")
        print(f"  Labels: {label_count}")
        
        if img_count == label_count:
            print(f"  ✅ Matched!")
        else:
            print(f"  ❌ Mismatch!")
    
    # Sample label file
    print("\n📄 Sample label file content:")
    sample_labels = list((yolo_path / 'labels' / 'train').glob('*.txt'))
    
    if sample_labels:
        with open(sample_labels[0], 'r') as f:
            content = f.read()
            print(f"  {content}")
    
    print("\n✅ Conversion verification complete!")

# ================================
# MAIN EXECUTION
# ================================

def main():
    """Convert dataset to YOLOv8 format"""
    
    print("\n" + "="*60)
    print("🛩️  DATASET CONVERSION TO YOLOv8 FORMAT")
    print("="*60)
    
    print(f"\nSource Dataset: {SOURCE_DATASET}")
    print(f"Output Dataset: {YOLO_DATASET}")
    print(f"Train/Val/Test Split: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    
    # Step 1: Create directory structure
    create_yolo_structure()
    
    # Step 2: Copy and convert
    converted = copy_to_yolo_format()
    
    if converted == 0:
        print("\n❌ No images were converted!")
        return False
    
    # Step 3: Create data.yaml
    yaml_path = create_data_yaml()
    
    # Step 4: Verify
    verify_conversion()
    
    print("\n" + "="*60)
    print("✅ DATASET CONVERSION COMPLETE!")
    print("="*60)
    print(f"\n✨ Your dataset is ready for YOLOv8 training!")
    print(f"   Data YAML: {yaml_path}")
    print(f"   Dataset path: {YOLO_DATASET}/")
    print("\n   Next step: python yolov8_training_optimized.py")
    
    return True

# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Conversion failed. Please check your dataset structure.")
        exit(1)