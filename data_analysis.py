import os
import random
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# 1. CHECK DATASET STRUCTURE
# -------------------------------
def check_structure(base_path):
    print("\n📂 Checking Dataset Structure...\n")
    
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            print(f"❌ Missing folder: {split_path}")
            continue
        
        print(f"✅ {split.upper()} folder found")
        
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            
            if os.path.isdir(category_path):
                print(f"   ├── {category}")

# -------------------------------
# 2. COUNT IMAGES PER CLASS
# -------------------------------
def count_images(base_path):
    print("\n📊 Image Count Per Class...\n")
    
    counts = {}
    
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(base_path, split)
        counts[split] = {}
        
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            
            if os.path.isdir(category_path):
                num_images = len(os.listdir(category_path))
                counts[split][category] = num_images
                print(f"{split.upper()} → {category}: {num_images}")
    
    return counts

# -------------------------------
# 3. CHECK CLASS IMBALANCE
# -------------------------------
def check_imbalance(counts):
    print("\n⚖️ Checking Class Distribution (Train Set)...\n")
    
    train_counts = counts["train"]
    
    total = sum(train_counts.values())
    
    for cls, count in train_counts.items():
        print(f"{cls}: {count/total:.2f} ({count} images)")

# -------------------------------
# 4. VISUALIZE SAMPLE IMAGES
# -------------------------------
def show_samples(base_path, label, num_samples=5):
    path = os.path.join(base_path, "train", label)
    
    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        return
    
    images = os.listdir(path)
    
    if len(images) == 0:
        print(f"❌ No images found in {path}")
        return
    
    plt.figure(figsize=(12,5))
    
    for i in range(num_samples):
        img_path = os.path.join(path, random.choice(images))
        
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    
    plt.show()

# -------------------------------
# 5. CHECK CORRUPTED IMAGES
# -------------------------------
def check_corrupted_images(base_path):
    print("\n🔍 Checking for Corrupted Images...\n")
    
    corrupted = 0
    
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(base_path, split)
        
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"❌ Corrupted: {file_path}")
                    corrupted += 1
    
    print(f"\nTotal Corrupted Images: {corrupted}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    
    base_path = "data"   # change if your folder name is different
    
    # Step 1: Structure
    check_structure(base_path)
    
    # Step 2: Count images
    counts = count_images(base_path)
    
    # Step 3: Imbalance
    check_imbalance(counts)
    
    # Step 4: Visualization
    show_samples(base_path, "bird")
    show_samples(base_path, "drone")
    
    # Step 5: Corruption check
    check_corrupted_images(base_path)