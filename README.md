# Aerial-Bird-Drone-Detection-System-YOLOv8-CNN-Classification-Deep-Learning
End-to-end deep learning system for aerial bird vs drone classification and detection using CNN, MobileNetV2, and YOLOv8. Features a full pipeline from data analysis to deployment, optimized for CPU, supports real-time inference, and includes an interactive Streamlit web app.
# 🛩️ AERIAL BIRD & DRONE DETECTION SYSTEM
---

# **1. ABSTRACT**

This project presents a complete deep learning-based system designed to detect and classify aerial objects, specifically birds and drones, from image data. The system integrates both image classification and object detection techniques to provide not only identification of objects but also their precise spatial localization using bounding boxes.

The solution leverages Convolutional Neural Networks (CNNs) for classification and the YOLOv8 (You Only Look Once) architecture for real-time object detection. A key focus of this project is computational efficiency, enabling the entire pipeline to run effectively on CPU-based systems without requiring high-end GPU hardware.

The developed system achieves classification accuracy of up to 96% and detection performance (mAP50) of up to 90%, making it suitable for real-world applications such as airspace monitoring, wildlife conservation, and surveillance systems.

---

# **2. INTRODUCTION**

## **2.1 Background**

Monitoring aerial environments is critical in multiple domains including aviation safety, wildlife conservation, and security. Manual monitoring methods are time-consuming, error-prone, and inefficient for large-scale data processing.

With advancements in deep learning and computer vision, automated systems can now analyze large volumes of visual data quickly and accurately. This project aims to build a practical system that bridges theoretical machine learning concepts with real-world applications.

## **2.2 Problem Statement**

Manual analysis of aerial imagery suffers from:

* Low processing speed (20–30 images/hour)
* Human fatigue and inconsistency
* High operational costs
* Difficulty in scaling

The goal is to design an automated system that:

* Detects aerial objects (birds and drones)
* Classifies them accurately
* Locates them using bounding boxes
* Operates efficiently on standard hardware

---

# **3. OBJECTIVES**

* Develop a classification model for identifying birds and drones
* Implement object detection using YOLOv8
* Design a complete pipeline from data preprocessing to deployment
* Ensure CPU-based optimization for accessibility
* Build an interactive web application for user interaction
* Evaluate system performance using standard metrics

---

# **4. SYSTEM ARCHITECTURE**

## **4.1 Overall Workflow**

1. Data Analysis
2. Data Preprocessing & Augmentation
3. Model Training (Classification & Detection)
4. Model Evaluation
5. Deployment using Web Interface

## **4.2 Architecture Diagram (Conceptual)**

Dataset → Data Analysis → Preprocessing →
→ Classification Model (CNN / MobileNetV2)
→ Detection Model (YOLOv8)
→ Evaluation → Web Application (Streamlit)

---

# **5. METHODOLOGY**

## **5.1 Dataset Analysis**

* Validates dataset structure
* Detects class imbalance
* Identifies corrupted images
* Visualizes sample images

## **5.2 Data Preprocessing**

* Image resizing
* Normalization
* Data augmentation techniques:

  * Rotation
  * Zoom
  * Horizontal flipping
  * Brightness adjustment

## **5.3 Classification Models**

### **Custom CNN**

* Multiple convolution layers
* Batch normalization
* Dropout for regularization

### **MobileNetV2 (Transfer Learning)**

* Pre-trained on ImageNet
* Fine-tuned for binary classification
* Faster convergence and higher accuracy

## **5.4 Object Detection (YOLOv8)**

* Uses YOLOv8 Nano architecture
* Detects objects and draws bounding boxes
* Provides confidence scores

### Key Features:

* Real-time detection
* Lightweight model (3.2M parameters)
* CPU-optimized

## **5.5 Dataset Conversion for YOLO**

* Converts dataset into YOLO format
* Generates label files with bounding boxes
* Creates configuration file (data.yaml)

---

# **6. IMPLEMENTATION DETAILS**

## **6.1 Technologies Used**

* Python 3.8+
* TensorFlow / Keras
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy, Pandas
* Scikit-learn
* Matplotlib
* Streamlit

## **6.2 System Modules**

### **Module 1: Data Analysis**

* Checks dataset quality
* Ensures readiness for training

### **Module 2: Classification Training**

* Trains CNN and MobileNetV2
* Compares performance

### **Module 3: Dataset Conversion**

* Converts dataset into YOLO format

### **Module 4: Detection Training**

* Trains YOLOv8 model
* Validates detection performance

### **Module 5: Web Application**

* Allows user interaction
* Displays predictions and results

---

# **7. RESULTS AND PERFORMANCE**

## **7.1 Classification Performance**

* Accuracy: 92% – 96%
* Precision: ~93%
* Recall: ~92%

## **7.2 Detection Performance**

* mAP50: 85% – 90%
* Detection speed: ~30 ms per image

## **7.3 System Efficiency**

* Processes 3000+ images per hour
* Runs on CPU without GPU support

---

# **8. APPLICATIONS**

* Airport bird-strike prevention
* Drone surveillance systems
* Wildlife monitoring
* Environmental research
* Border security systems

---

# **9. ADVANTAGES**

* High accuracy and speed
* CPU-based implementation
* Scalable and modular design
* User-friendly interface
* Real-time detection capability

---

# **10. LIMITATIONS**

* Performance decreases in low-light conditions
* Difficulty detecting very small or distant objects
* Limited to two classes (bird and drone)
* Accuracy depends on dataset quality

---

# **11. FUTURE ENHANCEMENTS**

* Extend to multi-class detection
* Integrate real-time video processing
* Deploy on edge devices (Raspberry Pi, Jetson Nano)
* Improve accuracy with larger datasets
* Cloud deployment for scalability

---

# **12. CONCLUSION**

This project successfully demonstrates the development of a complete deep learning pipeline for aerial object detection and classification. By combining CNN-based classification with YOLOv8 detection, the system achieves high accuracy while maintaining computational efficiency.

The implementation proves that practical and scalable machine learning solutions can be developed without relying on expensive hardware. The project has strong real-world relevance and can be extended further for advanced applications in surveillance and environmental monitoring.

---

# **13. REFERENCES**

1. TensorFlow Documentation
2. YOLOv8 Ultralytics Documentation
3. ImageNet Dataset
4. OpenCV Documentation
5. Scikit-learn Documentation

---

# **14. APPENDIX**

## **A. Hardware Requirements**

* Minimum 8GB RAM
* CPU-based system
* 20GB storage

## **B. Software Requirements**

* Python 3.8+
* Required libraries (TensorFlow, OpenCV, etc.)

## **C. Execution Commands**

```bash
python data_analysis.py
python yolodataset.py
python yolo_inference.py
streamlit run app.py
```

---
## 📂 Dataset

Due to GitHub storage limitations, the dataset (bird and drone images) is not included in this repository.

You can download the dataset from the link below:

🔗 **Dataset Link:** Classification Dataset
(https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh)
 Object Detection Dataset (YOLOv8 Format)
(https://drive.google.com/drive/folders/114wV_igIhWldcG0HftNIZZsivrs8G22p)

After downloading, extract the dataset and place it in the project root directory as follows:
dataset/
├── train/
│ ├── bird/
│ └── drone/
├── val/
├── test/


# **FINAL REMARK**

This project represents a complete, practical, and deployable machine learning system that effectively bridges theoretical knowledge with real-world implementation.

---
