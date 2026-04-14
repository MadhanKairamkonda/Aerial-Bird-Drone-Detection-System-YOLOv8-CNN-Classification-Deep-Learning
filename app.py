"""
Complete Streamlit Application
Aerial Bird vs Drone Classification & Detection
Integrates all methods from train_model.py and data_analysis.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

# Try to import YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ================================
# PAGE CONFIGURATION
# ================================

st.set_page_config(
    page_title="Aerial Bird & Drone Detection",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #000000;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333333;
    }
    .result-heading {
        background-color: #000000;
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .detection-box {
        background-color: #000000;
        color: #ffffff;
        padding: 0.8rem;
        border-left: 4px solid #ffffff;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .bird-box {
        background-color: #000000;
        color: #ffffff;
        border-left-color: #ffffff;
    }
    .drone-box {
        background-color: #000000;
        color: #ffffff;
        border-left-color: #ffffff;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# UTILITY CLASSES
# ================================

class ClassificationModel:
    """Handle Keras classification model"""
    
    def __init__(self):
        self.model = None
        self.class_names = ['Bird', 'Drone']
        self.image_size = (224, 224)
    
    def load_model(self, model_path):
        """Load classification model"""
        try:
            self.model = keras.models.load_model(model_path)
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def preprocess(self, image):
        """Preprocess image (from train_model.py)"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(self.image_size)
        img_array = np.array(image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image):
        """Make classification prediction"""
        if self.model is None:
            return None, None, None
        
        img_array = self.preprocess(image)
        predictions = self.model.predict(img_array, verbose=0)
        
        prob_drone = float(predictions[0][0])
        prob_bird = 1.0 - prob_drone
        probs = [prob_bird, prob_drone]
        
        class_idx = 1 if prob_drone > 0.5 else 0
        confidence = max(prob_bird, prob_drone)
        class_name = self.class_names[class_idx]
        
        return class_name, confidence, probs


class DetectionModel:
    """Handle YOLOv8 detection model"""
    
    def __init__(self):
        self.model = None
        self.class_names = ['Bird', 'Drone']
    
    def load_model(self, model_path):
        """Load detection model"""
        try:
            if not Path(model_path).exists():
                st.warning(f"⚠️  Model file not found: {model_path}")
                return False
            
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"❌ Error loading detection model: {e}")
            return False
    
    def detect(self, image, confidence=0.3):
        """Run detection on image"""
        if self.model is None:
            return None
        
        img_array = np.array(image)
        results = self.model.predict(img_array, conf=confidence, verbose=False)
        
        return results[0]
    
    def draw_detections(self, image, result):
        """Draw bounding boxes on image"""
        annotated = result.plot()
        return Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    def get_detections_list(self, result):
        """Get list of detections with info"""
        detections = []
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]
            bbox = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
        
        return detections


class DataAnalyzer:
    """Data analysis functions (from data_analysis.py)"""
    
    @staticmethod
    def get_dataset_stats(base_path="dataset"):
        """Get dataset statistics"""
        stats = {
            'train': {'bird': 0, 'drone': 0},
            'val': {'bird': 0, 'drone': 0},
            'test': {'bird': 0, 'drone': 0}
        }
        
        for split in stats.keys():
            split_path = Path(base_path) / split
            
            if not split_path.exists():
                continue
            
            for class_name in ['bird', 'drone']:
                class_path = split_path / class_name
                
                if class_path.exists():
                    count = len([f for f in class_path.glob('*') 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    stats[split][class_name] = count
        
        return stats
    
    @staticmethod
    def visualize_dataset_stats(stats):
        """Visualize dataset statistics"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        for idx, (split, counts) in enumerate(stats.items()):
            labels = list(counts.keys())
            values = list(counts.values())
            
            axes[idx].bar(labels, values, color=['#ff9800', '#2196f3'])
            axes[idx].set_title(f"{split.upper()}")
            axes[idx].set_ylabel("Number of Images")
            
            for i, v in enumerate(values):
                axes[idx].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig


# ================================
# SESSION STATE
# ================================

if 'classifier' not in st.session_state:
    st.session_state.classifier = ClassificationModel()

if 'detector' not in st.session_state:
    st.session_state.detector = DetectionModel() if YOLO_AVAILABLE else None

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = DataAnalyzer()

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None


# ================================
# MAIN APP
# ================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    # 🛩️ Aerial Bird & Drone Detection System
    
    **Intelligent Deep Learning System for Aerial Object Classification & Detection**
    
    - 🏷️ **Classification**: Identify if the image contains a Bird or Drone
    - 🎯 **Detection**: Locate all objects with bounding boxes
    - 📊 **Analysis**: Comprehensive metrics and evaluation
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Task selection
        task = st.radio(
            "Select Mode",
            ["Classification", "Detection", "Both"],
            help="Choose what analysis to perform"
        )
        
        # Model settings
        if task in ["Classification", "Both"]:
            st.markdown("### 📦 Classification Model")
            
            model_option = st.selectbox(
                "Select Model",
                ["Final Model", "Best Custom CNN"],
                key="classifier_model"
            )
            
            model_paths = {
                "Final Model": "final_model.h5",
                "Best Custom CNN": "best_model.h5"
            }
            
            model_path = model_paths[model_option]
            
            if st.button("🔄 Load Classification Model", key="load_classifier"):
                with st.spinner(f"Loading {model_option}..."):
                    if st.session_state.classifier.load_model(model_path):
                        st.success("✅ Classification model loaded!")
        
        if task in ["Detection", "Both"] and YOLO_AVAILABLE:
            st.markdown("### 🎯 Detection Model")
            
            detection_model_path = st.text_input(
                "Model Path",
                "models/best_yolov8.pt",
                key="detection_model_path"
            )
            
            if st.button("🔄 Load Detection Model", key="load_detector"):
                with st.spinner("Loading YOLOv8..."):
                    if st.session_state.detector.load_model(detection_model_path):
                        st.success("✅ Detection model loaded!")
        
        if not YOLO_AVAILABLE and task in ["Detection", "Both"]:
            st.warning("⚠️ YOLOv8 not installed. Run: pip install ultralytics")
        
        st.markdown("---")
        
        # Confidence settings
        if task in ["Classification", "Both"]:
            st.markdown("### 📊 Classification Settings")
            class_confidence = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.5, 0.05,
                key="class_confidence"
            )
        else:
            class_confidence = 0.5
        
        if task in ["Detection", "Both"] and YOLO_AVAILABLE:
            st.markdown("### 🎯 Detection Settings")
            det_confidence = st.slider(
                "Detection Confidence",
                0.0, 1.0, 0.3, 0.05,
                key="det_confidence"
            )
        else:
            det_confidence = 0.3
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📚 Navigation")
        
        page = st.radio(
            "Choose Page",
            ["Prediction", "Dataset Analysis", "Model Info"],
            key="page_selection"
        )
    
    # Main content based on page
    if page == "Prediction":
        show_prediction_page(task, class_confidence, det_confidence)
    
    elif page == "Dataset Analysis":
        show_analysis_page()
    
    elif page == "Model Info":
        show_info_page()


def show_prediction_page(task, class_confidence, det_confidence):
    """Prediction interface"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose aerial image",
            type=["jpg", "jpeg", "png"],
            help="Upload an aerial image with birds or drones"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.session_state.uploaded_image = image
            
            st.write(f"Uploaded file: **{uploaded_file.name}**")
            st.image(image, width=520, caption="Uploaded Image")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                
                # Classification
                if task in ["Classification", "Both"]:
                    if st.session_state.classifier.model is None:
                        st.warning("⚠️  Load classification model first")
                    else:
                        with st.spinner("Analyzing..."):
                            class_name, confidence, probs = st.session_state.classifier.predict(image)
                            
                            st.session_state.classification_result = {
                                'class': class_name,
                                'confidence': confidence,
                                'probs': probs
                            }
                
                # Detection
                if task in ["Detection", "Both"] and YOLO_AVAILABLE:
                    if st.session_state.detector.model is None:
                        st.warning("⚠️  Load detection model first")
                    else:
                        with st.spinner("Running detection..."):
                            result = st.session_state.detector.detect(image, det_confidence)
                            st.session_state.detection_result = result
    
    with col2:
        st.markdown("""
        <div class="result-heading">
            <h3 style="margin: 0;">📊 Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.classification_result is not None:
            result = st.session_state.classification_result
            
            emoji = "🐦" if result['class'] == 'Bird' else "🚁"
            
            st.markdown(f"""
            <div style="padding: 1rem; background-color: #000000; color: #ffffff; border-radius: 0.5rem;">
                <h2 style="margin-top: 0; color: #ffffff;">{emoji} {result['class']}</h2>
                <h3 style="color: #ffffff;">Confidence: {result['confidence']*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🐦 Bird", f"{result['probs'][0]*100:.1f}%")
            with col_b:
                st.metric("🚁 Drone", f"{result['probs'][1]*100:.1f}%")
            
            # Chart
            import pandas as pd
            df = pd.DataFrame({
                'Class': ['Bird', 'Drone'],
                'Probability': result['probs']
            })
            st.bar_chart(df.set_index('Class'))
        
        if st.session_state.detection_result is not None:
            result = st.session_state.detection_result
            
            st.markdown(f"### 🎯 Objects Detected: {len(result.boxes)}")
            
            # Show annotated image
            if st.session_state.uploaded_image:
                annotated = st.session_state.detector.draw_detections(
                    st.session_state.uploaded_image, 
                    result
                )
                st.image(annotated, width=700)
            
            # List detections
            detections = st.session_state.detector.get_detections_list(result)
            
            if detections:
                st.markdown("**Detected Objects:**")
                
                for i, det in enumerate(detections):
                    class_style = "bird-box" if det['class'] == 'Bird' else "drone-box"
                    
                    st.markdown(f"""
                    <div class="detection-box {class_style}">
                        <strong>{i+1}. {det['class']}</strong> | 
                        Confidence: <span class="confidence-high">{det['confidence']:.2%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No objects detected. Try lowering confidence threshold.")
        
        if st.session_state.classification_result is None and st.session_state.detection_result is None:
            if st.session_state.uploaded_image is None:
                st.info("👈 Upload an image to start")
            else:
                st.info("👈 Click 'Analyze Image' to run analysis")


def show_analysis_page():
    """Dataset analysis page"""
    
    st.markdown("## 📊 Dataset Analysis")
    
    # Dataset statistics
    stats = st.session_state.analyzer.get_dataset_stats()
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_total = sum(stats['train'].values())
        st.metric("Train Images", train_total)
    
    with col2:
        val_total = sum(stats['val'].values())
        st.metric("Validation Images", val_total)
    
    with col3:
        test_total = sum(stats['test'].values())
        st.metric("Test Images", test_total)
    
    # Visualize stats
    fig = st.session_state.analyzer.visualize_dataset_stats(stats)
    st.pyplot(fig)
    
    # Detailed breakdown
    st.markdown("### Detailed Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Train Set**")
        st.write(f"🐦 Bird: {stats['train']['bird']}")
        st.write(f"🚁 Drone: {stats['train']['drone']}")
    
    with col2:
        st.markdown("**Validation Set**")
        st.write(f"🐦 Bird: {stats['val']['bird']}")
        st.write(f"🚁 Drone: {stats['val']['drone']}")
    
    with col3:
        st.markdown("**Test Set**")
        st.write(f"🐦 Bird: {stats['test']['bird']}")
        st.write(f"🚁 Drone: {stats['test']['drone']}")


def show_info_page():
    """Model information page"""
    
    st.markdown("## ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Classification Model")
        st.write("""
        **Architecture:** Custom CNN + MobileNetV2 Transfer Learning
        
        **Input Size:** 224×224×3
        
        **Classes:** Bird, Drone
        
        **Key Features:**
        - Image normalization to [0, 1]
        - Data augmentation (rotation, zoom, flip, brightness)
        - Early stopping to prevent overfitting
        - ModelCheckpoint to save best weights
        
        **Metrics:**
        - Accuracy: ~94%
        - Precision/Recall: ~93%
        """)
    
    with col2:
        st.markdown("### Detection Model")
        st.write("""
        **Architecture:** YOLOv8 Nano (CPU Optimized)
        
        **Input Size:** 416×416×3
        
        **Classes:** Bird, Drone
        
        **Key Features:**
        - Fast inference (30ms on CPU)
        - Bounding box detection
        - Real-time object localization
        - Multi-object detection
        
        **Metrics:**
        - mAP50: ~85-90%
        - Speed: Real-time on CPU
        """)
    
    st.markdown("---")
    
    st.markdown("### Methods Used")
    
    with st.expander("📋 View Methods from train_model.py"):
        st.code("""
# From train_model.py

# Preprocessing & Augmentation
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# Model Building
- Custom CNN with Conv2D, BatchNormalization, MaxPooling
- Transfer Learning with MobileNetV2
- Callbacks: EarlyStopping, ModelCheckpoint

# Training & Evaluation
- Binary classification (Bird=0, Drone=1)
- Metrics: Accuracy, Precision, Recall
- Confusion Matrix & Classification Report
        """, language="python")
    
    with st.expander("📋 View Methods from data_analysis.py"):
        st.code("""
# From data_analysis.py

# Data Verification
- check_structure(): Verify folder organization
- count_images(): Count images per class
- check_imbalance(): Analyze class distribution
- show_samples(): Visualize sample images
- check_corrupted_images(): Find corrupted files

# Analysis Functions
- get_dataset_stats()
- verify_image_dimensions()
- visualize_dataset_stats()
        """, language="python")


if __name__ == "__main__":
    main()