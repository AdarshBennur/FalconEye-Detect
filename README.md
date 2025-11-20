# FalconEye-Detect 🦅🚁

AI-Powered Aerial Object Classification & Detection System for distinguishing between Birds and Drones.

## 📋 Project Overview

FalconEye-Detect is a comprehensive deep learning solution that combines image classification and object detection to identify and locate birds and drones in aerial imagery. The system uses multiple approaches including custom CNN, transfer learning, and YOLOv8 for robust performance.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training Models

**Data Preprocessing:**
```bash
python scripts/data_preprocessing.py
```

**Train Custom CNN:**
```bash
python scripts/train_custom_cnn.py
```

**Train Transfer Learning Models:**
```bash
python scripts/train_transfer_learning.py
```

**Train YOLOv8 Detection:**
```bash
python scripts/train_yolov8.py
```

### 3. Model Evaluation
```bash
python scripts/model_evaluation.py
```

### 4. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
FalconEye-Detect/
├── data/
│   ├── classification_dataset/    # Bird vs Drone classification data
│   ├── object_detection_dataset/  # YOLOv8 detection data
│   └── processed/                 # Preprocessed data
├── scripts/
│   ├── data_preprocessing.py      # Data preparation and augmentation
│   ├── train_custom_cnn.py       # Custom CNN training
│   ├── train_transfer_learning.py # Transfer learning training
│   ├── train_yolov8.py          # YOLOv8 training
│   ├── model_evaluation.py       # Model evaluation and comparison
│   └── inference_utils.py         # Inference utilities
├── weights/                       # Trained model weights
├── results/                       # Training results and evaluations
├── notebooks/                     # Jupyter notebooks (if needed)
├── streamlit_app.py              # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Features

- **Classification**: Binary classification (Bird vs Drone)
- **Object Detection**: Localize objects with bounding boxes
- **Multiple Models**: Custom CNN, Transfer Learning, YOLOv8
- **Web Interface**: Interactive Streamlit application
- **Batch Processing**: Process multiple images at once
- **Model Comparison**: Comprehensive evaluation and comparison

## 📊 Models Included

1. **Custom CNN**: Built-from-scratch convolutional neural network
2. **Transfer Learning**: Pre-trained models (ResNet50, MobileNetV2, EfficientNetB0)
3. **YOLOv8**: State-of-the-art object detection

## 🔧 Usage

### Command Line Interface
```python
# Import inference utilities
from scripts.inference_utils import ModelInference

# Initialize inference
inference = ModelInference()
inference.load_all_available_models()

# Make predictions
result = inference.predict_classification('path/to/image.jpg')
print(f"Prediction: {result['class_name']} (Confidence: {result['confidence']:.3f})")
```

### Web Interface
1. Run `streamlit run streamlit_app.py`
2. Open browser to `http://localhost:8501`
3. Upload images or use sample images
4. Select models and adjust settings
5. View predictions and visualizations

## 📈 Applications

- **Airport Bird Strike Prevention**: Monitor runway zones for bird activity
- **Wildlife Protection**: Detect birds near wind farms or protected areas  
- **Security & Defense**: Identify drones in restricted airspace
- **Environmental Research**: Track bird populations without misclassification

## 🛠 Technical Stack

- **Deep Learning**: TensorFlow/Keras, Ultralytics YOLOv8
- **Image Processing**: OpenCV, PIL
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Data Processing**: NumPy, Pandas

## 📝 Notes

- Models will be automatically saved to `weights/` directory during training
- Training results and evaluations are saved to `results/` directory
- All scripts include comprehensive error handling and progress tracking
- The Streamlit app automatically loads all available trained models

## 🎯 Performance Expectations

- **Classification Accuracy**: 85-95% (depending on model)
- **Detection mAP**: 70-85% (YOLOv8)
- **Inference Speed**: Real-time capable
- **Model Size**: Optimized for deployment

## 🔍 Troubleshooting

1. **No models found**: Train models first using the training scripts
2. **Memory errors**: Reduce batch size in training scripts
3. **CUDA issues**: Ensure proper GPU setup for TensorFlow
4. **Import errors**: Check all dependencies are installed via requirements.txt
