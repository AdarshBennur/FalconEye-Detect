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

- **Deep Learning**: PyTorch \u003e=2.0.0, Ultralytics YOLOv8
- **Image Processing**: OpenCV, PIL
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Data Processing**: NumPy, Pandas

## ✅ What Works (Classification-Only Deployment)

**Current Status**: The app is optimized for **classification-only deployment** for maximum speed and stability.

- ✅ **Auto-Discovery**: Automatically finds and loads ALL .pt files in `./weights` folder
- ✅ **Single Image Inference**: Upload, sample images, and webcam capture
- ✅ **Batch Processing**: Process multiple images with progress bar and CSV export
- ✅ **Smart Class Mapping**: 3-tier fallback (detection_data.yaml → dataset structure → hardcoded)
- ✅ **Device Detection**: Automatic MPS/CUDA/CPU selection with logging
- ✅ **Error Handling**: Robust error logging to `results/loading_warnings.log`
- ✅ **Testing Infrastructure**: Sanity checks and comprehensive test scripts

## 🚫 What's Disabled

**Intentionally disabled for speed and simplicity**:

- 🚫 **YOLO Object Detection**: Disabled to avoid GPU memory overhead
- 🚫 **Real-time Video Streaming**: Use single-shot webcam capture instead
- 🚫 **Detection/Both Modes**: UI automatically hides these when no YOLO model loaded

> **Note**: Detection can be easily re-enabled by training YOLOv8 and placing weights in `./weights` folder. The app will automatically detect and load it.

## 🧪 Validation \u0026 Testing

### Quick Start Validation

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run Streamlit app (should start without crashes)
streamlit run streamlit_app.py

# 3. Run quick sanity check (tests all models on 10 images)
python3 scripts/model_evaluation.py --sanity-check

# 4. Run comprehensive inference test
python3 scripts/tests/test_inference.py
```

### Expected Outputs

1. **Streamlit App**:
   - Should start at `http://localhost:8501` without errors
   - Shows accurate model count (e.g., "9 classification models")
   - Detection disabled badge appears if no YOLO model

2. **Sanity Check**:
   - Tests each model on 10 sample images
   - Prints accuracy table
   - Exit code 0 on success

3. **Inference Test**:
   - Generates `results/inference_sanity.json`
   - Shows per-model accuracy and predictions
   - Exit code 0 on success

### Mac (Apple Silicon - MPS) Notes

**Auto-Detection**: The app automatically detects and uses MPS if available.

Check device in use (look for this in terminal output):

```
Inference device: mps
Class names: ['Bird', 'Drone']
```

**Force CPU** (if MPS issues occur):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
streamlit run streamlit_app.py
```

**Common MPS Issues**:

- If you see MPS-related errors, check `results/loading_warnings.log`
- Some older Mac models may not support all MPS operations
- Fallback to CPU is automatic and safe

## 📝 Notes

- **Model Auto-Discovery**: All `.pt` files in `weights/` are automatically discovered
- **Logging**: Model loading warnings saved to `results/loading_warnings.log`
- **Class Names**: Loaded from `data/processed/detection_data.yaml` with fallback to dataset structure
- **Testing**: Run `python3 scripts/tests/test_inference.py` to generate `results/inference_sanity.json`
- **Training Results**: Training logs and checkpoints saved to `results/` directory

## 🎯 Performance Expectations

- **Classification Accuracy**: 85-95% (depending on model)
- **Inference Speed**: Real-time capable on MPS/GPU, ~100ms on CPU
- **Model Count**: 9 classification models auto-discovered
- **Device Priority**: MPS \u003e CUDA \u003e CPU (automatic selection)

## 🔍 Troubleshooting

1. **No models found**: Check `weights/` folder contains `.pt` files
2. **NameError crashes**: Fixed with auto-discovery and instance variables
3. **Class name mismatches**: Check `results/loading_warnings.log` for mapping issues
4. **MPS errors on Mac**: Set `export PYTORCH_ENABLE_MPS_FALLBACK=1`
5. **Model loading failures**: Check `results/loading_warnings.log` for details

## 📚 Additional Commands

```bash
# Full model evaluation (detailed metrics)
python scripts/model_evaluation.py

# Test single model
python -c "from scripts.inference_utils import ModelInference; m=ModelInference(); m.load_all_available_models(); print(m.get_model_info())"

# Check discovered models
ls -lh weights/*.pt

# View loading warnings
cat results/loading_warnings.log
```
