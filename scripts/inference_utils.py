"""
Inference Utilities Module for FalconEye-Detect (PyTorch)
Provides utilities for loading trained models and making predictions
on new images for both classification and object detection tasks.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors

# Import our preprocessing module and model architectures
import sys
from pathlib import Path as PathLib

# Add scripts directory to path dynamically
scripts_dir = PathLib(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from data_preprocessing import DataPreprocessor
from train_custom_cnn import CustomCNN
from train_transfer_learning import TransferLearningModel


class ModelInference:
    """Handles model loading and inference for both classification and detection"""
    
    def __init__(self, base_path=None):
        # Auto-detect project root if not provided
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.base_path = Path(base_path)
        self.weights_path = self.base_path / "weights"
        
        self.classification_models = {}
        self.detection_models = {}
        
        # Device selection: MPS (Mac M1/M2) -> CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Image preprocessing transform (same as validation transform)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.class_names = ['Bird', 'Drone']
        
        print(f"Inference device: {self.device}")
    
    def load_classification_model(self, model_path, model_name, model_type='custom_cnn'):
        """Load a PyTorch classification model"""
        
        try:
            # Determine model architecture
            if model_type == 'custom_cnn':
                model = CustomCNN(input_channels=3, num_classes=1)
            else:
                # Extract base model name from model_type (e.g., 'transfer_resnet50' -> 'resnet50')
                base_model_name = model_type.replace('transfer_', '')
                model = TransferLearningModel(base_model_name=base_model_name, num_classes=1, pretrained=False)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.classification_models[model_name] = {
                'model': model,
                'type': model_type
            }
            
            print(f"Classification model '{model_name}' loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading classification model '{model_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_detection_model(self, model_path, model_name):
        """Load a YOLOv8 detection model"""
        
        try:
            model = YOLO(str(model_path))
            self.detection_models[model_name] = model
            print(f"Detection model '{model_name}' loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading detection model '{model_name}': {str(e)}")
            return False
    
    def load_all_available_models(self):
        """Load all available trained models"""
        
        print("Loading available models...")
        
        # Classification models
        classification_model_configs = [
            ('custom_cnn', 'custom_cnn_best.pt', 'custom_cnn'),
            ('transfer_resnet50', 'transfer_resnet50_phase2_best.pt', 'transfer_resnet50'),
            ('transfer_mobilenetv2', 'transfer_mobilenetv2_phase2_best.pt', 'transfer_mobilenetv2'),
            ('transfer_efficientnetb0', 'transfer_efficientnetb0_phase2_best.pt', 'transfer_efficientnetb0'),
        ]
        
        for model_name, filename, model_type in classification_model_configs:
            model_path = self.weights_path / filename
            if model_path.exists():
                self.load_classification_model(model_path, model_name, model_type)
            else:
                # Try alternative paths (final models)
                alt_path = self.weights_path / filename.replace('_best', '_final')
                if alt_path.exists():
                    self.load_classification_model(alt_path, model_name, model_type)
        
        # Detection models
        yolo_path = self.weights_path / "yolov8s_trained.pt"
        if yolo_path.exists():
            self.load_detection_model(yolo_path, "yolov8")
        
        print(f"Loaded {len(self.classification_models)} classification model(s)")
        print(f"Loaded {len(self.detection_models)} detection model(s)")
    
    def preprocess_image_for_classification(self, image_input):
        """Preprocess image for classification models"""
        
        # Handle different input types
        if isinstance(image_input, str) or isinstance(image_input, Path):
            # Path to image file
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Numpy array (BGR from OpenCV)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
            else:
                image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transform
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_classification(self, image_input, model_name=None):
        """Make classification prediction"""
        
        if not self.classification_models:
            raise ValueError("No classification models loaded!")
        
        # Use specified model or first available
        if model_name is None:
            model_name = list(self.classification_models.keys())[0]
        
        if model_name not in self.classification_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.classification_models.keys())}")
        
        # Get model
        model_info = self.classification_models[model_name]
        model = model_info['model']
        
        # Preprocess image
        tensor = self.preprocess_image_for_classification(image_input)
        tensor = tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            probability = output.cpu().numpy()[0][0]
        
        # Convert to class prediction
        predicted_class = 1 if probability > 0.5 else 0
        confidence = probability if predicted_class == 1 else (1 - probability)
        
        result = {
            'model_name': model_name,
            'predicted_class': int(predicted_class),
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probability': float(probability),
            'probabilities': {
                'Bird': float(1 - probability),
                'Drone': float(probability)
            }
        }
        
        return result
    
    def predict_detection(self, image_input, model_name=None, conf_threshold=0.5, iou_threshold=0.5):
        """Make object detection prediction"""
        
        if not self.detection_models:
            raise ValueError("No detection models loaded!")
        
        # Use specified model or first available
        if model_name is None:
            model_name = list(self.detection_models.keys())[0]
        
        if model_name not in self.detection_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.detection_models.keys())}")
        
        # Get model
        model = self.detection_models[model_name]
        
        # Handle image input
        if isinstance(image_input, str) or isinstance(image_input, Path):
            image_path = str(image_input)
        elif isinstance(image_input, np.ndarray):
            image_path = image_input
        else:
            raise ValueError(f"Unsupported image input type for detection: {type(image_input)}")
        
        # Run prediction
        results = model.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': self.class_names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'bbox_normalized': box.xywhn[0].cpu().numpy().tolist()  # [x_center, y_center, width, height] normalized
                }
                detections.append(detection)
        
        detection_result = {
            'model_name': model_name,
            'num_detections': len(detections),
            'detections': detections
        }
        
        return detection_result
    
    def annotate_image(self, image, detections, show_labels=True):
        """Annotate image with detection results"""
        
        # Handle image input
        if isinstance(image, str) or isinstance(image, Path):
            img = cv2.imread(str(image))
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Create annotator
        annotator = Annotator(img)
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Label
            label = f"{class_name} {confidence:.2f}" if show_labels else ""
            
            # Draw box
            annotator.box_label(bbox, label, color=colors(det['class_id'], True))
        
        # Get result
        annotated_img = annotator.result()
        
        return annotated_img
    
    def predict_and_visualize(self, image_input, task='both', classification_model=None, 
                            detection_model=None, conf_threshold=0.5):
        """Predict and create visualization for both tasks"""
        
        results = {}
        
        # Classification
        if task in ['classification', 'both']:
            if self.classification_models:
                try:
                    classification_result = self.predict_classification(image_input, classification_model)
                    results['classification'] = classification_result
                except Exception as e:
                    print(f"Classification error: {str(e)}")
                    results['classification'] = None
        
        # Detection
        if task in ['detection', 'both']:
            if self.detection_models:
                try:
                    detection_result = self.predict_detection(
                        image_input, 
                        detection_model, 
                        conf_threshold=conf_threshold
                    )
                    results['detection'] = detection_result
                    
                    # Create annotated image
                    if detection_result['num_detections'] > 0:
                        annotated = self.annotate_image(image_input, detection_result['detections'])
                        results['annotated_image'] = annotated
                        
                except Exception as e:
                    print(f"Detection error: {str(e)}")
                    results['detection'] = None
        
        return results
    
    def batch_predict(self, image_list, task='classification', model_name=None, conf_threshold=0.5):
        """Make predictions on a batch of images"""
        
        results = []
        
        for i, image_path in enumerate(image_list):
            print(f"Processing image {i+1}/{len(image_list)}: {image_path}")
            
            try:
                if task == 'classification':
                    result = self.predict_classification(image_path, model_name)
                elif task == 'detection':
                    result = self.predict_detection(image_path, model_name, conf_threshold=conf_threshold)
                else:
                    result = self.predict_and_visualize(image_path, task, model_name, model_name, conf_threshold)
                
                results.append({
                    'image_path': str(image_path),
                    'result': result
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        
        info = {
            'classification_models': list(self.classification_models.keys()),
            'detection_models': list(self.detection_models.keys()),
            'device': str(self.device),
            'class_names': self.class_names
        }
        
        return info
    
    def save_predictions(self, predictions, output_path):
        """Save prediction results to JSON file"""
        
        def convert_numpy(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert predictions
        predictions_serializable = json.loads(
            json.dumps(predictions, default=convert_numpy)
        )
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(predictions_serializable, f, indent=2)
        
        print(f"Predictions saved to: {output_path}")


class StreamlitInferenceUtils:
    """Utilities specifically designed for Streamlit integration"""
    
    def __init__(self, model_inference):
        self.inference = model_inference
    
    def image_to_base64(self, image):
        """Convert image to base64 string for display"""
        
        import base64
        from io import BytesIO
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        # Convert to base64
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def format_classification_result(self, result):
        """Format classification result for Streamlit display"""
        
        formatted = {
            'Prediction': result['class_name'],
            'Confidence': f"{result['confidence']*100:.2f}%",
            'Model': result['model_name'],
            'Probabilities': {
                'Bird': f"{result['probabilities']['Bird']*100:.1f}%",
                'Drone': f"{result['probabilities']['Drone']*100:.1f}%"
            }
        }
        
        return formatted
    
    def format_detection_result(self, result):
        """Format detection result for Streamlit display"""
        
        formatted = {
            'Model': result['model_name'],
            'Detections': result['num_detections'],
            'Objects': []
        }
        
        for det in result['detections']:
            formatted['Objects'].append({
                'Class': det['class_name'],
                'Confidence': f"{det['confidence']*100:.2f}%",
                'BBox': det['bbox']
            })
        
        return formatted


def main():
    """Main function to demonstrate inference capabilities"""
    
    print("FalconEye-Detect Inference Utilities (PyTorch)")
    print("="*60)
    
    # Initialize inference
    inference = ModelInference()
    
    # Load all available models
    inference.load_all_available_models()
    
    # Get model info
    info = inference.get_model_info()
    print(f"\nLoaded models:")
    print(f"  Classification: {info['classification_models']}")
    print(f"  Detection: {info['detection_models']}")
    
    # Test with a sample image if available
    # Get project root dynamically
    project_root = Path(__file__).parent.parent
    test_image_path = project_root / "data" / "classification_dataset" / "test" / "bird"
    if test_image_path.exists():
        sample_images = list(test_image_path.glob("*.jpg"))[:1]
        
        if sample_images and inference.classification_models:
            print(f"\nTesting classification with sample image: {sample_images[0].name}")
            
            try:
                result = inference.predict_classification(sample_images[0])
                print("\nClassification Result:")
                print(f"  Predicted: {result['class_name']}")
                print(f"  Confidence: {result['confidence']*100:.2f}%")
                print(f"  Probabilities: Bird={result['probabilities']['Bird']*100:.1f}%, Drone={result['probabilities']['Drone']*100:.1f}%")
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        if sample_images and inference.detection_models:
            print(f"\nTesting detection with sample image: {sample_images[0].name}")
            
            try:
                result = inference.predict_detection(sample_images[0], conf_threshold=0.25)
                print("\nDetection Result:")
                print(f"  Number of detections: {result['num_detections']}")
                for i, det in enumerate(result['detections']):
                    print(f"  Detection {i+1}: {det['class_name']} ({det['confidence']*100:.2f}%)")
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    print("\nInference utilities ready!")


if __name__ == "__main__":
    main()
