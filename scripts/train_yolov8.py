"""
YOLOv8 Training Module for FalconEye-Detect
Handles YOLOv8 model training for bird and drone object detection.
"""

import os
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator

# Import our preprocessing module
import sys
from pathlib import Path

# Add scripts directory to path dynamically
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from data_preprocessing import DataPreprocessor

class YOLOv8Trainer:
    """YOLOv8 model trainer for object detection"""
    
    def __init__(self, base_path=None):
        # Auto-detect project root if not provided
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.base_path = Path(base_path)
        self.weights_path = self.base_path / "weights"
        self.results_path = self.base_path / "results"
        
        # Create directories if they don't exist
        self.weights_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        self.model = None
        self.data_yaml = None
        self.training_results = None
        
    def prepare_yolo_config(self):
        """Prepare YOLOv8 configuration file"""
        
        preprocessor = DataPreprocessor()
        data_yaml_path = preprocessor.prepare_detection_data()
        
        # Verify the configuration
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("YOLOv8 Configuration:")
        print(f"  Train images: {config['train']}")
        print(f"  Validation images: {config['val']}")
        print(f"  Test images: {config['test']}")
        print(f"  Number of classes: {config['nc']}")
        print(f"  Class names: {config['names']}")
        
        self.data_yaml = data_yaml_path
        return data_yaml_path
    
    def initialize_model(self, model_size='n'):
        """Initialize YOLOv8 model"""
        
        # Available model sizes: n (nano), s (small), m (medium), l (large), x (extra large)
        model_sizes = {
            'n': 'yolov8n.pt',  # Fastest, least accurate
            's': 'yolov8s.pt',  # Balanced
            'm': 'yolov8m.pt',  # Good accuracy
            'l': 'yolov8l.pt',  # Better accuracy
            'x': 'yolov8x.pt'   # Best accuracy, slowest
        }
        
        if model_size not in model_sizes:
            raise ValueError(f"Model size must be one of: {list(model_sizes.keys())}")
        
        model_name = model_sizes[model_size]
        print(f"Initializing YOLOv8{model_size.upper()} model...")
        
        # Load pre-trained YOLO model
        self.model = YOLO(model_name)
        print(f"Loaded {model_name} successfully")
        
        return self.model
    
    def train_model(self, epochs=100, imgsz=640, batch_size=16, model_size='s',
                   patience=50, save_period=10):
        """Train YOLOv8 model"""
        
        if self.data_yaml is None:
            raise ValueError("Data configuration not prepared. Call prepare_yolo_config() first.")
        
        # Initialize model
        self.initialize_model(model_size)
        
        print(f"Starting YOLOv8 training...")
        print(f"Model size: YOLOv8{model_size.upper()}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch_size}")
        
        # Train the model
        self.training_results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            save_period=save_period,
            project=str(self.results_path),
            name=f"yolov8{model_size}_training",
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            # Data augmentation parameters
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # Save the trained model
        model_save_path = self.weights_path / f"yolov8{model_size}_trained.pt"
        self.model.save(str(model_save_path))
        
        print(f"Training completed!")
        print(f"Model saved to: {model_save_path}")
        
        return self.training_results
    
    def validate_model(self, model_path=None):
        """Validate the trained model"""
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for validation")
        
        print("Validating model...")
        
        # Run validation
        validation_results = model.val(
            data=self.data_yaml,
            save_json=True,
            save_hybrid=False,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            device=None,
            dnn=False,
            plots=True,
            rect=False,
            split='val'
        )
        
        print("Validation completed!")
        return validation_results
    
    def export_model(self, model_path, export_formats=['onnx', 'torchscript']):
        """Export trained model to different formats"""
        
        model = YOLO(model_path)
        exported_models = {}
        
        for format_name in export_formats:
            try:
                print(f"Exporting to {format_name.upper()}...")
                exported_path = model.export(format=format_name, optimize=True)
                exported_models[format_name] = exported_path
                print(f"Exported to: {exported_path}")
            except Exception as e:
                print(f"Failed to export to {format_name}: {str(e)}")
        
        return exported_models
    
    def run_inference(self, image_path, model_path=None, conf_threshold=0.5):
        """Run inference on a single image"""
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for inference")
        
        # Run inference
        results = model(image_path, conf=conf_threshold, iou=0.5)
        
        return results[0] if results else None
    
    def visualize_predictions(self, image_path, model_path=None, conf_threshold=0.5, 
                            save_path=None):
        """Visualize model predictions on an image"""
        
        # Run inference
        result = self.run_inference(image_path, model_path, conf_threshold)
        
        if result is None:
            print("No predictions made")
            return None
        
        # Load original image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create annotator
        annotator = Annotator(image)
        
        # Get predictions
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Class names
                class_names = ['Bird', 'Drone']
                label = f"{class_names[cls]} {conf:.2f}"
                
                # Annotate
                annotator.box_label([x1, y1, x2, y2], label)
        
        # Get annotated image
        annotated_image = annotator.result()
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_rgb)
        plt.title(f'YOLOv8 Predictions - {Path(image_path).name}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return annotated_rgb
    
    def save_training_summary(self, model_size='s'):
        """Save training summary and metrics"""
        
        if self.training_results is None:
            print("No training results available")
            return None
        
        # Extract metrics from training results
        # Note: YOLOv8 saves results automatically, we'll create a summary
        
        summary = {
            "model_name": f"yolov8{model_size}",
            "training_completed": datetime.now().isoformat(),
            "model_type": "Object Detection",
            "architecture": f"YOLOv8{model_size.upper()}",
            "task": "Bird and Drone Detection",
            "classes": ["Bird", "Drone"],
            "num_classes": 2,
            "data_yaml": str(self.data_yaml),
            "training_framework": "Ultralytics YOLOv8"
        }
        
        # Save summary
        summary_path = self.results_path / f"yolov8{model_size}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
        return summary
    
    def test_on_sample_images(self, model_path, num_samples=5):
        """Test the trained model on sample images"""
        
        # Get test images
        detection_path = self.base_path / "data" / "object_detection_dataset" / "test" / "images"
        test_images = list(detection_path.glob("*.jpg"))[:num_samples]
        
        print(f"Testing model on {len(test_images)} sample images...")
        
        for i, img_path in enumerate(test_images):
            print(f"\nProcessing image {i+1}: {img_path.name}")
            
            # Create save path for visualization
            save_path = self.results_path / f"test_prediction_{i+1}.png"
            
            # Visualize predictions
            self.visualize_predictions(
                image_path=img_path,
                model_path=model_path,
                conf_threshold=0.3,
                save_path=save_path
            )
        
        print(f"\nTest results saved to: {self.results_path}")

def main():
    """Main training function"""
    
    print("FalconEye-Detect YOLOv8 Training")
    print("="*50)
    
    # Initialize trainer
    trainer = YOLOv8Trainer()
    
    # Prepare YOLO configuration
    print("Preparing YOLOv8 configuration...")
    data_yaml = trainer.prepare_yolo_config()
    
    # Train model
    print("\nStarting YOLOv8 training...")
    training_results = trainer.train_model(
        epochs=100,
        imgsz=640,
        batch_size=16,
        model_size='s',  # Use small model for balance of speed and accuracy
        patience=20,
        save_period=10
    )
    
    # Validate model
    print("\nValidating trained model...")
    validation_results = trainer.validate_model()
    
    # Save training summary
    summary = trainer.save_training_summary('s')
    
    # Test on sample images
    model_path = trainer.weights_path / "yolov8s_trained.pt"
    if model_path.exists():
        print("\nTesting model on sample images...")
        trainer.test_on_sample_images(str(model_path), num_samples=5)
    
    # Export model to different formats
    print("\nExporting model...")
    exported_models = trainer.export_model(
        str(model_path),
        export_formats=['onnx']
    )
    
    print("\nYOLOv8 training completed!")
    print(f"Model saved to: {model_path}")
    print("Check the results folder for training metrics and visualizations.")

if __name__ == "__main__":
    main()
