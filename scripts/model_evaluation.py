"""
Model Evaluation Module for FalconEye-Detect (PyTorch)
Comprehensive evaluation and comparison of trained models including 
custom CNN, transfer learning models, and YOLOv8.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Import our preprocessing module and model architectures
import sys
from pathlib import Path

# Add scripts directory to path dynamically
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from data_preprocessing import DataPreprocessor
from train_custom_cnn import CustomCNN
from train_transfer_learning import TransferLearningModel


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, base_path=None):
        # Auto-detect project root if not provided
        if base_path is None:
            base_path = Path(__file__).parent.parent
        
        self.base_path = Path(base_path)
        self.weights_path = self.base_path / "weights"
        self.results_path = self.base_path / "results"
        
        # Create directories if they don't exist
        self.results_path.mkdir(exist_ok=True)
        
        self.preprocessor = DataPreprocessor()
        self.class_names = ['Bird', 'Drone']
        
        # Device selection: MPS (Mac M1/M2) -> CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
    
    def load_test_data(self):
        """Load and preprocess test data for classification models"""
        
        print("Loading test data...")
        
        # Create test DataLoader
        _, _, test_loader = self.preprocessor.create_data_loaders(batch_size=32, num_workers=0)
        
        print(f"Test set: {len(test_loader.dataset)} images")
        
        # Get distribution
        dist = test_loader.dataset.get_class_distribution()
        print(f"Birds: {dist['birds']}")
        print(f"Drones: {dist['drones']}")
        
        return test_loader
    
    def load_pytorch_model(self, model_path, model_type='custom_cnn'):
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
            
            print(f"Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_classification_model(self, model_path, model_name, model_type='custom_cnn'):
        """Evaluate a classification model (CNN or Transfer Learning)"""
        
        print(f"\nEvaluating {model_name} model...")
        
        # Load model
        model = self.load_pytorch_model(model_path, model_type)
        if model is None:
            return None
        
        # Load test data
        test_loader = self.load_test_data()
        
        # Make predictions
        print("Making predictions...")
        all_predictions_prob = []
        all_predictions = []
        all_labels = []
        
        # Use inference_mode for faster evaluation
        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(self.device)
                
                # Forward pass
                outputs = model(images)
                predictions_prob = outputs.cpu().numpy()
                predictions = (predictions_prob > 0.5).astype(int).flatten()
                
                all_predictions_prob.extend(predictions_prob.flatten().tolist())
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.numpy().tolist())
        
        # Convert to numpy arrays
        predictions_prob = np.array(all_predictions_prob)
        predictions = np.array(all_predictions)
        test_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='binary')
        recall = recall_score(test_labels, predictions, average='binary')
        f1 = f1_score(test_labels, predictions, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Classification report
        report = classification_report(
            test_labels, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # ROC curve for binary classification
        fpr, tpr, _ = roc_curve(test_labels, predictions_prob)
        roc_auc = auc(fpr, tpr)
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': predictions.tolist(),
            'predictions_prob': predictions_prob.tolist(),
            'true_labels': test_labels.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def evaluate_yolo_model(self, model_path, model_name):
        """Evaluate YOLOv8 object detection model"""
        
        print(f"\nEvaluating {model_name} model...")
        
        try:
            model = YOLO(str(model_path))
            print(f"YOLOv8 model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            return None
        
        # Get detection dataset path
        detection_path = self.base_path / "data" / "object_detection_dataset"
        yaml_config = self.base_path / "data" / "processed" / "detection_data.yaml"
        
        # Prepare detection data if yaml doesn't exist
        if not yaml_config.exists():
            print("Preparing detection data configuration...")
            yaml_config = self.preprocessor.prepare_detection_data()
        
        # Run validation on test set
        print("Running YOLOv8 validation...")
        try:
            validation_results = model.val(
                data=str(yaml_config),
                split='test',
                save_json=True,
                conf=0.001,
                iou=0.6,
                max_det=300,
                verbose=False
            )
            
            # Extract metrics from validation results
            metrics = validation_results.results_dict
            
            results = {
                'model_name': model_name,
                'map50': float(metrics.get('metrics/mAP50(B)', 0)),
                'map50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
                'precision': float(metrics.get('metrics/precision(B)', 0)),
                'recall': float(metrics.get('metrics/recall(B)', 0)),
                'f1_score': float(metrics.get('metrics/F1(B)', 0)) if 'metrics/F1(B)' in metrics else 0,
                'model_type': 'object_detection',
                'classes': ['Bird', 'Drone']
            }
            
            print(f"Model: {model_name}")
            print(f"mAP@0.5: {results['map50']:.4f}")
            print(f"mAP@0.5:0.95: {results['map50_95']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error during YOLOv8 validation: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, cm, model_name, save_path=None):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curves(self, results_list, save_path=None):
        """Plot ROC curves for multiple models"""
        
        plt.figure(figsize=(10, 8))
        
        for results in results_list:
            if 'fpr' in results and 'tpr' in results:
                plt.plot(
                    results['fpr'], 
                    results['tpr'],
                    label=f"{results['model_name']} (AUC = {results['roc_auc']:.3f})",
                    linewidth=2
                )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def create_comparison_report(self, results_list):
        """Create comprehensive comparison report"""
        
        # Separate classification and detection results
        classification_results = [r for r in results_list if 'accuracy' in r]
        detection_results = [r for r in results_list if 'map50' in r]
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'classification_models': len(classification_results),
            'detection_models': len(detection_results),
            'models_compared': len(results_list)
        }
        
        # Classification models comparison
        if classification_results:
            classification_df = pd.DataFrame([
                {
                    'Model': r['model_name'],
                    'Accuracy': r['accuracy'],
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1-Score': r['f1_score'],
                    'ROC AUC': r['roc_auc']
                }
                for r in classification_results
            ])
            
            report['classification_comparison'] = classification_df.to_dict('records')
            
            # Find best classification model
            best_acc_model = classification_df.loc[classification_df['Accuracy'].idxmax()]
            best_f1_model = classification_df.loc[classification_df['F1-Score'].idxmax()]
            
            report['best_accuracy_model'] = {
                'name': best_acc_model['Model'],
                'accuracy': float(best_acc_model['Accuracy'])
            }
            report['best_f1_model'] = {
                'name': best_f1_model['Model'],
                'f1_score': float(best_f1_model['F1-Score'])
            }
        
        # Detection models comparison
        if detection_results:
            detection_df = pd.DataFrame([
                {
                    'Model': r['model_name'],
                    'mAP@0.5': r['map50'],
                    'mAP@0.5:0.95': r['map50_95'],
                    'Precision': r['precision'],
                    'Recall': r['recall']
                }
                for r in detection_results
            ])
            
            report['detection_comparison'] = detection_df.to_dict('records')
            
            # Find best detection model
            if len(detection_results) > 0:
                best_map_model = detection_df.loc[detection_df['mAP@0.5'].idxmax()]
                report['best_detection_model'] = {
                    'name': best_map_model['Model'],
                    'map50': float(best_map_model['mAP@0.5'])
                }
        
        return report, classification_df if classification_results else None, detection_df if detection_results else None
    
    def visualize_model_comparison(self, classification_df=None, detection_df=None):
        """Create visualization for model comparison"""
        
        if classification_df is not None and len(classification_df) > 0:
            # Classification models comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            axes[0, 0].bar(classification_df['Model'], classification_df['Accuracy'], color='skyblue')
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1-Score comparison
            axes[0, 1].bar(classification_df['Model'], classification_df['F1-Score'], color='lightgreen')
            axes[0, 1].set_title('Model F1-Score Comparison')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Precision vs Recall
            axes[1, 0].scatter(classification_df['Recall'], classification_df['Precision'], 
                              s=100, c=classification_df['Accuracy'], cmap='viridis')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision vs Recall (Color: Accuracy)')
            
            # Add model names as annotations
            for i, model in enumerate(classification_df['Model']):
                axes[1, 0].annotate(
                    model, 
                    (classification_df['Recall'].iloc[i], classification_df['Precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8
                )
            
            # Overall performance metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            best_model_idx = classification_df['Accuracy'].idxmax()
            best_model_values = [
                classification_df.iloc[best_model_idx]['Accuracy'],
                classification_df.iloc[best_model_idx]['Precision'],
                classification_df.iloc[best_model_idx]['Recall'],
                classification_df.iloc[best_model_idx]['F1-Score'],
                classification_df.iloc[best_model_idx]['ROC AUC']
            ]
            
            axes[1, 1].bar(metrics, best_model_values, color='orange')
            axes[1, 1].set_title(f'Best Model Performance - {classification_df.iloc[best_model_idx]["Model"]}')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save classification comparison
            comparison_path = self.results_path / "classification_models_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Classification comparison saved to: {comparison_path}")
        
        if detection_df is not None and len(detection_df) > 0:
            # Detection models comparison
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(detection_df))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x - width/2, detection_df['mAP@0.5'], width, label='mAP@0.5', color='skyblue')
            plt.bar(x + width/2, detection_df['mAP@0.5:0.95'], width, label='mAP@0.5:0.95', color='lightcoral')
            plt.xlabel('Models')
            plt.ylabel('mAP Score')
            plt.title('Detection Models - mAP Comparison')
            plt.xticks(x, detection_df['Model'], rotation=45)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.bar(x - width/2, detection_df['Precision'], width, label='Precision', color='lightgreen')
            plt.bar(x + width/2, detection_df['Recall'], width, label='Recall', color='orange')
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Detection Models - Precision vs Recall')
            plt.xticks(x, detection_df['Model'], rotation=45)
            plt.legend()
            
            plt.tight_layout()
            
            # Save detection comparison
            detection_comparison_path = self.results_path / "detection_models_comparison.png"
            plt.savefig(detection_comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Detection comparison saved to: {detection_comparison_path}")
    
    def evaluate_all_models(self):
        """Evaluate all available trained models"""
        
        print("FalconEye-Detect Model Evaluation (PyTorch)")
        print("="*50)
        
        results_list = []
        
        # Find available PyTorch models (.pt files)
        model_files = {
            'Custom CNN': {
                'path': self.weights_path / "custom_cnn_best.pt",
                'type': 'custom_cnn'
            },
            'Transfer ResNet50': {
                'path': self.weights_path / "transfer_resnet50_phase2_best.pt",
                'type': 'transfer_resnet50'
            },
            'Transfer MobileNetV2': {
                'path': self.weights_path / "transfer_mobilenetv2_phase2_best.pt",
                'type': 'transfer_mobilenetv2'
            },
            'Transfer EfficientNetB0': {
                'path': self.weights_path / "transfer_efficientnetb0_phase2_best.pt",
                'type': 'transfer_efficientnetb0'
            },
            'YOLOv8': {
                'path': self.weights_path / "yolov8s_trained.pt",
                'type': 'yolo'
            }
        }
        
        # Evaluate classification models
        for model_name, model_info in model_files.items():
            if model_name != 'YOLOv8' and model_info['path'].exists():
                results = self.evaluate_classification_model(
                    model_info['path'], 
                    model_name,
                    model_info['type']
                )
                if results:
                    results_list.append(results)
                    
                    # Plot confusion matrix for each model
                    cm = np.array(results['confusion_matrix'])
                    cm_path = self.results_path / f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
                    self.plot_confusion_matrix(cm, model_name, cm_path)
        
        # Evaluate YOLOv8 model
        if model_files['YOLOv8']['path'].exists():
            yolo_results = self.evaluate_yolo_model(model_files['YOLOv8']['path'], 'YOLOv8')
            if yolo_results:
                results_list.append(yolo_results)
        
        if not results_list:
            print("No trained models found for evaluation!")
            print(f"Looking in: {self.weights_path}")
            return None, None
        
        # Create comparison report
        report, classification_df, detection_df = self.create_comparison_report(results_list)
        
        # Save detailed results
        results_file = self.results_path / "detailed_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'report': report,
                'detailed_results': results_list
            }, f, indent=2)
        
        # Save comparison report
        report_file = self.results_path / "model_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot ROC curves for classification models
        classification_results = [r for r in results_list if 'fpr' in r]
        if classification_results:
            roc_path = self.results_path / "roc_curves_comparison.png"
            self.plot_roc_curves(classification_results, roc_path)
        
        # Create comparison visualizations
        self.visualize_model_comparison(classification_df, detection_df)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'best_accuracy_model' in report:
            best_acc = report['best_accuracy_model']
            print(f"Best Classification Model (Accuracy): {best_acc['name']} ({best_acc['accuracy']:.4f})")
        
        if 'best_f1_model' in report:
            best_f1 = report['best_f1_model']
            print(f"Best Classification Model (F1): {best_f1['name']} ({best_f1['f1_score']:.4f})")
        
        if 'best_detection_model' in report:
            best_det = report['best_detection_model']
            print(f"Best Detection Model: {best_det['name']} (mAP@0.5: {best_det['map50']:.4f})")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Comparison report saved to: {report_file}")
        print(f"Visualizations saved to: {self.results_path}")
        
        return results_list, report


def main():
    """Main evaluation function"""
    
    evaluator = ModelEvaluator()
    results, report = evaluator.evaluate_all_models()
    
    if results:
        print("\nModel evaluation completed successfully!")
    else:
        print("\nNo models were evaluated. Please train models first.")


if __name__ == "__main__":
    main()
