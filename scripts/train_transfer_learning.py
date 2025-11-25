"""
Transfer Learning Training Module for FalconEye-Detect (PyTorch)
Implements transfer learning with pre-trained models for bird vs drone classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Import our preprocessing module
import sys
from pathlib import Path

# Add scripts directory to path dynamically
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from data_preprocessing import DataPreprocessor


class TransferLearningModel(nn.Module):
    """Transfer learning model with custom classifier head"""
    
    def __init__(self, base_model_name='resnet50', num_classes=1, pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        self.base_model_name = base_model_name.lower()
        
        # Load pretrained base model
        if self.base_model_name == 'resnet50':
            self.base_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove original classifier
            
        elif self.base_model_name == 'mobilenetv2':
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
            
        elif self.base_model_name == 'mobilenetv3':
            self.base_model = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.base_model.classifier[0].in_features
            self.base_model.classifier = nn.Identity()
            
        elif self.base_model_name == 'efficientnetb0':
            self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
            
        elif self.base_model_name == 'vgg16':
            self.base_model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.base_model.classifier[0].in_features
            self.base_model.classifier = nn.Identity()
            
        elif self.base_model_name == 'inceptionv3':
            self.base_model = models.inception_v3(weights='IMAGENET1K_V1' if pretrained else None, aux_logits=False)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {base_model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output
    
    def freeze_base(self):
        """Freeze all base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self, num_layers=20):
        """Unfreeze the last num_layers of the base model"""
        # First freeze all
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Then unfreeze last layers
        base_params = list(self.base_model.parameters())
        for param in base_params[-num_layers:]:
            param.requires_grad = True


class TransferLearningTrainer:
    """Transfer learning model builder and trainer"""
    
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
        self.base_model_name = None
        
        # Device selection: MPS (Mac M1/M2) -> CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Available pre-trained models
        self.available_models = ['resnet50', 'mobilenetv2', 'mobilenetv3', 'efficientnetb0', 'vgg16', 'inceptionv3']
        
        print(f"Using device: {self.device}")
        
        # Initialize training history
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
        }
    
    def build_transfer_model(self, base_model_name='resnet50', num_classes=1, trainable_layers=0):
        """Build transfer learning model with specified base model"""
        
        if base_model_name.lower() not in self.available_models:
            raise ValueError(f"Model {base_model_name} not available. Choose from: {self.available_models}")
        
        self.base_model_name = base_model_name.lower()
        
        # Build model
        self.model = TransferLearningModel(
            base_model_name=self.base_model_name,
            num_classes=num_classes,
            pretrained=True
        )
        
        # Freeze base model initially
        self.model.freeze_base()
        
        # If trainable_layers > 0, unfreeze top layers
        if trainable_layers > 0:
            self.model.unfreeze_base(num_layers=trainable_layers)
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Transfer learning model built with {base_model_name.upper()} base")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, use_amp=True):
        """Setup optimizer, loss function, and AMP scaler"""
        
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=10,
            min_lr=1e-8
        )
        
        # Mixed Precision Training (AMP) for faster training
        # Note: MPS doesn't fully support FP16 yet (causes crashes with BatchNorm)
        # Only enable AMP for CUDA
        self.use_amp = use_amp and (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"✓ Mixed Precision (AMP) enabled for {self.device.type}")
        else:
            self.scaler = None
            if self.device.type == 'mps':
                print("ℹ️  MPS detected: Using FP32 (AMP disabled due to MPS limitations)")
            else:
                print("✗ Mixed Precision (AMP) disabled (CPU mode)")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print("Model compiled successfully!")
        print(f"Optimizer: Adam (lr={learning_rate})")
        print(f"Loss function: BCELoss")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def calculate_metrics(self, outputs, labels):
        """Calculate accuracy, precision, and recall"""
        
        predictions = (outputs > 0.5).float()
        
        # Accuracy
        accuracy = (predictions == labels).float().mean().item()
        
        # True Positives, False Positives, False Negatives
        tp = ((predictions == 1) & (labels == 1)).float().sum().item()
        fp = ((predictions == 1) & (labels == 0)).float().sum().item()
        fn = ((predictions == 0) & (labels == 1)).float().sum().item()
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return accuracy, precision, recall
    
    def train_epoch(self, train_loader):
        """Train for one epoch with AMP support"""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            # Mixed Precision Training
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                accuracy, precision, recall = self.calculate_metrics(outputs, labels)
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            epoch_precision += precision
            epoch_recall += recall
        
        # Average metrics
        num_batches = len(train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches,
            'precision': epoch_precision / num_batches,
            'recall': epoch_recall / num_batches
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch with optimized inference"""
        
        self.model.eval()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        
        # Use inference_mode for faster validation
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                accuracy, precision, recall = self.calculate_metrics(outputs, labels)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                epoch_precision += precision
                epoch_recall += recall
        
        # Average metrics
        num_batches = len(val_loader)
        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches,
            'precision': epoch_precision / num_batches,
            'recall': epoch_recall / num_batches
        }
    
    def train_model(self, train_loader, val_loader, epochs=50, fine_tune_epochs=25, model_name=None):
        """Train the transfer learning model with optional fine-tuning"""
        
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        if model_name is None:
            model_name = f"transfer_{self.base_model_name}"
        
        print(f"Starting transfer learning training...")
        print(f"Base model: {self.base_model_name.upper()}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Phase 1: Train with frozen base model
        print(f"\nPhase 1: Training with frozen base model for {epochs} epochs...")
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} (Phase 1)")
            
            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.history['loss'].append(train_metrics['loss'])
            self.history['accuracy'].append(train_metrics['accuracy'])
            self.history['precision'].append(train_metrics['precision'])
            self.history['recall'].append(train_metrics['recall'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_path = self.weights_path / f"{model_name}_phase1_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy']
                }, best_model_path)
                print(f"✓ Saved best model (val_acc: {best_val_accuracy:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Phase 2: Fine-tuning
        if fine_tune_epochs > 0:
            print(f"\nPhase 2: Fine-tuning for {fine_tune_epochs} epochs...")
            
            # Unfreeze top layers
            self.model.unfreeze_base(num_layers=20)
            
            # Lower learning rate for fine-tuning
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.00001
            
            print(f"Unfroze last 20 layers of base model")
            print(f"Learning rate reduced to 0.00001")
            
            # Reset patience for fine-tuning
            patience_counter = 0
            phase1_epochs = len(self.history['loss'])
            
            for epoch in range(fine_tune_epochs):
                print(f"\nEpoch {epoch + 1}/{fine_tune_epochs} (Phase 2)")
                
                # Train and validate
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate_epoch(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_metrics['loss'])
                
                # Store history
                self.history['loss'].append(train_metrics['loss'])
                self.history['accuracy'].append(train_metrics['accuracy'])
                self.history['precision'].append(train_metrics['precision'])
                self.history['recall'].append(train_metrics['recall'])
                
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
                
                # Print metrics
                print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    best_model_path = self.weights_path / f"{model_name}_phase2_best.pt"
                    torch.save({
                        'epoch': phase1_epochs + epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': val_metrics['accuracy']
                    }, best_model_path)
                    print(f"✓ Saved best model (val_acc: {best_val_accuracy:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs of fine-tuning")
                    break
        
        # Save final model
        final_model_path = self.weights_path / f"{model_name}_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self, model_name=None):
        """Plot training history with phase separation"""
        
        if not self.history['loss']:
            raise ValueError("No training history available")
        
        if model_name is None:
            model_name = f"transfer_{self.base_model_name}"
        
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(epochs, self.history['accuracy'], 'bo-', label='Training Accuracy', markersize=3)
        axes[0, 0].plot(epochs, self.history['val_accuracy'], 'ro-', label='Validation Accuracy', markersize=3)
        axes[0, 0].set_title(f'Model Accuracy - {model_name.upper()}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(epochs, self.history['loss'], 'bo-', label='Training Loss', markersize=3)
        axes[0, 1].plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss', markersize=3)
        axes[0, 1].set_title(f'Model Loss - {model_name.upper()}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        axes[1, 0].plot(epochs, self.history['precision'], 'bo-', label='Training Precision', markersize=3)
        axes[1, 0].plot(epochs, self.history['val_precision'], 'ro-', label='Validation Precision', markersize=3)
        axes[1, 0].set_title(f'Model Precision - {model_name.upper()}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        axes[1, 1].plot(epochs, self.history['recall'], 'bo-', label='Training Recall', markersize=3)
        axes[1, 1].plot(epochs, self.history['val_recall'], 'ro-', label='Validation Recall', markersize=3)
        axes[1, 1].set_title(f'Model Recall - {model_name.upper()}')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_path / f"{model_name}_training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {plot_path}")
    
    def save_training_summary(self, model_name=None):
        """Save training summary and metrics"""
        
        if not self.history['loss']:
            raise ValueError("No training history available")
        
        if model_name is None:
            model_name = f"transfer_{self.base_model_name}"
        
        # Get best metrics
        best_val_acc_epoch = np.argmax(self.history['val_accuracy']) + 1
        best_val_acc = max(self.history['val_accuracy'])
        best_val_loss = min(self.history['val_loss'])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            "model_name": model_name,
            "base_model": self.base_model_name,
            "training_completed": datetime.now().isoformat(),
            "total_epochs": len(self.history['loss']),
            "best_validation_accuracy": float(best_val_acc),
            "best_validation_loss": float(best_val_loss),
            "best_epoch": int(best_val_acc_epoch),
            "final_training_accuracy": float(self.history['accuracy'][-1]),
            "final_validation_accuracy": float(self.history['val_accuracy'][-1]),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "architecture": "Transfer Learning (PyTorch)",
            "input_shape": [224, 224, 3],
            "optimizer": "Adam",
            "loss_function": "BCELoss",
            "device": str(self.device)
        }
        
        # Save summary
        summary_path = self.results_path / f"{model_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_acc_epoch}")
        
        return summary
    
    def train_multiple_models(self, train_loader, val_loader, models_to_train=['resnet50', 'mobilenetv2']):
        """Train multiple transfer learning models for comparison"""
        
        results = {}
        
        for model_name in models_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} Transfer Learning Model")
            print(f"{'='*60}")
            
            try:
                # Reset history for new model
                self.history = {
                    'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
                    'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
                }
                
                # Build model
                self.build_transfer_model(base_model_name=model_name)
                
                # Compile model
                self.compile_model(learning_rate=0.001)
                
                # Train model
                history = self.train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=30,
                    fine_tune_epochs=15,
                    model_name=f"transfer_{model_name}"
                )
                
                # Plot results
                self.plot_training_history(f"transfer_{model_name}")
                
                # Save summary
                summary = self.save_training_summary(f"transfer_{model_name}")
                
                results[model_name] = summary
                
                print(f"{model_name.upper()} training completed successfully!")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return results


def main():
    """Main training function"""
    
    print("FalconEye-Detect Transfer Learning Training (PyTorch)")
    print("="*60)
    
    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor()
    trainer = TransferLearningTrainer()
    
    # Create optimized data loaders
    print("Creating optimized data loaders...")
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        batch_size=32  # num_workers auto-detected
    )
    
    # Train multiple models for comparison
    models_to_train = ['resnet50', 'mobilenetv2', 'efficientnetb0']
    
    print(f"\nTraining {len(models_to_train)} transfer learning models...")
    results = trainer.train_multiple_models(
        train_loader=train_loader,
        val_loader=val_loader,
        models_to_train=models_to_train
    )
    
    # Print comparison summary
    print("\n" + "="*60)
    print("TRANSFER LEARNING MODELS COMPARISON")
    print("="*60)
    
    for model_name, summary in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Best Validation Accuracy: {summary['best_validation_accuracy']:.4f}")
        print(f"  Total Parameters: {summary['total_parameters']:,}")
        print(f"  Trainable Parameters: {summary['trainable_parameters']:,}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['best_validation_accuracy'])
        print(f"\n{'='*60}")
        print(f"Best performing model: {best_model[0].upper()}")
        print(f"Best accuracy: {best_model[1]['best_validation_accuracy']:.4f}")
        print(f"{'='*60}")
    
    print("\nTransfer learning training completed!")


if __name__ == "__main__":
    main()
