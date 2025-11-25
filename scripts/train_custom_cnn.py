"""
Custom CNN Training Module for FalconEye-Detect (PyTorch)
Builds and trains a custom CNN architecture for bird vs drone classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


class CustomCNN(nn.Module):
    """Custom CNN architecture for bird vs drone classification"""
    
    def __init__(self, input_channels=3, num_classes=1):
        super(CustomCNN, self).__init__()
        
        # First Convolutional Block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Second Convolutional Block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Third Convolutional Block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Fourth Convolutional Block
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            # Output layer - sigmoid for binary classification
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


class CustomCNNTrainer:
    """Custom CNN model builder and trainer"""
    
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
        
        # Device selection: MPS (Mac M1/M2) -> CUDA -> CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        
        print(f"Using device: {self.device}")
        
        # Initialize training history
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
        }
    
    def build_cnn_model(self, input_channels=3, num_classes=1):
        """Build custom CNN architecture"""
        
        self.model = CustomCNN(input_channels=input_channels, num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("Model built successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
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
            factor=0.5, 
            patience=7, 
            min_lr=1e-7
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
        
        print("Model compiled successfully!")
        print(f"Optimizer: Adam (lr={learning_rate})")
        print(f"Loss function: BCELoss")
    
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
                # Forward pass with autocast
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training (CPU)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics (outside autocast to avoid precision issues)
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
        
        # Use inference_mode for faster validation (10-15% speedup vs no_grad)
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward pass (no autocast needed for inference)
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
    
    def train_model(self, train_loader, val_loader, epochs=100, model_name="custom_cnn"):
        """Train the CNN model with early stopping and model checkpointing"""
        
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate based on validation loss
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
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
            
            # Model checkpoint - save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_path = self.weights_path / f"{model_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                    'history': self.history
                }, best_model_path)
                print(f"✓ Saved best model (val_acc: {best_val_accuracy:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_val_accuracy:.4f}")
                break
        
        # Load best model
        best_checkpoint = torch.load(self.weights_path / f"{model_name}_best.pt")
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Save final model
        final_model_path = self.weights_path / f"{model_name}_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self, model_name="custom_cnn"):
        """Plot training history"""
        
        if not self.history['loss']:
            raise ValueError("No training history available")
        
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(epochs, self.history['accuracy'], 'bo-', label='Training Accuracy', markersize=3)
        axes[0, 0].plot(epochs, self.history['val_accuracy'], 'ro-', label='Validation Accuracy', markersize=3)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(epochs, self.history['loss'], 'bo-', label='Training Loss', markersize=3)
        axes[0, 1].plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss', markersize=3)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        axes[1, 0].plot(epochs, self.history['precision'], 'bo-', label='Training Precision', markersize=3)
        axes[1, 0].plot(epochs, self.history['val_precision'], 'ro-', label='Validation Precision', markersize=3)
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        axes[1, 1].plot(epochs, self.history['recall'], 'bo-', label='Training Recall', markersize=3)
        axes[1, 1].plot(epochs, self.history['val_recall'], 'ro-', label='Validation Recall', markersize=3)
        axes[1, 1].set_title('Model Recall')
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
    
    def save_training_summary(self, model_name="custom_cnn"):
        """Save training summary and metrics"""
        
        if not self.history['loss']:
            raise ValueError("No training history available")
        
        # Get best metrics
        best_val_acc_epoch = np.argmax(self.history['val_accuracy']) + 1
        best_val_acc = max(self.history['val_accuracy'])
        best_val_loss = min(self.history['val_loss'])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        summary = {
            "model_name": model_name,
            "training_completed": datetime.now().isoformat(),
            "total_epochs": len(self.history['loss']),
            "best_validation_accuracy": float(best_val_acc),
            "best_validation_loss": float(best_val_loss),
            "best_epoch": int(best_val_acc_epoch),
            "final_training_accuracy": float(self.history['accuracy'][-1]),
            "final_validation_accuracy": float(self.history['val_accuracy'][-1]),
            "model_parameters": int(total_params),
            "architecture": "Custom CNN (PyTorch)",
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


def main():
    """Main training function"""
    
    print("FalconEye-Detect Custom CNN Training (PyTorch)")
    print("="*50)
    
    # Initialize preprocessor and trainer
    preprocessor = DataPreprocessor()
    trainer = CustomCNNTrainer()
    
    # Create data loaders with optimized settings
    print("Creating optimized data loaders...")
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
        batch_size=32  # num_workers auto-detected in create_data_loaders
    )
    
    # Build model
    print("\nBuilding custom CNN model...")
    model = trainer.build_cnn_model()
    
    # Display model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Compile model with AMP enabled
    print("\nCompiling model with optimizations...")
    trainer.compile_model(learning_rate=0.001, use_amp=True)
    
    # Train model
    print("\nStarting model training...")
    history = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        model_name="custom_cnn"
    )
    
    # Plot training history
    print("\nGenerating training plots...")
    trainer.plot_training_history("custom_cnn")
    
    # Save training summary
    print("\nSaving training summary...")
    summary = trainer.save_training_summary("custom_cnn")
    
    print("\n" + "="*50)
    print("Custom CNN training completed!")
    print(f"Best validation accuracy: {summary['best_validation_accuracy']:.4f}")
    print(f"Model saved to: {trainer.weights_path}")
    print("="*50)


if __name__ == "__main__":
    main()
