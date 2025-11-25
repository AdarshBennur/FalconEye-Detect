"""
Data Preprocessing Module for FalconEye-Detect (PyTorch)
Handles image loading, normalization, augmentation, and data loaders
for bird vs drone classification and object detection tasks.
"""

import os
import sys

# Environment configuration
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
cv2.setNumThreads(0)

from pathlib import Path
import yaml
from PIL import Image

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class BirdDroneDataset(Dataset):
    """PyTorch Dataset for Bird vs Drone classification"""
    
    def __init__(self, data_dir, transform=None, max_images_per_class=None):
        """
        Args:
            data_dir: Directory containing 'bird' and 'drone' subdirectories
            transform: torchvision transforms to apply
            max_images_per_class: Optional limit on images per class
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load bird images (label 0)
        bird_path = self.data_dir / "bird"
        if bird_path.exists():
            bird_files = sorted(list(bird_path.glob("*.jpg")))
            if max_images_per_class:
                bird_files = bird_files[:max_images_per_class]
            
            for img_file in bird_files:
                self.samples.append(str(img_file))
                self.labels.append(0)
        
        # Load drone images (label 1)
        drone_path = self.data_dir / "drone"
        if drone_path.exists():
            drone_files = sorted(list(drone_path.glob("*.jpg")))
            if max_images_per_class:
                drone_files = drone_files[:max_images_per_class]
            
            for img_file in drone_files:
                self.samples.append(str(img_file))
                self.labels.append(1)
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and transform a single image"""
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Return class distribution statistics"""
        labels_array = np.array(self.labels)
        return {
            'total': len(self.labels),
            'birds': np.sum(labels_array == 0),
            'drones': np.sum(labels_array == 1)
        }


class DataPreprocessor:
    """Handles all data preprocessing tasks for classification and detection"""
    
    def __init__(self, base_path=None):
        # Auto-detect project root if not provided
        if base_path is None:
            # Get the directory containing this script (scripts/)
            # Then go up one level to get project root
            base_path = Path(__file__).parent.parent
        
        self.base_path = Path(base_path)
        self.classification_path = self.base_path / "data" / "classification_dataset"
        self.detection_path = self.base_path / "data" / "object_detection_dataset"
        self.processed_path = self.base_path / "data" / "processed"
        self.img_size = (224, 224)
        
        # Create processed directory if it doesn't exist
        self.processed_path.mkdir(exist_ok=True)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            # ImageNet normalization - can be changed to [0,1] normalization if needed
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_classification_data(self, subset="train", max_images_per_class=None):
        """Load and preprocess classification dataset (returns numpy arrays for compatibility)"""
        data_path = self.classification_path / subset
        images, labels = [], []
        
        print(f"Loading {subset} dataset...")
        
        try:
            # Load bird images (label 0)
            bird_path = data_path / "bird"
            if bird_path.exists():
                bird_files = list(bird_path.glob("*.jpg"))
                if max_images_per_class:
                    bird_files = bird_files[:max_images_per_class]
                
                print(f"Loading {len(bird_files)} bird images...")
                for i, img_file in enumerate(bird_files):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            images.append(img)
                            labels.append(0)  # Bird
                        
                        if (i + 1) % 100 == 0:
                            print(f"  Loaded {i + 1}/{len(bird_files)} bird images")
                            
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
                        continue
            
            # Load drone images (label 1)
            drone_path = data_path / "drone"
            if drone_path.exists():
                drone_files = list(drone_path.glob("*.jpg"))
                if max_images_per_class:
                    drone_files = drone_files[:max_images_per_class]
                
                print(f"Loading {len(drone_files)} drone images...")
                for i, img_file in enumerate(drone_files):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.img_size)
                            images.append(img)
                            labels.append(1)  # Drone
                        
                        if (i + 1) % 100 == 0:
                            print(f"  Loaded {i + 1}/{len(drone_files)} drone images")
                            
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
                        continue
            
            if not images:
                raise ValueError(f"No images found in {data_path}")
            
            # Convert to numpy arrays and normalize
            print("Converting to numpy arrays and normalizing...")
            images = np.array(images, dtype=np.float32) / 255.0
            labels = np.array(labels)
            
            print(f"Successfully loaded {len(images)} images from {subset} set")
            print(f"Birds: {np.sum(labels == 0)}, Drones: {np.sum(labels == 1)}")
            
            return images, labels
            
        except Exception as e:
            print(f"Error loading classification data: {e}")
            raise
    
    def create_data_loaders(self, batch_size=32, num_workers=None, max_images_per_class=None):
        """Create optimized PyTorch DataLoaders for training, validation, and testing"""
        
        print("Creating optimized PyTorch DataLoaders...")
        
        # Auto-detect optimal num_workers for Mac M2
        if num_workers is None:
            # Use 4 workers for faster data loading, fallback to 0 if issues arise
            # Set to 0 if experiencing multiprocessing issues on macOS
            num_workers = 0  # Safe default for macOS MPS
        
        # Detect if MPS is available for pin_memory optimization
        # Note: MPS doesn't support pin_memory yet, so disable it for MPS
        use_pin_memory = torch.cuda.is_available()  # Only enable for CUDA
        
        # Create datasets
        train_dataset = BirdDroneDataset(
            self.classification_path / "train",
            transform=self.train_transform,
            max_images_per_class=max_images_per_class
        )
        
        val_dataset = BirdDroneDataset(
            self.classification_path / "valid",
            transform=self.val_transform,
            max_images_per_class=max_images_per_class
        )
        
        test_dataset = BirdDroneDataset(
            self.classification_path / "test",
            transform=self.val_transform,
            max_images_per_class=max_images_per_class
        )
        
        # Create optimized DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        # Print statistics
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"  {train_dataset.get_class_distribution()}")
        print(f"Validation dataset: {len(val_dataset)} samples")
        print(f"  {val_dataset.get_class_distribution()}")
        print(f"Test dataset: {len(test_dataset)} samples")
        print(f"  {test_dataset.get_class_distribution()}")
        print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={use_pin_memory}")
        
        return train_loader, val_loader, test_loader
    
    def create_data_generators(self, batch_size=32, validation_split=0.2):
        """
        Legacy method for backward compatibility with old TensorFlow-style code.
        Returns DataLoaders but with a similar interface.
        Note: validation_split parameter is ignored as we use separate directories.
        """
        print("Note: create_data_generators() is deprecated. Use create_data_loaders() instead.")
        return self.create_data_loaders(batch_size=batch_size)
    
    def prepare_detection_data(self):
        """Prepare object detection data in YOLOv8 format"""
        
        # Read the existing data.yaml configuration
        yaml_path = self.detection_path / "data.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths to be absolute
            config['train'] = str(self.detection_path / "train" / "images")
            config['val'] = str(self.detection_path / "valid" / "images")
            config['test'] = str(self.detection_path / "test" / "images")
            
            # Save updated configuration
            updated_yaml_path = self.processed_path / "detection_data.yaml"
            with open(updated_yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print(f"Detection data configuration saved to {updated_yaml_path}")
            return str(updated_yaml_path)
        else:
            raise FileNotFoundError("data.yaml not found in object detection dataset")
    
    def visualize_samples(self, num_samples=8, show_plot=False):
        """Visualize sample images from the dataset"""
        
        try:
            # Load some samples from train set
            images, labels = self.load_classification_data("train", max_images_per_class=100)
            
            # Select random samples
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            
            class_names = ['Bird', 'Drone']
            
            for i, idx in enumerate(indices):
                axes[i].imshow(images[idx])
                axes[i].set_title(f"{class_names[labels[idx]]}")
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.processed_path / "sample_images.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Sample visualization saved to {save_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing without visualization...")
    
    def analyze_dataset(self):
        """Analyze dataset statistics and class distribution"""
        
        stats = {}
        
        for subset in ['train', 'valid', 'test']:
            try:
                images, labels = self.load_classification_data(subset, max_images_per_class=None)
                
                stats[subset] = {
                    'total_images': len(images),
                    'birds': np.sum(labels == 0),
                    'drones': np.sum(labels == 1),
                    'bird_ratio': np.sum(labels == 0) / len(images) if len(images) > 0 else 0
                }
            except Exception as e:
                print(f"Error analyzing {subset} set: {e}")
                stats[subset] = {'error': str(e)}
        
        # Print analysis
        print("\n=== Dataset Analysis ===")
        for subset, data in stats.items():
            if 'error' in data:
                print(f"\n{subset.upper()} SET: Error - {data['error']}")
            else:
                print(f"\n{subset.upper()} SET:")
                print(f"  Total images: {data['total_images']}")
                print(f"  Birds: {data['birds']}")
                print(f"  Drones: {data['drones']}")
                print(f"  Bird ratio: {data['bird_ratio']:.2%}")
        
        return stats
    
    def preprocess_single_image(self, image_path, for_detection=False):
        """Preprocess a single image for inference"""
        
        if for_detection:
            # For YOLOv8, return as-is (YOLOv8 handles preprocessing)
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            # For classification, return PyTorch tensor
            try:
                image = Image.open(image_path).convert('RGB')
                tensor = self.val_transform(image)
                # Add batch dimension
                return tensor.unsqueeze(0)
            except Exception as e:
                raise ValueError(f"Could not load image {image_path}: {e}")


def test_basic_functionality():
    """Test basic functionality with limited data to isolate issues"""
    
    print("Testing basic PyTorch preprocessing functionality...")
    
    try:
        preprocessor = DataPreprocessor()
        print("✓ DataPreprocessor initialized successfully")
        
        # Test with limited data first
        print("\nTesting with small dataset (10 images per class)...")
        train_images, train_labels = preprocessor.load_classification_data("train", max_images_per_class=10)
        print(f"✓ Loaded {len(train_images)} training images successfully")
        
        # Test DataLoader creation
        print("\nTesting DataLoader creation...")
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
            batch_size=4, 
            num_workers=0,
            max_images_per_class=10
        )
        print(f"✓ Created DataLoaders: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
        
        # Test batch loading
        print("\nTesting batch loading...")
        for images, labels in train_loader:
            print(f"✓ Loaded batch: images shape={images.shape}, labels shape={labels.shape}")
            print(f"  Image dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
            break
        
        # Test detection data preparation
        print("\nTesting detection data preparation...")
        yaml_path = preprocessor.prepare_detection_data()
        print(f"✓ Detection configuration prepared: {yaml_path}")
        
        print("\n✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to demonstrate preprocessing capabilities"""
    
    try:
        # First run basic functionality test
        if not test_basic_functionality():
            print("Exiting due to basic functionality test failure.")
            return
        
        print("\n" + "="*50)
        print("FalconEye-Detect Data Preprocessing - Full Run (PyTorch)")
        print("="*50)
        
        preprocessor = DataPreprocessor()
        
        # Analyze dataset with error handling
        print("\nAnalyzing dataset...")
        try:
            stats = preprocessor.analyze_dataset()
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            print("Continuing with basic preprocessing...")
        
        # Visualize samples (save only, no display)
        print("\nCreating sample visualization...")
        try:
            preprocessor.visualize_samples(num_samples=8, show_plot=False)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Continuing without visualization...")
        
        # Create data loaders
        print("\nCreating PyTorch DataLoaders...")
        try:
            train_loader, val_loader, test_loader = preprocessor.create_data_loaders(batch_size=32)
            
            print(f"Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
            print(f"Validation loader: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
            print(f"Test loader: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            print("This might be due to data path issues or insufficient memory.")
        
        # Prepare detection data
        print("\nPreparing detection data...")
        try:
            yaml_path = preprocessor.prepare_detection_data()
            print(f"Detection configuration ready: {yaml_path}")
        except Exception as e:
            print(f"Error preparing detection data: {e}")
        
        print("\nData preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Fatal error in preprocessing: {e}")
        print("Please check your data paths and ensure all dependencies are installed.")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    
    # Check if we should run in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in test mode...")
        success = test_basic_functionality()
        if success:
            print("Test completed successfully!")
        else:
            print("Test failed!")
            sys.exit(1)
    else:
        main()
