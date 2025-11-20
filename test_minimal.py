#!/usr/bin/env python3
"""
Minimal test to isolate the mutex lock issue on macOS
"""

import os
import sys

# Set matplotlib backend early to prevent GUI issues
import matplotlib
matplotlib.use('Agg')

print("Step 1: Basic imports...")
try:
    import numpy as np
    print("✓ numpy imported")
    
    import cv2
    print("✓ opencv imported")
    
    # Disable tensorflow warnings and threading issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    # Limit tensorflow threading to prevent mutex issues
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    print("✓ tensorflow imported and configured")
    
    from pathlib import Path
    print("✓ pathlib imported")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nStep 2: Testing path access...")
try:
    # Auto-detect project root
    base_path = Path(__file__).parent
    print(f"Base path exists: {base_path.exists()}")
    
    data_path = base_path / "data"
    print(f"Data path exists: {data_path.exists()}")
    
    classification_path = data_path / "classification_dataset"
    print(f"Classification path exists: {classification_path.exists()}")
    
    if classification_path.exists():
        train_path = classification_path / "train"
        print(f"Train path exists: {train_path.exists()}")
        
        if train_path.exists():
            bird_path = train_path / "bird"
            drone_path = train_path / "drone"
            print(f"Bird path exists: {bird_path.exists()}")
            print(f"Drone path exists: {drone_path.exists()}")
            
            if bird_path.exists():
                bird_images = list(bird_path.glob("*.jpg"))
                print(f"Found {len(bird_images)} bird images")
            
            if drone_path.exists():
                drone_images = list(drone_path.glob("*.jpg"))
                print(f"Found {len(drone_images)} drone images")
    
except Exception as e:
    print(f"✗ Path access error: {e}")
    sys.exit(1)

print("\nStep 3: Testing basic image loading...")
try:
    # Try to load one image
    test_img_path = None
    
    if classification_path.exists():
        train_path = classification_path / "train" / "bird"
        if train_path.exists():
            bird_images = list(train_path.glob("*.jpg"))
            if bird_images:
                test_img_path = bird_images[0]
    
    if test_img_path and test_img_path.exists():
        print(f"Testing with image: {test_img_path}")
        
        # Load image with OpenCV
        img = cv2.imread(str(test_img_path))
        if img is not None:
            print(f"✓ Image loaded successfully, shape: {img.shape}")
            
            # Convert color space
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"✓ Color conversion successful")
            
            # Resize
            img_resized = cv2.resize(img_rgb, (224, 224))
            print(f"✓ Resize successful: {img_resized.shape}")
            
            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0
            print(f"✓ Normalization successful")
            
        else:
            print("✗ Failed to load test image")
    else:
        print("No test images available")

except Exception as e:
    print(f"✗ Image loading error: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 4: Testing class initialization...")
try:
    # Add the scripts directory to Python path
    scripts_path = str(base_path / "scripts")
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    
    from data_preprocessing import DataPreprocessor
    print("✓ DataPreprocessor class imported")
    
    preprocessor = DataPreprocessor()
    print("✓ DataPreprocessor initialized")
    
except Exception as e:
    print(f"✗ Class initialization error: {e}")
    import traceback
    traceback.print_exc()

print("\nMinimal test completed!")
