"""
Inference sanity test for FalconEye-Detect classification models
Tests all discovered models on a small validation set
"""

import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

# Add scripts to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from inference_utils import ModelInference

def load_test_samples(n=10):
    """Load a small set of labeled test images for sanity checks"""
    
    test_path = project_root / "data" / "classification_dataset" / "test"
    
    if not test_path.exists():
        print(f"Test path not found: {test_path}")
        return []
    
    samples = []
    
    for class_dir in test_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        # Take first n//2 images from each class
        for img_path in images[:n//2]:
            samples.append((img_path, class_name))
    
    return samples[:n]

def run_inference_sanity_test():
    """Run sanity test on all loaded models"""
    
    print("=" * 60)
    print("FalconEye-Detect Inference Sanity Test")
    print("=" * 60)
    
    # Load models
    print("\nInitializing inference system...")
    inference = ModelInference()
    inference.load_all_available_models()
    
    if not inference.classification_models:
        print("\n❌ No classification models loaded")
        return 1
    
    # Load test samples
    print("\nLoading test samples...")
    test_samples = load_test_samples(n=10)
    
    if not test_samples:
        print("❌ No test samples found")
        return 1
    
    print(f"Loaded {len(test_samples)} test samples")
    class_counts = {}
    for _, label in test_samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f"Class distribution: {class_counts}")
    
    # Test each model
    results = {}
    
    print(f"\n{'Model':<45} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    print("=" * 75)
    
    for model_name in sorted(inference.classification_models.keys()):
        
        correct = 0
        predictions = []
        errors = []
        
        for img_path, true_label in test_samples:
            try:
                result = inference.predict_classification(str(img_path), model_name=model_name)
                
                if result:
                    pred_label = result['class_name']
                    predictions.append({
                        'image': img_path.name,
                        'true': true_label,
                        'predicted': pred_label,
                        'confidence': result['confidence']
                    })
                    
                    # Case-insensitive comparison
                    if pred_label.lower() == true_label.lower():
                        correct += 1
            except Exception as e:
                errors.append({
                    'image': img_path.name,
                    'error': str(e)
                })
        
        accuracy = correct / len(test_samples) if test_samples else 0
        
        results[model_name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_samples),
            'predictions': predictions,
            'errors': errors
        }
        
        # Display result
        status = "✅" if accuracy > 0.6 else "⚠️"
        print(f"{status} {model_name:<42} {accuracy:>9.1%} {correct:>8} {len(test_samples):>8}")
        
        if errors:
            print(f"   └─ {len(errors)} error(s)")
    
    # Save results
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "inference_sanity.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\nTest complete!")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = run_inference_sanity_test()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
