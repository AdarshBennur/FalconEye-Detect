"""
Weight Synchronization Script for FalconEye-Detect

Downloads missing model weights from remote URLs with checksum verification.
Supports weights/manifest.json for configuration.

Usage:
    python3 scripts/sync_weights.py [--force] [--verify-only]
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import urllib.request
import shutil

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MANIFEST_FILE = WEIGHTS_DIR / "manifest.json"

def compute_checksum(filepath, algorithm='sha256'):
    """Compute file checksum"""
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def check_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer"""
    if not filepath.exists():
        return False
    
    # LFS pointers are small text files
    if filepath.stat().st_size < 1024:
        try:
            with open(filepath, 'r') as f:
                content = f.read(200)
                if 'version https://git-lfs.github.com' in content:
                    return True
        except:
            pass
    return False

def download_file_with_progress(url, dest_path, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Create temp file
            temp_path = dest_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_length - filled)
                        print(f"  [{bar}] {progress:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
            
            # Move temp to final location
            shutil.move(str(temp_path), str(dest_path))
            print(f"\n✅ Downloaded {filename} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return True
            
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def create_default_manifest():
    """Create default manifest template"""
    
    manifest = {
        "version": "1.0",
        "description": "FalconEye-Detect model weights manifest",
        "weights": {}
    }
    
    # Scan existing weights
    if WEIGHTS_DIR.exists():
        for pt_file in WEIGHTS_DIR.glob("*.pt"):
            if not check_lfs_pointer(pt_file):
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                checksum = compute_checksum(pt_file)
                
                manifest["weights"][pt_file.name] = {
                    "url": f"https://github.com/YOUR_USERNAME/FalconEye-Detect/releases/download/v1.0.0/{pt_file.name}",
                    "size_mb": round(size_mb, 2),
                    "checksum": checksum,
                    "algorithm": "sha256",
                    "required": True
                }
    
    return manifest

def load_manifest():
    """Load weights manifest"""
    
    if not MANIFEST_FILE.exists():
        print(f"⚠️  Manifest not found: {MANIFEST_FILE}")
        print("   Creating template manifest...")
        
        manifest = create_default_manifest()
        
        # Save template
        WEIGHTS_DIR.mkdir(exist_ok=True)
        with open(MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Created: {MANIFEST_FILE}")
        print("   Please update URLs for deployment!")
        
        return manifest
    
    with open(MANIFEST_FILE) as f:
        return json.load(f)

def verify_weights(manifest, verbose=True):
    """Verify existing weights against manifest"""
    
    weights = manifest.get('weights', {})
    results = {}
    
    for filename, info in weights.items():
        filepath = WEIGHTS_DIR / filename
        
        status = {
            'exists': filepath.exists(),
            'is_lfs_pointer': False,
            'checksum_valid': False,
            'size_matches': False
        }
        
        if filepath.exists():
            status['is_lfs_pointer'] = check_lfs_pointer(filepath)
            
            if not status['is_lfs_pointer']:
                # Verify checksum
                if 'checksum' in info:
                    actual_checksum = compute_checksum(filepath, info.get('algorithm', 'sha256'))
                    status['checksum_valid'] = (actual_checksum == info['checksum'])
                else:
                    status['checksum_valid'] = None  # No checksum in manifest
                
                # Verify size
                actual_size_mb = filepath.stat().st_size / (1024 * 1024)
                expected_size_mb = info.get('size_mb', 0)
                size_diff = abs(actual_size_mb - expected_size_mb)
                status['size_matches'] = (size_diff < 0.1)  # Within 0.1 MB tolerance
        
        results[filename] = status
        
        if verbose:
            if not status['exists']:
                print(f"❌ {filename}: Missing")
            elif status['is_lfs_pointer']:
                print(f"⚠️  {filename}: LFS pointer (needs download)")
            elif status['checksum_valid'] == False:
                print(f"⚠️  {filename}: Checksum mismatch")
            elif status['size_matches'] == False:
                print(f"⚠️  {filename}: Size mismatch")
            else:
                print(f"✅ {filename}: Valid")
    
    return results

def sync_weights(force=False, verify_only=False):
    """Main synchronization function"""
    
    print("=" * 60)
    print("FalconEye-Detect Weight Synchronization")
    print("=" * 60)
    
    # Create weights directory
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # Load manifest
    manifest = load_manifest()
    weights = manifest.get('weights', {})
    
    if not weights:
        print("\n⚠️  No weights in manifest!")
        return 1
    
    print(f"\nManifest version: {manifest.get('version', 'unknown')}")
    print(f"Total weights: {len(weights)}\n")
    
    # Verify existing weights
    if verify_only:
        print("Verifying weights...")
        verify_weights(manifest, verbose=True)
        return 0
    
    # Sync weights
    download_count = 0
    skip_count = 0
    fail_count = 0
    
    for filename, info in weights.items():
        filepath = WEIGHTS_DIR / filename
        url = info.get('url', '')
        
        # Check if download needed
        needs_download = False
        
        if not filepath.exists():
            needs_download = True
            reason = "missing"
        elif check_lfs_pointer(filepath):
            needs_download = True
            reason = "LFS pointer"
        elif force:
            needs_download = True
            reason = "force flag"
        elif 'checksum' in info:
            actual_checksum = compute_checksum(filepath, info.get('algorithm', 'sha256'))
            if actual_checksum != info['checksum']:
                needs_download = True
                reason = "checksum mismatch"
        
        if not needs_download:
            print(f"✓ {filename} ({filepath.stat().st_size / (1024*1024):.1f} MB) - OK")
            skip_count += 1
            continue
        
        # Download
        print(f"\n{filename} ({reason})")
        
        if not url or 'YOUR_USERNAME' in url:
            print(f"⚠️  No valid URL in manifest (skipping)")
            fail_count += 1
            continue
        
        if download_file_with_progress(url, filepath, filename):
            # Verify checksum after download
            if 'checksum' in info:
                actual_checksum = compute_checksum(filepath, info.get('algorithm', 'sha256'))
                if actual_checksum == info['checksum']:
                    print(f"✅ Checksum verified")
                    download_count += 1
                else:
                    print(f"⚠️  Checksum mismatch after download!")
                    filepath.unlink()  # Remove bad file
                    fail_count += 1
            else:
                download_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Synchronization Summary")
    print("=" * 60)
    print(f"Downloaded: {download_count}")
    print(f"Skipped (already valid): {skip_count}")
    print(f"Failed: {fail_count}")
    
    if fail_count > 0:
        print("\n⚠️  Some weights failed to download")
        print("   Check URLs in weights/manifest.json")
        return 1
    else:
        print("\n✅ Weight synchronization complete!")
        return 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Synchronize model weights for FalconEye-Detect')
    parser.add_argument('--force', action='store_true',
                       help='Re-download even if files exist and checksums match')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing weights without downloading')
    
    args = parser.parse_args()
    
    try:
        exit_code = sync_weights(force=args.force, verify_only=args.verify_only)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
