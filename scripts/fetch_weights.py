"""
Weight Provisioning Script for FalconEye-Detect

Fetches model weights when deploying to Streamlit Cloud or other environments
where Git LFS objects may not be available.

Usage:
    python3 scripts/fetch_weights.py [--force]
"""

import os
import sys
import json
from pathlib import Path
import urllib.request
import shutil

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_MANIFEST = PROJECT_ROOT / "WEIGHTS_URLS.json"

def check_lfs_pointer(filepath):
    """Check if file is a Git LFS pointer (small text file)"""
    if not filepath.exists():
        return False
    
    # LFS pointers are small text files (~130 bytes)
    if filepath.stat().st_size < 1024:  # Less than 1KB
        try:
            with open(filepath, 'r') as f:
                content = f.read(200)
                if 'version https://git-lfs.github.com' in content:
                    return True
        except:
            pass
    return False

def download_file(url, dest_path, filename):
    """Download file with progress"""
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
                        print(f"  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
            
            # Move temp to final location
            shutil.move(str(temp_path), str(dest_path))
            print(f"\n✅ Downloaded {filename} ({dest_path.stat().st_size} bytes)")
            return True
            
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def load_weights_manifest():
    """Load weights manifest from WEIGHTS_URLS.json"""
    if not WEIGHTS_MANIFEST.exists():
        print(f"⚠️  Weights manifest not found: {WEIGHTS_MANIFEST}")
        print("   Creating manifest with GitHub Releases placeholder...")
        
        # Create placeholder manifest
        manifest = {
            "note": "Replace URLs with actual download links from GitHub Releases or S3",
            "weights": {}
        }
        
        # Find existing weight files
        if WEIGHTS_DIR.exists():
            for pt_file in WEIGHTS_DIR.glob("*.pt"):
                if not check_lfs_pointer(pt_file):
                    # Real file exists, add placeholder URL
                    manifest["weights"][pt_file.name] = {
                        "url": f"https://github.com/YOUR_USERNAME/FalconEye-Detect/releases/download/v1.0/{pt_file.name}",
                        "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2),
                        "required": False
                    }
        
        with open(WEIGHTS_MANIFEST, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   Created template: {WEIGHTS_MANIFEST}")
        print(f"   Please update URLs and run again.")
        return manifest
    
    with open(WEIGHTS_MANIFEST) as f:
        return json.load(f)

def fetch_weights(force=False):
    """Main function to fetch weights"""
    
    print("=" * 60)
    print("FalconEye-Detect Weight Provisioning")
    print("=" * 60)
    
    # Create weights directory
    WEIGHTS_DIR.mkdir(exist_ok=True)
    
    # Load manifest
    manifest = load_weights_manifest()
    weights = manifest.get('weights', {})
    
    if not weights:
        print("\n⚠️  No weights in manifest. Checking local files...")
        
        # List existing weights
        existing = list(WEIGHTS_DIR.glob("*.pt"))
        if existing:
            print(f"\nFound {len(existing)} local weight file(s):")
            for pt_file in existing:
                is_lfs = check_lfs_pointer(pt_file)
                status = "LFS pointer" if is_lfs else f"{pt_file.stat().st_size / (1024*1024):.1f} MB"
                print(f"  - {pt_file.name} ({status})")
            
            if any(check_lfs_pointer(f) for f in existing):
                print("\n⚠️  Some files are LFS pointers. Weights need to be downloaded.")
                print("   Update WEIGHTS_URLS.json with download URLs and run again.")
        else:
            print("  No weight files found.")
        
        return
    
    # Process each weight in manifest
    print(f"\nProcessing {len(weights)} weight file(s) from manifest...\n")
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for filename, info in weights.items():
        dest_path = WEIGHTS_DIR / filename
        url = info.get('url', '')
        
        # Check if already exists and is valid
        if dest_path.exists() and not check_lfs_pointer(dest_path):
            if not force:
                print(f"✓ {filename} already exists ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
                skipped_count += 1
                continue
        
        # Download
        if not url or 'YOUR_USERNAME' in url:
            print(f"⚠️  {filename}: No valid URL in manifest (skipping)")
            failed_count += 1
            continue
        
        if download_file(url, dest_path, filename):
            downloaded_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Weight Provisioning Summary")
    print("=" * 60)
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped (already present): {skipped_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\n⚠️  Some weights  failed to download. Check URLs in WEIGHTS_URLS.json")
        sys.exit(1)
    else:
        print("\n✅ Weight provisioning complete!")
        sys.exit(0)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch model weights for FalconEye-Detect')
    parser.add_argument('--force', action='store_true', help='Re-download even if files exist')
    
    args = parser.parse_args()
    
    fetch_weights(force=args.force)

if __name__ == "__main__":
    main()
