# Hosting Model Weights for Deployment

## Problem

Model weight files (`.pt`) are large (5-100 MB each) and cannot be stored directly in Git or reliably served via Git LFS on platforms like Streamlit Cloud.

## Solution

Host weights externally (S3, Google Cloud Storage, or GitHub Releases) and use the weight synchronization system to download them automatically.

---

## Option 1: GitHub Releases (Recommended for Public Projects)

### Step 1: Create a Release

```bash
# Create and push tag
git tag v1.0.0
git push origin v1.0.0
```

### Step 2: Upload Weights to Release

1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Choose tag `v1.0.0`
4. Upload all `.pt` files from `weights/` folder as release assets
5. Publish release

### Step 3: Update Manifest

The manifest was auto-generated at `weights/manifest.json`. Update URLs:

```json
{
  "version": "1.0",
  "weights": {
    "custom_cnn_best.pt": {
      "url": "https://github.com/YOUR_USERNAME/FalconEye-Detect/releases/download/v1.0.0/custom_cnn_best.pt",
      "size_mb": 16.5,
      "checksum": "abc123...",
      "algorithm": "sha256",
      "required": true
    },
    ...
  }
}
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Option 2: Amazon S3

### Step 1: Create S3 Bucket

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create bucket
aws s3 mb s3://falconeye-weights
```

### Step 2: Upload Weights

```bash
# Make bucket public (or use signed URLs)
aws s3api put-bucket-acl --bucket falconeye-weights --acl public-read

# Upload all weights
aws s3 cp weights/ s3://falconeye-weights/v1.0.0/ --recursive --exclude "*" --include "*.pt"

# Make files public
aws s3api put-object-acl --bucket falconeye-weights --key "v1.0.0/*" --acl public-read
```

### Step 3: Update Manifest

```json
{
  "weights": {
    "custom_cnn_best.pt": {
      "url": "https://falconeye-weights.s3.amazonaws.com/v1.0.0/custom_cnn_best.pt",
      ...
    }
  }
}
```

---

## Option 3: Google Cloud Storage

### Step 1: Create GCS Bucket

```bash
# Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# Create bucket
gsutil mb gs://falconeye-weights

# Make bucket public
gsutil iam ch allUsers:objectViewer gs://falconeye-weights
```

### Step 2: Upload Weights

```bash
# Upload all weights
gsutil -m cp weights/*.pt gs://falconeye-weights/v1.0.0/
```

### Step 3: Update Manifest

```json
{
  "weights": {
    "custom_cnn_best.pt": {
      "url": "https://storage.googleapis.com/falconeye-weights/v1.0.0/custom_cnn_best.pt",
      ...
    }
  }
}
```

---

## Testing Weight Sync Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify existing weights
python3 scripts/sync_weights.py --verify-only

# Sync weights (download if missing/corrupted)
python3 scripts/sync_weights.py

# Force re-download all weights
python3 scripts/sync_weights.py --force
```

---

## Deployment to Streamlit Cloud

### Automatic Sync

The Streamlit app automatically runs `sync_weights.py` on startup. It will:

1. Check for missing or corrupted weight files
2. Download from URLs in `weights/manifest.json`
3. Verify checksums after download
4. Show progress in UI

### Manual Configuration

If you need to disable automatic sync, set environment variable:

```bash
SKIP_WEIGHT_SYNC=1
```

---

## Generating Checksums for Manifest

The sync script auto-generates checksums when creating the manifest. To manually compute:

```bash
# For a single file
shasum -a 256 weights/custom_cnn_best.pt

# For all files
for f in weights/*.pt; do
  echo "$f: $(shasum -a 256 $f | cut -d' ' -f1)"
done
```

---

## Security Considerations

### Public Weights

If your model weights are not sensitive:

- ✅ Use GitHub Releases (free, easy)
- ✅ Use public S3/GCS buckets

### Private Weights

If your model weights are proprietary:

- 🔐 Use private S3/GCS buckets with signed URLs
- 🔐 Use authentication headers in download requests
- 🔐 Rotate access keys regularly

#### Example: S3 with Signed URLs

```python
# In sync_weights.py, modify download_file_with_progress:
import boto3

s3 = boto3.client('s3')
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'falconeye-weights', 'Key': 'v1.0.0/custom_cnn_best.pt'},
    ExpiresIn=3600  # 1 hour
)
```

---

## Troubleshooting

### Issue: Checksum mismatch after download

**Cause**: File corrupted during transfer

**Solution**:

```bash
# Force re-download
python3 scripts/sync_weights.py --force
```

### Issue: Download fails with 403/404

**Cause**: URL incorrect or file not public

**Solution**:

1. Verify URL is accessible in browser
2. Check bucket/release permissions
3. Test URL with `curl`:

   ```bash
   curl -I "https://your-url-here"
   ```

### Issue: Slow downloads on Streamlit Cloud

**Cause**: Large files, slow network

**Solution**:

1. Use CDN-backed storage (S3 with CloudFront)
2. Compress weights files (not recommended, affects model loading)
3. Consider smaller model architectures

---

## Cost Estimates

### GitHub Releases

- **Free** for public repositories
- Storage limit: 2 GB per file, unlimited files
- Bandwidth: Unlimited

### Amazon S3

- Storage: ~$0.023/GB/month
- Bandwidth: $0.09/GB (first 10 TB)
- **Example**: 500 MB weights = ~$0.01/month storage + ~$0.05/download

### Google Cloud Storage

- Storage: ~$0.020/GB/month
- Bandwidth: $0.12/GB
- **Example**: 500 MB weights = ~$0.01/month storage + ~$0.06/download

---

## Best Practices

1. ✅ **Version your weights**: Use v1.0.0, v1.1.0 in paths
2. ✅ **Always include checksums**: Prevents corrupted downloads
3. ✅ **Test locally first**: Run `sync_weights.py` before deploying
4. ✅ **Monitor bandwidth**: Track download costs if using paid storage
5. ✅ **Keep manifest in repo**: Commit `weights/manifest.json` to Git
6. ✅ **Document changes**: Update manifest version when weights change

---

## Example Workflow

```bash
# 1. Train new models
python scripts/train_custom_cnn.py

# 2. Generate manifest with checksums
python scripts/sync_weights.py --verify-only

# 3. Upload to GitHub Release
# (Manual: create release, upload files)

# 4. Update URLs in weights/manifest.json
# Replace YOUR_USERNAME with actual username

# 5. Commit and push
git add weights/manifest.json
git commit -m "Update weights manifest for v1.0.0"
git push

# 6. Deploy to Streamlit Cloud
# App will auto-download weights on first run
```

---

## Support

For issues with weight hosting:

- GitHub Releases: Check release permissions
- S3/GCS: Verify bucket ACLs and CORS settings
- Checksums: Re-generate manifest with `sync_weights.py --verify-only`
