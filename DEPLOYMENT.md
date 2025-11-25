# Streamlit Cloud Deployment Guide

## Prerequisites

1. GitHub repository with FalconEye-Detect code
2. Model weights (`.pt` files) available for download
3. Streamlit Cloud account

## Deployment Steps

### 1. Prepare Model Weights

**Important**: Streamlit Cloud does NOT reliably pull Git LFS objects. You must host weights externally.

#### Option A: GitHub Releases (Recommended)

1. Create a new release on GitHub:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. Go to GitHub Releases and upload all `.pt` files from `weights/` folder

3. Update `WEIGHTS_URLS.json` with download URLs:

   ```json
   {
     "weights": {
       "custom_cnn_best.pt": {
         "url": "https://github.com/YOUR_USERNAME/FalconEye-Detect/releases/download/v1.0.0/custom_cnn_best.pt",
         "size_mb": 16.5,
         "required": true
       },
       ...
     }
   }
   ```

#### Option B: Cloud Storage (S3, Google Cloud Storage)

1. Upload weights to your cloud storage
2. Generate public download URLs
3. Update `WEIGHTS_URLS.json` with URLs

### 2. Update Requirements

Ensure `requirements.txt` has:

```txt
numpy==2.1.3  # MUST be first
opencv-python-headless==4.10.0.84  # Headless for Linux
torch>=2.0.0
streamlit>=1.25.0
```

### 3. Configure Streamlit Cloud

1. Go to <https://share.streamlit.io>
2. Connect your GitHub repository
3. Set Main file path: `streamlit_app.py`
4. Python version: `3.12` (or `3.11` if issues)

### 4. Deploy

1. Click "Deploy"
2. Watch logs for:
   - ✅ `numpy` installed first
   - ✅ `opencv-python-headless` installed without errors
   - ✅ Weight provisioning runs successfully
   - ✅ Models loaded (check for "Loaded X/9 classification model(s)")

### 5. Verify

Once deployed:

1. Check sidebar shows correct model count
2. Expand "🔍 Loaded Models (Debug)" to see all 9 models
3. Upload a test image and verify inference works

## Common Issues

### Issue: `cv2` import error / numpy ABI mismatch

**Solution**:

- Ensure `numpy==2.1.3` is listed FIRST in requirements.txt
- Use `opencv-python-headless` (not `opencv-python`)
- Clear Streamlit Cloud cache and redeploy

### Issue: Only 2 models showing instead of 9

**Solution**:

- Check "Loaded Models (Debug)" expander in sidebar
- Verify `results/loading_warnings.log` for errors
- Check all `.pt` files are present in `weights/` folder
- Ensure weight provisioning completed successfully

### Issue: Git LFS files not available

**Solution**:

- Do NOT rely on Git LFS for Streamlit Cloud
- Host weights on GitHub Releases or S3
- Update `WEIGHTS_URLS.json` with download URLs
- Weight fetch runs automatically on app startup

### Issue: App crashes on startup

**Solution**:

- Check Streamlit Cloud logs
- Verify Python version (3.11 or 3.12)
- Ensure all dependencies install correctly
- Check for NameError or import errors

## Testing Locally Before Deploy

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify weights
python3 scripts/fetch_weights.py

# 4. Run sanity check
python3 scripts/model_evaluation.py --sanity-check

# 5. Run inference test
python3 scripts/tests/test_inference.py

# 6. Start app
streamlit run streamlit_app.py
```

Expected output:

- Sanity check shows 9 models
- Inference test creates `results/inference_sanity.json`
- App shows 9 classification models in sidebar

## Environment Variables (Optional)

You can set these in Streamlit Cloud settings:

- `WEIGHTS_MANIFEST_URL`: URL to download `WEIGHTS_URLS.json` if not in repo
- `SKIP_WEIGHT_FETCH`: Set to `1` to skip weight provisioning (if weights already present)

## Support

If deployment fails:

1. Check Streamlit Cloud logs
2. Verify `results/loading_warnings.log`
3. Test locally first
4. Check GitHub Issues for similar problems
