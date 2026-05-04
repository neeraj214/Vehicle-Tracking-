# 🔑 How to Get Your Free Roboflow API Key

## Steps (takes ~2 minutes)

### 1. Sign Up
Go to: https://app.roboflow.com
Click "Sign Up" — use your college email or Google account.
It is completely free.

### 2. Get API Key
After signing in:
- Click your profile icon (top right)
- Go to "Settings"
- Click "API Keys" in the left sidebar
- Copy the key under "Private API Key"

### 3. Paste into Script
Open scripts/download_dataset.py and replace:
  ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
with:
  ROBOFLOW_API_KEY = "your_copied_key_here"

### 4. Run Download
  cd vehicle-tracker
  python scripts/download_dataset.py

### 5. Verify
  python scripts/verify_dataset.py

---

## What Gets Downloaded
- ~10,000 highway vehicle images from UA-DETRAC
- COCO JSON annotations auto-converted to UA-DETRAC XML
- Organized into sequence folders under data/UA-DETRAC/

## After Download — Update backbone_config.py
  "ua_detrac_img_dir": "data/UA-DETRAC/images",
  "ua_detrac_ann_dir": "data/UA-DETRAC/annotations",

## Dataset Size
  ~500 MB download (much smaller than full 10GB official dataset)
  Perfect for training Re-ID head on RTX 2050 GPU
