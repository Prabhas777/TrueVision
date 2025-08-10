# Data Instructions

This project uses a FER-style dataset (e.g., FER-2013). We do **not** redistribute the data here.
Instead, use the script below to download and prepare it locally.

## Option A: Kaggle CLI (FER2013)

1) Install Kaggle CLI and set your API token:
- Create token at https://www.kaggle.com/settings/account -> "Create New API Token"
- Save `kaggle.json` to:
  - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
  - macOS/Linux: `~/.kaggle/kaggle.json`
- Ensure permissions: `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux)

2) Run the download script (from repo root):
```bash
python scripts/download_data.py --source kaggle --dataset msambare/fer2013 --out data/fer2013
