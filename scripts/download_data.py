
import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

def run_cmd(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_kaggle(dataset: str, out_dir: Path):
    ensure_dir(out_dir)
    zip_path = out_dir / "dataset.zip"
    # kaggle datasets download -d msambare/fer2013 -p data/fer2013
    run_cmd(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir)])
    # Find the downloaded zip (kaggle may append version)
    zips = list(out_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No zip downloaded. Check Kaggle auth/config.")
    z = zips[0]
    print(f"Extracting: {z}")
    with zipfile.ZipFile(z, 'r') as f:
        f.extractall(out_dir)
    # Optional: remove the zip
    try:
        z.unlink()
    except Exception:
        pass

def maybe_build_fer_layout(root: Path):
    """
    If the dataset comes as CSV or a different structure, you’d implement parsing here.
    For now, we assume the user obtains a FER-style directory or a zip that already includes folders.
    """
    train = root / "train"
    if train.exists():
        print("Detected train/ folder; assuming FER-style layout. Nothing to restructure.")
        return
    # If you want to implement CSV-to-folders conversion, do it here.
    print("NOTE: No train/ folder found. Please arrange data into FER-style folders.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle"], required=True, help="Data source")
    ap.add_argument("--dataset", help="Kaggle dataset id, e.g. msambare/fer2013")
    ap.add_argument("--out", default="data/fer2013", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if args.source == "kaggle":
        if not args.dataset:
            raise ValueError("--dataset is required for Kaggle source")
        download_kaggle(args.dataset, out_dir)
        maybe_build_fer_layout(out_dir)

    print("Done. Data at:", out_dir.resolve())

if __name__ == "__main__":
    main()
