#!/usr/bin/env python3
"""
Google Colab setup and run script for the jersey-number-pipeline.

Usage in Colab:
  1. Runtime -> Change runtime type -> GPU
  2. Clone the repo and cd into it (or upload this script and run from repo root):
       !git clone https://github.com/superbolt08/jersey-number-pipeline.git
       %cd jersey-number-pipeline
  3. Edit the CONFIG below to match your Google Drive paths.
  4. Run: !python run_colab.py

The script will: mount Drive, copy dataset and weights from Drive, clone SAM if needed,
install dependencies, and run the pipeline (SoccerNet test).
"""

import os
import subprocess
import sys

# ============== CONFIG: Edit these to match your Google Drive layout ==============
# After mounting, Drive is at /content/drive/MyDrive (or your shared drive path).
DRIVE_BASE = "/content/drive/MyDrive"
# Folder under Drive containing your files (no leading/trailing slash).
DRIVE_PROJECT_FOLDER = "jersey-number-pipeline"
# Name of the dataset zip file in that folder (e.g. jersey-2023.zip).
DATASET_ZIP = "jersey-2023.zip"
# Subfolder under DRIVE_PROJECT_FOLDER for model weights (reid .ckpt, pose vitpose-h.pth, models/*).
WEIGHTS_FOLDER = "weights"
# Set to False to skip mounting (e.g. if you already mounted in a previous cell).
MOUNT_DRIVE = True
# ==================================================================================


def run(cmd, check=True, shell=True, cwd=None):
    """Run a shell command; raise on failure if check=True."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=shell, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    return result.returncode


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    print(f"Working directory: {repo_root}")

    # 1. Mount Google Drive
    if MOUNT_DRIVE:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            print("Not in Colab (google.colab not found). Skipping Drive mount.")
        except Exception as e:
            print(f"Drive mount failed (you may need to authenticate in the browser): {e}")
            sys.exit(1)

    drive_project = os.path.join(DRIVE_BASE, DRIVE_PROJECT_FOLDER)
    drive_zip = os.path.join(drive_project, DATASET_ZIP)
    drive_weights = os.path.join(drive_project, WEIGHTS_FOLDER)

    # 2. Clone SAM into sam2 if missing
    if not os.path.isdir(os.path.join(repo_root, "sam2")):
        print("Cloning SAM into sam2/ ...")
        run("git clone https://github.com/davda54/sam.git sam2", cwd=repo_root)
    else:
        print("sam2/ already present, skipping clone.")

    # 3. Dataset: copy zip from Drive and unzip
    os.makedirs(os.path.join(repo_root, "data", "SoccerNet"), exist_ok=True)
    if os.path.isfile(drive_zip):
        run(f'cp "{drive_zip}" "{repo_root}/"', cwd=repo_root)
        run(f'unzip -o -q "{repo_root}/{DATASET_ZIP}" -d "{repo_root}/data/SoccerNet"', cwd=repo_root)
        # If zip contained jersey-2023/ we're done; if it contained train/ and test/ at top level, move them
        data_sn = os.path.join(repo_root, "data", "SoccerNet")
        if os.path.isdir(os.path.join(data_sn, "train")) and not os.path.isdir(os.path.join(data_sn, "jersey-2023")):
            run(f'mkdir -p "{data_sn}/jersey-2023"', cwd=repo_root)
            run(f'mv "{data_sn}/train" "{data_sn}/jersey-2023/"', cwd=repo_root)
            run(f'mv "{data_sn}/test" "{data_sn}/jersey-2023/"', cwd=repo_root)
        print("Dataset copied and extracted.")
    else:
        print(f"Dataset zip not found at {drive_zip}. Please upload it to Drive or set CONFIG.")

    # 4. Weights: copy from Drive into repo
    if os.path.isdir(drive_weights):
        models_dst = os.path.join(repo_root, "models")
        reid_dst = os.path.join(repo_root, "reid", "centroids-reid", "models")
        pose_dst = os.path.join(repo_root, "pose", "ViTPose", "checkpoints")
        os.makedirs(models_dst, exist_ok=True)
        os.makedirs(reid_dst, exist_ok=True)
        os.makedirs(pose_dst, exist_ok=True)
        for src_sub, dst in [
            (os.path.join(drive_weights, "models"), models_dst),
            (os.path.join(drive_weights, "reid"), reid_dst),
            (os.path.join(drive_weights, "pose"), pose_dst),
        ]:
            if os.path.isdir(src_sub):
                run(f'cp "{src_sub}"/* "{dst}/" 2>/dev/null || true', cwd=repo_root)
        print("Weights copied from Drive.")
    else:
        print(f"Weights folder not found at {drive_weights}. Copy weights manually or set CONFIG.")

    # 5. Install dependencies (no version pin so Colab gets a valid torch)
    print("Installing dependencies...")
    run(
        "pip install -q torch torchvision opencv-python Pillow numpy pandas scipy tqdm pytorch-lightning yacs",
        cwd=repo_root,
    )

    # 6. Run the pipeline
    print("Running pipeline: python main.py SoccerNet test")
    run("python main.py SoccerNet test", cwd=repo_root)
    print("Done. Outputs are in out/SoccerNetResults/")


if __name__ == "__main__":
    main()
