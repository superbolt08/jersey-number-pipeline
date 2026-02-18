#!/usr/bin/env python3
"""
Google Colab setup and run script for the jersey-number-pipeline.

Usage in Colab:
  1. Runtime -> Change runtime type -> GPU
  2. Either:
     - PERSIST_TO_DRIVE = True: run this script from any directory (e.g. after uploading it).
       It will mount Drive, clone the main repo and all sub-repos to Drive, then run.
     - PERSIST_TO_DRIVE = False: clone the repo and cd into it, then run:
       !git clone https://github.com/superbolt08/jersey-number-pipeline.git
       %cd jersey-number-pipeline
       !python run_colab.py
  3. Edit the CONFIG below to match your Google Drive paths.
  4. Run: !python run_colab.py

The script clones the same repos as setup.py: main repo, sam2, reid/centroids-reid,
pose/ViTPose, str/parseq; copies dataset and weights from Drive; installs deps; runs the pipeline.
"""

import os
import subprocess
import sys

# ============== CONFIG: Edit these to match your Google Drive layout ==============
DRIVE_BASE = "/content/drive/MyDrive"
DRIVE_PROJECT_FOLDER = "jersey-number-pipeline"
DATASET_ZIP = "jersey-2023.zip"
WEIGHTS_FOLDER = "weights"
MOUNT_DRIVE = True
# True = clone main repo + sub-repos to Drive (persistent). False = run from current dir (must be repo root).
PERSIST_TO_DRIVE = True
# ==================================================================================


def run(cmd, check=True, shell=True, cwd=None):
    """Run a shell command; raise on failure if check=True."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=shell, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    return result.returncode


def main():
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

    # 2. Repo root: either on Drive (persistent) or current script directory
    if PERSIST_TO_DRIVE:
        os.makedirs(drive_project, exist_ok=True)
        repo_root = os.path.join(drive_project, "jersey-number-pipeline")
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            print("Cloning main repo to Drive ...")
            run(f'cd "{drive_project}" && git clone https://github.com/superbolt08/jersey-number-pipeline.git')
        else:
            print("Main repo already on Drive, skipping clone.")
    else:
        repo_root = os.path.dirname(os.path.abspath(__file__))

    os.chdir(repo_root)
    print(f"Working directory: {repo_root}")

    # 3. Clone sub-repos (same as setup.py): sam2, reid/centroids-reid, pose/ViTPose, str/parseq
    if not os.path.isdir(os.path.join(repo_root, "sam2")):
        print("Cloning SAM into sam2/ ...")
        run("git clone --recurse-submodules https://github.com/davda54/sam.git sam2", cwd=repo_root)

    os.makedirs(os.path.join(repo_root, "reid"), exist_ok=True)
    if not os.path.isdir(os.path.join(repo_root, "reid", "centroids-reid")):
        print("Cloning Centroid-ReID into reid/centroids-reid/ ...")
        run(
            "git clone --recurse-submodules https://github.com/mikwieczorek/centroids-reid.git reid/centroids-reid",
            cwd=repo_root,
        )
        os.makedirs(os.path.join(repo_root, "reid", "centroids-reid", "models"), exist_ok=True)

    os.makedirs(os.path.join(repo_root, "pose"), exist_ok=True)
    if not os.path.isdir(os.path.join(repo_root, "pose", "ViTPose")):
        print("Cloning ViTPose into pose/ViTPose/ ...")
        run(
            "git clone --recurse-submodules https://github.com/ViTAE-Transformer/ViTPose.git pose/ViTPose",
            cwd=repo_root,
        )

    os.makedirs(os.path.join(repo_root, "str"), exist_ok=True)
    if not os.path.isdir(os.path.join(repo_root, "str", "parseq")):
        print("Cloning PARSeq into str/parseq/ ...")
        run(
            "git clone --recurse-submodules https://github.com/baudm/parseq.git str/parseq",
            cwd=repo_root,
        )

    # 4. Dataset: copy zip from Drive and unzip
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

    # 5. Weights: copy from Drive into repo
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

    # 6. Install dependencies (no version pin so Colab gets a valid torch)
    print("Installing dependencies...")
    run(
        "pip install -q torch torchvision opencv-python Pillow numpy pandas scipy tqdm pytorch-lightning yacs",
        cwd=repo_root,
    )

    # 7. Run the pipeline
    print("Running pipeline: python main.py SoccerNet test")
    run("python main.py SoccerNet test", cwd=repo_root)
    print("Done. Outputs are in out/SoccerNetResults/")

    # 8. Copy outputs to Drive so they persist
    out_dir = os.path.join(repo_root, "out")
    if os.path.isdir(out_dir) and os.path.isdir(drive_project):
        run(f'cp -r "{out_dir}" "{drive_project}/"', cwd=repo_root)
        print("Outputs copied to Drive.")


if __name__ == "__main__":
    main()
