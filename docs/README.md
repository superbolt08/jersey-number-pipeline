# Jersey Number Pipeline — Setup Guide (CPU-only)

This guide walks through setting up the jersey-number-pipeline for **CPU-only** use (e.g. on a laptop without an NVIDIA GPU). The **same code** uses the GPU when one is available (e.g. on Google Colab or a machine with CUDA), so you do not need to change anything when moving to a GPU environment.

## Table of contents

- [1. Create a conda environment](#1-create-a-conda-environment)
- [2. Install base dependencies](#2-install-base-dependencies)
- [3. Clone the repository](#3-clone-the-repository)
- [4. Place the dataset](#4-place-the-dataset)
- [5. Run the setup script](#5-run-the-setup-script)
- [6. Download required model weights](#6-download-required-model-weights)
- [7. Install dependencies in auxiliary environments (CPU-only)](#7-install-dependencies-in-auxiliary-environments-cpu-only)
- [8. Run inference](#8-run-inference)
- [Running on Google Colab (GPU)](#running-on-google-colab-gpu)
- [Final directory structure](#final-directory-structure)
- [Appendix](#appendix)
  - [Installing conda](#installing-conda)
  - [Installing dataset](#installing-dataset)
- [Why these files were changed](#why-these-files-were-changed)

---

## 1. Create a conda environment

```bash
conda create -n jersey python=3.8 -y
conda activate jersey
```

> Use `conda activate jersey` every time you open this project.

If you don't have conda installed, see [Installing conda](#installing-conda) below.

---

## 2. Install base dependencies

**PyTorch 1.9.0 (CPU):**

```bash
pip install torch==1.9.0 torchvision==0.10.0
```

**OpenCV:**

```bash
pip install opencv-python
```

---

## 3. Clone the repository

From your COSC419 project folder:

```bash
git clone https://github.com/mkoshkina/jersey-number-pipeline.git
cd jersey-number-pipeline
```

---

## 4. Place the dataset

Ensure the dataset is under:

```
jersey-number-pipeline/
└── data/
    └── SoccerNet/
        ├── train/
        └── test/
```

If you don't have the dataset yet, see [Installing dataset](#installing-dataset) below.

---

## 5. Run the setup script

1. Replace the project's `setup.py` with the `setup.py` from this folder.
2. Run:

```bash
python setup.py SoccerNet
```

5.1 Replace the listed files in the project with the files in this folder
configuration.py
helpers.py
legibility_classifer.py
requirements.txt
setup.py
---

## 6. Download required model weights

You must **manually download** these weights and place them as follows:

| What to download | Put in |
|------------------|--------|
| Centroid-ReID weights | `reid/centroids-reid/models/` |
| ViTPose weights | `pose/ViTPose/checkpoints/` |
| PARSeq (SoccerNet fine-tuned) | `models/` |
| Legibility classifier (SoccerNet) | `models/` |

Links and details are in the main pipeline [README](https://github.com/mkoshkina/jersey-number-pipeline). **This step is mandatory.**

---

## 7. Install dependencies in auxiliary environments (CPU-only)

The pipeline runs some steps in separate conda envs (`centroids`, `vitpose`, `parseq2`). On a **CPU-only machine**, do **not** use the sub-repos’ `requirements.txt` files—they often pin CUDA builds. Install CPU-only packages instead.

If you see **"Generate features"** then an error like `ModuleNotFoundError: No module named 'numpy'`, install deps in the **centroids** env (CPU only):

```bash
conda activate centroids
pip install numpy torch torchvision tqdm opencv-python Pillow
```

Use the same idea for other envs if you hit similar import errors: activate that env and install the missing packages with CPU PyTorch (no `+cu*` versions).

---

## 8. Run inference

When everything is in place:

```bash
pip install -r requirements.txt
python main.py SoccerNet test
```

This runs the full pipeline:

- Re-ID filtering  
- Legibility classifier  
- Pose-guided crop  
- STR (scene text recognition)  
- Tracklet consolidation  

---

## Running on Google Colab (GPU)

The pipeline automatically uses the GPU when available. To run on Colab with a GPU:

1. **Runtime → Change runtime type → GPU** (e.g. T4).
2. Clone your repo and install dependencies in one environment (no separate conda envs):
   ```python
   !git clone https://github.com/superbolt08/jersey-number-pipeline.git
   %cd jersey-number-pipeline
   !pip install -q -r requirements.txt
   !pip install -q numpy torch torchvision tqdm opencv-python Pillow pandas scipy
   ```
3. Put data and model weights in the expected paths (e.g. upload to Drive and mount, or download in the notebook).
4. Run: `!python main.py SoccerNet test`

Re-ID, pose, and STR steps will use the GPU when it is available. For a full Colab workflow (develop locally, push to GitHub, pull and run on Colab), see your project’s Colab instructions.

---

## Final directory structure

```
jersey-number-pipeline/
├── data/
│   └── SoccerNet/
├── models/
├── pose/
├── reid/
├── sam/
├── str/
└── main.py
```

If you see this structure and the commands above work, you're done.

---

# Appendix

## Installing conda

### Option 1 — Miniconda (recommended)

**Step 1: Download**

- Go to: [Miniconda — conda documentation](https://docs.conda.io/en/latest/miniconda.html)
- Download: **Miniconda3 Windows 64-bit** (Python 3.x installer)

**Step 2: Install**

- Run the installer.
- Choose **"Just Me"**.
- Use the default install location.
- **Check "Add Miniconda to PATH"** (important).
- Finish installation.

**Step 3: Verify**

Open a new PowerShell window and run:

```bash
conda --version
```

You should see something like `conda 23.x.x`. If you get "command not found", restart your computer and try again.

**Step 4: Create the project environment**

```bash
conda create -n jersey python=3.8 -y
conda activate jersey
```

---

## Installing dataset

**Step 1: Install SoccerNet package**

```bash
pip install SoccerNet
```

Verify:

```bash
python -c "import SoccerNet; print('OK')"
```

**Step 2: Download jersey dataset**

From inside `jersey-number-pipeline`:

```bash
python -c "from SoccerNet.Downloader import SoccerNetDownloader; d=SoccerNetDownloader(LocalDirectory='./data/SoccerNet'); d.downloadDataTask(task='jersey-2023', split=['train','test'])"
```

This downloads into `jersey-number-pipeline/data/SoccerNet/`.

**Step 3: Resulting layout**

```
jersey-number-pipeline/
└── data/
    └── SoccerNet/
        ├── train/
        └── test/
```

---

## Why these files were changed

The files in this `setup-documentation` folder differ from the original pipeline so that setup and inference work on **Windows** and **CPU-only** machines, and with the **SoccerNet jersey-2023** dataset layout. Summary:

| File | Why it was changed |
|------|--------------------|
| **setup.py** | Git clone destination paths are quoted so Windows does not split them into multiple arguments. Directory creation uses `os.makedirs(..., exist_ok=True)` so `pose/ViTPose/checkpoints` and similar paths exist before downloads, and so `./pose`, `./reid`, `./str` exist before listing. |
| **configuration.py** | `root_dir` for SoccerNet is set to `./data/SoccerNet/jersey-2023` because the SoccerNet downloader puts the task data in that subfolder; the original path expected `./data/SoccerNet/test/...` directly. |
| **helpers.py** | In `identify_soccer_balls`, entries under the images folder are skipped unless they are directories. This avoids treating macOS `.DS_Store` (or other files) as tracklet folders, which would cause "The directory name is invalid" on Windows. |
| **legibility_classifier.py** | The SAM repo is cloned as `sam2/` with `sam.py` inside, but the code expected `from sam.sam import SAM`. The script now adds `sam2` to `sys.path` and uses `from sam import SAM` so the import resolves without renaming the clone. |
| **main.py** | All `conda run -n <env> python3` commands were changed to use `python` instead of `python3`. On Windows this ensures the correct env's interpreter (and its packages) is used; otherwise the re-id, pose, and STR steps could run with the wrong environment and miss modules like `numpy`. |
| **centroid_reid.py** | Before loading the Re-ID checkpoint, the file is checked (size and first bytes). If it looks like HTML or is too small, a clear error is raised so you re-download the real weights instead of getting a pickle error. Tracklet iteration skips non-directory entries (e.g. `.DS_Store`) so feature extraction does not crash. |
| **reid/centroids-reid/losses/center_loss.py** | `use_gpu` is effectively set to `use_gpu and torch.cuda.is_available()`. The original code always called `.cuda()` when `use_gpu=True`, which crashes on CPU-only machines; now CUDA is only used when available. |
| **requirements.txt** | Added for the main pipeline environment so you can install the jersey env's dependencies in one step (`pip install -r requirements.txt`) with CPU-friendly versions of PyTorch and other packages. |
