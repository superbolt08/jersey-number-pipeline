# Jersey Number Pipeline — Setup Guide

**Primary way to run:** on **Google Colab** using its GPU. This guide is written for Colab first: all main steps are commands you run inside Colab cells. Running locally with conda is described at the end as an alternative.

This repository ([superbolt08/jersey-number-pipeline](https://github.com/superbolt08/jersey-number-pipeline)) contains the pipeline code and PARSeq (`str/parseq`). You must also clone the SAM repo into `sam2/`, and provide the dataset and model weights (e.g. from Google Drive).

---

## Table of contents

- [Prerequisites (Google Drive)](#prerequisites-google-drive)
- [1. Open Colab and enable GPU](#1-open-colab-and-enable-gpu)
- [2. Clone this repo and the SAM sub-repo](#2-clone-this-repo-and-the-sam-sub-repo)
- [3. Mount Google Drive and bring in the dataset](#3-mount-google-drive-and-bring-in-the-dataset)
- [4. Model weights](#4-model-weights)
- [5. Install dependencies](#5-install-dependencies)
- [6. Run the pipeline](#6-run-the-pipeline)
- [Expected directory layout in Colab](#expected-directory-layout-in-colab)
- [Getting results out of Colab](#getting-results-out-of-colab)
- [Running locally (alternative)](#running-locally-alternative)
- [Appendix: Dataset and weights](#appendix-dataset-and-weights)

---

## Prerequisites (Google Drive)

Before running in Colab, prepare the following and put them on **Google Drive** (paths below are examples; adjust in the Colab cells to match your Drive layout):

1. **SoccerNet jersey-2023 dataset**  
   - Zip the folder that contains `train/` and `test/` (each with `images/` and labels as per SoccerNet).  
   - Upload the zip to Drive, e.g. `My Drive/jersey-number-pipeline/jersey-2023.zip`.

2. **Model weights** (see [Appendix: Dataset and weights](#appendix-dataset-and-weights) for links):  
   - Centroid-ReID weights → e.g. put in a folder `My Drive/jersey-number-pipeline/weights/reid/` (the `.ckpt` files).  
   - ViTPose weights → e.g. `My Drive/jersey-number-pipeline/weights/pose/vitpose-h.pth`.  
   - PARSeq (SoccerNet) and legibility (SoccerNet) → e.g. `My Drive/jersey-number-pipeline/weights/models/` (the `.ckpt` and `.pth` files for the pipeline’s `models/` directory).

You will mount Drive in Colab and copy these into the cloned repo (see below).

---

## 1. Open Colab and enable GPU

1. Go to [Google Colab](https://colab.research.google.com).
2. **File → New notebook**.
3. **Runtime → Change runtime type**:
   - **Hardware accelerator:** **GPU** (e.g. T4).
4. Click **Save**.

---

## 2. Clone this repo and the SAM sub-repo

Run these in Colab cells (one cell per block is fine):

```python
# Clone the pipeline repo
!git clone https://github.com/superbolt08/jersey-number-pipeline.git
%cd jersey-number-pipeline
```

```python
# Clone SAM (required for legibility step). Must be in a folder named sam2.
!git clone https://github.com/davda54/sam.git sam2
```

---

## 3. Mount Google Drive and bring in the dataset

Mount Drive so the notebook can read your files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create the data directory and copy the dataset from Drive. If your zip is at `My Drive/jersey-number-pipeline/jersey-2023.zip`:

```python
%cd /content/jersey-number-pipeline
!mkdir -p data/SoccerNet
# Copy the zip from Drive (change the path to match your Drive folder)
!cp "/content/drive/MyDrive/jersey-number-pipeline/jersey-2023.zip" ./
!unzip -q jersey-2023.zip -d data/SoccerNet
```

If the zip expands to a folder named `jersey-2023`, you should now have `data/SoccerNet/jersey-2023/` with `train/` and `test/` inside. If your zip expands to `train/` and `test/` at the top level, move them into `jersey-2023`:

```python
# Only if unzip created data/SoccerNet/train and data/SoccerNet/test directly:
# !mkdir -p data/SoccerNet/jersey-2023
# !mv data/SoccerNet/train data/SoccerNet/jersey-2023/
# !mv data/SoccerNet/test data/SoccerNet/jersey-2023/
```

Verify the layout (pipeline expects `data/SoccerNet/jersey-2023/test/images`, etc.):

```python
!ls data/SoccerNet/jersey-2023/
!ls data/SoccerNet/jersey-2023/test/ | head -5
```

---

## 4. Model weights

Copy the required weights from Drive into the paths the pipeline expects. Adjust the Drive paths to where you uploaded them.

```python
%cd /content/jersey-number-pipeline

# Pipeline models/ (PARSeq SoccerNet + legibility SoccerNet)
!mkdir -p models
!cp "/content/drive/MyDrive/jersey-number-pipeline/weights/models/"* models/ 2>/dev/null || true
# Or copy files one by one if you prefer, e.g.:
# !cp "/content/drive/MyDrive/jersey-number-pipeline/weights/models/parseq_epoch=24-step=....ckpt" models/
# !cp "/content/drive/MyDrive/jersey-number-pipeline/weights/models/legibility_resnet34_soccer_20240215.pth" models/

# Re-ID weights (reid/centroids-reid/models/)
!mkdir -p reid/centroids-reid/models
!cp "/content/drive/MyDrive/jersey-number-pipeline/weights/reid/"* reid/centroids-reid/models/ 2>/dev/null || true

# ViTPose (pose/ViTPose/checkpoints/)
!mkdir -p pose/ViTPose/checkpoints
!cp "/content/drive/MyDrive/jersey-number-pipeline/weights/pose/vitpose-h.pth" pose/ViTPose/checkpoints/ 2>/dev/null || true
```

If you prefer to download weights in Colab instead of Drive, use the links in the [Appendix](#appendix-dataset-and-weights) and `gdown` or similar, then place the files into `models/`, `reid/centroids-reid/models/`, and `pose/ViTPose/checkpoints/` as above.

---

## 5. Install dependencies

Colab has no conda. Install everything in the current environment (run from the repo root):

```python
%cd /content/jersey-number-pipeline
# Do not use -r requirements.txt here: it pins torch==1.9.0, which Colab may not have.
!pip install -q torch torchvision opencv-python Pillow numpy pandas scipy tqdm
```

For the Re-ID step you may also need `pytorch-lightning` and `yacs`:

```python
!pip install -q pytorch-lightning yacs
```

(If you hit missing modules for pose or STR, install them in the same way in a new cell.)

---

## 6. Run the pipeline

From the repo root:

```python
%cd /content/jersey-number-pipeline
!python main.py SoccerNet test
```

The pipeline will use the GPU automatically. It runs: soccer-ball filter → Re-ID features → outlier filter → legibility → pose → crops → STR → combine → evaluation.

---

## Expected directory layout in Colab

After the steps above, under `/content/jersey-number-pipeline/` you should have:

```
jersey-number-pipeline/
├── data/
│   └── SoccerNet/
│       └── jersey-2023/
│           ├── train/
│           └── test/
├── models/
│   ├── parseq_epoch=24-step=...ckpt
│   └── legibility_resnet34_soccer_20240215.pth
├── pose/
│   └── ViTPose/
│       └── checkpoints/
│           └── vitpose-h.pth
├── reid/
│   └── centroids-reid/
│       └── models/
│           └── (*.ckpt)
├── sam2/
├── str/
│   └── parseq/
├── main.py
└── ...
```

---

## Getting results out of Colab

Outputs are written under `out/SoccerNetResults/` (e.g. `test/final_results.json`). To download to your machine:

- Use **Files** in the left sidebar to browse to `jersey-number-pipeline/out/`, then right‑click and download, or  
- Zip and download from a cell:

```python
!cd /content/jersey-number-pipeline && zip -r out.zip out/
from google.colab import files
files.download('out.zip')
```

You can also copy `out/` to Drive so it persists after the session:

```python
!cp -r /content/jersey-number-pipeline/out "/content/drive/MyDrive/jersey-number-pipeline/"
```

---

## Running locally (alternative)

If you want to run on your own machine with conda instead of Colab:

1. Clone this repo: `git clone https://github.com/superbolt08/jersey-number-pipeline.git && cd jersey-number-pipeline`
2. Clone SAM into `sam2`: `git clone https://github.com/davda54/sam.git sam2`
3. Create a conda env (e.g. `conda create -n jersey python=3.8 -y`) and install: `pip install -r requirements.txt` (or install PyTorch/OpenCV etc. manually).
4. Run the setup script to clone Centroid-ReID and ViTPose and create envs `centroids`, `vitpose`, `parseq2`: `python setup.py SoccerNet`
5. Download and place the dataset under `data/SoccerNet/jersey-2023/` and all model weights as in the table in the Appendix.
6. On CPU-only machines, install dependencies in the `centroids` (and other) envs without CUDA pins: e.g. `conda activate centroids && pip install numpy torch torchvision tqdm opencv-python Pillow`.
7. Run: `conda activate jersey && python main.py SoccerNet test`

The pipeline will use the GPU if available, or CPU otherwise.

---

# Appendix: Dataset and weights

## Dataset

- **SoccerNet jersey-2023:** [SoccerNet jersey](https://github.com/SoccerNet/sn-jersey). You can download via the SoccerNet package (e.g. on your PC or in Colab):
  ```python
  pip install SoccerNet
  python -c "from SoccerNet.Downloader import SoccerNetDownloader; d=SoccerNetDownloader(LocalDirectory='./data/SoccerNet'); d.downloadDataTask(task='jersey-2023', split=['train','test'])"
  ```
  Then zip the resulting `data/SoccerNet/jersey-2023` (or the whole `data` folder) and upload to Drive for Colab.

## Model weights (mandatory)

Download and place as below. Links are in the [main README](https://github.com/superbolt08/jersey-number-pipeline) or the [original repo](https://github.com/mkoshkina/jersey-number-pipeline).

| What to download | Put in (relative to repo root) |
|-------------------|---------------------------------|
| Centroid-ReID weights | `reid/centroids-reid/models/` |
| ViTPose (`vitpose-h.pth`) | `pose/ViTPose/checkpoints/` |
| PARSeq SoccerNet fine-tuned | `models/` |
| Legibility classifier (SoccerNet) | `models/` |

For Colab, upload these to a folder on Drive (e.g. `weights/reid/`, `weights/pose/`, `weights/models/`) and copy them into the repo as in [Section 4](#4-model-weights).
