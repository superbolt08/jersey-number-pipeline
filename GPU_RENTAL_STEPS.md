# Vast.ai GPU Rental: SSH + Setup + Run (Jersey Pipeline)

## 0) SSH into the GPU VM

Use the public port shown by Vast.ai as `-p`.

### Try this (adjust user if Vast suggests `ubuntu`)
```powershell
ssh -i "C:\Users\SyedS\Documents\UBCO Courses\COSC419\project\ssh keys\yes" -p <PUBLIC_SSH_PORT> root@<PUBLIC_IP> -L 8080:localhost:8080
```

If `root` doesn’t work:
```powershell
ssh -i "C:\Users\SyedS\Documents\UBCO Courses\COSC419\project\ssh keys\yes" -p <PUBLIC_SSH_PORT> ubuntu@<PUBLIC_IP> -L 8080:localhost:8080
```

If you see `Permission denied (publickey)`, it means your key wasn’t accepted by the server (wrong key path, wrong username, or different key than the one instance expects).

## 1) Get the repo on the VM

```bash
mkdir -p ~/projects
cd ~/projects

# Option A: git clone
git clone https://github.com/superbolt08/jersey-number-pipeline.git
cd jersey-number-pipeline
```

## 2) Create a Python virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel
pip install -r requirements.txt

# Only needed if you run STR (PARSeq) code paths:
pip install -r str/parseq/requirements/inference.txt
pip install -e str/parseq
```

## 3) Ensure required sub-repos exist

From repo root (`~/projects/jersey-number-pipeline`):

```bash
[ -d sam2 ] || git clone --recurse-submodules https://github.com/davda54/sam.git sam2
[ -d reid/centroids-reid ] || git clone --recurse-submodules https://github.com/mikwieczorek/centroids-reid.git reid/centroids-reid
[ -d pose/ViTPose ] || git clone --recurse-submodules https://github.com/ViTAE-Transformer/ViTPose.git pose/ViTPose
[ -d str/parseq ] || git clone --recurse-submodules https://github.com/baudm/parseq.git str/parseq
```

## 4) Install ViTPose’s `mmcv-full` (required for `import mmcv`)

```bash
python scripts/install_mmcv_full_vitpose.py

# Verify:
python -c "import mmcv; print('mmcv ok:', mmcv.__version__)"
```

If `import mmcv` fails after installation, rerun the install script and check the output for whether a matching wheel was found/installed.

## 5) Put the dataset on local disk (recommended)

### Can we mount Google Drive instead?
You cannot “mount the GPU” to Google Drive. What you *can* mount is the **filesystem** (e.g., via `rclone mount` or Google Drive FUSE). In practice for ML image pipelines, Drive mounts are often much slower and can be flaky (latency + many small file reads), so the reliable approach is:
- **Sync/copy from Drive → VM local SSD**
- **Run the pipeline from local disk**

### Expected layout (as used by this repo)

For SoccerNet jersey-2023:

```text
data/SoccerNet/jersey-2023/test/images/<tracklet_id>/...
```

Your `configuration.py` sets:
- `root_dir = ./data/SoccerNet/jersey-2023`
- `images = test/images`

### If you have split archives `train.zip` and `test.zip`

Assuming you have them locally on the VM:

```bash
mkdir -p data/SoccerNet/jersey-2023
unzip -o test.zip  -d data/SoccerNet/jersey-2023
unzip -o train.zip -d data/SoccerNet/jersey-2023
```

If the zip expands with an extra nesting level (e.g. `test/test/images`), fix/flatten so the final location is:
`data/SoccerNet/jersey-2023/test/images/`.

### Verify quickly

```bash
ls data/SoccerNet/jersey-2023/test/images | head
```

### If you only have `train.zip` / `test.zip` on Google Drive
1. Copy the zips from Drive to the VM local disk (example paths; update the `gdrive:` remote and folder):
```bash
mkdir -p ~/data/SoccerNet/jersey-2023
# Example: if zips are in Drive at:
#   MyDrive/<project>/<repo>/data/SoccerNet/jersey-2023/
# adjust the right-hand local path as you like.
rclone sync "gdrive:jersey-number-pipeline/jersey-number-pipeline/data/SoccerNet/jersey-2023" \
  ~/data/SoccerNet/jersey-2023 --progress
```
2. Unzip into the repo’s expected dataset location:
```bash
cd ~/projects/jersey-number-pipeline
mkdir -p data/SoccerNet/jersey-2023
unzip -o ~/data/SoccerNet/jersey-2023/test.zip  -d data/SoccerNet/jersey-2023
unzip -o ~/data/SoccerNet/jersey-2023/train.zip -d data/SoccerNet/jersey-2023
```

3. If the zip created an extra nesting level (e.g. `test/test/images`), flatten so the final path is:
`data/SoccerNet/jersey-2023/test/images/`.

4. Verify:
```bash
ls data/SoccerNet/jersey-2023/test/images | head
```

## 6) Put weights in the expected locations

You need these folders at repo root (paths vary by your setup, but this is the expected structure):

```text
models/
reid/centroids-reid/models/
pose/ViTPose/checkpoints/
```

## 7) Run the pipeline

From repo root:

```bash
python main.py SoccerNet test --resume
```

Use `--force` if you want to recompute everything for the step(s):

```bash
python main.py SoccerNet test --force
```

## 8) (Optional) Sync results back to Drive/S3

Once the run is done, copy `out/` back to your persistent storage.

Example with `rclone` (update remote/path):
```bash
rclone sync ./out your_remote:jersey-number-pipeline/out --progress
```

---

## Notes / common pitfalls

1. **If your dataset extraction was partial**, the pipeline can still run but later steps may fail or process the wrong set of tracklets.
2. **`mmcv` is the ViTPose blocker** on modern Python/torch combos. Always verify `python -c "import mmcv"` after installation.
3. **Drive persistence** helps keep *data + code + outputs*, but you still need to install Python packages again on each new VM runtime.

