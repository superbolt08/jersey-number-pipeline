#!/usr/bin/env bash
# Idempotent setup for a Linux GPU VM (Vast.ai, etc.): repo, venv, deps, sub-repos,
# mmcv (ViTPose), optional dataset download + extract, optional weights via rclone.
#
# Does NOT: pick a GPU, install NVIDIA drivers, or SSH — assume you're already on the machine.
#
# Usage (examples):
#   # Minimal: repo + env only (dataset/weights you handle yourself)
#   ./scripts/setup_vast_gpu_environment.sh
#
#   # Pull zips from rclone then unzip under data/SoccerNet/jersey-2023/
#   export RCLONE_REMOTE=gdrive
#   export RCLONE_ZIPS_PATH='jersey-number-pipeline/jersey-number-pipeline/data/SoccerNet/jersey-2023'
#   export DATASET_SOURCE=rclone_zips
#   ./scripts/setup_vast_gpu_environment.sh
#
#   # Sync already-extracted jersey-2023 tree from rclone
#   export DATASET_SOURCE=rclone_extracted
#   export RCLONE_EXTRACTED_PATH='jersey-number-pipeline/data/SoccerNet/jersey-2023'
#   ./scripts/setup_vast_gpu_environment.sh
#
#   # Zips already on disk
#   export DATASET_SOURCE=local_zips
#   export LOCAL_ZIPS_DIR="$HOME/data/jersey_zips"
#   ./scripts/setup_vast_gpu_environment.sh
#
# Env overrides:
#   JERSEY_PROJECTS_DIR   default ~/projects
#   REPO_URL              git URL
#   REPO_NAME             directory name
#   SKIP_MMLAB_MMCV       1 to skip mmcv installer (not recommended)
#   SKIP_PARSEQ           1 to skip PARSeq editable install
#   RCLONE_REMOTE         e.g. gdrive (required for rclone_* dataset modes)
#   RCLONE_ZIPS_PATH      remote path (no remote: prefix) for zips folder
#   RCLONE_EXTRACTED_PATH remote path for extracted jersey-2023 tree
#   RCLONE_WEIGHTS_PATH   optional remote folder to sync into repo (models + reid + pose)
#   DATASET_SOURCE        skip | rclone_zips | rclone_extracted | local_zips
#   LOCAL_ZIPS_DIR        directory containing train.zip / test.zip (local_zips mode)

set -euo pipefail

: "${JERSEY_PROJECTS_DIR:=$HOME/projects}"
: "${REPO_URL:=https://github.com/superbolt08/jersey-number-pipeline.git}"
: "${REPO_NAME:=jersey-number-pipeline}"
: "${DATASET_SOURCE:=skip}" # skip | rclone_zips | rclone_extracted | local_zips
: "${SKIP_MMLAB_MMCV:=0}"
: "${SKIP_PARSEQ:=0}"

REPO_DIR="${JERSEY_PROJECTS_DIR}/${REPO_NAME}"
DATASET_ROOT="${REPO_DIR}/data/SoccerNet/jersey-2023"
MARKER_DIR="${REPO_DIR}/.setup_markers"

log() { echo "[setup] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

dataset_ready() {
  [[ -d "${DATASET_ROOT}/test/images" ]]
}

ensure_dataset_layout() {
  DATASET_ROOT="${DATASET_ROOT}" python3 <<'PY'
import os, shutil

root = os.environ["DATASET_ROOT"]
if not os.path.isdir(root):
    raise SystemExit(f"Missing dataset root: {root}")
for split in ("train", "test", "challenge"):
    nested = os.path.join(root, split, split)
    if not os.path.isdir(nested):
        continue
    parent = os.path.join(root, split)
    for name in os.listdir(nested):
        if name.startswith("."):
            continue
        src = os.path.join(nested, name)
        dst = os.path.join(parent, name)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    try:
        os.rmdir(nested)
    except OSError:
        pass
if not os.path.isdir(os.path.join(root, "test", "images")):
    raise SystemExit(f"Expected {root}/test/images after normalize — check zip layout.")
print("Dataset layout OK:", root)
PY
}

mkdir -p "${JERSEY_PROJECTS_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Cloning ${REPO_URL} → ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
else
  log "Repo exists: ${REPO_DIR} (skipping clone)"
fi

cd "${REPO_DIR}"
mkdir -p "${MARKER_DIR}"

if [[ ! -d .venv ]]; then
  log "Creating .venv"
  python3 -m venv .venv
else
  log ".venv exists (skipping venv create)"
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip_install_req() {
  local stamp="${MARKER_DIR}/pip_requirements"
  if [[ -f requirements.txt ]] && { [[ ! -f "${stamp}" ]] || [[ requirements.txt -nt "${stamp}" ]]; }; then
    log "pip install -r requirements.txt"
    pip install -U pip wheel
    pip install -r requirements.txt
    touch "${stamp}"
  else
    log "requirements.txt up to date (skip)"
  fi
}

pip_install_parseq() {
  [[ "${SKIP_PARSEQ}" == "1" ]] && { log "SKIP_PARSEQ=1 (skip PARSeq)"; return 0; }
  local stamp="${MARKER_DIR}/pip_parseq"
  local need=0
  [[ ! -f "${stamp}" ]] && need=1
  [[ -f str/parseq/requirements/inference.txt ]] && [[ str/parseq/requirements/inference.txt -nt "${stamp}" ]] && need=1
  if [[ "${need}" -eq 1 ]]; then
    log "PARSeq inference + editable install"
    pip install -r str/parseq/requirements/inference.txt
    pip install -e str/parseq
    touch "${stamp}"
  else
    log "PARSeq deps up to date (skip)"
  fi
}

clone_subrepo() {
  local dir="$1"
  local url="$2"
  mkdir -p "$(dirname "${dir}")"
  if [[ ! -e "${dir}/.git" ]]; then
    log "git clone ${url} → ${dir}"
    git clone --recurse-submodules "${url}" "${dir}"
  else
    log "Subrepo exists: ${dir} (skip)"
  fi
}

patch_vitpose_mmcv_cap() {
  local patch_py="${REPO_DIR}/scripts/patch_vitpose_mmpose_mmcv_range.py"
  local vit="${REPO_DIR}/pose/ViTPose"
  if [[ -f "${patch_py}" ]] && [[ -d "${vit}/mmpose" ]]; then
    python3 "${patch_py}" "${vit}" || true
  fi
}

install_mmcv_if_needed() {
  [[ "${SKIP_MMLAB_MMCV}" == "1" ]] && { log "SKIP_MMLAB_MMCV=1 (skip mmcv install)"; return 0; }
  if python3 -c "import mmcv" 2>/dev/null; then
    log "mmcv already importable (skip installer)"
    python3 -c "import mmcv; print('mmcv:', mmcv.__version__)"
    return 0
  fi
  log "Running scripts/install_mmcv_full_vitpose.py (may take a while)"
  python3 scripts/install_mmcv_full_vitpose.py
  python3 -c "import mmcv; print('mmcv:', mmcv.__version__)"
}

sync_weights_optional() {
  if [[ -z "${RCLONE_WEIGHTS_PATH:-}" ]]; then
    log "RCLONE_WEIGHTS_PATH unset (skip weights sync)"
    return 0
  fi
  [[ -n "${RCLONE_REMOTE:-}" ]] || { echo "RCLONE_REMOTE required for weights sync" >&2; return 1; }
  need_cmd rclone || { echo "rclone not installed; install it or unset RCLONE_WEIGHTS_PATH" >&2; return 1; }
  log "rclone sync weights: ${RCLONE_REMOTE}:${RCLONE_WEIGHTS_PATH}/ → ${REPO_DIR}/"
  mkdir -p models reid/centroids-reid/models pose/ViTPose/checkpoints
  # Remote folder should mirror needed paths under repo root (e.g. models/, reid/..., pose/...).
  rclone sync "${RCLONE_REMOTE}:${RCLONE_WEIGHTS_PATH}/" "${REPO_DIR}/" --progress
}

dataset_rclone_zips() {
  need_cmd rclone || { echo "rclone required for DATASET_SOURCE=rclone_zips" >&2; exit 1; }
  [[ -n "${RCLONE_REMOTE:-}" ]] || { echo "Set RCLONE_REMOTE" >&2; exit 1; }
  [[ -n "${RCLONE_ZIPS_PATH:-}" ]] || { echo "Set RCLONE_ZIPS_PATH (remote folder with train.zip/test.zip)" >&2; exit 1; }
  local stage="${HOME}/.cache/jersey_rclone_zips"
  mkdir -p "${stage}"
  log "rclone sync zips → ${stage}"
  rclone sync "${RCLONE_REMOTE}:${RCLONE_ZIPS_PATH}" "${stage}" --progress
  mkdir -p "${DATASET_ROOT}"
  for z in train.zip test.zip challenge.zip; do
    if [[ -f "${stage}/${z}" ]]; then
      log "unzip ${z}"
      unzip -o -q "${stage}/${z}" -d "${DATASET_ROOT}"
    fi
  done
  export DATASET_ROOT
  ensure_dataset_layout
}

dataset_rclone_extracted() {
  need_cmd rclone || { echo "rclone required" >&2; exit 1; }
  [[ -n "${RCLONE_REMOTE:-}" ]] || { echo "Set RCLONE_REMOTE" >&2; exit 1; }
  [[ -n "${RCLONE_EXTRACTED_PATH:-}" ]] || { echo "Set RCLONE_EXTRACTED_PATH (remote jersey-2023 tree)" >&2; exit 1; }
  mkdir -p "${REPO_DIR}/data/SoccerNet"
  log "rclone sync extracted dataset → ${DATASET_ROOT}"
  rclone sync "${RCLONE_REMOTE}:${RCLONE_EXTRACTED_PATH}" "${DATASET_ROOT}" --progress
  export DATASET_ROOT="${DATASET_ROOT}"
  ensure_dataset_layout
}

dataset_local_zips() {
  [[ -n "${LOCAL_ZIPS_DIR:-}" ]] || { echo "Set LOCAL_ZIPS_DIR" >&2; exit 1; }
  mkdir -p "${DATASET_ROOT}"
  for z in train.zip test.zip challenge.zip; do
    if [[ -f "${LOCAL_ZIPS_DIR}/${z}" ]]; then
      log "unzip ${LOCAL_ZIPS_DIR}/${z}"
      unzip -o -q "${LOCAL_ZIPS_DIR}/${z}" -d "${DATASET_ROOT}"
    fi
  done
  export DATASET_ROOT="${DATASET_ROOT}"
  ensure_dataset_layout
}

# --- main steps ---
pip_install_req
pip_install_parseq

mkdir -p sam2 reid pose str 2>/dev/null || true
clone_subrepo sam2 https://github.com/davda54/sam.git
clone_subrepo reid/centroids-reid https://github.com/mikwieczorek/centroids-reid.git
clone_subrepo pose/ViTPose https://github.com/ViTAE-Transformer/ViTPose.git
clone_subrepo str/parseq https://github.com/baudm/parseq.git

if [[ -f scripts/patch_centroids_reid_lightning2.py ]] && [[ -d reid/centroids-reid ]]; then
  log "Patch centroids-reid for PyTorch Lightning 2.x (if needed)"
  python3 scripts/patch_centroids_reid_lightning2.py reid/centroids-reid || true
fi

patch_vitpose_mmcv_cap
install_mmcv_if_needed

if dataset_ready; then
  log "Dataset already at ${DATASET_ROOT} (skip download/extract)"
else
  case "${DATASET_SOURCE}" in
    skip)
      log "DATASET_SOURCE=skip and dataset missing at ${DATASET_ROOT} — set DATASET_SOURCE or extract manually"
      ;;
    rclone_zips)
      dataset_rclone_zips
      ;;
    rclone_extracted)
      dataset_rclone_extracted
      ;;
    local_zips)
      dataset_local_zips
      ;;
    *)
      echo "Unknown DATASET_SOURCE=${DATASET_SOURCE}" >&2
      exit 1
      ;;
  esac
fi

sync_weights_optional

if dataset_ready; then
  n="$(find "${DATASET_ROOT}/test/images" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  log "Done. Tracklet dirs under test/images: ${n}"
else
  log "Warning: ${DATASET_ROOT}/test/images still missing — configure DATASET_SOURCE or extract manually."
fi

log "Next: place weights if needed, then: python main.py SoccerNet test --resume"
