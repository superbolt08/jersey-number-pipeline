from pathlib import Path
import sys
import os
import json
import argparse
import re

# Absolute path + insert at front so `import datasets` resolves to centroids-reid/datasets/,
# not Hugging Face `datasets` from site-packages (Colab has both).
_ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reid', 'centroids-reid')
)
sys.path.insert(0, _ROOT_DIR)
ROOT = _ROOT_DIR + os.sep


def _patch_centroids_reid_for_pl2():
    """Upstream centroids-reid targets PL 1.x; patch vendored files for PyTorch Lightning 2.x."""
    root = os.path.normpath(ROOT)

    def _write_if_changed(path, old, new, skip_if):
        if not os.path.isfile(path):
            return
        try:
            text = open(path, encoding='utf-8').read()
        except OSError:
            return
        if skip_if(text):
            return
        if old not in text:
            return
        try:
            open(path, 'w', encoding='utf-8').write(text.replace(old, new, 1))
        except OSError:
            pass

    _write_if_changed(
        os.path.join(root, 'utils', 'misc.py'),
        'from pytorch_lightning.utilities.seed import seed_everything',
        (
            'try:\n'
            '    from pytorch_lightning.utilities.seed import seed_everything\n'
            'except ImportError:\n'
            '    from lightning_fabric.utilities.seed import seed_everything'
        ),
        lambda t: 'lightning_fabric.utilities.seed' in t,
    )
    _write_if_changed(
        os.path.join(root, 'callbacks', 'chechpointer_callback.py'),
        'from pytorch_lightning.callbacks.base import Callback',
        'from pytorch_lightning.callbacks import Callback',
        lambda t: 'pytorch_lightning.callbacks import Callback' in t and 'callbacks.base' not in t,
    )
    # PL 2.x: self.hparams has no setter; only save_hyperparameters(...) is allowed.
    _bases = os.path.join(root, 'modelling', 'bases.py')
    if os.path.isfile(_bases):
        try:
            _bt = open(_bases, encoding='utf-8').read()
        except OSError:
            pass
        else:
            if 'save_hyperparameters(AttributeDict(hparams))' in _bt and 'self.hparams = AttributeDict(hparams)' not in _bt:
                pass
            else:
                _new_bt, _n = re.subn(
                    r'^([ \t]*)self\.hparams = AttributeDict\(hparams\)\s*\n[ \t]*self\.save_hyperparameters\(self\.hparams\)',
                    r'\1self.save_hyperparameters(AttributeDict(hparams))',
                    _bt,
                    count=1,
                    flags=re.MULTILINE,
                )
                if _n:
                    try:
                        open(_bases, 'w', encoding='utf-8').write(_new_bt)
                    except OSError:
                        pass


_patch_centroids_reid_for_pl2()

import numpy as np
import torch

# PyTorch 2.6+ defaults torch.load(weights_only=True). Lightning .ckpt files need full pickle
# (callbacks, etc.); trust only checkpoints you placed in reid/centroids-reid/models/.
_orig_torch_load = torch.load


def _torch_load_for_pl_ckpt(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_for_pl_ckpt

from tqdm import tqdm
import cv2
from PIL import Image

from config import cfg
from train_ctl_model import CTLModel

from datasets.transforms import ReidTransforms



# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT+'/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT+'/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/market1501_resnet50_256_128_epoch_120.ckpt')
ver_to_specs["res50_duke"]   = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt')


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights


def _checkpoint_valid(path):
    """Raise a clear error if the checkpoint is missing or invalid (e.g. HTML from failed download)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Centroid-ReID checkpoint not found: {path}\n"
            "Download the model weights from the pipeline README (Centroid-ReID weights) and place them in reid/centroids-reid/models/"
        )
    size = os.path.getsize(path)
    if size < 1024 * 1024:  # < 1 MB is not a real ResNet50 checkpoint
        raise ValueError(
            f"Centroid-ReID checkpoint looks invalid (file too small: {size} bytes): {path}\n"
            "The file may be an HTML page from a failed Google Drive download.\n"
            "Re-download the weights from the pipeline README and place the real .ckpt file(s) in reid/centroids-reid/models/"
        )
    with open(path, "rb") as f:
        head = f.read(20)
    if head.startswith(b"<") or head.startswith(b"{") or head.startswith(b"<!DOCTYPE"):
        raise ValueError(
            f"Centroid-ReID checkpoint looks like HTML/text, not a PyTorch file: {path}\n"
            "Re-download the weights from the pipeline README (Centroid-ReID) and place them in reid/centroids-reid/models/"
        )


def generate_features(input_folder, output_folder, model_version='res50_market', subset_tracklets=None):
    # load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)
    
    _checkpoint_valid(cfg.MODEL.PRETRAIN_PATH)
    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH, cfg=cfg)

    # print("Loading from " + MODEL_FILE)
    if use_cuda:
        model.to('cuda')
        print("using GPU")
    model.eval()

    tracks = os.listdir(input_folder)
    if subset_tracklets is not None:
        allow = set(subset_tracklets)
        tracks = [t for t in tracks if t in allow]
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)

    for track in tqdm(tracks):
        track_path = os.path.join(input_folder, track)
        if not os.path.isdir(track_path):
            continue  # skip .DS_Store and other non-directory entries
        features = []
        images = os.listdir(track_path)
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        for img_path in images:
            img = cv2.imread(os.path.join(track_path, img_path))
            input_img = Image.fromarray(img)
            input_img = torch.stack([val_transforms(input_img)])
            with torch.no_grad():
                _, global_feat = model.backbone(input_img.cuda() if use_cuda else input_img)
                global_feat = model.bn(global_feat)
            features.append(global_feat.cpu().numpy().reshape(-1,))

        np_feat = np.array(features)
        with open(output_file, 'wb') as f:
            np.save(f, np_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    parser.add_argument('--subset_file', default='', help="Optional JSON list of tracklet directory names to process only")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    subset = None
    if args.subset_file and os.path.isfile(args.subset_file):
        with open(args.subset_file, 'r') as sf:
            subset = json.load(sf)

    generate_features(args.tracklets_folder, args.output_folder, subset_tracklets=subset)



