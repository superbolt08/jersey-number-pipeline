# A General Framework for Jersey Number Recognition in Sports
Code, data, and model weights for paper  [A General Framework for Jersey Number Recognition in Sports](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) (Maria Koshkina, James H. Elder).

![Pipeline](docs/soccer_pipeline.png)

Image-level detection, localization and recognition (experiments on Hockey dataset):
  - legibility classifier
  - scene text recognition for jersey numbers

Tracklet-level detection, localization and recognition (experiments on SoccerNet dataset):
  - occlusion/outlier removal using re-id features and fitting a Gaussian
  - legibility classifier
  - pose-guided RoI cropping
  - scene text recognition for jersey numbers
  - tracklet prediction consolidation

## Requirements:
* pytorch 1.9.0
* opencv

## Setup:
Clone current repo.
Create conda environment and install requirements.
Code makes use of the several repositories. Run 
```
python3 setup.py 
```

to automatically clone, setup a separate conda environment for each and fetch models. 

Alternatively,  clone each of the following repo, setup conda environments for each following documentation in corresponding repo, and download models:
### SAM:
Should be in jersey-number-pipeline/sam. Repo: [https://github.com/davda54/sam](https://github.com/davda54/sam)

### Centroid-Reid:
Should be in jersey-number-pipeline/reid/centroids-reid. Repo: [https://github.com/mikwieczorek/centroids-reid](https://github.com/mikwieczorek/centroids-reid).
Download [centroid-reid model weights](https://drive.google.com/file/d/1bSUNpvMfJkvCFOu-TK-o7iGY1p-9BxmO/view?usp=sharing) and place 
them under jersey-number-pipeline/reid/centroids-reid/models.

### ViTPose:
Should be in jersey-number-pipeline/pose/ViTPose. Repo: [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).
Download [ViTPose model weights](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) and place 
them under jersey-number-pipeline/pose/ViTPose/checkpoints/.

### PARSeq:
We include the version of the PARSeq code that was used to fine-tune the jersey number model as part of this repo. The original PARSeq repo is [https://github.com/baudm/parseq](https://github.com/baudm/parseq). Model weights should be downloaded and placed under jersey-number-pipeline/models/. 
* [Original model weights](https://drive.google.com/file/d/1AK_GnM6pIYyfIf3tBYSKIyR3Fa3Z46Cx/view?usp=sharing)
* [Hockey fine-tuned](https://drive.google.com/file/d/1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE/view?usp=sharing)
* [SoccerNet fine-tuned](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing)


## Data:
SoccerNet Jersey Number Recognition:
[https://github.com/SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey)
Download and save under /data subfolder. 

* Weakly-labelled player images used to train legibility classifier can be downloaded [here](https://drive.google.com/file/d/1CmJfUmS_ZudgEiCT14b2CbyMA3nEO_uy/view?usp=sharing). 
* Weakly-labelled jersey number crops used to fine-tune STR in LMDB format can be downloaded [here](https://drive.google.com/file/d/1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs/view?usp=sharing).

Hockey (comprised of legibility dataset and jersey number dataset): 
* Request access by contacting [Maria Koshkina](mailto:koshkina@hotmail.com?subject=Hockey). Extract under data/Hockey subfolder.

### Trained Legibility Classifier Weights:
Download and place under jersey-number-pipeline/models/.
* [Hockey](https://drive.google.com/file/d/1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp/view?usp=sharing)
* [SoccerNet](https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing)


## Configuration:
Update configuration.py if required to set custom path to data or dependencies.

## Changes in this fork (COSC 419 proposal and tooling)

The table below summarizes extensions relative to the upstream Koshkina et al. baseline. Toggle behavior mainly via `configuration.proposal` and related keys.

| The change | Where it is | Why it was made |
|------------|-------------|-----------------|
| Central **proposal** settings: combine mode, STR threshold, tracklet confidence floor, legibility gate for crops, color filter on/off | `configuration.py` (`proposal`, `legibility_train_*`) | Single place to align the pipeline with the course proposal (confidence gating, digit-wise combine, crop quality). |
| **Digit-wise** tracklet aggregation from STR logits | `main.py` (combine step), `helpers.py` (`process_jersey_id_predictions_bayesian`, `predict_jersey_number`) | Combine frames using per-digit likelihoods instead of only whole-number voting, as proposed for generalization. |
| **Confidence-weighted** class aggregation and low-confidence frame drop at tracklet level | `helpers.py` (`find_best_prediction`, `process_jersey_id_predictions`) | Reduces impact of blurry or occluded frames when `combine_mode` is `confidence_weighted`. |
| Skip **low-confidence** crops before recording STR outputs | `str.py` (`run_inference`), `main.py` (`--min_str_confidence`) | Cuts noise into PARSeq and unnecessary work; matches SoccerNet-style confidence gating from the literature review. |
| **Stricter legibility** threshold when building crops (uses saved per-image scores) | `main.py` (`get_soccer_net_legibility_results`, legibility JSON), `helpers.py` (`generate_crops`) | Ensures torso crops sent to STR are more likely to contain readable numbers. |
| **HSV-style color filtering** on crops | `helpers.py` (`color_filter_jersey_digits`) | Emphasize digit-like colors before STR, as sketched in the proposal preprocessing section. |
| **Training-time augmentations** (rotation, color jitter, optional blur) | `jersey_number_dataset.py` | Simulate motion and lighting variation on jersey crops without a separate offline clone-20% dataset step. |
| **25 epochs, lr 0.001, momentum 0.85** for legibility training defaults | `configuration.py`, `legibility_classifier.py` | Match proposal section 3.2 hyperparameters where training is run from this repo. |
| **PARSeq training** epoch default | `main.py` (`train_parseq`, `trainer.max_epochs=25`) | Keep STR fine-tuning schedule consistent with the proposal’s epoch count. |
| **Per-image legibility scores** persisted to JSON | `main.py`, `configuration.py` / dataset `working_dir` | Feed crop gating and make legibility debugging reproducible. |
| **Step timing and unbuffered subprocess output** | `main.py` (`_StepTimer`, `_run_shell_with_updates`) | Long GPU runs give clearer progress; addresses replication feedback issues noted in the proposal. |
| **Resume** pipeline stages when outputs already exist | `main.py` (`--resume`, `--force`) | Faster iteration on SoccerNet without redoing finished stages. |
| **Nested SoccerNet jersey-2023 paths** for classifiers | `helpers.py`, `legibility_classifier.py`, `number_classifier.py` | Match current dataset layout under `train/` / `test/`. |
| **ViTPose / Colab install helpers** and Lightning 2 patches for Re-ID | `colab/`, `scripts/install_mmcv_full_vitpose.py`, `centroid_reid.py`, `scripts/patch_centroids_reid_lightning2.py` | End-to-end runs on Colab and modern dependency stacks. |
| **GPU VM setup** (e.g. rented cloud GPUs) | `scripts/setup_vast_gpu_environment.sh`, `GPU_RENTAL_STEPS.md` | One-shot Linux environment setup for remote GPUs. |
| **Windows / CPU / Python 3.12** compatibility (e.g. COCO tools, conda vs plain `python`) | `requirements.txt`, `pose.py`, `main.py`, `str.py`, etc. | Allow local development and platforms without NVIDIA CUDA. |
| **`gdown` and Drive downloads** | `requirements.txt`, `setup.py` | Fetch Google Drive checkpoints and assets during setup or in notebooks. |
| **SAM checked out as `sam2/`** | `setup.py`, `legibility_classifier.py`, `colab/`, `scripts/setup_vast_gpu_environment.sh` | Keeps SAM alongside the repo and avoids import/name clashes; matches Colab and rental-VM docs. |
| **Tracklet lists = subdirectories only** (ignore `.DS_Store` and other files) | `main.py` (`_tracklet_dir_names`), `gaussian_outliers.py`, `centroid_reid.py`, `helpers.py` (`consolidated_results`) | Stops non-folders from breaking filtering, Re-ID, and final consolidation on macOS/Windows. |
| **Optional SoccerNet tracklet subset** | `configuration.py` (`soccer_net_max_tracklets`), `main.py`, `centroid_reid.py`, `gaussian_outliers.py`, `helpers.identify_soccer_balls` (`allowed_tracklets`) | Run smoke tests or partial splits without processing every track folder. |
| **ViTPose mmcv range patch** | `scripts/patch_vitpose_mmpose_mmcv_range.py` | Relaxes mmcv upper bounds when upstream ViTPose/mmpose pins block install. |
| **Kaggle notebook** | `colab/kaggle.ipynb` | Mirror of Colab-style setup for Kaggle kernels. |
| **PARSeq / `strhub` extra deps** | `str/parseq/requirements/core.txt`, `str/parseq/requirements/inference.txt` | Adds `lmdb` and inference pins so `python str.py` works in a minimal env. |
| **PARSeq env naming documented** | `configuration.py` (`str_env`, `str_platform`, comments) | Clarifies which conda env and CUDA tag the STR step expects. |
| **Centroid-ReID import path and `torch.load` for `.ckpt`** | `centroid_reid.py` | Resolves `datasets` and loads Lightning 2 checkpoints from different working directories. |
| **Updated PyTorch stack in requirements** | `requirements.txt` | Moves beyond paper-era torch 1.9 for current GPUs and wheels. |
| **`pycocotools` for pose COCO API** | `requirements.txt`, `pose.py` | Replaces `xtcocotools` where builds fail (e.g. Python 3.12). |
| **Safer directory creation in bootstrap** | `setup.py` | Uses `os.makedirs(..., exist_ok=True)` when creating model dirs. |
| **`.gitignore` updates** | `.gitignore` | Excludes local data, caches, and OS junk from commits. |
| **COSC 419 proposal PDF** | `docs/COSC 419 Project Proposal - Group 4.docx.pdf` | Course submission artifact for the team. |
| **Extended setup guide** | `docs/README.md` | Colab-first walkthrough, Drive layout, and weight paths for this fork. |
| **`GPU_RENTAL_STEPS.md`** | repo root | Human-readable checklist complementing the Vast setup shell script. |

For Colab-focused instructions, see `docs/README.md`.

## Inference:
To run the full inference pipeline for SoccerNet:
```
python3 main.py SoccerNet test
```
To run legibility and jersey number inference for hockey:
```
python3 main.py Hockey test
```
Update actions in main.py actions list to run steps selectively.

## Train (Hockey)
Train legibility classifier:
```
python3 legibility_classifier.py --train --arch resnet34 --sam --data <new-dataset-directory> --trained_model_path ./experiments/hockey_legibility.pth
```

Fine-tune PARSeq STR for hockey number recognition:
```
python3 main.py Hockey train --train_str
```

Trained model will be under str/parseq/outputs

## Train (SoccerNet)
To train legibility classifier and jersey number recognition for SoccerNet, we first generate weakly labelled datasets and then use them to fine-tune.
Weak labels are obtained by using models trained on hockey data.

Train legibility classifier for it:
```
python3 legibility_classifier.py --finetune --arch resnet34 --sam --data <new-dataset-directory>  --full_val_dir
<new-dataset-directory>/val --trained_model_path ./experiments/hockey_legibility.pth --new_trained_model_path ./experiments/sn_legibility.pth
```

Fine-tune PARSeq on weakly-labelled SoccerNet data:
```
python3 main.py SoccerNet train --train_str
```

Trained model will be under str/parseq/outputs.

## Citation
```
@InProceedings{Koshkina_2024_CVPR,
    author    = {Koshkina, Maria and Elder, James H.},
    title     = {A General Framework for Jersey Number Recognition in Sports Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3235-3244}
}
```

## Acknowledgements
We would like to thank authors of the following repositories: 
* [PARSeq](https://github.com/baudm/parseq)
* [Centroid-Reid](https://github.com/mikwieczorek/centroids-reid)
* [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
* [SoccerNet](https://github.com/SoccerNet/sn-jersey)
* [McGill Hockey Player Tracking Dataset](https://github.com/grant81/hockeyTrackingDataset)
* [SAM](https://github.com/davda54/sam)

## License
[![License](https://i.creativecommons.org/l/by-nc/3.0/88x31.png)](http://creativecommons.org/licenses/by-nc/3.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 3.0 Unported License](http://creativecommons.org/licenses/by-nc/3.0/).
