# Mini Ablation Report: Confidence-Weighted STR Training

## Objective
Evaluate whether confidence-weighted pseudo-label training improves STR fine-tuning performance over a standard baseline on SoccerNet weak labels.

## Experimental Setup
- Model: PARSeq (configured for jersey numbers with max label length 2)
- Dataset: `data/SoccerNet/jersey-2023/lmdb`
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU
- Training schedule: 25 epochs, single GPU
- Validation: end of each epoch (`trainer.val_check_interval=1.0`)

## Weight Generation Method
I computed per-image sample weights from the teacher STR predictions on LMDB training samples:
- Compare teacher prediction to pseudo-label (agreement/disagreement)
- Compute confidence (excluding EOS)
- Assign sample weight:
  - agreement: `weight = confidence`
  - disagreement: `weight = confidence * disagree_multiplier`
- I tried two weighting settings:
  - v1: `disagree_multiplier = 0.1`
  - v2: `disagree_multiplier = 0.5`

Teacher-vs-pseudo agreement during scoring: **99.44%** on 13031/13104 image samples.

## Commands Used
From `str/parseq`:

```bash
# Generate weights (v1)
python tools/score_lmdb_consistency.py ../../models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt --data_root ../../data/SoccerNet/jersey-2023/lmdb --output_json ../../out/SoccerNetResults/train/sample_weights_v1.json --device cuda

# Generate weights (v2, softer disagreement penalty)
python tools/score_lmdb_consistency.py ../../models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt --data_root ../../data/SoccerNet/jersey-2023/lmdb --output_json ../../out/SoccerNetResults/train/sample_weights_v2_dm05.json --device cuda --disagree_multiplier 0.5

# Baseline training (no weights)
python train.py +experiment=parseq dataset=real data.root_dir=../../data/SoccerNet/jersey-2023/lmdb pretrained=null trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=25 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=2 model.max_label_length=2 data.num_workers=8

# Weighted training (v1)
python train.py +experiment=parseq dataset=real data.root_dir=../../data/SoccerNet/jersey-2023/lmdb pretrained=null trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=25 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=2 model.max_label_length=2 data.num_workers=8 data.train_weights_path=../../out/SoccerNetResults/train/sample_weights_v1.json

# Weighted training (v2)
python train.py +experiment=parseq dataset=real data.root_dir=../../data/SoccerNet/jersey-2023/lmdb pretrained=null trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=25 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=2 model.max_label_length=2 data.num_workers=8 data.train_weights_path=../../out/SoccerNetResults/train/sample_weights_v2_dm05.json
```

## Results

| Run | Best val_accuracy | Best val_NED | Output run folder |
|---|---:|---:|---|
| Baseline (no weights) | **42.1017** | **50.0343** | `str/parseq/outputs/parseq/2026-03-27_13-00-08` |
| Weighted v1 (`disagree_multiplier=0.1`) | 38.6676 | 46.5316 | `str/parseq/outputs/parseq/2026-03-27_13-27-17` |
| Weighted v2 (`disagree_multiplier=0.5`) | 37.6374 | 46.6690 | `str/parseq/outputs/parseq/2026-03-27_13-54-26` |

Compared to baseline:
- Weighted v1: `-3.4341` percentage points in `val_accuracy`
- Weighted v2: `-4.4643` percentage points in `val_accuracy`

## Discussion
The confidence weighted perforemed worse than the baseline, this is probably because STR already agreed with the pseudo-labels at a very high accuracy 99.44% so this suggest that the pseudo-labeling doesn't actually introduce noise.

## Conclusion
For the tested settings, confidence-weighted pseudo-label training did not improve performance over standard training. The baseline model is currently the best-performing option.


