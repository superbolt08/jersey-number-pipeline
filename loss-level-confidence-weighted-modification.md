# Loss-Level Confidence-Weighted Pseudo-Label Training Modification

## What this is
This modification changes STR training so each training sample can have a different loss weight.

Idea:
- if teacher prediction agrees with pseudo-label -> higher weight
- if teacher prediction disagrees -> lower weight

So noisy pseudo-labels affect training less.

## Setup before running
Use the PARSeq environment and run commands from `str/parseq`.

```bash
conda activate parseq2
cd str\parseq
```

Note:
- Hydra breaks on checkpoint filenames with `=` in command overrides.
- We copied the checkpoint to a simple name: `models/parseq95.ckpt`.

## General command flow

### 1) Generate sample weights JSON
This scores pseudo-label consistency and writes per-sample weights.

```bash
python tools\score_lmdb_consistency.py ..\..\models\parseq95.ckpt --data_root ..\..\data\SoccerNet\jersey-2023\lmdb --output_json ..\..\out\SoccerNetResults\train\sample_weights_v1.json --device cuda
```

Observed from run:
- `Agree: 13031 / 13104 (99.44%)`

### 2) Continue training from strong checkpoint (weighted run)
This is the working setup we used:

```bash
python train.py +experiment=parseq dataset=real data.root_dir=..\..\data\SoccerNet\jersey-2023\lmdb ckpt_path=..\..\models\parseq95.ckpt trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=30 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=25 model.max_label_length=25 data.num_workers=8 data.train_weights_path=..\..\out\SoccerNetResults\train\sample_weights_v1.json
```

Why `max_label_length=25`:
- the checkpoint has `pos_queries` built for length 25
- using length 2 caused a size mismatch (`[1,26,384]` vs `[1,3,384]`)
- run resumes around epoch 25 and goes to epoch 30

### 3) Matched no-weight control run (for fair comparison)

```bash
python train.py +experiment=parseq dataset=real data.root_dir=..\..\data\SoccerNet\jersey-2023\lmdb ckpt_path=..\..\models\parseq95.ckpt trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=30 trainer.val_check_interval=1.0 data.batch_size=128 data.max_label_length=25 model.max_label_length=25 data.num_workers=8
```

### 4) Run pipeline inference with the new checkpoint
After training, point SoccerNet `str_model` in `configuration.py` to the new checkpoint.

Then rerun only STR/combine with resume:

```bash
del ..\..\out\SoccerNetResults\jersey_id_results.json
del ..\..\out\SoccerNetResults\final_results.json
cd ..\..
python main.py SoccerNet test --resume
python evaluate.py --pred out/SoccerNetResults/final_results.json --gt data/SoccerNet/jersey-2023/test/test_gt.json
```

## What changed in code
- `str/parseq/configs/main.yaml`
  - added `data.train_weights_path` (default null)
- `str/parseq/strhub/data/module.py`
  - passes optional weights path to train dataset
- `str/parseq/strhub/data/dataset.py`
  - can load sample weights and return `(img, label, weight)` during training
- `str/parseq/strhub/models/parseq/system.py`
  - training step supports weighted per-token loss
- `str/parseq/tools/score_lmdb_consistency.py`
  - new script to generate the weights JSON from teacher consistency

## Quick summary of pipeline impact
This does not change ReID, pose, crop generation, or test-time aggregation logic.
It only changes how STR is trained when using pseudo-labels.

## Known limitation we found
- Directly resuming `parseq95.ckpt` with `max_label_length=2` is not possible due to tensor shape mismatch.
- So I used the architecture-compatible route (`max_label_length=25`) to continue from that checkpoint.

## Results 


### Metrics
- Baseline STR val accuracy: `95.6044`
- Weighted v1 STR val accuracy (`disagree_multiplier=0.1`): `95.6044` (epoch 29)
- Baseline STR val NED: `96.3255`
- Weighted v1 STR val NED (`disagree_multiplier=0.1`): `96.4286` (epoch 29)
- v1 val deltas (weighted - baseline): accuracy `+0.0000`, NED `+0.1031`
- Baseline end-to-end test accuracy: `86.71%`
- Weighted v1 end-to-end test accuracy: `85.88%`
- v1 delta vs baseline: `-0.83%`
- v1 error analysis (baseline -> weighted):
  - `2-digit -> 1-digit`: `6 -> 8`
  - `1-digit -> 2-digit`: `3 -> 4`
  - `Illegible mismatch (-1 disagreement)`: `96 -> 96`
- Weighted v2 STR val accuracy (`disagree_multiplier=0.5`): `95.7418` (epoch 29)
- Weighted v2 STR val NED (`disagree_multiplier=0.5`): `96.4629` (epoch 29)
- v2 val deltas (weighted - baseline): accuracy `+0.1374`, NED `+0.1374`
- Weighted v2 end-to-end test accuracy: `86.46%`
- v2 delta vs baseline: `-0.25%`
- v2 error analysis (baseline -> weighted v2):
  - `2-digit -> 1-digit`: `6 -> 5`
  - `1-digit -> 2-digit`: `3 -> 3`
  - `Illegible mismatch (-1 disagreement)`: `96 -> 100`

### Notes
- Best weighted v1 checkpoint: `checkpoints/epoch=29-step=3090-val_accuracy=95.6044-val_NED=96.4286.ckpt`
- Best weighted v2 checkpoint: `checkpoints/epoch=29-step=3090-val_accuracy=95.7418-val_NED=96.4629.ckpt`
- Any training issues:
  - Direct path `checkpoints/...` failed in `str.py` loader because filename did not include `parseq`.
  - Fixed by copying to parseq-named files in `models/` and using those in config.
- Any runtime differences:
- Final take:
  - v2 (`disagree_multiplier=0.5`) was clearly better than v1 (`0.1`) on end-to-end accuracy (`86.46%` vs `85.88%`), but still slightly below baseline (`86.71%`).
  - Softer weighting helped the 2-digit -> 1-digit confusion issue, but increased `-1` mismatch errors.
