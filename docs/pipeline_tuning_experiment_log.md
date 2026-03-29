# Pipeline Tuning Experiment Log

This log captures tuning experiments run on branch work around combine/crop/STR preprocessing and weak-label weighting.

## Baseline (Reference)

- Evaluation command:
  - `python evaluate.py --pred out/SoccerNetResults/final_results_baseline.json --gt data/SoccerNet/jersey-2023/test/test_gt.json`
- Baseline test accuracy: **85.38%**
- Error profile (from `analyze_errors.py`):
  - Total tracklets: 1211
  - Errors: 177
  - 2-digit -> 1-digit: 6
  - 1-digit -> 2-digit: 5
  - Illegible mismatches (`-1` disagreements): 105

Observation: most errors are illegibility mismatches, not one-vs-two-digit confusion.

---

## Experiment Group A: Confidence-Weighted STR Training (Mini Ablation)

Teacher-vs-pseudo agreement during STR sample scoring:
- **99.44%** (13031 / 13104)

Training results (validation metrics):

| Run | Best val_accuracy | Best val_NED | Notes |
|---|---:|---:|---|
| Baseline training (no sample weights) | **42.1017** | **50.0343** | `outputs/parseq/2026-03-27_13-00-08` |
| Weighted v1 (`disagree_multiplier=0.1`) | 38.6676 | 46.5316 | worse than baseline |
| Weighted v2 (`disagree_multiplier=0.5`) | 37.6374 | 46.6690 | worse than baseline |

Conclusion: confidence-weighted pseudo-label training did not improve this setup.

---

## Experiment Group B: Length-Aware Digit Combine Rule

Goal: reduce 2-digit collapse to 1-digit in tracklet combine.

Tried:
- Added `length_aware_two_digit_rule` and `length_aware_two_digit_bonus`.
- Tested bonuses including extreme values to validate path.

Results:
- Bonus `1.0`: no improvement vs baseline.
- Bonus `100.0`: severe degradation (accuracy around **72%**), confirming rule path executed but over-biased.
- Practical bonuses did not produce a net gain.

Conclusion: this heuristic did not improve top-line accuracy for this dataset.

---

## Experiment Group C: Confidence Filtering in Digit-Wise Combine

Goal: ignore low-confidence STR frame predictions before digit-wise tracklet aggregation.

Change:
- Applied `min_tracklet_frame_confidence` filtering in digit-wise combine path.

Tried:
- `min_tracklet_frame_confidence = 0.20`, `0.25`, `0.30`

Results:
- `0.20` / `0.25`: no improvement.
- `0.30`: degraded confidence/quality.

Conclusion: threshold filtering in digit-wise combine did not improve final accuracy.

---

## Experiment Group D: STR Aspect-Ratio Letterbox Padding

Goal: preserve crop aspect ratio via black padding before PARSeq resize.

Change:
- Added `str_letterbox_pad` toggle.

Result:
- With `str_letterbox_pad=True` (padding-only test): test accuracy dropped to about **49%**.

Conclusion: this preprocessing strongly harms current STR performance and should remain disabled.

---



