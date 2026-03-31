# Skip Low Confidence STR Crops Modification

This change adds a confidence threshold to the SoccerNet STR step. For each cropped torso image, PARSeq predicts a jersey string and returns token-level confidence values for the decoded characters. The pipeline multiplies those token confidences together, ignoring the final end-token confidence, to produce one per-crop score.

If that per-crop score is below `min_str_frame_confidence`, the crop is skipped during STR inference.
 
## Estimation Command

Used the saved STR output file to estimate how many crops would be skipped at different thresholds without rerunning STR:

```bash
python -c "import json, math; t=0.35; d=json.load(open('out/SoccerNetResults/jersey_id_results.json')); n=sum(1 for v in d.values() if math.prod(float(x) for x in v['confidence'][:-1]) < t); print(f'threshold={t} total={len(d)} skipped={n} kept={len(d)-n}')"
```

## Threshold

- `0.12` -> `0` skipped, `97472` kept
- `0.20` -> `34` skipped, `97438` kept
- `0.25` -> `95` skipped, `97377` kept
- `0.30` -> `233` skipped, `97239` kept
- `0.35` -> `479` skipped, `96993` kept
- `0.40` -> `843` skipped, `96629` kept
- `0.50` -> `2040` skipped, `95432` kept

## Current Setting

- Updated `configuration.py` to `min_str_frame_confidence = 0.35`.

## Baseline Evaluation Result

- Ground-truth samples: `1211`
- Predictions provided: `1211`
- Correct predictions: `1050`
- Missing predictions: `0`
- Final accuracy: `86.71%`

## Evaluation Result at 0.12

- Ground-truth samples: `1211`
- Predictions provided: `1211`
- Correct predictions: `1050`
- Missing predictions: `0`
- Final accuracy: `86.71%`

## Evaluation Result at 0.35

- Ground-truth samples: `1211`
- Predictions provided: `1211`
- Correct predictions: `1051`
- Missing predictions: `0`
- Final accuracy: `86.79%`

## Evaluation Result at 0.40

- STR skipped crops: `843`
- Ground-truth samples: `1211`
- Predictions provided: `1211`
- Correct predictions: `1051`
- Missing predictions: `0`
- Final accuracy: `86.79%`
