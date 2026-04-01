# Digit-Wise Tracklet Aggregation Modification

This modification changes the final tracklet aggregation step after STR. Instead of taking each crop's full predicted jersey number and combining those whole-number votes across the tracklet, it aggregates evidence for the tens digit and units digit separately using the STR model's per-position outputs, then reconstructs the final jersey number from those two digit decisions.

## Result Compared To Baseline

- Baseline result: `1050 / 1211` correct (`86.71%`)
- Digit-wise aggregation result: `1034 / 1211` correct (`85.38%`)

That means the digit-wise version was worse by `16` tracklets, or about `1.32` percentage points.
