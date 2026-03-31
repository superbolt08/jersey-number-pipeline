import argparse
import json
import os
from collections import Counter


def analyze(pred_path, gt_path, top_k, out_txt):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    correct = 0
    errors = []
    two_to_one = 0
    one_to_two = 0
    illegible_mismatch = 0

    for track_id, gt_label_raw in gt.items():
        gt_label = str(gt_label_raw)
        pred_label = str(pred.get(track_id, -1))

        if pred_label == gt_label:
            correct += 1
        else:
            errors.append((track_id, gt_label, pred_label))

        if len(gt_label) == 2 and gt_label != "-1" and len(pred_label) == 1 and pred_label != "-1":
            two_to_one += 1
        if len(gt_label) == 1 and gt_label != "-1" and len(pred_label) == 2 and pred_label != "-1":
            one_to_two += 1
        if (gt_label == "-1") != (pred_label == "-1"):
            illegible_mismatch += 1

    total = len(gt)
    accuracy = 100.0 * correct / total if total else 0.0
    confusion = Counter((g, p) for _, g, p in errors)

    print(f"Total tracklets: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Errors: {len(errors)}")
    print(f"2-digit -> 1-digit errors: {two_to_one}")
    print(f"1-digit -> 2-digit errors: {one_to_two}")
    print(f"Illegible mismatches (-1 disagreements): {illegible_mismatch}")
    print("")
    print(f"Top {top_k} confusion pairs (GT -> Pred):")
    for (g, p), count in confusion.most_common(top_k):
        print(f"{g} -> {p}: {count}")

    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Jersey Number Error Analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total tracklets: {total}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Errors: {len(errors)}\n")
        f.write(f"2-digit -> 1-digit errors: {two_to_one}\n")
        f.write(f"1-digit -> 2-digit errors: {one_to_two}\n")
        f.write(f"Illegible mismatches (-1 disagreements): {illegible_mismatch}\n")
        f.write("\n")
        f.write(f"Top {top_k} confusion pairs (GT -> Pred):\n")
        for (g, p), count in confusion.most_common(top_k):
            f.write(f"{g} -> {p}: {count}\n")
        f.write("\n")
        f.write("All error tracklets:\n")
        for track_id, g, p in errors:
            f.write(f"{track_id}: GT={g}, Pred={p}\n")

    print(f"\nWrote full error list to: {out_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JerseyNet prediction errors.")
    parser.add_argument(
        "--pred",
        default="out/SoccerNetResults/final_results_baseline.json",
        help="Path to predictions JSON",
    )
    parser.add_argument(
        "--gt",
        default="data/SoccerNet/jersey-2023/test/test_gt.json",
        help="Path to ground-truth JSON",
    )
    parser.add_argument("--top_k", type=int, default=15, help="Top confusion pairs to print")
    parser.add_argument(
        "--out_txt",
        default="out/error_analysis.txt",
        help="Path to output TXT file with all errors",
    )
    args = parser.parse_args()
    analyze(args.pred, args.gt, args.top_k, args.out_txt)
