import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from strhub.data.dataset import build_tree_dataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


def _parse_jersey_digits(value: str) -> str:
    digits = ''.join(ch for ch in value if ch.isdigit())
    return digits if digits else '-1'


def _confidence_product(token_confidences) -> float:
    conf = token_confidences.detach().cpu().tolist()
    total = 1.0
    for c in conf[:-1]:
        total *= float(c)
    return total


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Score LMDB pseudo-label consistency using STR teacher.")
    parser.add_argument("checkpoint", help="PARSeq checkpoint path")
    parser.add_argument("--data_root", required=True, help="Root containing train/<train_dir>/... LMDBs")
    parser.add_argument("--output_json", required=True, help="Path to write sample weights JSON")
    parser.add_argument("--train_dir", default="real", help="Train subdir under train/ (default: real)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disagree_multiplier", type=float, default=0.1)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap for smoke tests; 0 = all")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
    hp = model.hparams
    transform = SceneTextDataModule.get_transform(tuple(hp.img_size))
    lmdb_root = Path(args.data_root).resolve() / "train" / args.train_dir
    dataset = build_tree_dataset(
        str(lmdb_root),
        hp.charset_test,
        hp.max_label_length,
        0,
        True,
        True,
        transform=transform,
        return_sample_key=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    weights = {}
    stats = {
        "total": 0,
        "agree": 0,
        "disagree": 0,
        "mean_weight": 0.0,
        "mean_confidence": 0.0,
    }

    pbar = tqdm(loader, desc="Scoring LMDB consistency")
    for images, labels, sample_keys in pbar:
        logits = model(images.to(model.device))
        probs_full = logits[:, :3, :11].softmax(-1)
        preds, probs = model.tokenizer.decode(probs_full)

        for pred, prob, gt_label, sample_key in zip(preds, probs, labels, sample_keys):
            pred_num = _parse_jersey_digits(pred)
            gt_num = _parse_jersey_digits(gt_label)
            conf = _confidence_product(prob)
            agree = pred_num == gt_num
            weight = conf if agree else conf * args.disagree_multiplier
            weight = max(args.min_weight, min(args.max_weight, weight))

            weights[sample_key] = float(weight)
            stats["total"] += 1
            stats["agree"] += int(agree)
            stats["disagree"] += int(not agree)
            stats["mean_weight"] += float(weight)
            stats["mean_confidence"] += float(conf)

            if args.max_samples > 0 and stats["total"] >= args.max_samples:
                break
        if args.max_samples > 0 and stats["total"] >= args.max_samples:
            break

    if stats["total"] > 0:
        stats["mean_weight"] /= stats["total"]
        stats["mean_confidence"] /= stats["total"]

    payload = {
        "format_version": 1,
        "params": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "train_dir": args.train_dir,
            "disagree_multiplier": args.disagree_multiplier,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
        },
        "stats": stats,
        "weights": weights,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {len(weights)} sample weights -> {out_path}")
    print(f"Agree: {stats['agree']}/{stats['total']} ({(100.0 * stats['agree'] / max(stats['total'], 1)):.2f}%)")


if __name__ == "__main__":
    main()
