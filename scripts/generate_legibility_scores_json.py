#!/usr/bin/env python3
"""
Build legibility_scores*.json for SoccerNet (per-frame raw legibility scores).

Output format matches helpers._lookup_legibility_score expectations: each image is
stored under both its absolute path and basename -> float score.

Default behavior matches main.soccer_net_pipeline legibility step:
  gaussian-filtered image lists, soccer ball tracks excluded, same model/arch as configuration.py.

Run from repository root:
  python scripts/generate_legibility_scores_json.py test
  python scripts/generate_legibility_scores_json.py val --max-tracklets 5

Requires: Gaussian (or sim) filter JSON and soccer_ball JSON already exist under working_dir
unless you pass --no-filter / --include-balls as documented below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Repo root = parent of scripts/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import legibility_classifier as lc  # noqa: E402
from tqdm import tqdm  # noqa: E402

import configuration as config  # noqa: E402


def _tracklet_dir_names(images_root: str) -> list[str]:
    return [
        name
        for name in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, name))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "part",
        choices=("test", "val", "train", "challenge"),
        help="SoccerNet split (must exist under configuration.dataset['SoccerNet']).",
    )
    parser.add_argument(
        "--max-tracklets",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N tracklet folders (sorted by name). None = all.",
    )
    parser.add_argument(
        "--filter",
        choices=("gauss", "sim"),
        default="gauss",
        help="Which main-subject JSON to use when --no-filter is not set (default: gauss).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Use all files in each tracklet folder (no gauss/sim JSON).",
    )
    parser.add_argument(
        "--include-balls",
        action="store_true",
        help="Do not exclude tracklets listed in soccer_ball_list.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Override output JSON path (default: working_dir + split legibility_scores from config).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        metavar="N",
        help="Optional json.dump indent (default: None for compact one-line file).",
    )
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)

    sn = config.dataset["SoccerNet"]
    part_cfg = sn[args.part]
    root_dir = sn["root_dir"]
    image_dir = part_cfg["images"]
    path_to_images = os.path.join(root_dir, image_dir)
    if not os.path.isdir(path_to_images):
        print(f"Images root not found: {path_to_images}", file=sys.stderr)
        return 1

    working_dir = sn["working_dir"]
    tracklets = _tracklet_dir_names(path_to_images)
    filtered = None
    if not args.no_filter:
        key = "gauss_filtered" if args.filter == "gauss" else "sim_filtered"
        path_to_filter = os.path.join(working_dir, part_cfg[key])
        if not os.path.isfile(path_to_filter):
            print(f"Filter JSON missing: {path_to_filter}", file=sys.stderr)
            return 1
        with open(path_to_filter, "r", encoding="utf-8") as f:
            filtered = json.load(f)

    if not args.include_balls:
        soccer_ball_list = os.path.join(working_dir, part_cfg["soccer_ball_list"])
        if not os.path.isfile(soccer_ball_list):
            print(f"soccer_ball_list missing: {soccer_ball_list}", file=sys.stderr)
            return 1
        with open(soccer_ball_list, "r", encoding="utf-8") as f:
            ball_json = json.load(f)
        ball_list = set(ball_json["ball_tracks"])
        tracklets = [t for t in tracklets if t not in ball_list]

    tracklets = sorted(tracklets)
    if args.max_tracklets is not None and args.max_tracklets > 0:
        tracklets = tracklets[: int(args.max_tracklets)]

    model_path = sn["legibility_model"]
    arch = sn["legibility_model_arch"]
    if not os.path.isfile(model_path):
        print(f"Legibility model not found: {model_path}", file=sys.stderr)
        return 1

    legibility_scores: dict[str, float] = {}
    for directory in tqdm(tracklets, desc="tracklets"):
        track_dir = os.path.join(path_to_images, directory)
        if filtered is not None:
            if directory not in filtered:
                continue
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        if not images_full_path:
            continue
        _binary, track_raw = lc.run(
            images_full_path,
            model_path,
            arch=arch,
            threshold=0.5,
            return_raw_scores=True,
        )
        for p, s in zip(images_full_path, track_raw):
            v = float(s)
            legibility_scores[p] = v
            legibility_scores[os.path.basename(p)] = v

    scores_name = part_cfg.get("legibility_scores", "legibility_scores.json")
    out_path = args.output or os.path.join(working_dir, scores_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_scores:
        json.dump(legibility_scores, out_scores, indent=args.indent)
        if args.indent is not None:
            out_scores.write("\n")

    n_keys_unique_paths = len(legibility_scores) // 2
    print(f"Wrote {out_path} ({len(legibility_scores)} entries, ~{n_keys_unique_paths} images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
