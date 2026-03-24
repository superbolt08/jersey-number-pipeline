#!/usr/bin/env python3
"""Patch mikwieczorek/centroids-reid for PyTorch Lightning 2.x."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCHES: list[tuple[str, str, str, str]] = [
    (
        "utils/misc.py",
        "from pytorch_lightning.utilities.seed import seed_everything",
        (
            "try:\n"
            "    from pytorch_lightning.utilities.seed import seed_everything\n"
            "except ImportError:\n"
            "    from lightning_fabric.utilities.seed import seed_everything"
        ),
        "lightning_fabric.utilities.seed",
    ),
    (
        "callbacks/chechpointer_callback.py",
        "from pytorch_lightning.callbacks.base import Callback",
        "from pytorch_lightning.callbacks import Callback",
        "pytorch_lightning.callbacks import Callback",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "centroids_root",
        nargs="?",
        default="reid/centroids-reid",
        help="Path to centroids-reid clone",
    )
    args = p.parse_args()
    root = Path(args.centroids_root)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    for rel, needle, repl, done_mark in PATCHES:
        path = root / rel
        if not path.is_file():
            print(f"Skip (missing): {path}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        if done_mark in text and needle not in text:
            print("Already patched:", path)
            continue
        if needle not in text:
            print(f"No expected line in {path}; leave unchanged.", file=sys.stderr)
            continue
        path.write_text(text.replace(needle, repl, 1), encoding="utf-8")
        print("Patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
