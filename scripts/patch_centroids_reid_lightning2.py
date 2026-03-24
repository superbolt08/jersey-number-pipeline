#!/usr/bin/env python3
"""Patch mikwieczorek/centroids-reid for PyTorch Lightning 2.x (seed_everything import path)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

NEEDLE = "from pytorch_lightning.utilities.seed import seed_everything"
REPLACEMENT = (
    "try:\n"
    "    from pytorch_lightning.utilities.seed import seed_everything\n"
    "except ImportError:\n"
    "    from lightning_fabric.utilities.seed import seed_everything"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "misc_py",
        nargs="?",
        default="reid/centroids-reid/utils/misc.py",
        help="Path to centroids-reid utils/misc.py",
    )
    args = p.parse_args()
    path = Path(args.misc_py)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    if "lightning_fabric.utilities.seed" in text:
        print("Already patched or compatible:", path)
        return 0
    if NEEDLE not in text:
        print(f"No expected import line in {path}; leave unchanged.", file=sys.stderr)
        return 1
    path.write_text(text.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
    print("Patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
