#!/usr/bin/env python3
"""Relax ViTPose vendored mmpose mmcv upper bound (1.5.0 -> 1.7.2) for mmcv-full + torch2 wheels."""
from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "vitpose_root",
        nargs="?",
        default="pose/ViTPose",
        help="Path to ViTPose clone (contains mmpose/__init__.py)",
    )
    args = p.parse_args()
    path = pathlib.Path(args.vitpose_root) / "mmpose" / "__init__.py"
    if not path.is_file():
        print(f"Skip: not found {path}", file=sys.stderr)
        return 0
    text = path.read_text(encoding="utf-8")
    old = "mmcv_maximum_version = '1.5.0'"
    new = "mmcv_maximum_version = '1.7.2'"
    old2 = 'mmcv_maximum_version = "1.5.0"'
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print("Patched:", path)
        return 0
    if old2 in text:
        path.write_text(text.replace(old2, 'mmcv_maximum_version = "1.7.2"', 1), encoding="utf-8")
        print("Patched:", path)
        return 0
    if "mmcv_maximum_version = '1.7.2'" in text or 'mmcv_maximum_version = "1.7.2"' in text:
        print("Already patched:", path)
        return 0
    print(f"No expected mmcv_maximum_version line in {path}; leave unchanged.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
