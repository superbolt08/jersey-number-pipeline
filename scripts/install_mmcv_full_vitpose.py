#!/usr/bin/env python3
"""
Install mmcv-full 1.7.x (provides `import mmcv`) for ViTPose's vendored mmpose.

Colab often uses Python 3.12 + a recent torch/CUDA combo where `mim install` has no
matching prebuilt wheel. This script retries several OpenMMLab -f URLs, then optional
source build, and finally checks `import mmcv` so failures are obvious.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _run_pip(args: list[str], *, quiet: bool = True, timeout: int | None = 600) -> int:
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(args)
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("pip install timed out.", flush=True)
        return 1


def _have_mmcv() -> bool:
    try:
        import mmcv

        print("mmcv import OK:", mmcv.__version__, flush=True)
        return True
    except Exception as e:
        print("mmcv not importable:", e, flush=True)
        return False


def _cuda_tag(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    parts = cuda_version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None
    return f"cu{major}{minor}"


def main() -> int:
    if _have_mmcv():
        return 0

    _run_pip(["-U", "pip", "setuptools", "wheel"], quiet=True)

    print("Trying openmim + mim install mmcv-full==1.7.2 ...", flush=True)
    _run_pip(["-U", "openmim"], quiet=True)
    subprocess.call("mim install mmcv-full==1.7.2", shell=True)
    if _have_mmcv():
        return 0

    try:
        import torch
    except ImportError:
        print("torch is not installed; install requirements.txt first.", flush=True)
        return 1

    cuda_v = torch.version.cuda
    my_cu = _cuda_tag(cuda_v)
    cu_order: list[str] = []
    if my_cu:
        cu_order.append(my_cu)
    for x in ("cu128", "cu126", "cu124", "cu121", "cu118", "cu117"):
        if x not in cu_order:
            cu_order.append(x)

    torch_tags = [
        "2.6.0",
        "2.5.0",
        "2.4.0",
        "2.3.0",
        "2.2.0",
        "2.1.0",
        "2.0.0",
    ]
    tv = torch.__version__.split("+")[0].rsplit(".", 1)[0] + ".0"
    if tv not in torch_tags:
        torch_tags.insert(0, tv)

    for cu in cu_order:
        for torch_ver in torch_tags:
            url = f"https://download.openmmlab.com/mmcv/dist/{cu}/torch{torch_ver}/index.html"
            print("Trying prebuilt index:", url, flush=True)
            code = _run_pip(
                ["mmcv-full==1.7.2", "-f", url],
                quiet=False,
                timeout=900,
            )
            if code != 0:
                continue
            if _have_mmcv():
                return 0

    print(
        "No prebuilt mmcv-full wheel matched. Building from source (10–40+ min on Colab) ...",
        flush=True,
    )
    env = os.environ.copy()
    env["MMCV_WITH_OPS"] = "1"
    _run_pip(["ninja", "packaging"], quiet=True)
    try:
        code = subprocess.call(
            [sys.executable, "-m", "pip", "install", "mmcv-full==1.7.2"],
            env=env,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("Source build timed out.", flush=True)
        code = 1
    if code != 0:
        print("pip build mmcv-full failed.", flush=True)
        return code
    if _have_mmcv():
        return 0

    print(
        "Still cannot import mmcv. Try a Colab runtime with an older Python (3.10) or use conda.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
