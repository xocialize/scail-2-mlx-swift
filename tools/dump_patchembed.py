#!/usr/bin/env python
"""S1 patch-embed fixtures — ORACLE venv.

The SCAIL dual-mask patch embeds are non-overlapping Conv3d (kernel=stride=
patch_size). wan-core does its single patch embed as a reshaped linear with a
different key, so SCAIL owns its three Conv3d embeds. This dumps random-init
Conv3d weights + an NDHWC input + the conv output for two channel widths
(20-ch latent path, 28-ch mask path), so the Swift Conv3d wrapper is gated
directly. Weights stored NDHWC [O,kt,kh,kw,I] — the layout both nn.Conv3d
(MLX) and MLX-swift Conv3d use, and what SCAIL's checkpoint stores.
"""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/patchembed"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

DIM = 8
PATCH = (1, 2, 2)
T, H, W = 2, 4, 4


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


def dump(tag, inCh):
    mx.random.seed(hash(tag) % 10_000)
    conv = nn.Conv3d(inCh, DIM, kernel_size=PATCH, stride=PATCH)
    rng = np.random.default_rng(len(tag))
    x_ncdhw = rng.standard_normal((1, inCh, T, H, W)).astype("float32")  # NCDHW
    x_ndhwc = mx.array(x_ncdhw).transpose(0, 2, 3, 4, 1)
    out = conv(x_ndhwc)  # NDHWC out
    mx.eval(out, conv.weight, conv.bias)
    npy(f"{tag}_weight", conv.weight)   # [O,kt,kh,kw,I]
    npy(f"{tag}_bias", conv.bias)
    npy(f"{tag}_x", x_ncdhw)            # NCDHW (Swift transposes)
    npy(f"{tag}_out", out)             # NDHWC
    return inCh


widths = {"latent": dump("latent", 20), "mask": dump("mask", 28)}
(OUT / "meta.json").write_text(json.dumps(
    {"dim": DIM, "patch": list(PATCH), "T": T, "H": H, "W": W, "widths": widths}, indent=1))
print("wrote patchembed fixtures", widths)
