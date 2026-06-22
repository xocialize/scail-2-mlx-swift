#!/usr/bin/env python
"""S2 mask-compress + half-bilinear fixtures — ORACLE venv, CPU stream."""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/maskcompress"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)
from scail2_mlx.utils.scail_utils import extract_and_compress_mask_to_latent  # noqa: E402
from scail2_mlx.scail import _half_bilinear  # noqa: E402


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


# mask: [3, T, H, W] in [-1,1] — H,W divisible by 8 (3x /2). T such that 4k+1.
T, H, W = 9, 64, 64
rng = np.random.default_rng(11)
raw = rng.choice([0, 200, 230, 255], size=(3, T, H, W)).astype("float32")
mask = (raw - 127.5) / 127.5
out = extract_and_compress_mask_to_latent(mx.array(mask), additional_spatial_downsample=1)
npy("mask", mask)
npy("compress_out", out)

# half-bilinear: [T, C, H, W]
hb_in = rng.standard_normal((5, 3, 48, 96)).astype("float32")
npy("hb_in", hb_in)
npy("hb_out", _half_bilinear(mx.array(hb_in)))

(OUT / "meta.json").write_text(json.dumps({"T": T, "H": H, "W": W}, indent=1))
print("wrote maskcompress fixtures; compress_out", list(np.array(out).shape))
