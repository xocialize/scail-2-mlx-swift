#!/usr/bin/env python
"""S2 CLIP-preprocess fixtures — ORACLE venv, CPU stream. Bicubic matrices +
full CLIPModel.visual preprocessing (resize 224 + normalize)."""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/clippre"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)
from scail2_mlx.modules.clip import bicubic_resize_matrix, _CLIP_MEAN, _CLIP_STD  # noqa: E402


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


# bicubic matrices for a few in-sizes -> 224
for h in (123, 224, 360):
    npy(f"bicubic_{h}", bicubic_resize_matrix(h, 224))

# full preprocess: img [C,1,H,W] in [-1,1] (the pipeline's img[:,None,:,:])
rng = np.random.default_rng(6)
img = np.tanh(rng.standard_normal((3, 1, 90, 70))).astype("float32")
npy("pre_in", img)
size = 224
mh = bicubic_resize_matrix(90, size)
mw = bicubic_resize_matrix(70, size)
x = mx.array(img).transpose(1, 0, 2, 3)  # [1,3,H,W]
v = mx.matmul(mx.matmul(mh, x), mw.T)
mean = mx.array(_CLIP_MEAN).reshape(1, 3, 1, 1)
std = mx.array(_CLIP_STD).reshape(1, 3, 1, 1)
out = (v * 0.5 + 0.5 - mean) / std
npy("pre_out", out)

(OUT / "meta.json").write_text(json.dumps({"size": size, "sizes": [123, 224, 360]}, indent=1))
print("wrote clippre fixtures; pre_out", list(np.array(out).shape))
