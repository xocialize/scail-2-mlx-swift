#!/usr/bin/env python
"""S1 CLIP ViT fixtures — ORACLE venv, CPU stream (fast SDPA carries M5 GPU
noise on either side — see the i2v lesson).

SCAIL's image conditioning is the open-clip xlm-roberta ViT-H/14 VISUAL tower,
called with use_31_block=True (returns the 2nd-to-last block output, [B,257,
1280] at production). Net-new to the Wan family. Tiny config here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/clip"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten  # noqa: E402
from scail2_mlx.modules.clip import VisionTransformer  # noqa: E402

CFG = dict(
    image_size=28, patch_size=14, dim=32, mlp_ratio=4, out_dim=16,
    num_heads=2, num_layers=3, pool_type="token", pre_norm=True,
    post_norm=False, activation="gelu",
)


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


mx.random.seed(0)
vit = VisionTransformer(**CFG)
mx.eval(vit.parameters())

rng = np.random.default_rng(5)
x = rng.standard_normal((1, 3, CFG["image_size"], CFG["image_size"])).astype("float32")
out = vit(mx.array(x), use_31_block=True)  # [B, tokens, dim], skips last block
mx.eval(out)

for key, val in tree_flatten(vit.parameters()):
    npy("w_" + key.replace(".", "__"), val)
npy("x", x)
npy("out", out)
(OUT / "meta.json").write_text(json.dumps(CFG, indent=1))
print("wrote clip fixtures", {k: CFG[k] for k in ("dim", "num_layers", "image_size")},
      "out", list(out.shape))
