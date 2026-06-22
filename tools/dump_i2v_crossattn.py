#!/usr/bin/env python
"""S1 i2v cross-attn fixtures — ORACLE venv.

SCAIL's cross-attn is Wan2.1-I2V: CLIP image tokens prepended to the text
context, attended via a separate k_img/v_img path sharing q, summed with the
text attention. Net-new to the Wan family (wan-core cross-attn is text-only).
Dumps the 9 projection weights + input x + context (img tokens ‖ 512 text
tokens) + oracle output, tiny config.
"""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/i2vcross"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)  # exact fp32 — match the Swift CPU-stream gate
from mlx.utils import tree_flatten  # noqa: E402
from scail2_mlx.modules.model_scail2 import WanI2VCrossAttention, T5_CONTEXT_TOKEN_NUMBER  # noqa: E402

DIM, HEADS = 64, 4
L1 = 32          # query length
IMG_LEN = 8      # CLIP image tokens
TEXT_LEN = T5_CONTEXT_TOKEN_NUMBER  # 512, hardcoded upstream


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


mx.random.seed(0)
attn = WanI2VCrossAttention(DIM, HEADS, qk_norm=True)
# materialize all params
mx.eval(attn.parameters())

rng = np.random.default_rng(3)
x = rng.standard_normal((1, L1, DIM)).astype("float32")
context = rng.standard_normal((1, IMG_LEN + TEXT_LEN, DIM)).astype("float32")

out = attn(mx.array(x), mx.array(context), None)
mx.eval(out)

# dump weights under their SCAIL key names (cross_attn.* without the prefix)
for key, val in tree_flatten(attn.parameters()):
    npy("w_" + key.replace(".", "__"), val)

npy("x", x)
npy("context", context)
npy("out", out)
(OUT / "meta.json").write_text(json.dumps(
    {"dim": DIM, "heads": HEADS, "L1": L1, "imgLen": IMG_LEN, "textLen": TEXT_LEN}, indent=1))
print("wrote i2vcross fixtures", {"dim": DIM, "heads": HEADS, "imgLen": IMG_LEN})
