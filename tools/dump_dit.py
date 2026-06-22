#!/usr/bin/env python
"""S2 full-DiT fixture — ORACLE venv, CPU stream.

Tiny-config SCAIL2Model forward (mirrors the Python test_model_forward_parity):
random weights (zero-init head/modulation re-randomized — the vacuous-comparison
trap), single batch, both shift regimes via replace_flag, history off. Dumps all
state-dict tensors under their checkpoint keys + the inputs + output.
"""
import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/dit"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ORACLE))

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten  # noqa: E402
from scail2_mlx.modules.model_scail2 import SCAIL2Model  # noqa: E402

CFG = dict(
    model_type="i2v", patch_size=(1, 2, 2), text_len=512, in_dim=20, mask_dim=28,
    dim=64, ffn_dim=128, freq_dim=32, text_dim=48, out_dim=16, num_heads=4,
    num_layers=2, qk_norm=True, cross_attn_norm=True,
)
T_, H_, W_ = 2, 8, 8


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


mx.random.seed(0)
model = SCAIL2Model(**CFG)
mx.eval(model.parameters())

# re-randomize zero/constant-init weights (head.head, modulations) so the
# comparison isn't vacuous (the upstream init zeros the head)
rng = np.random.default_rng(99)
flat = dict(tree_flatten(model.parameters()))
new = {}
for k, val in flat.items():
    a = np.array(val)
    if np.abs(a).sum() == 0 or a.std() == 0:
        a = (0.02 * rng.standard_normal(a.shape)).astype(np.float32)
    new[k] = mx.array(a)
from mlx.utils import tree_unflatten  # noqa: E402
model.update(tree_unflatten(list(new.items())))
mx.eval(model.parameters())

g = lambda shape, seed: mx.array(np.random.default_rng(seed).standard_normal(shape).astype("float32"))
inp = dict(
    x=g((20, T_, H_, W_), 1),                # in_dim 20 (16 + 4 i2v mask appended in fwd-> but tiny uses inDim=20 already)
    ref_latents=g((16, 1, H_, W_), 2),       # ref gets ones-mask -> 20
    ref_masks=g((28, 1 + T_, H_, W_), 3),
    pose_latents=g((16, T_, H_ // 2, W_ // 2), 4),  # pose gets ones-mask -> 20
    driving_masks=g((28, T_, H_ // 2, W_ // 2), 5),
    clip_fea=g((1, 257, 1280), 6),
    context=g((7, CFG["text_dim"]), 8),
    t=mx.array(np.array([500.0], dtype="float32")),
)
# NOTE: oracle appends 4 mask channels to x/ref/pose inside forward, so the
# raw x here is 16-ch (zeros-mask -> 20). Fix: x raw is 16-ch.
inp["x"] = g((16, T_, H_, W_), 1)

for replace in [False, True]:
    out = model(
        x=[inp["x"]], pose_latents=[inp["pose_latents"]], driving_masks=[inp["driving_masks"]],
        ref_latents=[inp["ref_latents"]], ref_masks=[inp["ref_masks"]], t=inp["t"],
        context=[inp["context"]], seq_len=int(1e10), replace_flag=replace,
        clip_fea=inp["clip_fea"],
    )[0]
    mx.eval(out)
    npy(f"out_{'replace' if replace else 'anim'}", out)

for k, val in tree_flatten(model.parameters()):
    npy("w_" + k.replace(".", "__"), val)
for k, val in inp.items():
    npy("in_" + k, val)
(OUT / "meta.json").write_text(json.dumps({**{k: CFG[k] for k in (
    "patch_size", "text_len", "in_dim", "mask_dim", "dim", "ffn_dim", "freq_dim",
    "text_dim", "out_dim", "num_heads", "num_layers")}, "T": T_, "H": H_, "W": W_}, indent=1))
print("wrote dit fixtures; out shape", list(np.load(OUT / "out_anim.npy").shape))
