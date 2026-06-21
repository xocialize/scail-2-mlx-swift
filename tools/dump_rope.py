#!/usr/bin/env python
"""S1 RoPE fixtures — run with the ORACLE venv:

  /Volumes/DEV_ARCHIVE/scail-2-mlx/.venv/bin/python tools/dump_rope.py

Dumps, for a tiny config and BOTH shift regimes (animation + replace), the
freqs table + input + `rope_apply_scail` output as .npy (fp32) plus a JSON
sidecar of scalar params, into Tests/SCAIL2Tests/Fixtures/rope/.

The Swift S1 gate reuses wan-core `ropeParams` for the freq table (band
construction is identical) and translates only the 3-segment application.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ORACLE = Path("/Volumes/DEV_ARCHIVE/scail-2-mlx")
OUT = Path(__file__).resolve().parents[1] / "Tests/SCAIL2Tests/Fixtures/rope"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ORACLE))
import mlx.core as mx  # noqa: E402

# model_scail2 imports `.attention` at module top — stub it (rope needs no GPU).
pkg = types.ModuleType("wr"); pkg.__path__ = []
mp = types.ModuleType("wr.modules"); mp.__path__ = []
attn = types.ModuleType("wr.modules.attention"); attn.flash_attention = lambda *a, **k: None
sys.modules.update({"wr": pkg, "wr.modules": mp, "wr.modules.attention": attn})
spec = importlib.util.spec_from_file_location(
    "wr.modules.model_scail2", ORACLE / "scail2_mlx/modules/model_scail2.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
spec.loader.exec_module(m)

# Tiny config: heads=4, head_dim=32 -> c=16, band split [6,5,5].
N, D = 4, 32
F_, H_, W_ = 4, 8, 8


def _band_f64(max_seq_len, dim, theta=10000.0):
    # float64 freq table (numpy) — matches the PyTorch upstream (torch.polar,
    # float64) AND wan-core's ropeParams (Double loop). The Python oracle's own
    # rope_params uses float32; the canonical Swift/family path is float64, so
    # the FIXTURE uses float64 (skill: compute freq tables in Double before fp32).
    pos = np.arange(max_seq_len, dtype=np.float64)[:, None]
    inv = 1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim)[None, :]
    f = pos * inv
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)


def mx_freqs(d=D):
    ct, st = _band_f64(8192, d - 4 * (d // 6))
    ch, sh = _band_f64(8192, 2 * (d // 6))
    cw, sw = _band_f64(8192, 2 * (d // 6))
    cos = mx.array(np.concatenate([ct, ch, cw], axis=1))
    sin = mx.array(np.concatenate([st, sh, sw], axis=1))
    return cos, sin


def shift_kwargs(replace):
    return dict(
        rope_T=F_, rope_H=H_, rope_W=W_,
        rope_T_shift={"ref": 0, "pose": 0 if replace else 1, "video": 0 if replace else 1},
        rope_H_shift={"ref": 120 if replace else 0, "pose": 0, "video": 0},
        rope_W_shift={"ref": 0, "pose": 120, "video": 0},
    )


def npy(name, arr):
    np.save(OUT / f"{name}.npy", np.asarray(arr, dtype=np.float32))


# Freq table (cos, sin) — what wan-core ropeParams must reproduce.
cos, sin = mx_freqs()
npy("freqs_cos", cos)
npy("freqs_sin", sin)

ref_len, vid_len = H_ * W_, F_ * H_ * W_
pose_len = F_ * (H_ // 2) * (W_ // 2)
seq = ref_len + vid_len + pose_len
rng = np.random.default_rng(7)
x = rng.standard_normal((2, seq, N, D)).astype("float32")
npy("x", x)

for regime, replace in [("anim", False), ("replace", True)]:
    kw = shift_kwargs(replace)
    kw.update(ref_length=ref_len, seq_length=vid_len, pose_length=pose_len)
    out = m.rope_apply_scail(mx.array(x), freqs=(cos, sin), **kw)
    npy(f"out_{regime}", out)

meta = dict(
    numHeads=N, headDim=D, F=F_, H=H_, W=W_,
    refLen=ref_len, vidLen=vid_len, poseLen=pose_len, seq=seq,
    halfD=D // 2,
)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("wrote rope fixtures to", OUT, meta)
