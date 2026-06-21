# scail-2-mlx-swift — Porting Spec (S0–S7)

**Wan-family member, targeting MLXEngine.** SCAIL-2 is a Wan2.1-I2V-14B fork; it
joins bernini/ti2v/helios/vace/phantom in `WAN_DEV/` as a **`wan-core` consumer**
(local-path dep) and ends at an MLXEngine `ModelPackage` (S7). The substrate
(WanModel DiT blocks · 16-ch WanVAE + StreamingDecode · umT5-XXL · RoPE base ·
FlowUniPC/Euler/DPM++ · WeightLoader) is **reused, not re-ported** — only the
SCAIL delta is net-new. Validation home: the `WAN_TESTING` app + the masking tool
(SCAIL consumes 28-ch color-coded driving masks, a natural fit).

## Ground truth

| role | location |
|---|---|
| **Substrate (REUSE)** | `WAN_DEV/wan-core-mlx-swift` — `WanCore` module; bit-exact, family-shared |
| **Oracle (Python MLX)** | `/Volumes/DEV_ARCHIVE/scail-2-mlx` — parity-locked vs PyTorch (27 tests; DiT fp32-exact on CPU oracle; CLIP real-weights 2.7e-4; chunked VAE decode <5e-4/frame) |
| **Weights** | `xocialize/SCAIL-2-bf16` on HF / local `scail-2-mlx/weights/mlx/` — dit (bf16, 1307 keys), umt5 (bf16), clip (fp16), vae (fp32) |
| **Family analog** | `vace-mlx-swift` — the other net-new-branch consumer (Context Adapter); mirror its consumer/branch shape |
| **PT upstream** | `scail-2-mlx/refs/SCAIL-2` (branch `wan-scail2`) |

## Reuse map (consume wan-core; net-new = the SCAIL delta only)

`wan-core`'s `WanModel` is config-driven I2V via channel-concat `y:` (TI2V
mask-blend style); `WanCrossAttention` is **text-only**. SCAIL-2's CLIP-conditioned
i2v cross-attn + visual tower is therefore **genuinely net-new to the family**
(no existing consumer uses CLIP). The 16-ch `WanVAE` already carries
`StreamingDecode` with the `Rep` first-chunk cache — the same fix shipped as
mlx-video PR #38, so the Python oracle's `vae_stream.py` is already in wan-core.

| component | plan | source |
|---|---|---|
| Wan self-attn / FFN / norms / modulation | **REUSE** wan-core `Attention.swift` + `Transformer.swift`; SCAIL FFN keys `ffn.layers.{0,2}` already match | WanCore |
| umT5-XXL encoder + TextEncode | **REUSE** verbatim; verify bit-identity vs SCAIL `umt5.safetensors` by value-fingerprint at S1 | WanCore |
| 16-ch WanVAE + StreamingDecode (Rep cache) + StreamingEncode | **REUSE**; verify VAE bit-identity (fingerprint) + confirm frame mapping 1+(T−1)·4 | WanCore |
| FlowUniPC / Euler / DPM++ schedulers | **REUSE** wan-core `Scheduler.swift` | WanCore |
| WeightLoader / RoPE precompute base | **REUSE**; extend RoPE for 3-segment application | WanCore |
| WanConfig | **EXTEND** with SCAIL fields (maskDim 28, pose/mask embeds, i2v flag) — or a `SCAILConfig` wrapping `WanConfig` | WanCore + SCAIL2 |
| **3-segment ref/video/pose RoPE** | **NET-NEW** in SCAIL2 — translate oracle `rope_apply_{ref,video,pose}` + pose avg-pool; build freq tables in Swift `Double` before fp32 cast. (Heed family E16: multi-frame temporal-RoPE *application* drifts ~0.35% — gate `<0.02`, flag quality follow-up) | oracle `model_scail2.py` |
| **dual mask patch embeds** (`patch_embedding_pose`, `patch_embedding_mask`) | **NET-NEW** in SCAIL2 — Conv3d NDHWC | oracle |
| **i2v WanCrossAttention** (`k_img`/`v_img`/`norm_k_img`) + **`img_emb` MLPProj** | **NET-NEW** in SCAIL2 (SCAIL-local for now; promote to wan-core only if a 2nd Wan2.1-I2V consumer appears) | oracle |
| **CLIP xlm-roberta ViT-H visual tower** + bicubic resize matrices | **NET-NEW** in SCAIL2 — translate oracle `clip.py`; matrices in `Double` | oracle |
| **28-ch mask compression** + **segmented pipeline** (history overlap, dual-mask conditioning, CFG) | **NET-NEW** in SCAIL2 — translate oracle `scail.py` / `scail_utils.py` | oracle |

**Architecture note (flagged, default chosen):** the net-new i2v-CLIP surface
lives SCAIL-local, not in wan-core, since no other family member uses CLIP. If a
generic Wan2.1-I2V consumer later lands, promote `img_emb` + the i2v cross-attn
variant up to wan-core (the "net-new lands in wan-core so all consumers inherit"
rule). Revisit at S7.

## Phase gates

| phase | scope | gate | status |
|---|---|---|---|
| **S0** | scaffold; config decode; key contract vs `dit/umt5/clip/vae.safetensors` headers (Foundation-only, offline) | 0 missing / 0 unused per component | **PASSED 2026-06-12** — dit 1307 / umt5 242 / clip 393 / vae 194, generators cross-checked vs pinned fixture |
| **S0b** | **wan-core consumer pivot**: relocate into `WAN_DEV/`, depend on `WanCore`, builds green; fingerprint SCAIL `vae`/`umt5` vs a canonical sibling (Bernini) → confirm REUSE vs convert | builds; VAE/umT5 all-keys value-match (or documented delta) | in progress |
| **S1** | net-new forwards on real weights vs oracle fixtures (CPU stream): 3-seg RoPE units (both shift regimes + pose pool), dual-mask embeds, i2v cross-attn, CLIP visual, plus a reuse-sanity forward of the wan-core Wan block on SCAIL weights | oracle's own thresholds (≤1e-4 RoPE, ≤1e-3 component; multi-frame temporal-RoPE `<0.02` per family E16) | — |
| **S2** | few-step e2e golden: injected numpy noise/contexts, animation + replace modes, history on/off | match oracle e2e fixture ≤0.05 | — |
| **S2b** | tokenizer wiring (`google/umt5-xxl`) + ONE real GPU generation (animation_001, dpm++ 16) — the eyeball gate | visually clean; checkerboard detector clean | — |
| **S3** | multi-segment long video (history overlap); preprocessing (bicubic, mask compress) parity | bit-match oracle utils | — |
| **S5** | memory machinery: reuse wan-core decode-memory levers (StreamingDecode + `Memory.cacheLimit` cap); peak-`phys_footprint` report at 480p envelope → `residentBytes` | flat per-step active memory; measured phys | — |
| **S6** | quantized variant — **BLOCKED on Python side**: q4 fails its own gate (CPU-true cosine 0.9498 vs ≥0.99); q8 CPU verification pending. Port whichever tier the oracle certifies; cross-validate same-fixture | oracle's certified gate | blocked |
| **S7** | **MLXEngine wrap** (`MLXSCAIL2` target + MLXToolKit dep + engine dep): `ModelPackage`, capability (character-animation / textToVideo lane), two-layer license gate, C0–C13, measured `QuantFootprint`. Wire into `WAN_TESTING` app harness. | C0–C13 pass; runs in the Wan test app | target |

## Known traps carried from the Python port + donor

- **M5 GPU fp32 matmul ≈ 8e-4 relative (TF32-class).** ALL parity gates on the
  CPU stream; GPU-vs-CPU cosine noise floor measured at 0.99927 on the 14B —
  GPU cannot resolve quant-grade comparisons.
- **Metal watchdog family:** weight loads on CPU stream (`Device.withDefaultDevice(.cpu)`),
  never eval giant constant fills, ARC-scope big models, per-step eval + cache clear.
- **Long-run ops:** heavy gates / generations run detached
  (`nohup … & disown`), stderr progress markers, release builds for CPU gates.
- **SPM test product metallib is fragile** — every Metal-context gate is a CLI
  mode of `RunSCAIL2` (`--s1-gate` etc.); `swift test` keeps only never-eval tests.
- **Upstream zero-init head:** re-randomize zero/constant weights in any
  random-weights parity fixture (vacuous-comparison trap).
- mlx-video VAE decode semantics: use wan-core `StreamingDecode` (the `Rep`
  first-chunk cache); whole-sequence decode emits 4·T frames with a divergent
  head. Frame mapping caveat (family): `1+(T−1)·4` holds only for T>1; T=1→3 frames.
- **Family quirks that bite every wan-core consumer** (wan-video skill): NFKC the
  negative prompt (`precomposedStringWithCompatibilityMapping`); `ditDType:.float32`
  at video seqLen (bf16 NaNs ≥1024); a hand-written branch loop (the SCAIL 3-seg
  RoPE / dual-mask path) must replicate `runBlocks`' per-block `eval` at
  `seqLen ≥ wanLargeSeq` or it builds one unbounded fp32 graph (plateaued+glacial).
  16-ch VAE hits the large-seqLen wall *earlier* than vae22 (~4× tokens per H×W).

## Fixtures

`tools/dump_*.py` scripts run with the ORACLE'S venv
(`/Volumes/DEV_ARCHIVE/scail-2-mlx/.venv`), dumping `.npy` (bf16 as fp32, ids
int32) into `Tests/SCAIL2Tests/Fixtures/`. RNG note: mx seed streams are
bit-identical across bindings — pin with a one-shot seeded-normal test, keep
injection as belt-and-braces.
