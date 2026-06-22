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
| **S0b** | **wan-core consumer pivot**: relocate into `WAN_DEV/`, depend on `WanCore`, builds green; structural-reuse gate (instantiate wan-core `WanVAE`/`UMT5EncoderModel` weight-free, key sets vs SCAIL safetensors) | builds; VAE/umT5 0 missing / 0 unused | **PASSED 2026-06-15** — builds green vs WanCore; `RunSCAIL2 --s0b-gate`: vae 194 + umt5 242 keys both 0/0 → **REUSE substrate loaders verbatim**. Provenance (stock Wan2.1_VAE.pth = family 16-ch VAE) covers values; permutation-invariant value fingerprint vs a downloaded sibling is the S1 belt-and-braces |
| **S1** | net-new forwards vs oracle fixtures (CPU stream). **3-seg RoPE ✅** (`--s1-rope-gate`: freq-table reuse 0.0 + anim/replace 0.0; `RoPE3Seg` reuses wan-core `ropeParams`). **dual-mask Conv3d embeds ✅** (`--s1-patchembed-gate`: latent 20-ch + mask 28-ch, ~3e-7; `SCAILPatchEmbeds` = MLX-swift Conv3d). **i2v cross-attn ✅** (`--s1-i2vcross-gate`: 0.0; dumper pinned to mx.cpu — fast SDPA carries M5 GPU noise either side). **CLIP ViT-H ✅** (`--s1-clip-gate`: 0.0; `CLIPVisionTower` use_31_block path, exact-GELU mlp via `Sequential` → native `layers.0/.2`). **ALL 4 NET-NEW LOCKED (0.0).** Remaining: wan-block reuse-sanity (low-risk); CLIP preprocessing bicubic matrices are pipeline-level (S2/S3) | ≤1e-4 RoPE / ≤1e-3 component | **net-new ✅** |
| **S2** | full DiT forward ✅ (`--s2-dit-gate`: anim 1.7e-6 / replace 1.6e-6, 91 params verify .all). **All net-new pipeline utils ✅**: mask compress + half-bilinear 0.0 (`--s2-mask-gate`), CLIP bicubic preprocess ~1e-7 (`--s2-clippre-gate`, Double-built matrices). **Every computational unit parity-locked.** Remaining: pipeline ORCHESTRATION — wire reused wan-core (FlowUniPC/DPM++ scheduler, WanVAE encode/decode, umT5+tokenizer) + proven components into the segment loop + CFG → first generation. This converges with S2b. | units ✅; e2e ≤0.05 | utils done |
| **S2b** | tokenizer wiring (`google/umt5-xxl`) + ONE real GPU generation (animation_001, dpm++ 16) — the eyeball gate | visually clean; checkerboard detector clean | — |
| **S3** | multi-segment long video (history overlap); preprocessing (bicubic, mask compress) parity | bit-match oracle utils | — |
| **S5** | memory machinery: reuse wan-core decode-memory levers (StreamingDecode + `Memory.cacheLimit` cap); peak-`phys_footprint` report at 480p envelope → `residentBytes` | flat per-step active memory; measured phys | — |
| **S6** | quantized variant — **BLOCKED on Python side**: q4 fails its own gate (CPU-true cosine 0.9498 vs ≥0.99); q8 CPU verification pending. Port whichever tier the oracle certifies; cross-validate same-fixture | oracle's certified gate | blocked |
| **S7** | **MLXEngine wrap** (`MLXSCAIL2` target + MLXToolKit + engine dep): `ModelPackage` (C13 engine-owned lifecycle, `@InferenceActor`), `SCAIL2Configuration` (C9 Codable/defaultable), `RequirementsManifest` with measured per-quant `residentBytes` (C10), two-layer license gate (C7 weight Apache-2.0 / C8 port-code Apache-2.0), contract version (C0), `@unknown default` discipline (C12). Wire into `WAN_TESTING` app harness; quantify a real run. | C0–C13 pass; runs in Wan test app | target |

### S7 open design question (STOP-AND-ASK before wrapping)

**Which canonical capability?** SCAIL-2's I/O is *ref image + driving video → video
performing that motion* — motion transfer, not text-to-video and not editing the
driving clip. The 1.2.0 enum (18 cases: `textToVideo`, `imageEdit`, `videoAnalysis`,
…) has no clean fit; `videoEdit` is the nearest but wrong semantics (we don't edit
the driving video). Wan2.2-Animate is the same shape (family table: "different lane,
new capability"). Per the skill, **a capability not in the enum is core-owned →
stop-and-ask the engine maintainer**. Options to put to them: (a) new
`characterAnimation` capability, (b) a motion-transfer `videoToVideo`, or (c) map to
an existing case with mode/specialty tags. Resolve WITH the maintainer at S7 entry,
shared with the Animate lane (don't invent an enum case unilaterally). Driving-mask
input arrives via the user's masking tool (28-ch color-coded), a SCAIL request field.

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
