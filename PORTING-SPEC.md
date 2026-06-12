# scail-2-mlx-swift — Porting Spec (S0–S7)

**Standalone port.** MLXEngine integration (S7) is explicitly deferred: the 14B
pipeline (~34 GB active bf16) is too heavy for engine admission as-is. This repo
iterates standalone to discover the right usage surface (preset configs, memory
modes, maybe a trimmed capability) before any `ModelPackage` wrap.

## Ground truth

| role | location |
|---|---|
| **Oracle (Python MLX)** | `/Volumes/DEV_ARCHIVE/scail-2-mlx` — parity-locked vs PyTorch (27 tests; DiT fp32-exact on CPU oracle; CLIP real-weights 2.7e-4; chunked VAE decode <5e-4/frame) |
| **Weights** | `xocialize/SCAIL-2-bf16` on HF / local `scail-2-mlx/weights/mlx/` — dit (bf16, 1307 keys), umt5 (bf16), clip (fp16), vae (fp32) |
| **Swift donor** | `~/Development/MLXEngine/bernini-r-mlx-swift` — Wan2.2-family, S0–S6 parity-locked 2026-06-12 (Transformer/Attention/UMT5/UniPC/RoPE machinery) |
| **PT upstream** | `scail-2-mlx/refs/SCAIL-2` (branch `wan-scail2`) |

## Donor map (decide lift-vs-translate at S0 by KEY PATHS, not architecture)

| component | plan |
|---|---|
| Wan attention block / FFN / norms | bernini `Transformer.swift` — likely lift; verify `blocks.N.self_attn.q` / `ffn.layers.{0,2}` paths against `dit.safetensors` headers |
| umT5-XXL encoder | bernini `UMT5EncoderModel.swift` — lift candidate; same mlx-video lineage. Check donor LOAD-TIME dtype policy |
| FlowUniPC + DPM++ schedulers | bernini `Scheduler.swift` — bit-exact in donor; lift + add dpm++ if absent |
| 3-segment SCAIL RoPE | **translate from oracle** `scail2_mlx/modules/model_scail2.py` (rope_apply_ref/video/pose + pose avg-pool); donor `RoPESA3D.swift` is the Swift-idiom reference only. Frequency tables in Swift `Double` before fp32 cast |
| i2v cross-attn (k_img/v_img), dual mask patch embeds | **translate from oracle** (net-new vs donor) |
| CLIP xlm-roberta ViT-H visual | **translate from oracle** `modules/clip.py` incl. bicubic resize matrices (build in `Double`) |
| Wan**2.1** VAE + chunked causal decode (Rep sentinel) | **translate** — donor has Wan2.2 VAE (different). Source: oracle `utils/vae_stream.py` + mlx-video `wan_2/vae.py` (post-PR-#38 semantics: T latents -> 1+(T−1)·4 frames) |
| 28-ch mask compression, pipeline (segments + history), CFG | **translate from oracle** `scail.py` / `utils/scail_utils.py` |

## Phase gates

| phase | scope | gate | status |
|---|---|---|---|
| **S0** | scaffold; config decode; key contract vs `dit/umt5/clip/vae.safetensors` headers (Foundation-only, offline); donor key-path comparison → lift/translate decisions | 0 missing / 0 unused per component | **PASSED 2026-06-12** — dit 1307 / umt5 242 / clip 393 / vae 194, generators cross-checked vs pinned fixture. Donor verdicts: **umT5 LIFT verbatim** (native key match incl. `gate_proj`/`pos_embedding`); **schedulers LIFT** (donor ships Euler+DPM++(2M)+UniPC); **Wan block LIFT + adapt** — FFN keys are `ffn.layers.{0,2}` here vs donor `fc1/fc2` (fix natively with literal "0"/"2" ModuleInfo keys under a "layers" child, NO load-time remap), plus i2v `k_img/v_img/norm_k_img` and affine `norm3`; RoPE/CLIP/VAE2.1/pipeline TRANSLATE from oracle as planned |
| **S1** | substrate forwards on real weights vs oracle fixtures (CPU stream): Wan block, umT5, CLIP visual, VAE encode + chunked decode, RoPE units (incl. both shift regimes + pose pool) | oracle's own thresholds (≤1e-4 RoPE, ≤1e-3 component) | — |
| **S2** | few-step e2e golden: injected numpy noise/contexts, animation + replace modes, history on/off | match oracle e2e fixture ≤0.05 | — |
| **S2b** | tokenizer wiring (`google/umt5-xxl`) + ONE real GPU generation (animation_001, dpm++ 16) — the eyeball gate | visually clean; checkerboard detector clean | — |
| **S3** | multi-segment long video (history overlap); preprocessing (bicubic, mask compress) parity | bit-match oracle utils | — |
| **S5** | memory machinery: per-step eval + cache clear (Metal watchdog discipline); peak-memory report at 480p envelope | flat per-step active memory | — |
| **S6** | quantized variant — **BLOCKED on Python side**: q4 fails its own gate (CPU-true cosine 0.9498 vs ≥0.99); q8 CPU verification pending. Port whichever tier the oracle certifies; cross-validate same-fixture | oracle's certified gate | blocked |
| **S7** | engine wrap (`ModelPackage`, C0–C13, memory report) | **deferred by design** | deferred |

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
- mlx-video VAE decode semantics: use the chunked causal path (oracle
  `vae_stream.py`); whole-sequence decode emits 4·T frames with a divergent head.

## Fixtures

`tools/dump_*.py` scripts run with the ORACLE'S venv
(`/Volumes/DEV_ARCHIVE/scail-2-mlx/.venv`), dumping `.npy` (bf16 as fp32, ids
int32) into `Tests/SCAIL2Tests/Fixtures/`. RNG note: mx seed streams are
bit-identical across bindings — pin with a one-shot seeded-normal test, keep
injection as belt-and-braces.
