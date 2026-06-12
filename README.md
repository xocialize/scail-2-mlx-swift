# scail-2-mlx-swift

Standalone Swift/MLX port of **SCAIL-2** (zai-org) — end-to-end controlled
character animation (reference character image + driving video → animated
video) on Apple Silicon.

> ## ⚠️ Status: S0 scaffold — port not started
>
> This repo exists to iterate on SCAIL-2 in Swift **standalone**: the 14B
> pipeline (~34 GB active at bf16) is deliberately **not** wrapped for
> MLXEngine yet — it's too heavy for engine admission as-is. We learn the
> right usage surface here first (presets, memory modes, possibly a trimmed
> capability), then decide what an engine wrap should even look like.
> See [PORTING-SPEC.md](PORTING-SPEC.md) for the S0–S7 plan, donor map,
> and carried-over traps.

## Lineage

- **Oracle:** [`xocialize/scail-2-mlx`](https://github.com/xocialize/scail-2-mlx)
  (Python MLX, parity-locked vs PyTorch; weights at
  [`xocialize/SCAIL-2-bf16`](https://huggingface.co/xocialize/SCAIL-2-bf16))
- **Swift donor:** `bernini-r-mlx-swift` (Wan2.2 family — transformer blocks,
  umT5, bit-exact UniPC)
- **Upstream:** [zai-org/SCAIL-2](https://github.com/zai-org/SCAIL-2)
  (Wan2.1-I2V-14B fork; arXiv 2512.05905)

## Layout

```
Sources/SCAIL2/        # core: models, pipeline (S1+)
Sources/RunSCAIL2/     # CLI: generation + all Metal-context gates (--sN-gate)
Tests/SCAIL2Tests/     # never-eval tests only (key paths, config, scalars)
tools/                 # fixture dumpers (run with the oracle's Python venv)
```

## License

Apache-2.0. Derived from SCAIL-2 (Zhipu AI), Wan2.1 (Alibaba), and the
scail-2-mlx Python port. See NOTICE.
