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
>
> **Current state:** S0 only. `Sources/SCAIL2/` holds the config, key
> contract, and safetensors-header reader (`SCAIL2Config.swift`,
> `KeyContract.swift`, `SafetensorsHeader.swift`); the model and pipeline are
> not written yet. `Sources/RunSCAIL2/` has the S0 Metal-context gate
> (`S0Gate.swift`); later `--sN` gates land as the port progresses. `tools/`
> (fixture dumpers) is currently an empty placeholder.

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
Sources/SCAIL2/        # core: config, key contract, safetensors header (S0);
                       #   models + pipeline land at S1+
Sources/RunSCAIL2/     # CLI: RunSCAIL2 + Metal-context gates (S0Gate now; --sN later)
Tests/SCAIL2Tests/     # never-eval tests only (config, key contract; Fixtures/)
tools/                 # fixture dumpers (empty placeholder; run with the oracle venv)
```

## Build

```bash
swift build
swift test          # never-eval tests (config + key contract)
```

Targets (`Package.swift`): library `SCAIL2` and executable `RunSCAIL2`.
Platform `macOS 15+`. Dependencies: `ml-explore/mlx-swift` (≥0.30.0) and
`huggingface/swift-transformers` (umT5 sentencepiece tokenizer only; weight
download is the port's own loader).

## License

Apache-2.0. Derived from SCAIL-2 (Zhipu AI), Wan2.1 (Alibaba), and the
scail-2-mlx Python port. See NOTICE.
