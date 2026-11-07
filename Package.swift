// swift-tools-version: 6.2
// scail-2-mlx-swift — Swift/MLX port of SCAIL-2 (zai-org): end-to-end controlled
// character animation (reference image + driving video -> video). A Wan-family
// member: a Wan2.1-I2V-14B fork consuming the shared `wan-core` substrate
// (WanModel DiT · 16-ch WanVAE + StreamingDecode · umT5-XXL · RoPE · schedulers).
//
// Net-new SCAIL delta over the substrate (lives in the SCAIL2 target): 3-segment
// ref/video/pose RoPE, dual mask patch embeddings, CLIP-conditioned i2v
// cross-attention + the open-clip xlm-roberta ViT-H visual tower, and the
// segmented animation pipeline. Python oracle (parity-locked): DEV_ARCHIVE/scail-2-mlx.
//
// The MLXEngine wrapper target (`MLXSCAIL2` + MLXToolKit dep) lands at S7 — kept
// out of the graph through the parity phases to keep iteration light. See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "SCAIL2",
    platforms: [
        // v26 to match the MLXEngine contract (MLXToolKit) the S7 wrapper will link.
        .macOS(.v26)
    ],
    products: [
        .library(name: "SCAIL2", targets: ["SCAIL2"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the SCAIL pipeline.
        .library(name: "MLXSCAIL2", targets: ["MLXSCAIL2"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", "0.32.3"..<"0.33.0"),  // NAX split-K fix (AB-R-0778)
        // Tokenizers (umT5 sentencepiece) only; weight download is our own loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // The neutral Wan substrate, shared with bernini/ti2v/helios/vace/phantom.
        // Local path dep; a wan-core edit recompiles into every consumer.
        .package(url: "https://github.com/xocialize/wan-core-mlx-swift", from: "0.6.0"),
        // MLXEngine contract (MLXToolKit) for the S7 wrapper target; ≥0.27.0 for the CAN
        // cancellation gate (MLXServeConformance.CancellationConformance).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.27.0"),
    ],
    targets: [
        .target(
            name: "SCAIL2",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/SCAIL2"
        ),
        .target(
            name: "MLXSCAIL2",
            dependencies: [
                "SCAIL2",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXSCAIL2"
        ),
        .executableTarget(
            name: "RunSCAIL2",
            dependencies: [
                "SCAIL2",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
            ],
            path: "Sources/RunSCAIL2"
        ),
        .testTarget(
            name: "SCAIL2Tests",
            dependencies: ["SCAIL2"],
            path: "Tests/SCAIL2Tests",
            resources: [.copy("Fixtures")]
        ),
        // Wrapper conformance (offline): the CAN cancellation gate.
        .testTarget(
            name: "MLXSCAIL2Tests",
            dependencies: [
                "MLXSCAIL2",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXSCAIL2Tests"
        ),
    ]
)
