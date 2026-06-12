// swift-tools-version: 6.2
// scail-2-mlx-swift — STANDALONE Swift/MLX port of SCAIL-2 (zai-org): end-to-end
// controlled character animation (reference image + driving video -> video).
// Deliberately NOT MLXEngine-integrated yet (the 14B pipeline is too heavy for
// engine admission as-is); this repo iterates on the model standalone to learn
// the right usage surface first. Python oracle: DEV_ARCHIVE/scail-2-mlx
// (parity-locked vs PyTorch). Swift component donor: bernini-r-mlx-swift
// (Wan2.2 family, S0-S6 parity-locked). See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "SCAIL2",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "SCAIL2", targets: ["SCAIL2"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Tokenizers (umT5 sentencepiece) only; weight download is our own loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "SCAIL2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/SCAIL2"
        ),
        .executableTarget(
            name: "RunSCAIL2",
            dependencies: ["SCAIL2"],
            path: "Sources/RunSCAIL2"
        ),
        .testTarget(
            name: "SCAIL2Tests",
            dependencies: ["SCAIL2"],
            path: "Tests/SCAIL2Tests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
