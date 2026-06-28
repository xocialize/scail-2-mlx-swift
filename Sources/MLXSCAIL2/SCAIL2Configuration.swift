import Foundation
import MLXToolKit

/// Init-time configuration for `MLXSCAIL2Package` (C9): which variant + where the flat checkpoint
/// lives. Per-request inputs (reference image, driving video, mask, prompt, size, steps) ride the
/// canonical `CharacterAnimationRequest`, not here.
///
/// SCAIL-2's DiT runs **fp32** (the Wan-family rule — bf16 NaNs the DiT at video sequence lengths);
/// the on-disk weights are bf16 (`xocialize/SCAIL-2-bf16`) and upcast at load. `quant` therefore
/// describes the *stored* checkpoint, not the runtime dtype. int4 is BLOCKED upstream (the Python
/// q4 fails its own parity gate), so the only shipping variant today is `.bf16`.
public struct SCAIL2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var repo: String
    public var revision: String?
    public var quant: Quant
    /// Resolved local checkpoint folder (holds dit/vae/umt5/clip safetensors). Environment-specific
    /// → excluded from `Codable`.
    public var modelDirectory: URL?
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "xocialize/SCAIL-2-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the resolved flat checkpoint (dit/vae/umt5/clip)
/// into the OS file cache before `load()`'s GPU evals, so a cold load-time `eval` never faults
/// weights off slow/external storage inside a live Metal command buffer (the cold-load GPU watchdog,
/// `kIOGPUCommandBufferCallbackErrorTimeout`). SCAIL-2's loaders already page on the CPU stream
/// (`Device.withDefaultDevice(.cpu)`) and a cold 32.8 GB DiT was measured not to trip the watchdog
/// (102 GB peak run), so this is belt-and-suspenders. The whole resolved `modelDirectory` is paged;
/// only the config knows the path, execution is the engine's (`WeightPrewarmer`, best-effort). Nil on
/// the HF-download path → no-op.
extension SCAIL2Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] { [modelDirectory].compactMap { $0 } }
}
