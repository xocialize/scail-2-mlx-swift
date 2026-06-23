import Foundation
import SCAIL2

/// The resident SCAIL-2 runtime: owns the loaded models (SCAILModel DiT + wan-core WanVAE + umT5 +
/// CLIP tower + tokenizer) and runs the full media → video path, so `MLXSCAIL2Package` stays a thin
/// contract adapter (the same split Phantom uses with `PhantomPipeline`).
///
/// ## Scaffold status (S7 step 3 — contract side)
/// The contract adapter (`MLXSCAIL2Package`), `SCAIL2Configuration`, and the manifest are complete
/// and build/register against MLXToolKit. This runtime is the **step-4 seam**: `fromPretrained` and
/// `generate` declare the real surface but are not wired yet, because the weight loaders
/// (`loadDiT`/`loadVAE`/`loadUMT5`/`loadCLIP`/`encodeText`) and media IO
/// (`loadImageCHW`/`loadVideoTCHW`/`writeMP4`) currently live in the **`RunSCAIL2` executable**
/// target and aren't importable. Step 4 lifts those two files (`Loaders.swift` + `MediaIO.swift`)
/// from `RunSCAIL2` into the `SCAIL2` library — both the CLI and this wrapper then share them — and
/// fills in the bodies below, reusing the proven `GenerateMode.run` orchestration verbatim
/// (encode CLIP + umT5 first and EVICT before the fp32 DiT load; per-step eval + `Memory.cacheLimit`
/// cap inside `SCAILPipeline`).
final class SCAILRuntime {
    private let pipeline: SCAILPipeline
    private let config: SCAIL2Config

    private init(pipeline: SCAILPipeline, config: SCAIL2Config) {
        self.pipeline = pipeline
        self.config = config
    }

    /// Page the working set in: load DiT (fp32 — the Wan-family rule) + VAE (+ tokenizer); CLIP/umT5
    /// are paged per request and evicted before denoise (the family memory discipline).
    static func fromPretrained(directory: URL, config: SCAIL2Config) throws -> SCAILRuntime {
        throw SCAILRuntimeError.wiringPending(
            "SCAILRuntime.fromPretrained — lift Loaders.swift from RunSCAIL2 into SCAIL2 (S7 step 4)")
    }

    /// Decoded canonical inputs → an mp4 (`Data`). Mirrors `GenerateMode.run`: media decode → CLIP
    /// visual → umT5 encode (evict both) → VAE-encode ref/mask → `SCAILPipeline.generate` → writeMP4.
    func generate(
        referenceImage: Data,
        drivingVideo: Data,
        drivingMask: Data?,
        prompt: String,
        width: Int,
        height: Int,
        steps: Int,
        guidanceScale: Float,
        shift: Double,
        solver: String,
        segmentLen: Int,
        segmentOverlap: Int,
        seed: UInt64,
        replaceFlag: Bool,
        onStep: ((Int, Int) -> Void)?
    ) throws -> (mp4: Data, frameCount: Int, fps: Double) {
        throw SCAILRuntimeError.wiringPending(
            "SCAILRuntime.generate — lift MediaIO.swift from RunSCAIL2 into SCAIL2 + reuse "
            + "GenerateMode.run orchestration (S7 step 4)")
    }
}

/// Errors from the not-yet-wired runtime seam. Distinct from `MLXToolKit.PackageError` so the
/// engine surfaces a precise, actionable message rather than a generic load failure.
enum SCAILRuntimeError: Error, CustomStringConvertible {
    case wiringPending(String)

    var description: String {
        switch self {
        case .wiringPending(let what): return "SCAIL-2 runtime not wired: \(what)"
        }
    }
}
