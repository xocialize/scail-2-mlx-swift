import Foundation
import MLXToolKit
import SCAIL2
import WanCore

/// MLXEngine package: SCAIL-2 (Wan2.1-I2V-14B character animation) over the canonical
/// `characterAnimation` surface (contract 1.6.0). SCAIL is the lead consumer of the lane: a
/// reference character image + a driving performance video → a video of that character performing
/// the motion. `.animation` keeps the reference identity and transfers the motion; `.replacement`
/// swaps the reference identity into the driving clip (SCAIL's `replaceFlag`).
///
/// Engine-owned lifecycle (C13, `@InferenceActor`): construct from `SCAIL2Configuration`, page the
/// working set in with `load()`, drive `run(_:)`, reclaim with `unload()`. The non-`Sendable`
/// `SCAILRuntime` never crosses the actor boundary. Cancellation is honored per denoising step.
///
/// SCAIL is `.poseless` — no skeleton/pose extractor dependency (the driving mask is a plain RGB
/// video the package compresses internally), which is its deploy-simplicity edge over the
/// `.poseDriven` Wan2.2-Animate that will register as a second package on this same capability.
@InferenceActor
public final class MLXSCAIL2Package: ModelPackage {
    public typealias Configuration = SCAIL2Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "zai-org/SCAIL-2",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // DERIVED, pending the S5 in-app re-measure. The DiT runs fp32 (bf16 NaNs the DiT at
                // video seqLen); measured at 512×288/65f on the 128 GB box: ~65.6 GB DiT resident,
                // ~70 GB steady denoise, ~99 GB decode peak (CLIP+umT5 encoded then EVICTED before
                // the DiT loads). Native-res is higher → pro tier. int4 is BLOCKED upstream (Python
                // q4 fails its own parity gate), so bf16 (fp32-runtime) is the only variant today.
                // Re-ground residentBytes on the measured phys after the live run (S5).
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 100_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max  // 128 GB-class; the fp32 14B DiT + decode peak is a pro-tier load
            ),
            specialties: [
                // Ranks SCAIL within the characterAnimation lane: poseless (no extractor) is its
                // distinguishing trait vs the poseDriven Wan2.2-Animate.
                SpecialtyWeight(.poseless, strength: 0.7),
                SpecialtyWeight(.general, strength: 0.4),
            ],
            surfaces: [
                CharacterAnimationContract.descriptor(
                    name: "scail-2-character-animation",
                    summary: "SCAIL-2 (MLX): a reference character image + a driving video → a video "
                        + "of that character performing the driving motion. Mode `.animation` keeps "
                        + "the reference identity; `.replacement` swaps it into the driving clip. "
                        + "Poseless (the driving mask is a plain RGB video).",
                    modes: [.animation, .replacement, .quality, .fast]
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident runtime (fp32 DiT + 16-ch WanVAE + tokenizer), paged in by `load()`.
    private var runtime: SCAILRuntime?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard runtime == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        runtime = try await SCAILRuntime.fromPretrained(directory: directory, config: SCAIL2Config())
    }

    public func unload() async {
        runtime = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let runtime else { throw PackageError.notLoaded }
        switch request.capability {
        case .characterAnimation:
            guard let req = request as? CharacterAnimationRequest else {
                throw PackageError.configurationMismatch(
                    expected: "CharacterAnimationRequest",
                    got: String(describing: type(of: request)))
            }
            return try await runAnimation(req, runtime: runtime)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surface

    private func runAnimation(
        _ request: CharacterAnimationRequest, runtime: SCAILRuntime
    ) async throws -> CharacterAnimationResponse {
        try Task.checkCancellation()

        // Resolve the request envelope against the SCAIL defaults (oracle truths). Size must be
        // divisible by 32 (the patch×VAE-stride grid); the CLI defaults are 512×288.
        let width = request.width ?? 512
        let height = request.height ?? 288
        guard width % 32 == 0, height % 32 == 0 else {
            throw PackageError.configurationMismatch(
                expected: "width & height divisible by 32",
                got: "\(width)x\(height)")
        }
        let steps = resolveSteps(mode: request.mode, steps: request.steps)
        let replaceFlag = resolveReplaceFlag(mode: request.mode)

        // SCAIL needs TWO masks: the per-frame `drivingMask` (canonical, but optional in the
        // lane-ready schema — required here) AND a still foreground mask of the reference. The
        // latter has no canonical field yet, so it rides `metaData["referenceMask"]` as a base64
        // PNG (a known C5 smell — flagged for promotion to a canonical `referenceMask: Image?`, or
        // auto-derivation via the `matting` capability; see PORTING-SPEC §S7).
        guard let drivingMaskVideo = request.drivingMask else {
            throw PackageError.configurationMismatch(
                expected: "drivingMask (per-frame driving mask video — required by SCAIL)", got: "nil")
        }
        guard case let .string(refMaskB64)? = request.metaData["referenceMask"],
              let referenceMask = Data(base64Encoded: refMaskB64) else {
            throw PackageError.configurationMismatch(
                expected: "metaData[\"referenceMask\"] = base64 PNG of the still reference foreground "
                    + "mask (SCAIL conditions on a ref mask + the driving mask)",
                got: request.metaData["referenceMask"].map { "\($0)" } ?? "absent")
        }

        let onStep: @Sendable (Int, Int) throws -> Void = { _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }

        let result = try await runtime.generate(
            referenceImage: request.referenceImage.data,
            referenceMask: referenceMask,
            drivingVideo: request.drivingVideo.data,
            drivingMask: drivingMaskVideo.data,
            prompt: request.prompt ?? "",
            width: width,
            height: height,
            steps: steps,
            guidanceScale: Float(request.guidanceScale ?? 5.0),
            shift: 5.0,
            solver: "dpm++",
            segmentLen: 81,
            segmentOverlap: 5,
            seed: request.seed ?? 42,
            replaceFlag: replaceFlag,
            maxFrames: request.numFrames,
            onStep: onStep)

        return CharacterAnimationResponse(
            video: Video(format: .mp4, data: result.mp4,
                         durationSeconds: Double(result.frameCount) / result.fps,
                         frameRate: result.fps))
    }
}

extension MLXSCAIL2Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXSCAIL2Package.self)
    }
}
