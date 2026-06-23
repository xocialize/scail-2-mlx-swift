// The resident SCAIL-2 runtime: owns the loaded DiT + VAE (+ tokenizer) and runs the full
// media → video path, so the MLXEngine wrapper (`MLXSCAIL2Package`) stays a thin contract adapter
// — the split Phantom uses with `PhantomPipeline`. This is the headed counterpart to
// `SCAILPipeline` (which is headless / tensor-in-tensor-out and stays parity-testable).
//
// 1:1 with the proven `GenerateMode.run` orchestration: CLIP visual + umT5 encode (encoded then
// EVICTED), VAE-encode ref/mask, `SCAILPipeline.generate`, mp4 write. The only deltas vs the CLI
// are the engine envelope — inputs arrive as canonical-artifact `Data` (round-tripped through temp
// files for the AVFoundation paths) and the DiT/VAE are resident across requests (loaded once in
// `fromPretrained`), with CLIP/umT5 paged per request and evicted (the family memory discipline).
import Foundation
import MLX
import MLXRandom
import Tokenizers
import WanCore

/// `@unchecked Sendable`: the engine drives this strictly serially behind `@InferenceActor` (one
/// `run` at a time per package), the same single-owner guarantee the MLX model classes rely on.
public final class SCAILRuntime: @unchecked Sendable {
    private let directory: URL
    private let config: SCAIL2Config
    private let tokenizer: any Tokenizer
    /// Resident across requests (loaded once). fp32 DiT (~65 GB at 14B) + 16-ch WanVAE.
    private let dit: SCAILModel
    private let vae: WanVAE

    private init(directory: URL, config: SCAIL2Config, tokenizer: any Tokenizer,
                 dit: SCAILModel, vae: WanVAE) {
        self.directory = directory
        self.config = config
        self.tokenizer = tokenizer
        self.dit = dit
        self.vae = vae
    }

    /// Page the resident working set in: tokenizer + VAE + the fp32 DiT (loaded on the CPU stream
    /// inside the loaders; the family watchdog rule). CLIP/umT5 are NOT resident — paged per request.
    public static func fromPretrained(directory: URL, config: SCAIL2Config = SCAIL2Config()) async throws -> SCAILRuntime {
        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)

        let vae = WanVAE(zDim: config.outDim, encoder: true)
        try loadVAE(vae, url: directory.appendingPathComponent("vae.safetensors"))

        let dit = SCAILModel(
            patchSize: (config.patchSize[0], config.patchSize[1], config.patchSize[2]),
            textLen: config.textLen, inDim: config.inDim, maskDim: config.maskDim,
            dim: config.dim, ffnDim: config.ffnDim, freqDim: config.freqDim,
            textDim: 4096, outDim: config.outDim,
            numHeads: config.numHeads, numLayers: config.numLayers, eps: config.eps)
        try loadDiT(dit, url: directory.appendingPathComponent("dit.safetensors"))
        // Reclaim the freed bf16 source buffers lingering in MLX's cache after the cast-to-fp32 load
        // (load-time phys peaks ~1.5× the fp32 weight size until this reclaim).
        Memory.clearCache()

        return SCAILRuntime(directory: directory, config: config, tokenizer: tokenizer, dit: dit, vae: vae)
    }

    /// Canonical inputs (artifact `Data`) → an mp4 (`Data`) + its frame count and fps. Mirrors
    /// `GenerateMode.run`. `onStep(step, total)` is invoked per denoising step (cancellation hook).
    public func generate(
        referenceImage: Data,
        referenceMask: Data,
        drivingVideo: Data,
        drivingMask: Data,
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
        maxFrames: Int? = nil,
        onStep: (@Sendable (Int, Int) throws -> Void)? = nil
    ) async throws -> (mp4: Data, frameCount: Int, fps: Double) {
        // ── inputs: images decode straight from Data; the AVFoundation paths need a file URL,
        //    so the two driving videos round-trip through temp files. ───────────────────────────
        let refImg = try loadImageCHW(data: referenceImage, targetH: height, targetW: width)     // [3,H,W]
        let maskImg = try loadImageCHW(data: referenceMask, targetH: height, targetW: width)      // [3,H,W]
        let poseURL = try Self.writeTemp(drivingVideo, ext: "mp4")
        let maskURL = try Self.writeTemp(drivingMask, ext: "mp4")
        defer { try? FileManager.default.removeItem(at: poseURL); try? FileManager.default.removeItem(at: maskURL) }

        let poseVideo = try await loadVideoTCHW(poseURL.path, targetH: height, targetW: width, maxFrames: maxFrames) // [T,3,H,W]
        let drivingMaskVideo = try await loadVideoTCHW(maskURL.path, targetH: height, targetW: width, maxFrames: maxFrames)
            .transposed(1, 0, 2, 3)                                                               // [3,T,H,W]
        eval(refImg, maskImg, poseVideo, drivingMaskVideo)

        // ── CLIP visual on ref → clipFea [1,257,1280], then EVICT ──────────────────────────────
        var clipTower: CLIPVisionTower? = CLIPVisionTower(
            imageSize: 224, patchSize: 14, dim: 1280, mlpRatio: 4,
            numHeads: 16, numLayers: 32, preNorm: true, eps: 1e-5)
        try loadCLIP(clipTower!, url: directory.appendingPathComponent("clip.safetensors"))
        let refCTHW = refImg.reshaped(3, 1, height, width)
        let clipFea = clipTower!(CLIPPreprocess.preprocess([refCTHW]).asType(.float16), use31Block: true)
        eval(clipFea)
        clipTower = nil
        Memory.clearCache()

        // ── umT5 text encode (cond + uncond), then EVICT ───────────────────────────────────────
        var umt5: UMT5EncoderModel? = UMT5EncoderModel()
        try loadUMT5(umt5!, url: directory.appendingPathComponent("umt5.safetensors"))
        let context = encodeText(encoder: umt5!, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
        let contextNull = encodeText(encoder: umt5!, tokenizer: tokenizer, prompt: "", textLen: config.textLen)
        eval(context, contextNull)
        umt5 = nil
        Memory.clearCache()

        // ── VAE encode (resident): ref latent [16,1,h,w] + ref mask 28-ch [28,1,h,w] ───────────
        let refLatent = vae.encode(refImg.reshaped(1, 3, 1, height, width))[0]
        let ref28 = MaskCompress.extract(maskImg.reshaped(3, 1, height, width), additionalSpatialDownsample: 1)
        eval(refLatent, ref28)

        // ── denoise + streaming decode ─────────────────────────────────────────────────────────
        var opts = SCAILGenerateOptions()
        opts.segmentLen = segmentLen
        opts.segmentOverlap = segmentOverlap
        opts.shift = shift
        opts.samplingSteps = steps
        opts.guideScale = guidanceScale
        opts.solver = solver

        MLXRandom.seed(seed)
        let pipeline = SCAILPipeline(model: dit, vae: vae, vaeStrideT: config.vaeStride[0])
        let video = try pipeline.generate(
            refLatent: refLatent, refMask28: ref28, clipFea: clipFea,
            context: context, contextNull: contextNull,
            poseVideo: poseVideo, drivingMask: drivingMaskVideo,
            replaceFlag: replaceFlag, options: opts, onStep: onStep)
        eval(video)

        // ── write mp4 → read back as Data ──────────────────────────────────────────────────────
        let outURL = try Self.tempURL(ext: "mp4")
        defer { try? FileManager.default.removeItem(at: outURL) }
        let fps = Double(config.sampleFps)
        try await writeMP4(frames: video, to: outURL, fps: fps)
        let mp4 = try Data(contentsOf: outURL)
        return (mp4: mp4, frameCount: video.dim(1), fps: fps)
    }

    // MARK: - Temp-file helpers (AVFoundation needs file URLs)

    private static func tempURL(ext: String) throws -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("scail2-\(UUID().uuidString).\(ext)")
    }

    private static func writeTemp(_ data: Data, ext: String) throws -> URL {
        let url = try tempURL(ext: ext)
        try data.write(to: url)
        return url
    }
}
