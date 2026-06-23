// --generate: the S2b eyeball gate. The first real GPU generation — wires the
// parity-locked SCAILModel DiT + reused wan-core VAE/umT5 + net-new CLIP/mask
// utils through SCAILPipeline.generate and writes an mp4.
//
// Memory discipline (family levers): encode CLIP + umT5 FIRST and EVICT them
// before the 14B DiT is even loaded, so the heavy fp32 DiT never co-resides
// with the encoders' working sets. Per-step eval + the Memory.cacheLimit cap
// live inside the pipeline.
import Foundation
import MLX
import MLXNN
import MLXRandom
import SCAIL2
import Tokenizers
import WanCore

enum GenerateMode {
    struct Args {
        var weightsDir = "/Volumes/DEV_ARCHIVE/scail-2-mlx/weights/mlx"
        var image = "", maskImage = "", pose = "", maskVideo = ""
        var prompt = ""
        var replaceFlag = false
        var height = 288, width = 512
        var maxFrames: Int? = nil
        var steps = 16
        var solver = "dpm++"
        var guide: Float = 5.0
        var shift = 5.0
        var segmentLen = 81
        var segmentOverlap = 5
        var seed: UInt64 = 42
        var out = "scail2_out.mp4"
        var dryEncode = false  // stop after encoders + VAE-encode (skip DiT + denoise)
    }

    static func err(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }

    static func parse(_ argv: [String]) -> Args {
        var a = Args()
        var it = argv.makeIterator()
        while let k = it.next() {
            func val() -> String { it.next() ?? "" }
            switch k {
            case "--weights-dir": a.weightsDir = val()
            case "--image": a.image = val()
            case "--mask-image": a.maskImage = val()
            case "--pose": a.pose = val()
            case "--mask-video": a.maskVideo = val()
            case "--prompt": a.prompt = val()
            case "--replace-flag": a.replaceFlag = true
            case "--target-h": a.height = Int(val()) ?? a.height
            case "--target-w": a.width = Int(val()) ?? a.width
            case "--max-frames": a.maxFrames = Int(val())
            case "--steps": a.steps = Int(val()) ?? a.steps
            case "--solver": a.solver = val()
            case "--guide": a.guide = Float(val()) ?? a.guide
            case "--shift": a.shift = Double(val()) ?? a.shift
            case "--segment-len": a.segmentLen = Int(val()) ?? a.segmentLen
            case "--segment-overlap": a.segmentOverlap = Int(val()) ?? a.segmentOverlap
            case "--seed": a.seed = UInt64(val()) ?? a.seed
            case "--out": a.out = val()
            case "--dry-encode": a.dryEncode = true
            default: err("warn: ignoring unknown arg \(k)")
            }
        }
        return a
    }

    static func run(_ argv: [String]) async -> Int32 {
        let a = parse(argv)
        guard a.height % 32 == 0, a.width % 32 == 0 else {
            err("target-h/target-w must be divisible by 32"); return 2
        }
        for (label, path) in [("image", a.image), ("mask-image", a.maskImage),
                              ("pose", a.pose), ("mask-video", a.maskVideo)] {
            guard !path.isEmpty else { err("missing --\(label)"); return 2 }
        }
        let dir = URL(fileURLWithPath: a.weightsDir)
        let cfg = SCAIL2Config()
        let fps = Double(cfg.sampleFps)

        do {
            // ── inputs (oracle generate.py _load_image/_load_video) ──────────
            err("[gen] loading media @ \(a.width)x\(a.height) …")
            let refImg = try loadImageCHW(a.image, targetH: a.height, targetW: a.width)      // [3,H,W]
            let maskImg = try loadImageCHW(a.maskImage, targetH: a.height, targetW: a.width)  // [3,H,W]
            let poseVideo = try await loadVideoTCHW(
                a.pose, targetH: a.height, targetW: a.width, maxFrames: a.maxFrames)          // [T,3,H,W]
            // mask video: [T,3,H,W] → [3,T,H,W] (oracle .transpose(1,0,2,3))
            let drivingMask = try await loadVideoTCHW(
                a.maskVideo, targetH: a.height, targetW: a.width, maxFrames: a.maxFrames)
                .transposed(1, 0, 2, 3)                                                       // [3,T,H,W]
            eval(refImg, maskImg, poseVideo, drivingMask)
            err("[gen] pose \(poseVideo.shape) mask \(drivingMask.shape)")

            // ── CLIP visual on ref → clipFea [1,257,1280], then EVICT ────────
            err("[gen] CLIP visual …")
            var clipTower: CLIPVisionTower? = CLIPVisionTower(
                imageSize: 224, patchSize: 14, dim: 1280, mlpRatio: 4,
                numHeads: 16, numLayers: 32, preNorm: true, eps: 1e-5)
            try loadCLIP(clipTower!, url: dir.appendingPathComponent("clip.safetensors"))
            // oracle clip.visual([img[:,None]]): preprocess [C,T,H,W] then tower (fp16)
            let refCTHW = refImg.reshaped(3, 1, a.height, a.width)  // [C,T,H,W], T=1
            let clipIn = CLIPPreprocess.preprocess([refCTHW]).asType(.float16)
            var clipFea = clipTower!(clipIn, use31Block: true)  // [1,257,1280]
            eval(clipFea)
            clipTower = nil
            Memory.clearCache()
            err("[gen] clipFea \(clipFea.shape)")

            // ── umT5 text encode (cond + uncond), then EVICT ─────────────────
            err("[gen] umT5 text encode …")
            let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
            var umt5: UMT5EncoderModel? = UMT5EncoderModel()  // umT5-xxl defaults = SCAIL config
            try loadUMT5(umt5!, url: dir.appendingPathComponent("umt5.safetensors"))
            var context = encodeText(
                encoder: umt5!, tokenizer: tokenizer, prompt: a.prompt, textLen: cfg.textLen)
            var contextNull = encodeText(
                encoder: umt5!, tokenizer: tokenizer, prompt: "", textLen: cfg.textLen)
            eval(context, contextNull)
            umt5 = nil
            Memory.clearCache()
            err("[gen] context \(context.shape) / null \(contextNull.shape)")

            // ── VAE (kept resident: ref/pose/history encode + decode) ────────
            err("[gen] VAE …")
            let vae = WanVAE(zDim: cfg.outDim, encoder: true)
            try loadVAE(vae, url: dir.appendingPathComponent("vae.safetensors"))
            // ref image → ref latent [16,1,h,w] (oracle vae.encode([ori_img.T(1,0,2,3)]))
            var refLatent = vae.encode(refImg.reshaped(1, 3, 1, a.height, a.width))[0]
            // ref mask → 28-ch latent [28,1,h,w]
            var ref28 = MaskCompress.extract(
                maskImg.reshaped(3, 1, a.height, a.width), additionalSpatialDownsample: 1)
            eval(refLatent, ref28)
            err("[gen] refLatent \(refLatent.shape) ref28 \(ref28.shape)")

            if a.dryEncode {
                err("[gen] --dry-encode OK: all encoders + VAE-encode wired "
                    + "(clipFea \(clipFea.shape), context \(context.shape), "
                    + "refLatent \(refLatent.shape), ref28 \(ref28.shape)); skipping DiT.")
                return 0
            }

            // ── DiT (fp32, ~64 GB) loaded LAST ───────────────────────────────
            err("[gen] DiT load (fp32) …")
            let dit = SCAILModel(
                patchSize: (cfg.patchSize[0], cfg.patchSize[1], cfg.patchSize[2]),
                textLen: cfg.textLen, inDim: cfg.inDim, maskDim: cfg.maskDim,
                dim: cfg.dim, ffnDim: cfg.ffnDim, freqDim: cfg.freqDim,
                textDim: 4096, outDim: cfg.outDim,
                numHeads: cfg.numHeads, numLayers: cfg.numLayers, eps: cfg.eps)
            try loadDiT(dit, url: dir.appendingPathComponent("dit.safetensors"))
            // The freed bf16 source buffers linger in MLX's cache until the
            // pipeline's cacheLimit cap reclaims them (load-time phys peaked
            // ~104 GB vs ~70 GB steady-state at 512×288/fp32). Reclaim now so
            // the load-time transient doesn't crowd the denoise headroom.
            Memory.clearCache()
            err(String(format: "[gen] DiT resident — phys %.1f GB",
                       Double(physFootprintBytes()) / 1e9))

            // ── generate ─────────────────────────────────────────────────────
            var opts = SCAILGenerateOptions()
            opts.segmentLen = a.segmentLen
            opts.segmentOverlap = a.segmentOverlap
            opts.shift = a.shift
            opts.samplingSteps = a.steps
            opts.guideScale = a.guide
            opts.solver = a.solver

            MLXRandom.seed(a.seed)
            let pipeline = SCAILPipeline(model: dit, vae: vae, vaeStrideT: cfg.vaeStride[0])
            err("[gen] denoise: solver=\(a.solver) steps=\(a.steps) guide=\(a.guide) "
                + "replace=\(a.replaceFlag) seed=\(a.seed)")
            // free the now-dead encoder outputs' container refs (latents kept)
            let video = pipeline.generate(
                refLatent: refLatent, refMask28: ref28, clipFea: clipFea,
                context: context, contextNull: contextNull,
                poseVideo: poseVideo, drivingMask: drivingMask,
                replaceFlag: a.replaceFlag, options: opts,
                onStep: { i, n in
                    err(String(format: "  step %d/%d  phys %.1f GB",
                               i, n, Double(physFootprintBytes()) / 1e9))
                })
            eval(video)
            refLatent = MLXArray(0); ref28 = MLXArray(0)
            clipFea = MLXArray(0); context = MLXArray(0); contextNull = MLXArray(0)
            err("[gen] video \(video.shape) — phys peak \(String(format: "%.1f GB", Double(Memory.peakMemory) / 1e9))")

            // ── write mp4 ────────────────────────────────────────────────────
            let outURL = URL(fileURLWithPath: a.out)
            try await writeMP4(frames: video, to: outURL, fps: fps)
            err("[gen] WROTE \(a.out)  (\(video.dim(1)) frames @ \(fps) fps)")
            return 0
        } catch {
            err("[gen] FAILED: \(error)")
            return 1
        }
    }
}
