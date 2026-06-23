// SCAIL-2 generation pipeline — orchestrates the proven SCAILModel DiT with the
// reused wan-core substrate (WanVAE encode/decode, FlowUniPC/DPM++ scheduler)
// and the net-new utils (MaskCompress, CLIPPreprocess + CLIPVisionTower). 1:1
// translation of the oracle SCAIL2Pipeline.generate (scail2_mlx/scail.py):
// segmented long-video with clean-history overlap, CFG, per-step eval.
//
// I/O (image/video decode, mp4 encode) and weight loading live at the call site
// (the RunSCAIL2 CLI / the engine wrapper); this type takes tensors + loaded
// models so it stays testable headless.
import Foundation
import MLX
import MLXRandom
import WanCore

public struct SCAILGenerateOptions: Sendable {
    public var segmentLen: Int = 81
    public var segmentOverlap: Int = 5
    public var shift: Double = 5.0
    public var samplingSteps: Int = 40
    public var guideScale: Float = 5.0
    public var solver: String = "unipc"  // or "dpm++"
    public init() {}
}

public final class SCAILPipeline {
    let model: SCAILModel
    let vae: WanVAE
    let vaeStrideT: Int

    public init(model: SCAILModel, vae: WanVAE, vaeStrideT: Int = 4) {
        self.model = model
        self.vae = vae
        self.vaeStrideT = vaeStrideT
    }

    private func buildSegments(_ total: Int, _ o: SCAILGenerateOptions) -> [(Int, Int)] {
        if total <= o.segmentLen {
            let keep = ((total - 1) / vaeStrideT) * vaeStrideT + 1
            return [(0, keep)]
        }
        var segs: [(Int, Int)] = []
        var start = 0
        let stride = o.segmentLen - o.segmentOverlap
        while start < total {
            let end = start + o.segmentLen
            if end > total { break }
            segs.append((start, end)); start += stride
        }
        return segs
    }

    /// Inputs are already VAE-encoded / preprocessed (the pipeline's encode side
    /// is thin wan-core reuse and is exercised separately):
    ///   refLatent [16,1,h,w], refMask28 [28,1,h,w], clipFea [1,257,1280],
    ///   context/contextNull [L,textDim] (umT5-embedded), poseVideo/drivingMask
    ///   raw [T,3,H,W]/[3,T,H,W] in [-1,1]. Returns decoded video [3, Tout, H, W].
    public func generate(
        refLatent: MLXArray, refMask28: MLXArray, clipFea: MLXArray,
        context: MLXArray, contextNull: MLXArray,
        poseVideo: MLXArray, drivingMask: MLXArray,
        replaceFlag: Bool, options o: SCAILGenerateOptions,
        noise noiseFn: (([Int]) -> MLXArray)? = nil,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let numFrames = poseVideo.dim(0)
        let segments = buildSegments(numFrames, o)
        let latC = refLatent.dim(0)
        let (latH, latW) = (refLatent.dim(2), refLatent.dim(3))
        var outputs: [MLXArray] = []
        var prevHistoryPixel: MLXArray? = nil

        for (segIdx, seg) in segments.enumerated() {
            let (s0, s1) = seg
            // pose / driving-mask -> half-res -> VAE-encode pose, compress mask
            let poseSeg = poseVideo[s0..<s1]                          // [T,3,H,W]
            let poseHalf = MaskCompress.halfBilinear(poseSeg)         // [T,3,H/2,W/2]
            let poseLatent = vae.encode(poseHalf.transposed(1, 0, 2, 3)[.newAxis])[0]  // [16,t,h,w]
            let latT = poseLatent.dim(1)

            let drivingSeg = drivingMask[0..., s0..<s1]               // [3,T,H,W]
            let drivingHalf = MaskCompress.halfBilinear(drivingSeg)
            let drivingMasks = MaskCompress.extract(drivingHalf)      // [28,t,h/?,w/?]

            // ref_masks = cat([ref_mask_28ch, zeros(28, lat_t, lat_h, lat_w)], dim=1)
            let nullMask = MLXArray.zeros([refMask28.dim(0), latT, latH, latW], dtype: refMask28.dtype)
            let refMasks = concatenated([refMask28, nullMask], axis: 1)

            var historyMask: MLXArray? = nil
            var historyLatent: MLXArray? = nil
            if segIdx > 0, let prev = prevHistoryPixel {
                historyLatent = vae.encode(prev[.newAxis])[0]
                let ht = min(historyLatent!.dim(1), latT)
                historyMask = concatenated([
                    MLXArray.ones([4, ht, latH, latW], dtype: .float32),
                    MLXArray.zeros([4, latT - ht, latH, latW], dtype: .float32),
                ], axis: 1)
            }

            let noise = (noiseFn ?? { MLXRandom.normal($0) })([latC, latT, latH, latW])

            let video = sampleSegment(
                noise: noise, poseLatent: poseLatent, drivingMasks: drivingMasks,
                refLatent: refLatent, refMasks: refMasks, clipFea: clipFea,
                context: context, contextNull: contextNull, historyMask: historyMask,
                historyLatent: historyLatent, replaceFlag: replaceFlag, options: o,
                onStep: onStep)

            if segIdx == 0 { outputs.append(video) }
            else { outputs.append(video[0..., o.segmentOverlap...]) }
            if segIdx < segments.count - 1 {
                prevHistoryPixel = video[0..., (video.dim(1) - o.segmentOverlap)...]
            }
        }
        return concatenated(outputs, axis: 1)
    }

    private func sampleSegment(
        noise: MLXArray, poseLatent: MLXArray, drivingMasks: MLXArray,
        refLatent: MLXArray, refMasks: MLXArray, clipFea: MLXArray,
        context: MLXArray, contextNull: MLXArray, historyMask: MLXArray?,
        historyLatent: MLXArray?, replaceFlag: Bool, options o: SCAILGenerateOptions,
        onStep: ((Int, Int) -> Void)?
    ) -> MLXArray {
        let unipc = o.solver == "unipc" ? FlowUniPCScheduler() : nil
        let dpmpp = o.solver == "dpm++" ? FlowDPMPP2MScheduler() : nil
        unipc?.setTimesteps(o.samplingSteps, shift: o.shift)
        dpmpp?.setTimesteps(o.samplingSteps, shift: o.shift)
        let timesteps: [Float] = unipc?.timesteps ?? dpmpp!.timesteps
        Memory.cacheLimit = 2 * 1024 * 1024 * 1024  // decode/denoise cache cap (family lever)

        func applyClean(_ latent: MLXArray) -> MLXArray {
            guard let h = historyLatent else { return latent }
            let ht = h.dim(1)
            return concatenated([h.asType(latent.dtype), latent[0..., ht...]], axis: 1)
        }

        var latent = applyClean(noise)
        for (i, t) in timesteps.enumerated() {
            let modelInput = applyClean(latent)
            let tArr = MLXArray([t])
            let cond = model(
                x: modelInput, poseLatents: poseLatent, drivingMasks: drivingMasks,
                refLatents: refLatent, refMasks: refMasks, t: tArr, context: context,
                clipFea: clipFea, replaceFlag: replaceFlag, historyMask: historyMask)
            var noisePred = cond
            if o.guideScale > 1.0 {
                let uncond = model(
                    x: modelInput, poseLatents: poseLatent, drivingMasks: drivingMasks,
                    refLatents: refLatent, refMasks: refMasks, t: tArr, context: contextNull,
                    clipFea: clipFea, replaceFlag: replaceFlag, historyMask: historyMask)
                noisePred = uncond + o.guideScale * (cond - uncond)
            }
            let stepped = unipc?.step(modelOutput: noisePred[.newAxis], timestep: t, sample: latent[.newAxis])
                ?? dpmpp!.step(modelOutput: noisePred[.newAxis], timestep: t, sample: latent[.newAxis])
            latent = applyClean(stepped[0])
            eval(latent)  // bound the graph per step (family runBlocks discipline)
            onStep?(i + 1, timesteps.count)
        }
        // Streaming decode (the oracle's decode_chunked): the Rep first-chunk
        // cache gives the upstream-faithful 1+(T-1)·4 frame count with flat
        // peak memory. Whole-sequence `vae.decode` would emit 4·T frames with a
        // divergent head + phase shift (oracle _WanVAEAdapter.decode comment).
        return decodeStreaming(vae: vae, latent[.newAxis])[0]  // [3, Tout, H, W]
    }
}
