// SCAIL-2 DiT — full forward assembly. 1:1 translation of the oracle
// SCAIL2Model (scail2_mlx/modules/model_scail2.py). Reuses the parity-locked
// net-new pieces (RoPE3Seg, SCAILPatchEmbeds, I2VCrossAttention) and wan-core
// `ropeParams` (inside RoPE3Seg). The blocks mirror the oracle WanAttentionBlock
// (6-way AdaLN modulation, fp32 norm/modulation islands) — NOT wan-core's
// WanAttentionBlock, which uses single-grid RoPE + text-only cross-attn.
import Foundation
import MLX
import MLXFast
import MLXNN

// fp32-internal norms (oracle WanRMSNorm / WanLayerNorm).
func layerNorm(_ x: MLXArray, _ w: MLXArray?, _ b: MLXArray?, _ eps: Float) -> MLXArray {
    MLXFast.layerNorm(x.asType(.float32), weight: w, bias: b, eps: eps).asType(x.dtype)
}

/// RMSNorm as a Module so its parameter loads under `<name>.weight` (matching
/// the checkpoint's `*.norm_q.weight`). Oracle WanRMSNorm == MLXFast.rmsNorm.
public final class RMSNormW: Module, @unchecked Sendable {
    let eps: Float
    public var weight: MLXArray
    public init(_ dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: eps).asType(x.dtype)
    }
}

func sinusoidalEmbedding1D(_ dim: Int, _ position: MLXArray) -> MLXArray {
    let half = dim / 2
    let exponents = (0..<half).map { Double($0) / Double(half) }
    let inv = MLXArray(exponents.map { Float(pow(10000.0, -$0)) })  // [half]
    let sinusoid = position.asType(.float32).reshaped(-1, 1) * inv.reshaped(1, half)
    return concatenated([cos(sinusoid), sin(sinusoid)], axis: 1)
}

final class SCAILSelfAttention: Module, @unchecked Sendable {
    let numHeads: Int, headDim: Int, scale: Float, eps: Float
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNormW
    @ModuleInfo(key: "norm_k") var normK: RMSNormW

    init(_ dim: Int, _ numHeads: Int, eps: Float) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(dim / numHeads), -0.5)
        self.eps = eps
        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = RMSNormW(dim, eps: eps)
        self._normK.wrappedValue = RMSNormW(dim, eps: eps)
        super.init()
    }

    // applyRope: q,k -> rotated (the 3-segment rope closure from the model)
    func callAsFunction(_ x: MLXArray, applyRope: (MLXArray) -> MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        let qh = normQ(q(x)).reshaped(b, s, numHeads, headDim)
        let kh = normK(k(x)).reshaped(b, s, numHeads, headDim)
        let vh = v(x).reshaped(b, s, numHeads, headDim)
        // RoPE applied on [B,S,N,D], then to [B,N,S,D] for SDPA
        let qr = applyRope(qh).transposed(0, 2, 1, 3)
        let kr = applyRope(kh).transposed(0, 2, 1, 3)
        let vt = vh.transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: qr, keys: kr, values: vt, scale: scale, mask: nil)
        return o(out.transposed(0, 2, 1, 3).reshaped(b, s, numHeads * headDim))
    }
}

final class SCAILBlock: Module, @unchecked Sendable {
    let eps: Float
    @ModuleInfo(key: "norm1") var norm1Dummy: Identity  // parameterless (key carries nothing)
    @ModuleInfo(key: "self_attn") var selfAttn: SCAILSelfAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm       // affine (cross_attn_norm)
    @ModuleInfo(key: "cross_attn") var crossAttn: I2VCrossAttention
    @ModuleInfo(key: "norm2") var norm2Dummy: Identity   // parameterless
    @ModuleInfo(key: "ffn") var ffn: Sequential          // Linear, GELU(tanh), Linear -> layers.0/.2
    @ModuleInfo(key: "modulation") var modulation: MLXArray  // [1,6,dim]

    init(_ dim: Int, _ ffnDim: Int, _ numHeads: Int, eps: Float, textLen: Int) {
        self.eps = eps
        self._norm1Dummy.wrappedValue = Identity()
        self._selfAttn.wrappedValue = SCAILSelfAttention(dim, numHeads, eps: eps)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._crossAttn.wrappedValue = I2VCrossAttention(dim, numHeads, qkNorm: true, eps: eps, textLen: textLen)
        self._norm2Dummy.wrappedValue = Identity()
        self._ffn.wrappedValue = Sequential {
            Linear(dim, ffnDim); GELU(approximation: .tanh); Linear(ffnDim, dim)
        }
        self._modulation.wrappedValue = MLXArray.zeros([1, 6, dim])
        super.init()
    }

    func callAsFunction(
        _ x0: MLXArray, e: MLXArray, context: MLXArray, applyRope: (MLXArray) -> MLXArray
    ) -> MLXArray {
        // e: [B,6,dim]; modulation fp32
        let mod = modulation.asType(.float32) + e.asType(.float32)
        let e0 = mod[0..., 0], e1 = mod[0..., 1], e2 = mod[0..., 2]
        let e3 = mod[0..., 3], e4 = mod[0..., 4], e5 = mod[0..., 5]

        // self-attention (norm1 is parameterless WanLayerNorm)
        let n1 = layerNorm(x0, nil, nil, eps).asType(.float32) * (1 + e1[0..., .newAxis, 0...]) + e0[0..., .newAxis, 0...]
        var x = x0 + selfAttn(n1, applyRope: applyRope) * e2[0..., .newAxis, 0...]

        // cross-attention (norm3 affine) + ffn (norm2 parameterless)
        x = x + crossAttn(norm3(x), context: context)
        let n2 = layerNorm(x, nil, nil, eps).asType(.float32) * (1 + e4[0..., .newAxis, 0...]) + e3[0..., .newAxis, 0...]
        x = x + ffn(n2) * e5[0..., .newAxis, 0...]
        return x
    }
}

final class SCAILHead: Module, @unchecked Sendable {
    let eps: Float
    @ModuleInfo(key: "head") var head: Linear
    @ModuleInfo(key: "modulation") var modulation: MLXArray  // [1,2,dim]
    init(_ dim: Int, _ outDim: Int, _ patch: (Int, Int, Int), eps: Float) {
        self.eps = eps
        self._head.wrappedValue = Linear(dim, patch.0 * patch.1 * patch.2 * outDim)
        self._modulation.wrappedValue = MLXArray.zeros([1, 2, dim])
        super.init()
    }
    func callAsFunction(_ x: MLXArray, e: MLXArray) -> MLXArray {
        // e: [B,dim]; modulation [1,2,dim] + e[:,None,:]
        let mod = modulation.asType(.float32) + e.asType(.float32)[0..., .newAxis, 0...]
        let e0 = mod[0..., 0], e1 = mod[0..., 1]
        let n = layerNorm(x, nil, nil, eps) * (1 + e1[0..., .newAxis, 0...]) + e0[0..., .newAxis, 0...]
        return head(n)
    }
}

// img_emb MLPProj: Sequential(LayerNorm, Linear, GELU(exact), Linear, LayerNorm)
// -> proj.layers.0/1/3/4. Keys: img_emb.proj.layers.{0,1,3,4}.
final class MLPProj: Module, @unchecked Sendable {
    @ModuleInfo(key: "proj") var proj: Sequential
    init(_ inDim: Int, _ outDim: Int) {
        self._proj.wrappedValue = Sequential {
            LayerNorm(dimensions: inDim); Linear(inDim, inDim); GELU()
            Linear(inDim, outDim); LayerNorm(dimensions: outDim)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

public final class SCAILModel: Module, @unchecked Sendable {
    public let dim: Int, numHeads: Int, numLayers: Int, freqDim: Int, outDim: Int, textLen: Int
    public let patch: (Int, Int, Int)
    let eps: Float
    let rope: RoPE3Seg

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "patch_embedding_pose") var patchEmbeddingPose: Conv3d
    @ModuleInfo(key: "patch_embedding_mask") var patchEmbeddingMask: Conv3d
    @ModuleInfo(key: "text_embedding") var textEmbedding: Sequential
    @ModuleInfo(key: "time_embedding") var timeEmbedding: Sequential
    @ModuleInfo(key: "time_projection") var timeProjection: Sequential
    @ModuleInfo(key: "img_emb") var imgEmb: MLPProj
    @ModuleInfo(key: "blocks") var blocks: [SCAILBlock]
    @ModuleInfo(key: "head") var head: SCAILHead

    public init(
        patchSize: (Int, Int, Int) = (1, 2, 2), textLen: Int = 512,
        inDim: Int = 20, maskDim: Int = 28, dim: Int = 5120, ffnDim: Int = 13824,
        freqDim: Int = 256, textDim: Int = 4096, outDim: Int = 16,
        numHeads: Int = 40, numLayers: Int = 40, eps: Float = 1e-6
    ) {
        self.dim = dim; self.numHeads = numHeads; self.numLayers = numLayers
        self.freqDim = freqDim; self.outDim = outDim; self.textLen = textLen
        self.patch = patchSize; self.eps = eps
        self.rope = RoPE3Seg(headDim: dim / numHeads)

        let ks = IntOrTriple([patchSize.0, patchSize.1, patchSize.2])
        self._patchEmbedding.wrappedValue = Conv3d(inputChannels: inDim, outputChannels: dim, kernelSize: ks, stride: ks)
        self._patchEmbeddingPose.wrappedValue = Conv3d(inputChannels: inDim, outputChannels: dim, kernelSize: ks, stride: ks)
        self._patchEmbeddingMask.wrappedValue = Conv3d(inputChannels: maskDim, outputChannels: dim, kernelSize: ks, stride: ks)
        self._textEmbedding.wrappedValue = Sequential {
            Linear(textDim, dim); GELU(approximation: .tanh); Linear(dim, dim)
        }
        self._timeEmbedding.wrappedValue = Sequential { Linear(freqDim, dim); SiLU(); Linear(dim, dim) }
        self._timeProjection.wrappedValue = Sequential { SiLU(); Linear(dim, dim * 6) }
        self._imgEmb.wrappedValue = MLPProj(1280, dim)
        self._blocks.wrappedValue = (0..<numLayers).map { _ in SCAILBlock(dim, ffnDim, numHeads, eps: eps, textLen: textLen) }
        self._head.wrappedValue = SCAILHead(dim, outDim, patchSize, eps: eps)
        super.init()
    }

    private func conv3d(_ c: Conv3d, _ xNCDHW: MLXArray) -> MLXArray {
        c(xNCDHW.transposed(0, 2, 3, 4, 1).asType(c.weight.dtype))
    }
    private func onesMask(_ x: MLXArray, _ md: Int) -> MLXArray {
        let s = x.shape
        return concatenated([x, MLXArray.ones([s[0], md, s[2], s[3], s[4]], dtype: x.dtype)], axis: 1)
    }
    private func zerosMask(_ x: MLXArray, _ md: Int) -> MLXArray {
        let s = x.shape
        return concatenated([x, MLXArray.zeros([s[0], md, s[2], s[3], s[4]], dtype: x.dtype)], axis: 1)
    }

    /// Single-batch forward. x/refLatents/etc are [C,F,H,W]; context [L,textDim];
    /// clipFea [1,257,1280]; t scalar. Returns [outDim, F, H, W].
    public func callAsFunction(
        x x0: MLXArray, poseLatents: MLXArray, drivingMasks: MLXArray,
        refLatents: MLXArray, refMasks: MLXArray, t: MLXArray,
        context ctx0: MLXArray, clipFea: MLXArray, replaceFlag: Bool,
        historyMask: MLXArray? = nil
    ) -> MLXArray {
        var x = x0[.newAxis]                 // [1,C,F,H,W]
        var refLat = refLatents[.newAxis]
        var poseLat = poseLatents[.newAxis]
        let driving = drivingMasks[.newAxis]
        let refM = refMasks[.newAxis]

        if let hm = historyMask {
            x = concatenated([x, hm[.newAxis]], axis: 1)
        } else {
            x = zerosMask(x, 4)
        }
        refLat = onesMask(refLat, 4)
        poseLat = onesMask(poseLat, 4)

        let (B, _, T, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let pcount = patch.0 * patch.1 * patch.2
        let refLength = (1 * H * W) / pcount
        let seqLength = T * refLength
        let poseLength = T * (H / 2) * (W / 2) / pcount

        x = concatenated([refLat, x], axis: 2)
        var emb = conv3d(patchEmbedding, x)
        emb = emb + conv3d(patchEmbeddingMask, refM)
        var poseEmb = conv3d(patchEmbeddingPose, poseLat)
        poseEmb = poseEmb + conv3d(patchEmbeddingMask, driving)
        var h = concatenated([emb.reshaped(B, -1, dim), poseEmb.reshaped(B, -1, dim)], axis: 1)

        // time embedding (fp32)
        let eVec = timeEmbedding(sinusoidalEmbedding1D(freqDim, t))   // [B,dim]
        let e0 = timeProjection(eVec).reshaped(-1, 6, dim)            // [B,6,dim]

        // context: text-embed padded to textLen, prepend CLIP img tokens
        let pad = textLen - ctx0.dim(0)
        let ctxPadded = concatenated([ctx0, MLXArray.zeros([pad, ctx0.dim(1)], dtype: ctx0.dtype)], axis: 0)[.newAxis]
        var context = textEmbedding(ctxPadded)
        context = concatenated([imgEmb(clipFea), context], axis: 1)

        let (rt, rh, rw) = (T / patch.0, H / patch.1, W / patch.2)
        let tShift = SegShifts(ref: 0, video: replaceFlag ? 0 : 1, pose: replaceFlag ? 0 : 1)
        let hShift = SegShifts(ref: replaceFlag ? 120 : 0, video: 0, pose: 0)
        let wShift = SegShifts(ref: 0, video: 0, pose: 120)
        let applyRope: (MLXArray) -> MLXArray = { q in
            self.rope.apply(q, F: rt, H: rh, W: rw,
                            refLen: refLength, vidLen: seqLength, poseLen: poseLength,
                            shifts: tShift, hShift, wShift)
        }

        for block in blocks {
            h = block(h, e: e0, context: context, applyRope: applyRope)
            eval(h)  // bound the fp32 graph at large seqLen (family runBlocks discipline)
        }
        h = head(h, e: eVec)
        return unpatchify(h, rt, rh, rw, offset: refLength)
    }

    // keep only the denoised video segment (skip ref tokens; pose tokens fall
    // beyond offset+prod(grid)). einsum 'fhwpqrc->cfphqwr'.
    private func unpatchify(_ x: MLXArray, _ f: Int, _ h: Int, _ w: Int, offset: Int) -> MLXArray {
        let c = outDim
        let u = x[0, offset ..< (offset + f * h * w)]
            .reshaped(f, h, w, patch.0, patch.1, patch.2, c)
            .transposed(6, 0, 3, 1, 4, 2, 5)
        return u.reshaped(c, f * patch.0, h * patch.1, w * patch.2).asType(.float32)
    }
}
