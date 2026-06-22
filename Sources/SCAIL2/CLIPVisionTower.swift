// SCAIL CLIP visual tower — open-clip xlm-roberta ViT-H/14 VISUAL branch only
// (the checkpoint is "onlyvisual"; the pipeline calls `visual(use_31_block:)`).
// Net-new to the Wan family. 1:1 translation of the oracle clip.py VISUAL path
// (VisionTransformer + AttentionBlock + SelfAttention + pre-norm LayerNorm).
// `use_31_block` returns the 2nd-to-last block output ([B,257,1280] at prod) —
// the text branch, AttentionPool, post_norm and head are unused, so omitted.
//
// Conv2d patch embed: kernel=stride=patch_size (non-overlapping), bias=false
// (pre_norm=true). ViT-H activation='gelu' => EXACT gelu (not tanh approx).
import Foundation
import MLX
import MLXFast
import MLXNN

final class CLIPSelfAttention: Module, @unchecked Sendable {
    let numHeads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ dim: Int, _ numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(dim / numHeads), -0.5)
        self._toQKV.wrappedValue = Linear(dim, dim * 3)
        self._proj.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, s, c) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = toQKV(x).reshaped(b, s, 3, numHeads, headDim)
        let q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        return proj(o.transposed(0, 2, 1, 3).reshaped(b, s, c))
    }
}

final class CLIPAttentionBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: CLIPSelfAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    // mlp = Sequential(Linear, GELU, Linear) -> native keys layers.0 / layers.2
    // (oracle's nn.Sequential has Dropout at 3 too, but it carries no params).
    @ModuleInfo(key: "mlp") var mlp: Sequential

    init(_ dim: Int, _ mlpRatio: Int, _ numHeads: Int, eps: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._attn.wrappedValue = CLIPSelfAttention(dim, numHeads)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps)
        self._mlp.wrappedValue = Sequential {
            Linear(dim, dim * mlpRatio); GELU(); Linear(dim * mlpRatio, dim)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // pre_norm (post_norm=false for ViT-H): x + attn(norm1(x)); x + mlp(norm2(x))
        var h = x + attn(norm1(x))
        h = h + mlp(norm2(h))
        return h
    }
}

public final class CLIPVisionTower: Module, @unchecked Sendable {
    public let dim: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "cls_embedding") var clsEmbedding: MLXArray
    @ModuleInfo(key: "pos_embedding") var posEmbedding: MLXArray
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm?
    @ModuleInfo(key: "transformer") var transformer: [CLIPAttentionBlock]
    // post_norm / head exist in the checkpoint but are unused under use_31_block.

    public init(
        imageSize: Int = 224, patchSize: Int = 14, dim: Int = 1280,
        mlpRatio: Int = 4, numHeads: Int = 16, numLayers: Int = 32,
        preNorm: Bool = true, eps: Float = 1e-5
    ) {
        self.dim = dim
        let ks = IntOrPair([patchSize, patchSize])
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: dim, kernelSize: ks, stride: ks, bias: !preNorm)
        let numPatches = (imageSize / patchSize) * (imageSize / patchSize)
        self._clsEmbedding.wrappedValue = MLXArray.zeros([1, 1, dim])
        self._posEmbedding.wrappedValue = MLXArray.zeros([1, numPatches + 1, dim])
        self._preNorm.wrappedValue = preNorm ? LayerNorm(dimensions: dim, eps: eps) : nil
        self._transformer.wrappedValue = (0..<numLayers).map {
            _ in CLIPAttentionBlock(dim, mlpRatio, numHeads, eps: eps)
        }
        super.init()
    }

    /// x: [B,3,H,W] NCHW. use_31_block=true returns the 2nd-to-last block output
    /// (the SCAIL conditioning tokens). Conv runs NHWC.
    public func callAsFunction(_ x: MLXArray, use31Block: Bool = true) -> MLXArray {
        let b = x.dim(0)
        var h = patchEmbedding(x.transposed(0, 2, 3, 1))  // [B,h',w',dim]
        h = h.reshaped(b, -1, dim)
        h = concatenated([broadcast(clsEmbedding, to: [b, 1, dim]), h], axis: 1)
        h = h + posEmbedding
        if let preNorm { h = preNorm(h) }
        let upper = use31Block ? transformer.count - 1 : transformer.count
        for i in 0..<upper { h = transformer[i](h) }
        return h
    }
}
