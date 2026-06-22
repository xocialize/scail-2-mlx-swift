// SCAIL i2v cross-attention — Wan2.1-I2V: CLIP image tokens prepended to the
// text context, attended via a separate k_img/v_img path that SHARES q with
// the text attention, the two summed before the output projection. Net-new to
// the Wan family (wan-core `WanCrossAttention` is text-only). 1:1 translation
// of the oracle `WanI2VCrossAttention` (scail2_mlx/modules/model_scail2.py).
//
// Keys match the checkpoint's `blocks.N.cross_attn.{q,k,v,o,k_img,v_img,
// norm_q,norm_k,norm_k_img}`.
import Foundation
import MLX
import MLXFast
import MLXNN

public final class I2VCrossAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let eps: Float
    let textLen: Int  // T5_CONTEXT_TOKEN_NUMBER (512) — img tokens are the prefix

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "k_img") var kImg: Linear
    @ModuleInfo(key: "v_img") var vImg: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNormW
    @ModuleInfo(key: "norm_k") var normK: RMSNormW
    @ModuleInfo(key: "norm_k_img") var normKImg: RMSNormW

    public init(_ dim: Int, _ numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6, textLen: Int = 512) {
        precondition(dim % numHeads == 0)
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(dim / numHeads), -0.5)
        self.eps = eps
        self.textLen = textLen
        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._kImg.wrappedValue = Linear(dim, dim)
        self._vImg.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = RMSNormW(dim, eps: eps)
        self._normK.wrappedValue = RMSNormW(dim, eps: eps)
        self._normKImg.wrappedValue = RMSNormW(dim, eps: eps)
        super.init()
    }

    // q/k/v: [B, L, dim] -> [B, heads, L, headDim] for SDPA
    private func heads(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        return x.reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let imgLen = context.dim(1) - textLen
        let contextImg = context[0..., 0..<imgLen]
        let contextText = context[0..., imgLen...]
        let b = x.dim(0)

        let qT = heads(normQ(q(x)))
        let kT = heads(normK(k(contextText)))
        let vT = heads(v(contextText))
        let kI = heads(normKImg(kImg(contextImg)))
        let vI = heads(vImg(contextImg))

        let imgX = MLXFast.scaledDotProductAttention(queries: qT, keys: kI, values: vI, scale: scale, mask: nil)
        let txtX = MLXFast.scaledDotProductAttention(queries: qT, keys: kT, values: vT, scale: scale, mask: nil)

        func merge(_ a: MLXArray) -> MLXArray {
            a.transposed(0, 2, 1, 3).reshaped(b, -1, numHeads * headDim)
        }
        return o(merge(txtX) + merge(imgX))
    }
}
