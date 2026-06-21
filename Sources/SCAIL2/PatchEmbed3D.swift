// SCAIL dual-mask patch embeds — three non-overlapping Conv3d (kernel=stride=
// patch_size) over NCDHW input. wan-core does its single patch embed as a
// reshaped linear (`patch_embedding_proj`), which doesn't match SCAIL's
// `patch_embedding{,_pose,_mask}.weight` Conv3d keys — so these are SCAIL-side.
//
// Mirrors the oracle `_conv3d_ncdhw`: transpose NCDHW->NDHWC, cast to weight
// dtype, conv. MLX-swift Conv3d weight layout [O,kt,kh,kw,I] == the stored
// checkpoint layout == the oracle nn.Conv3d layout.
import Foundation
import MLX
import MLXNN

public final class SCAILPatchEmbeds: Module, @unchecked Sendable {
    @ModuleInfo(key: "patch_embedding") public var patchEmbedding: Conv3d
    @ModuleInfo(key: "patch_embedding_pose") public var patchEmbeddingPose: Conv3d
    @ModuleInfo(key: "patch_embedding_mask") public var patchEmbeddingMask: Conv3d

    public init(inDim: Int, maskDim: Int, dim: Int, patch: (Int, Int, Int)) {
        let ks = IntOrTriple([patch.0, patch.1, patch.2])
        let st = IntOrTriple([patch.0, patch.1, patch.2])
        self._patchEmbedding.wrappedValue = Conv3d(
            inputChannels: inDim, outputChannels: dim, kernelSize: ks, stride: st)
        self._patchEmbeddingPose.wrappedValue = Conv3d(
            inputChannels: inDim, outputChannels: dim, kernelSize: ks, stride: st)
        self._patchEmbeddingMask.wrappedValue = Conv3d(
            inputChannels: maskDim, outputChannels: dim, kernelSize: ks, stride: st)
        super.init()
    }

    /// NCDHW [B,C,T,H,W] -> NDHWC embedded [B,T',H',W',dim], oracle `_conv3d_ncdhw`.
    public static func embed(_ conv: Conv3d, _ xNCDHW: MLXArray) -> MLXArray {
        let ndhwc = xNCDHW.transposed(0, 2, 3, 4, 1).asType(conv.weight.dtype)
        return conv(ndhwc)
    }
}
