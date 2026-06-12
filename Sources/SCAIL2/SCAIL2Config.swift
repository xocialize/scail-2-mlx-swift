// Resolved model configuration — mirror of the oracle's
// scail2_mlx/configs/scail_config_14B.py. Values are oracle truths; do not
// introduce defaults that differ from the converted checkpoint's config.
import Foundation

public struct SCAIL2Config: Codable, Sendable {
    public var modelType: String = "i2v"
    public var patchSize: [Int] = [1, 2, 2]
    public var textLen: Int = 512
    public var inDim: Int = 20  // 16 latent + 4 i2v mask channels
    public var maskDim: Int = 28
    public var dim: Int = 5120
    public var ffnDim: Int = 13824
    public var freqDim: Int = 256
    public var outDim: Int = 16
    public var numHeads: Int = 40
    public var numLayers: Int = 40
    public var qkNorm: Bool = true
    public var crossAttnNorm: Bool = true
    public var eps: Float = 1e-6
    public var vaeStride: [Int] = [4, 8, 8]
    public var numTrainTimesteps: Int = 1000
    public var sampleFps: Int = 16

    public init() {}
}
