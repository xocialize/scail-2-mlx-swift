// CLIP visual preprocessing — torch-kernel-exact bicubic resize to 224 as a
// separable matrix, then [-1,1] -> [0,1] -> CLIP mean/std normalize. 1:1 from
// oracle clip.py (bicubic_resize_matrix + CLIPModel.visual). Matrix built in
// Double (skill: freq/interp tables in Double before fp32 cast).
import Foundation
import MLX

public enum CLIPPreprocess {
    public static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public static let std: [Float] = [0.26862954, 0.26130258, 0.27577711]

    private static func cubicKernel(_ x: Double, a: Double = -0.75) -> Double {
        let ax = abs(x)
        if ax <= 1 { return (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1 }
        if ax < 2 { return a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a }
        return 0
    }

    /// [outSize, inSize] matrix M with (M @ x) == torch F.interpolate(bicubic,
    /// align_corners=false), border-clamped. Built in Double.
    public static func bicubicMatrix(_ inSize: Int, _ outSize: Int) -> MLXArray {
        let scale = Double(inSize) / Double(outSize)
        var m = [Float](repeating: 0, count: outSize * inSize)
        for i in 0..<outSize {
            let center = (Double(i) + 0.5) * scale - 0.5
            let t0 = Int(floor(center))
            for tap in (t0 - 1)...(t0 + 2) {
                let w = cubicKernel(center - Double(tap))
                let col = min(max(tap, 0), inSize - 1)
                m[i * inSize + col] += Float(w)
            }
        }
        return MLXArray(m, [outSize, inSize])
    }

    /// videos: list of [C, T, H, W] in [-1, 1]. Returns [sum_T, 3, size, size]
    /// CLIP-normalized, ready for CLIPVisionTower. Mirrors CLIPModel.visual.
    public static func preprocess(_ videos: [MLXArray], size: Int = 224) -> MLXArray {
        let meanA = MLXArray(mean).reshaped(1, 3, 1, 1)
        let stdA = MLXArray(std).reshaped(1, 3, 1, 1)
        var resized: [MLXArray] = []
        for u in videos {
            // u: [C,T,H,W] -> [T,C,H,W]
            let x = u.transposed(1, 0, 2, 3).asType(.float32)
            let (h, w) = (x.dim(2), x.dim(3))
            let mh = bicubicMatrix(h, size)   // [size, h]
            let mw = bicubicMatrix(w, size)   // [size, w]
            resized.append(matmul(matmul(mh, x), mw.transposed()))  // [T,C,size,size]
        }
        let v = concatenated(resized, axis: 0)
        return (v * 0.5 + 0.5 - meanA) / stdA
    }
}
