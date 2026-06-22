// SCAIL 28-channel mask compression — net-new. RGB segmentation mask
// [3,T,H,W] in [-1,1] -> 28-ch binary latent [28, T_lat, H_lat, W_lat] in
// {0,1}, no VAE. 7 color-coded channels (white/red/green/blue/yellow/magenta/
// cyan) x 4 packed frames. 1:1 translation of oracle
// extract_and_compress_mask_to_latent (scail2_mlx/utils/scail_utils.py).
import Foundation
import MLX

public enum MaskCompress {
    /// mask: [3, T, H, W] in [-1, 1]. Returns [28, T_lat, H_lat, W_lat].
    public static func extract(
        _ mask: MLXArray, additionalSpatialDownsample: Int = 1, temporalStride: Int = 4
    ) -> MLXArray {
        let (_, T, H, W) = (mask.dim(0), mask.dim(1), mask.dim(2), mask.dim(3))
        let onThresh: Float = (225.0 - 127.5) / 127.5  // pixel >= 225 counts as "on"

        let m = mask.transposed(1, 0, 2, 3).asType(.float32)  // [T,3,H,W]
        let R = (m[0..., 0..<1] .> onThresh).asType(.float32)
        let G = (m[0..., 1..<2] .> onThresh).asType(.float32)
        let B = (m[0..., 2..<3] .> onThresh).asType(.float32)
        let nR = 1 - R, nG = 1 - G, nB = 1 - B
        var binary7 = concatenated(
            [R * G * B, R * nG * nB, nR * G * nB, nR * nG * B, R * G * nB, R * nG * B, nR * G * B],
            axis: 1)  // [T,7,H,W]

        var hLat = H, wLat = W
        if additionalSpatialDownsample > 1 {
            hLat /= additionalSpatialDownsample
            wLat /= additionalSpatialDownsample
        }
        for _ in 0..<3 { hLat = (hLat + 1) / 2; wLat = (wLat + 1) / 2 }
        // area interpolate == uniform mean pool at integral ratio (divisible-by-32 input)
        precondition(H % hLat == 0 && W % wLat == 0, "area interpolate needs integral ratio")
        let (kh, kw) = (H / hLat, W / wLat)
        let (t, c7) = (binary7.dim(0), binary7.dim(1))
        binary7 = binary7.reshaped(t, c7, hLat, kh, wLat, kw).mean(axes: [3, 5])  // [T,7,hLat,wLat]

        let tLat = (T - 1) / temporalStride + 1
        let padded = concatenated(
            [repeated(binary7[0..<1], count: temporalStride, axis: 0), binary7[1...]], axis: 0)
        return padded.reshaped(tLat, temporalStride * 7, hLat, wLat).transposed(1, 0, 2, 3)
    }

    /// F.interpolate(scale_factor=0.5, mode='bilinear', align_corners=False) at
    /// an exact 1/2 ratio reduces to 2x2 mean pooling (oracle _half_bilinear).
    public static func halfBilinear(_ x: MLXArray) -> MLXArray {
        let n = x.ndim
        let (h, w) = (x.dim(n - 2), x.dim(n - 1))
        precondition(h % 2 == 0 && w % 2 == 0)
        var shape = x.shape
        shape.removeLast(2)
        shape.append(contentsOf: [h / 2, 2, w / 2, 2])
        return x.reshaped(shape).mean(axes: [-3, -1])
    }
}
