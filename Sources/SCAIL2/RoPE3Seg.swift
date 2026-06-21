// SCAIL 3-segment RoPE — the net-new application over the shared Wan frequency
// table. 1:1 translation of the oracle `rope_apply_{ref,video,pose}` +
// `rope_apply_scail` (scail2_mlx/modules/model_scail2.py). The freq table itself
// is wan-core `ropeParams` (band construction 44/42/42 is identical — verified
// against WanModel.swift), so this file owns ONLY the segment dispatch:
//   - per-segment (t,h,w) shifts (animation vs replace regimes)
//   - the pose segment's 2x2 avg-pool of the complex grid (cos/sin pooled
//     separately — intentionally non-unit rotations, replicated exactly)
//
// Layout note: wan-core `ropeParams(maxSeqLen, dim)` returns [seq, halfD, 2]
// with cos at [...,0], sin at [...,1]. We split that into plain cos/sin arrays
// so the math mirrors the oracle's (cos, sin) tuple structure.
import Foundation
import MLX
import WanCore

public struct SegShifts: Sendable {
    public var ref: Int, video: Int, pose: Int
    public init(ref: Int, video: Int, pose: Int) { self.ref = ref; self.video = video; self.pose = pose }
}

public struct RoPE3Seg {
    public let cos: MLXArray   // [maxSeq, halfD]
    public let sin: MLXArray   // [maxSeq, halfD]

    /// Build the SCAIL freq table from wan-core `ropeParams` (reused; bands
    /// d-4*(d/6) / 2*(d/6) / 2*(d/6) match the oracle).
    public init(headDim d: Int, maxSeqLen: Int = 8192) {
        let p0 = ropeParams(maxSeqLen, d - 4 * (d / 6))  // [seq, half0, 2]
        let p1 = ropeParams(maxSeqLen, 2 * (d / 6))
        let p2 = ropeParams(maxSeqLen, 2 * (d / 6))
        let table = concatenated([p0, p1, p2], axis: 1)   // [seq, halfD, 2]
        self.cos = table[.ellipsis, 0]
        self.sin = table[.ellipsis, 1]
    }

    /// Init directly from a precomputed (cos, sin) table — used by the S1 gate
    /// to inject the oracle's exact table.
    public init(cos: MLXArray, sin: MLXArray) { self.cos = cos; self.sin = sin }

    // c = halfD; band split [c - 2*(c/3), c/3, c/3]
    private func bands(_ grid: MLXArray, _ c: Int) -> (MLXArray, MLXArray, MLXArray) {
        let s1 = c / 3
        let s0 = c - 2 * s1
        return (grid[0..., 0..<s0], grid[0..., s0..<(s0 + s1)], grid[0..., (s0 + s1)...])
    }

    // Build the [f,h,w,c] grid for one component (cos or sin) from shifted bands.
    private func grid(
        _ comp: MLXArray, _ c: Int, _ f: Int, _ h: Int, _ w: Int,
        _ sf: Int, _ sh: Int, _ sw: Int
    ) -> MLXArray {
        let (bt, bh, bw) = bands(comp, c)
        let s1 = c / 3
        let s0 = c - 2 * s1
        let gt = broadcast(bt[sf..<(sf + f)].reshaped(f, 1, 1, s0), to: [f, h, w, s0])
        let gh = broadcast(bh[sh..<(sh + h)].reshaped(1, h, 1, s1), to: [f, h, w, s1])
        let gw = broadcast(bw[sw..<(sw + w)].reshaped(1, 1, w, s1), to: [f, h, w, s1])
        return concatenated([gt, gh, gw], axis: -1)  // [f,h,w,c]
    }

    // Complex rotation: x [B,S,N,D] with cos/sin grids [S,c] (c = D/2).
    private func rotate(_ x: MLXArray, _ cosG: MLXArray, _ sinG: MLXArray) -> MLXArray {
        let (b, s, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xr = x.asType(.float32).reshaped(b, s, n, d / 2, 2)
        let xRe = xr[.ellipsis, 0]
        let xIm = xr[.ellipsis, 1]
        let fc = cosG.reshaped(1, s, 1, -1)
        let fs = sinG.reshaped(1, s, 1, -1)
        let outRe = xRe * fc - xIm * fs
        let outIm = xRe * fs + xIm * fc
        return stacked([outRe, outIm], axis: -1).reshaped(b, s, n, d)
    }

    private func segmentRef(_ x: MLXArray, _ f: Int, _ h: Int, _ w: Int, _ sh: SegShifts2) -> MLXArray {
        let c = x.dim(3) / 2
        let cosG = grid(cos, c, f, h, w, sh.t, sh.h, sh.w).reshaped(f * h * w, -1)
        let sinG = grid(sin, c, f, h, w, sh.t, sh.h, sh.w).reshaped(f * h * w, -1)
        return rotate(x, cosG, sinG)
    }

    private func segmentPose(_ x: MLXArray, _ f: Int, _ h: Int, _ w: Int, _ sh: SegShifts2) -> MLXArray {
        let c = x.dim(3) / 2
        let seqLen = f * (h / 2) * (w / 2)
        var cosG = grid(cos, c, f, h, w, sh.t, sh.h, sh.w)  // [f,h,w,c]
        var sinG = grid(sin, c, f, h, w, sh.t, sh.h, sh.w)
        // avg_pool2d(k=2,s=2) over (h,w), cos/sin separately
        cosG = cosG.reshaped(f, h / 2, 2, w / 2, 2, c).mean(axes: [2, 4])
        sinG = sinG.reshaped(f, h / 2, 2, w / 2, 2, c).mean(axes: [2, 4])
        return rotate(x, cosG.reshaped(seqLen, -1), sinG.reshaped(seqLen, -1))
    }

    /// `rope_apply_scail`: split [ref | video | pose], apply each with its shifts.
    public func apply(
        _ x: MLXArray, F: Int, H: Int, W: Int,
        refLen: Int, vidLen: Int, poseLen: Int,
        shifts t: SegShifts, _ hsh: SegShifts, _ wsh: SegShifts
    ) -> MLXArray {
        let xRef = x[0..., 0..<refLen]
        let xVid = x[0..., refLen..<(refLen + vidLen)]
        let xPose = x[0..., (x.dim(1) - poseLen)...]
        let outRef = segmentRef(xRef, 1, H, W, SegShifts2(t.ref, hsh.ref, wsh.ref))
        let outVid = segmentRef(xVid, F, H, W, SegShifts2(t.video, hsh.video, wsh.video))
        let outPose = segmentPose(xPose, F, H, W, SegShifts2(t.pose, hsh.pose, wsh.pose))
        return concatenated([outRef, outVid, outPose], axis: 1)
    }
}

struct SegShifts2 { let t: Int, h: Int, w: Int; init(_ t: Int, _ h: Int, _ w: Int) { self.t = t; self.h = h; self.w = w } }
