// S1 RoPE gate: SCAIL 3-segment RoPE vs oracle fixtures, CPU stream.
// Two checks per regime:
//   (a) wan-core ropeParams reproduces the oracle freq table (band reuse proof)
//   (b) RoPE3Seg.apply matches rope_apply_scail (the net-new application)
// CPU stream per the M5 reduced-precision-GPU-matmul rule + family VAE doctrine.
import Foundation
import MLX
import SCAIL2
import WanCore

enum S1RoPEGate {
    static func run(fixtureDir: String) -> Int32 {
        let err = FileHandle.standardError
        func log(_ s: String) { err.write(Data((s + "\n").utf8)) }
        let dir = URL(fileURLWithPath: fixtureDir)

        func load(_ n: String) -> MLXArray {
            try! loadNumpy(url: dir.appendingPathComponent("\(n).npy"))
        }
        func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
            abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        }

        guard
            let metaData = FileManager.default.contents(
                atPath: dir.appendingPathComponent("meta.json").path),
            let meta = try? JSONDecoder().decode([String: Int].self, from: metaData)
        else { log("S1 RoPE: cannot read meta.json"); return 2 }

        let (F, H, W) = (meta["F"]!, meta["H"]!, meta["W"]!)
        let (refLen, vidLen, poseLen) = (meta["refLen"]!, meta["vidLen"]!, meta["poseLen"]!)
        let headDim = meta["headDim"]!

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            let oCos = load("freqs_cos"), oSin = load("freqs_sin")
            let x = load("x")

            // (a) band-construction reuse: wan-core ropeParams vs oracle table
            let built = RoPE3Seg(headDim: headDim, maxSeqLen: oCos.dim(0))
            let dCos = maxAbs(built.cos, oCos), dSin = maxAbs(built.sin, oSin)
            if dCos < 1e-6 && dSin < 1e-6 {
                log("S1 RoPE PASS freq-table: ropeParams reproduces oracle "
                    + "(cos \(dCos), sin \(dSin)) — band REUSE confirmed")
            } else {
                log("S1 RoPE FAIL freq-table: cos=\(dCos) sin=\(dSin)"); result = 1
            }

            // (b) the net-new 3-segment application, injecting the oracle table
            let rope = RoPE3Seg(cos: oCos, sin: oSin)
            for (regime, replace) in [("anim", false), ("replace", true)] {
                let t = SegShifts(ref: 0, video: replace ? 0 : 1, pose: replace ? 0 : 1)
                let hsh = SegShifts(ref: replace ? 120 : 0, video: 0, pose: 0)
                let wsh = SegShifts(ref: 0, video: 0, pose: 120)
                let out = rope.apply(
                    x, F: F, H: H, W: W,
                    refLen: refLen, vidLen: vidLen, poseLen: poseLen,
                    shifts: t, hsh, wsh)
                let d = maxAbs(out, load("out_\(regime)"))
                // RoPE units gate ≤1e-4 (multi-frame temporal still bit-tight
                // here at 4 frames; family E16 <0.02 is the e2e-scale caveat)
                if d < 1e-4 {
                    log("S1 RoPE PASS \(regime): max_abs=\(d)")
                } else {
                    log("S1 RoPE FAIL \(regime): max_abs=\(d) (gate 1e-4)"); result = 1
                }
            }
        }
        log(result == 0 ? "S1 ROPE GATE PASSED" : "S1 ROPE GATE FAILED")
        return result
    }
}
