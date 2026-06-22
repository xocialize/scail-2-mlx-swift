// S2 CLIP-preprocess gate: bicubic matrices + full preprocess vs oracle, CPU.
import Foundation
import MLX
import SCAIL2
import WanCore

enum S2CLIPPreGate {
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

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            for h in [123, 224, 360] {
                // identical taps/weights; residual is fp32 rounding-order skew
                // (Double-built matrix vs torch's two separable passes), <5e-4
                let d = maxAbs(CLIPPreprocess.bicubicMatrix(h, 224), load("bicubic_\(h)"))
                if d < 5e-4 { log("S2 clippre PASS bicubic \(h)->224: max_abs=\(d)") }
                else { log("S2 clippre FAIL bicubic \(h): \(d)"); result = 1 }
            }
            // full preprocess: pre_in is [3,1,H,W] -> wrap as one "video" [C,T,H,W]
            let out = CLIPPreprocess.preprocess([load("pre_in")])
            let d = maxAbs(out, load("pre_out"))
            if d < 5e-4 { log("S2 clippre PASS preprocess: max_abs=\(d)") }
            else { log("S2 clippre FAIL preprocess: \(d)"); result = 1 }
        }
        log(result == 0 ? "S2 CLIPPRE GATE PASSED" : "S2 CLIPPRE GATE FAILED")
        return result
    }
}
