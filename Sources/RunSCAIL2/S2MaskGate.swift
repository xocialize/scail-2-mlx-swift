// S2 mask-compress + half-bilinear gate vs oracle, CPU stream.
import Foundation
import MLX
import SCAIL2
import WanCore

enum S2MaskGate {
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
            let c = maxAbs(MaskCompress.extract(load("mask")), load("compress_out"))
            if c < 1e-6 { log("S2 mask PASS compress: max_abs=\(c)") }
            else { log("S2 mask FAIL compress: \(c)"); result = 1 }

            let h = maxAbs(MaskCompress.halfBilinear(load("hb_in")), load("hb_out"))
            if h < 1e-6 { log("S2 mask PASS half_bilinear: max_abs=\(h)") }
            else { log("S2 mask FAIL half_bilinear: \(h)"); result = 1 }
        }
        log(result == 0 ? "S2 MASK GATE PASSED" : "S2 MASK GATE FAILED")
        return result
    }
}
