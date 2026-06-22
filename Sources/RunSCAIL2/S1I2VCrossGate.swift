// S1 i2v cross-attn gate: SCAIL I2VCrossAttention vs oracle, CPU stream.
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum S1I2VCrossGate {
    static func run(fixtureDir: String) -> Int32 {
        let err = FileHandle.standardError
        func log(_ s: String) { err.write(Data((s + "\n").utf8)) }
        let dir = URL(fileURLWithPath: fixtureDir)
        func load(_ n: String) -> MLXArray {
            try! loadNumpy(url: dir.appendingPathComponent("\(n).npy"))
        }

        guard
            let md = FileManager.default.contents(
                atPath: dir.appendingPathComponent("meta.json").path),
            let meta = try? JSONSerialization.jsonObject(with: md) as? [String: Int]
        else { log("S1 i2vcross: cannot read meta.json"); return 2 }
        let (dim, heads, textLen) = (meta["dim"]!, meta["heads"]!, meta["textLen"]!)

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            let attn = I2VCrossAttention(dim, heads, qkNorm: true, textLen: textLen)
            // load dumped weights under SCAIL key names (w_<key__with__dunders>)
            let names = ["q", "k", "v", "o", "k_img", "v_img"]
            var params: [String: MLXArray] = [:]
            for p in names {
                params["\(p).weight"] = load("w_\(p)__weight")
                params["\(p).bias"] = load("w_\(p)__bias")
            }
            for n in ["norm_q", "norm_k", "norm_k_img"] {
                params[n] = load("w_\(n)__weight")
            }
            try! attn.update(parameters: ModuleParameters.unflattened(params), verify: [.all])

            let out = attn(load("x"), context: load("context"))
            let d = abs(out.asType(.float32) - load("out").asType(.float32)).max().item(Float.self)
            if d < 1e-3 {
                log("S1 i2vcross PASS: max_abs=\(d)")
            } else {
                log("S1 i2vcross FAIL: max_abs=\(d) (gate 1e-3)"); result = 1
            }
        }
        log(result == 0 ? "S1 I2VCROSS GATE PASSED" : "S1 I2VCROSS GATE FAILED")
        return result
    }
}
