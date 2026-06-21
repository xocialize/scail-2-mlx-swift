// S1 patch-embed gate: SCAIL Conv3d patchify vs oracle, CPU stream, both
// channel widths (20-ch latent path, 28-ch mask path).
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum S1PatchEmbedGate {
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
            let meta = try? JSONSerialization.jsonObject(with: md) as? [String: Any],
            let dim = meta["dim"] as? Int,
            let p = meta["patch"] as? [Int]
        else { log("S1 patchembed: cannot read meta.json"); return 2 }

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            for (tag, inCh) in [("latent", 20), ("mask", 28)] {
                let conv = Conv3d(
                    inputChannels: inCh, outputChannels: dim,
                    kernelSize: IntOrTriple(p), stride: IntOrTriple(p))
                try! conv.update(parameters: ModuleParameters.unflattened([
                    "weight": load("\(tag)_weight"), "bias": load("\(tag)_bias"),
                ]), verify: [.all])
                let out = SCAILPatchEmbeds.embed(conv, load("\(tag)_x"))
                let d = abs(out.asType(.float32) - load("\(tag)_out").asType(.float32))
                    .max().item(Float.self)
                if d < 1e-4 {
                    log("S1 patchembed PASS \(tag) (in=\(inCh)): max_abs=\(d)")
                } else {
                    log("S1 patchembed FAIL \(tag): max_abs=\(d) (gate 1e-4)"); result = 1
                }
            }
        }
        log(result == 0 ? "S1 PATCHEMBED GATE PASSED" : "S1 PATCHEMBED GATE FAILED")
        return result
    }
}
