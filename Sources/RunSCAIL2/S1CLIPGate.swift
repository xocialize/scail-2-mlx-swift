// S1 CLIP visual gate: SCAIL CLIPVisionTower (use_31_block) vs oracle, CPU stream.
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum S1CLIPGate {
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
            let dim = meta["dim"] as? Int, let heads = meta["num_heads"] as? Int,
            let layers = meta["num_layers"] as? Int, let imSize = meta["image_size"] as? Int,
            let patch = meta["patch_size"] as? Int, let mlpR = meta["mlp_ratio"] as? Int
        else { log("S1 clip: cannot read meta.json"); return 2 }

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            let vit = CLIPVisionTower(
                imageSize: imSize, patchSize: patch, dim: dim, mlpRatio: mlpR,
                numHeads: heads, numLayers: layers, preNorm: true, eps: 1e-5)

            // assemble params from dumped w_<key__dunders> (the tree-flattened keys)
            let want = Set(vit.parameters().flattened().map { $0.0 })
            var params: [String: MLXArray] = [:]
            for key in want {
                params[key] = load("w_" + key.replacingOccurrences(of: ".", with: "__"))
            }
            try! vit.update(parameters: ModuleParameters.unflattened(params), verify: [.all])

            let out = vit(load("x"), use31Block: true)
            let d = abs(out.asType(.float32) - load("out").asType(.float32)).max().item(Float.self)
            if d < 1e-3 {
                log("S1 clip PASS (use_31_block): max_abs=\(d)")
            } else {
                log("S1 clip FAIL: max_abs=\(d) (gate 1e-3)"); result = 1
            }
        }
        log(result == 0 ? "S1 CLIP GATE PASSED" : "S1 CLIP GATE FAILED")
        return result
    }
}
