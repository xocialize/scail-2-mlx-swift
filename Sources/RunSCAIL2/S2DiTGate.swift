// S2 full-DiT gate: SCAILModel forward vs oracle, CPU stream, both regimes.
// Loads all weights by the module's own flattened keys (verify .all) — this
// also re-proves the S0/S0b key contract against the live module.
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum S2DiTGate {
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
            let dim = meta["dim"] as? Int, let ffn = meta["ffn_dim"] as? Int,
            let heads = meta["num_heads"] as? Int, let layers = meta["num_layers"] as? Int,
            let freqDim = meta["freq_dim"] as? Int, let textDim = meta["text_dim"] as? Int,
            let inDim = meta["in_dim"] as? Int, let maskDim = meta["mask_dim"] as? Int,
            let outDim = meta["out_dim"] as? Int, let textLen = meta["text_len"] as? Int
        else { log("S2 dit: cannot read meta.json"); return 2 }

        var result: Int32 = 0
        Device.withDefaultDevice(.cpu) {
            let model = SCAILModel(
                patchSize: (1, 2, 2), textLen: textLen, inDim: inDim, maskDim: maskDim,
                dim: dim, ffnDim: ffn, freqDim: freqDim, textDim: textDim, outDim: outDim,
                numHeads: heads, numLayers: layers, eps: 1e-6)

            let want = Set(model.parameters().flattened().map { $0.0 })
            var params: [String: MLXArray] = [:]
            for key in want {
                params[key] = load("w_" + key.replacingOccurrences(of: ".", with: "__"))
            }
            do {
                try model.update(parameters: ModuleParameters.unflattened(params), verify: [.all])
            } catch {
                log("S2 dit: weight load FAILED — \(error)"); result = 1; return
            }
            log("S2 dit: key contract OK (\(want.count) params, verify .all)")

            for (regime, replace) in [("anim", false), ("replace", true)] {
                let out = model(
                    x: load("in_x"), poseLatents: load("in_pose_latents"),
                    drivingMasks: load("in_driving_masks"), refLatents: load("in_ref_latents"),
                    refMasks: load("in_ref_masks"), t: load("in_t"),
                    context: load("in_context"), clipFea: load("in_clip_fea"),
                    replaceFlag: replace)
                let d = abs(out.asType(.float32) - load("out_\(regime)").asType(.float32)).max().item(Float.self)
                // full-DiT pass threshold (mlx-porting: < 1e-2; fast-SDPA on CPU here)
                if d < 5e-3 {
                    log("S2 dit PASS \(regime): max_abs=\(d)")
                } else {
                    log("S2 dit FAIL \(regime): max_abs=\(d) (gate 5e-3)"); result = 1
                }
            }
        }
        log(result == 0 ? "S2 DIT GATE PASSED" : "S2 DIT GATE FAILED")
        return result
    }
}
