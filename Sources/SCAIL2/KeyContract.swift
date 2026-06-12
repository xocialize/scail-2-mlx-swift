// S0 key contract — the expected parameter key set for every component of
// xocialize/SCAIL-2-bf16, generated programmatically for the regular
// components (dit / umt5 / clip). The VAE's sequential indices are irregular
// and live pinned in Tests/SCAIL2Tests/Fixtures/key_contract.json.
//
// Naming notes (vs the bernini-r-mlx-swift donor):
//   - DiT FFN keys are `ffn.layers.{0,2}` (oracle's mlx Sequential), NOT the
//     donor's `ffn.{fc1,fc2}` — Swift modules use literal "0"/"2" ModuleInfo
//     keys under a "layers" child (native match; no load-time remap).
//   - cross_attn carries the i2v extras: k_img / v_img / norm_k_img.
import Foundation

public enum KeyContract {
    static func wb(_ p: String) -> [String] { ["\(p).weight", "\(p).bias"] }

    public static func dit(numLayers: Int = 40) -> Set<String> {
        var keys: [String] = []
        keys += wb("head.head") + ["head.modulation"]
        for i in [0, 1, 3, 4] { keys += wb("img_emb.proj.layers.\(i)") }
        for p in ["patch_embedding", "patch_embedding_mask", "patch_embedding_pose"] {
            keys += wb(p)
        }
        for i in [0, 2] { keys += wb("text_embedding.layers.\(i)") }
        for i in [0, 2] { keys += wb("time_embedding.layers.\(i)") }
        keys += wb("time_projection.layers.1")

        for b in 0..<numLayers {
            let blk = "blocks.\(b)"
            for p in ["q", "k", "v", "o"] { keys += wb("\(blk).self_attn.\(p)") }
            for n in ["norm_q", "norm_k"] { keys.append("\(blk).self_attn.\(n).weight") }
            for p in ["q", "k", "v", "o", "k_img", "v_img"] {
                keys += wb("\(blk).cross_attn.\(p)")
            }
            for n in ["norm_q", "norm_k", "norm_k_img"] {
                keys.append("\(blk).cross_attn.\(n).weight")
            }
            keys += wb("\(blk).norm3")
            for i in [0, 2] { keys += wb("\(blk).ffn.layers.\(i)") }
            keys.append("\(blk).modulation")
        }
        return Set(keys)
    }

    public static func umt5(numLayers: Int = 24) -> Set<String> {
        var keys = ["token_embedding.weight", "norm.weight"]
        for b in 0..<numLayers {
            let blk = "blocks.\(b)"
            for p in ["q", "k", "v", "o"] { keys.append("\(blk).attn.\(p).weight") }
            for p in ["fc1", "fc2", "gate_proj"] { keys.append("\(blk).ffn.\(p).weight") }
            keys.append("\(blk).norm1.weight")
            keys.append("\(blk).norm2.weight")
            keys.append("\(blk).pos_embedding.embedding.weight")
        }
        return Set(keys)
    }

    public static func clip(numLayers: Int = 32) -> Set<String> {
        var keys = [
            "log_scale", "visual.cls_embedding", "visual.head",
            "visual.patch_embedding.weight", "visual.pos_embedding",
        ]
        keys += wb("visual.pre_norm") + wb("visual.post_norm")
        for b in 0..<numLayers {
            let blk = "visual.transformer.\(b)"
            keys += wb("\(blk).attn.to_qkv") + wb("\(blk).attn.proj")
            for i in [0, 2] { keys += wb("\(blk).mlp.layers.\(i)") }
            keys += wb("\(blk).norm1") + wb("\(blk).norm2")
        }
        return Set(keys)
    }

    /// Compare a component's on-disk keys against an expected set.
    public static func check(
        component: String, expected: Set<String>, actual: Set<String>
    ) -> String? {
        let missing = expected.subtracting(actual).sorted()
        let unused = actual.subtracting(expected).sorted()
        if missing.isEmpty && unused.isEmpty { return nil }
        return "\(component): missing=\(missing.prefix(5)) unused=\(unused.prefix(5)) "
            + "(\(missing.count) missing / \(unused.count) unused)"
    }
}
