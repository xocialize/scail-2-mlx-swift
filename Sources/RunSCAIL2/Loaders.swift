// Component weight loaders for the --generate CLI. Follows the bernini
// fromPretrained pattern: load on the CPU stream (Device.withDefaultDevice(.cpu)
// — a multi-GB disk read on the GPU stream trips the ~10 s Metal command-buffer
// watchdog), verify the on-disk key set against the MODULE'S OWN flattened keys
// (0 missing / 0 unused after remap), chunked-materialize, then update.
//
// DiT + umT5 are cast to fp32 (family rule: bf16 NaNs the DiT at video seqLen
// ≥1024 tokens; mlx-video runs the umT5 softmax in fp32). The cast is lazy and
// materialized in chunks so the bf16 source is never fully co-resident with the
// fp32 result — peak ≈ the fp32 size alone, not bf16+fp32.
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum SCAILLoadError: Error {
    case keyContract(component: String, missing: [String], unused: [String])
}

/// Load `url` into `module`, verifying the post-remap on-disk key set equals the
/// module's own flattened parameter keys. `keyMap` rewrites on-disk names to
/// module names (CLIP strips its `visual.` prefix); names that map to nil are
/// dropped as tolerated extras (CLIP's unused head/post_norm/log_scale).
private func loadInto(
    _ module: Module, url: URL, component: String,
    castFP32: Bool, chunk: Int = 32,
    keyMap: ((String) -> String?)? = nil
) throws {
    let expected = Set(module.parameters().flattened().map { $0.0 })
    try Device.withDefaultDevice(.cpu) {
        let raw = try MLX.loadArrays(url: url)   // lazy; not materialized
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(expected.count)
        for (k, v) in raw {
            guard let mapped = keyMap.map({ $0(k) }) ?? k else { continue }  // nil → drop
            if expected.contains(mapped) {
                weights[mapped] = castFP32 ? v.asType(.float32) : v
            }
        }
        let present = Set(weights.keys)
        let missing = expected.subtracting(present)
        guard missing.isEmpty else {
            throw SCAILLoadError.keyContract(
                component: component, missing: Array(missing).sorted(), unused: [])
        }
        WeightLoader.materialize(weights, chunk: chunk)
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
    }
}

/// CLIP visual tower (fp16 on disk, kept fp16 like the oracle clip_dtype). On
/// disk keys are `visual.<…>` + the unused log_scale/head/post_norm — strip the
/// prefix and drop everything the visual-only module doesn't carry.
func loadCLIP(_ tower: CLIPVisionTower, url: URL) throws {
    try loadInto(tower, url: url, component: "clip", castFP32: false) { k in
        guard k.hasPrefix("visual.") else { return nil }
        return String(k.dropFirst("visual.".count))
    }
}

/// 16-ch WanVAE (fp32 on disk). Encoder enabled (we VAE-encode ref/pose/history).
func loadVAE(_ vae: WanVAE, url: URL) throws {
    try loadInto(vae, url: url, component: "vae", castFP32: false)
}

/// umT5-XXL encoder — cast to fp32 (mlx-video load_t5_encoder parity).
func loadUMT5(_ enc: UMT5EncoderModel, url: URL) throws {
    try loadInto(enc, url: url, component: "umt5", castFP32: true)
}

/// SCAIL DiT — cast to fp32 (family fp32-DiT rule at video seqLen). ~64 GB
/// resident at the 14B/fp32 config; loaded last so it never co-resides with the
/// (already-evicted) umT5/CLIP encoders.
func loadDiT(_ model: SCAILModel, url: URL) throws {
    try loadInto(model, url: url, component: "dit", castFP32: true, chunk: 24)
}
