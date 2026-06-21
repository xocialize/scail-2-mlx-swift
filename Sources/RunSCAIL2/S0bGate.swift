// S0b structural-reuse gate: prove wan-core's WanVAE + UMT5EncoderModel accept
// SCAIL's converted weights directly (0 missing / 0 unused), confirming REUSE
// of the substrate loaders rather than a SCAIL-local re-port.
//
// This is the offline, no-download half of the wan-video "is the VAE/T5 shared?"
// check: it proves the KEY CONTRACT matches. Provenance covers values (SCAIL's
// VAE is the stock Wan2.1_VAE.pth, the family's 16-ch VAE); the permutation-
// invariant VALUE fingerprint vs a sibling is the belt-and-braces S1 follow-up
// once a sibling VAE is downloaded for parity fixtures.
//
// Weight-free instantiation is lazy (MLXArray.zeros, never eval'd) — runs in
// milliseconds, touches no Metal.
import Foundation
import MLX
import MLXNN
import SCAIL2
import WanCore

enum S0bGate {
    static func run(weightsDir: String) -> Int32 {
        let err = FileHandle.standardError
        func log(_ s: String) { err.write(Data((s + "\n").utf8)) }

        func flattenedKeys(_ module: Module) -> Set<String> {
            Set(module.parameters().flattened().map { $0.0 })
        }

        func compare(_ name: String, _ swiftKeys: Set<String>) -> Bool {
            let url = URL(fileURLWithPath: weightsDir)
                .appendingPathComponent("\(name).safetensors")
            guard let diskKeys = try? SafetensorsHeader.keys(url) else {
                log("S0b \(name): cannot read \(url.path)")
                return false
            }
            let missing = swiftKeys.subtracting(diskKeys).sorted()  // module wants, disk lacks
            let unused = diskKeys.subtracting(swiftKeys).sorted()   // disk has, module ignores
            if missing.isEmpty && unused.isEmpty {
                log("S0b PASS \(name): wan-core loader matches \(diskKeys.count) keys "
                    + "(0 missing / 0 unused) — REUSE confirmed")
                return true
            }
            log("S0b FAIL \(name): missing=\(missing.prefix(6)) unused=\(unused.prefix(6)) "
                + "(\(missing.count) missing / \(unused.count) unused)")
            return false
        }

        // Family-standard constructors (mirror bernini's BerniniPipeline build).
        let vae = WanVAE(zDim: 16, encoder: true)
        let umt5 = UMT5EncoderModel()  // umT5-XXL defaults, sharedPos:false

        var ok = true
        ok = compare("vae", flattenedKeys(vae)) && ok
        ok = compare("umt5", flattenedKeys(umt5)) && ok

        log(ok ? "S0b GATE PASSED" : "S0b GATE FAILED")
        return ok ? 0 : 1
    }
}
