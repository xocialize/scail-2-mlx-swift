// CLI entry. Per PORTING-SPEC.md, every Metal-context gate runs as a mode of
// this executable (--s0-gate, --s1-gate, ...) — the SPM test product's
// metallib is unreliable; plain `swift run` does GPU inference fine.
import Foundation
import SCAIL2

@main
struct RunSCAIL2 {
    static func main() {
        let args = CommandLine.arguments.dropFirst()
        guard let mode = args.first else {
            FileHandle.standardError.write(Data("""
            scail-2-mlx-swift — S0 scaffold. Gates land per PORTING-SPEC.md.
            usage: RunSCAIL2 --s0-gate | --s0b-gate | --generate <args>

            """.utf8))
            exit(1)
        }
        switch mode {
        case "--s0-gate":
            let rest = Array(args.dropFirst())
            let weightsDir = rest.first
                ?? "/Volumes/DEV_ARCHIVE/scail-2-mlx/weights/mlx"
            let fixture = rest.count > 1
                ? rest[1]
                : "Tests/SCAIL2Tests/Fixtures/key_contract.json"
            exit(S0Gate.run(weightsDir: weightsDir, fixturePath: fixture))
        case "--s0b-gate":
            let weightsDir = Array(args.dropFirst()).first
                ?? "/Volumes/DEV_ARCHIVE/scail-2-mlx/weights/mlx"
            exit(S0bGate.run(weightsDir: weightsDir))
        case "--s1-rope-gate":
            let dir = Array(args.dropFirst()).first
                ?? "Tests/SCAIL2Tests/Fixtures/rope"
            exit(S1RoPEGate.run(fixtureDir: dir))
        case "--s1-patchembed-gate":
            let dir = Array(args.dropFirst()).first
                ?? "Tests/SCAIL2Tests/Fixtures/patchembed"
            exit(S1PatchEmbedGate.run(fixtureDir: dir))
        default:
            FileHandle.standardError.write(Data("unknown mode \(mode)\n".utf8))
            exit(1)
        }
    }
}
