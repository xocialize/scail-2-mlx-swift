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
            usage: RunSCAIL2 --s0-gate | --generate <args>

            """.utf8))
            exit(1)
        }
        switch mode {
        case "--s0-gate":
            FileHandle.standardError.write(Data("S0 key-contract gate: not implemented yet\n".utf8))
            exit(2)
        default:
            FileHandle.standardError.write(Data("unknown mode \(mode)\n".utf8))
            exit(1)
        }
    }
}
