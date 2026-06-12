// S0 gate: on-disk safetensors keys == generated contract (dit/umt5/clip)
// and == the pinned fixture (vae). 0 missing / 0 unused per component.
import Foundation
import SCAIL2

enum S0Gate {
    static func run(weightsDir: String, fixturePath: String) -> Int32 {
        let err = FileHandle.standardError
        func log(_ s: String) { err.write(Data((s + "\n").utf8)) }

        guard
            let fixtureData = FileManager.default.contents(atPath: fixturePath),
            let pinned = try? JSONDecoder().decode(
                [String: [String]].self, from: fixtureData)
        else {
            log("S0: cannot read fixture \(fixturePath)")
            return 2
        }

        let expected: [String: Set<String>] = [
            "dit": KeyContract.dit(),
            "umt5": KeyContract.umt5(),
            "clip": KeyContract.clip(),
            "vae": Set(pinned["vae"] ?? []),
        ]

        var failures = 0
        for (component, exp) in expected.sorted(by: { $0.key < $1.key }) {
            let url = URL(fileURLWithPath: weightsDir)
                .appendingPathComponent("\(component).safetensors")
            guard let actual = try? SafetensorsHeader.keys(url) else {
                log("S0 \(component): cannot read \(url.path)")
                failures += 1
                continue
            }
            // cross-check: generated contracts must also match the pinned
            // fixture (catches generator bugs independently of the weights)
            if let pinnedKeys = pinned[component], component != "vae" {
                if Set(pinnedKeys) != exp {
                    log("S0 \(component): generator disagrees with pinned fixture")
                    failures += 1
                }
            }
            if let msg = KeyContract.check(
                component: component, expected: exp, actual: actual)
            {
                log("S0 FAIL \(msg)")
                failures += 1
            } else {
                log("S0 PASS \(component): \(actual.count) keys, 0 missing / 0 unused")
            }
        }
        log(failures == 0 ? "S0 GATE PASSED" : "S0 GATE FAILED (\(failures))")
        return failures == 0 ? 0 : 1
    }
}
