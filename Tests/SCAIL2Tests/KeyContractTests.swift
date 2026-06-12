// Never-eval structural tests: generated contracts vs the pinned fixture.
// No weights, no MLX — runs anywhere `swift test` does.
import Foundation
import Testing

@testable import SCAIL2

private func pinned() throws -> [String: [String]] {
    let url = Bundle.module.url(
        forResource: "key_contract", withExtension: "json",
        subdirectory: "Fixtures")!
    return try JSONDecoder().decode(
        [String: [String]].self, from: Data(contentsOf: url))
}

@Test func ditContractMatchesPinned() throws {
    #expect(KeyContract.dit() == Set(try pinned()["dit"]!))
}

@Test func umt5ContractMatchesPinned() throws {
    #expect(KeyContract.umt5() == Set(try pinned()["umt5"]!))
}

@Test func clipContractMatchesPinned() throws {
    #expect(KeyContract.clip() == Set(try pinned()["clip"]!))
}

@Test func contractSizes() throws {
    #expect(KeyContract.dit().count == 1307)
    #expect(KeyContract.umt5().count == 242)
    #expect(KeyContract.clip().count == 393)
    #expect(Set(try pinned()["vae"]!).count == 194)
}
