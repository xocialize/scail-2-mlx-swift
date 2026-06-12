// Never-eval tests only in `swift test` (metallib fragility — see
// PORTING-SPEC.md); Metal-context gates are CLI modes of RunSCAIL2.
import Testing

@testable import SCAIL2

@Test func configDefaultsMatchOracle() {
    let c = SCAIL2Config()
    #expect(c.dim == 5120)
    #expect(c.numLayers == 40)
    #expect(c.inDim == 20)
    #expect(c.maskDim == 28)
    #expect(c.patchSize == [1, 2, 2])
}
