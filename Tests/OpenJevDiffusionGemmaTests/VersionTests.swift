import OpenJevDiffusionGemma
import Testing

@Test("OpenJevDiffusionGemma reports the package version")
func diffusionGemmaVersion() {
    #expect(openJevDiffusionGemmaVersion == "0.1.0-dev")
}
