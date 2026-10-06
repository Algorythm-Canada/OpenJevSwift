import OpenJevCore
import Testing

@Test("OpenJevCore reports the package version")
func coreVersion() {
    #expect(openJevCoreVersion == "0.1.0")
}
