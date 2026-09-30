import OpenJevServer
import Testing

@Test("OpenJevServer reports the package version")
func serverVersion() {
    #expect(openJevServerVersion == "0.1.0-dev")
}
