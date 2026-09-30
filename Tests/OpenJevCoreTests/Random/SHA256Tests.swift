import OpenJevCore
import Testing

#if canImport(CryptoKit)
    import CryptoKit
#endif

/// Checks ``SHA256`` against the FIPS 180-4 example vectors and, on Apple platforms, CryptoKit.
@Suite("SHA-256")
struct SHA256Tests {
    @Test(
        "Standard vectors",
        arguments: [
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
            ),
        ])
    func standardVectors(message: String, expected: String) {
        #expect(SeedFixtures.hex(OpenJevCore.SHA256.digest(message.utf8)) == expected)
    }

    @Test("One million a")
    func millionA() {
        let digest = OpenJevCore.SHA256.digest(repeatElement(UInt8(ascii: "a"), count: 1_000_000))
        let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        #expect(SeedFixtures.hex(digest) == expected)
    }

    #if canImport(CryptoKit)
        /// Every length from 0 to 300 bytes crosses the padding boundaries at 55, 56 and 64 bytes
        /// and their multiples.
        @Test("Agrees with CryptoKit for lengths 0 to 300")
        func matchesCryptoKit() {
            for length in 0...300 {
                let message = (0..<length).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) }
                let expected = Array(CryptoKit.SHA256.hash(data: message))
                #expect(OpenJevCore.SHA256.digest(message) == expected, "length \(length)")
            }
        }
    #endif
}
