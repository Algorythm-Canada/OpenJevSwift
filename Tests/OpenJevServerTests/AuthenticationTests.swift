#if canImport(HummingbirdTesting)
    import Foundation
    import HTTPTypes
    import Hummingbird
    import HummingbirdTesting
    import OpenJevCore
    @testable import OpenJevServer
    import OpenJevTestSupport
    import Testing

    /// Upstream's `check_auth` (issue #36). The recorded `auth_*` exchanges, which include every
    /// case of upstream's `test_auth`, are replayed by ``ErrorContractTests``; these tests take
    /// the check apart, with the answers upstream's own `check_auth` gave for the same headers.
    @Suite("Authentication")
    struct AuthenticationTests {
        /// Headers from `(name, bytes)` pairs, in order, the bytes kept as they are.
        static func headers(_ pairs: [(String, [UInt8])]) throws -> HTTPFields {
            var fields = HTTPFields()
            for (name, value) in pairs {
                fields.append(HTTPField(name: try #require(HTTPField.Name(name)), value: value))
            }
            return fields
        }

        /// The status `check_auth` answers with `sk-test` and `s3` configured, or `nil`.
        static func status(_ pairs: [(String, [UInt8])]) throws -> Int? {
            AuthenticationMiddleware.refusal(
                for: try headers(pairs), originSecret: "s3", apiKey: "sk-test")?.status
        }

        static let secret: (String, [UInt8]) = ("x-origin-secret", Array("s3".utf8))

        static func key(_ value: [UInt8]) -> (String, [UInt8]) {
            ("authorization", value)
        }

        static func key(_ value: String) -> (String, [UInt8]) {
            key(Array(value.utf8))
        }

        /// Upstream's `test_non_ascii_credentials_are_rejected_not_crashed`.
        @Test("Non-ASCII credentials are refused, never crash")
        func nonASCIICredentials() throws {
            let euro = [("x-origin-secret", Array("s€".utf8)), Self.key("Bearer sk-test")]
            #expect(try Self.status(euro) == 403)
            #expect(try Self.status([Self.secret, Self.key("Bearer sk-t€st")]) == 401)
            #expect(try Self.status([Self.secret, Self.key("Bearer sk-test")]) == nil)
        }

        /// The whitespace around a field value is not part of it: swift-http-types drops it, as
        /// h11 does for upstream behind uvicorn, so `" s3"` arrives as `s3`.
        @Test("The origin secret is checked first and compared exactly")
        func originSecret() throws {
            let refusal = AuthenticationMiddleware.refusal(
                for: HTTPFields(), originSecret: "s3", apiKey: "sk-test")
            #expect(refusal == .permission403)
            for value in ["", "S3", "s33", "s", "s 3"] {
                let pairs = [("x-origin-secret", Array(value.utf8)), Self.key("Bearer sk-test")]
                #expect(try Self.status(pairs) == 403, "\(value)")
            }
            for value in [" s3", "s3 ", "\ts3"] {
                let pairs = [("x-origin-secret", Array(value.utf8)), Self.key("Bearer sk-test")]
                #expect(try Self.status(pairs) == nil, "\(value)")
            }
        }

        /// `auth.removeprefix("Bearer ").strip()` over the bytes `refusal(for:)` is given: the
        /// exact prefix, then Python's whitespace, which among Latin-1 characters includes U+001C,
        /// U+0085 and U+00A0. A lone 0x85 or 0xA0 never reaches it over HTTP/1, where NIO turns
        /// it into U+FFFD (``nioDecodedValues()``).
        @Test("The key follows Bearer and Python's strip, as upstream reads it")
        func apiKey() throws {
            // A value of spaces is empty once the field drops its surrounding whitespace, so it
            // is the missing key's 403, as behind uvicorn.
            let cases: [([UInt8], Int?)] = [
                (Array("".utf8), 403),
                (Array(" ".utf8), 403),
                (Array("Bearer".utf8), 401),
                (Array("Bearer ".utf8), 401),
                (Array("Bearer  ".utf8), 401),
                (Array("Bearer\tsk-test".utf8), 401),
                (Array("Bearer sk-test\t".utf8), nil),
                (Array("Bearer sk-test".utf8) + [0xA0], nil),
                (Array("Bearer sk-test".utf8) + [0x85], nil),
                (Array("Bearer sk-test".utf8) + [0x1C], nil),
                ([0xA0] + Array("Bearer sk-test".utf8), 401),
                (Array("Bearer Bearer sk-test".utf8), 401),
                (Array("BEARER sk-test".utf8), 401),
                (Array("Basic sk-test".utf8), 401),
                (Array("sk-test ".utf8), nil),
                (Array("  sk-test".utf8), nil),
                (Array("Bearer sk-tes".utf8), 401),
                (Array("Bearer sk-testx".utf8), 401),
            ]
            for (value, expected) in cases {
                #expect(try Self.status([Self.secret, Self.key(value)]) == expected, "\(value)")
            }
            let missing = AuthenticationMiddleware.refusal(
                for: try Self.headers([Self.secret]), originSecret: "s3", apiKey: "sk-test")
            #expect(missing == .authenticationMissing403)
            let wrong = AuthenticationMiddleware.refusal(
                for: try Self.headers([Self.secret, Self.key("Bearer nope")]), originSecret: "s3",
                apiKey: "sk-test")
            #expect(wrong == .authentication401)
        }

        /// A header is Latin-1 to Starlette, so given the bytes, a key with an accent matches the
        /// Latin-1 byte and not its UTF-8 spelling, as upstream does. Over HTTP/1 the lone byte
        /// arrives as U+FFFD, so such a key is never matched (``nioDecodedValues()``).
        @Test("A non-ASCII key compares as upstream compares the bytes it is given")
        func nonASCIIKey() throws {
            let refusal = { (value: [UInt8]) in
                AuthenticationMiddleware.refusal(
                    for: try Self.headers([Self.key(value)]), originSecret: "", apiKey: "sk-é")
            }
            #expect(try refusal(Array("Bearer sk-".utf8) + [0xE9]) == nil)
            #expect(try refusal(Array("Bearer sk-é".utf8)) == .authentication401)
        }

        /// NIO's HTTP/1 decoder reads each header value with `String(decoding:as: UTF8.self)`,
        /// and Hummingbird's field is made from that string, so a byte that is not UTF-8 arrives
        /// as U+FFFD. Such a value is refused, never accepted and never a crash, where upstream,
        /// which reads the raw bytes, would strip a 0xA0 or match a Latin-1 key (D-031).
        @Test("Header bytes that are not UTF-8 arrive as NIO decodes them, and are refused")
        func nioDecodedValues() throws {
            func decoded(_ bytes: [UInt8]) -> HTTPFields {
                [.authorization: String(decoding: bytes, as: UTF8.self)]
            }
            let padded = decoded(Array("Bearer sk-test".utf8) + [0xA0])
            #expect(
                AuthenticationMiddleware.refusal(for: padded, originSecret: "", apiKey: "sk-test")
                    == .authentication401)
            let latin1 = decoded(Array("Bearer sk-".utf8) + [0xE9])
            #expect(
                AuthenticationMiddleware.refusal(for: latin1, originSecret: "", apiKey: "sk-é")
                    == .authentication401)
            let utf8 = decoded(Array("Bearer sk-test".utf8))
            #expect(
                AuthenticationMiddleware.refusal(for: utf8, originSecret: "", apiKey: "sk-test")
                    == nil)
        }

        @Test("The first of repeated headers counts, as Starlette's Headers.get reads it")
        func repeatedHeaders() throws {
            let wrongFirst = [Self.secret, Self.key("Bearer nope"), Self.key("Bearer sk-test")]
            #expect(try Self.status(wrongFirst) == 401)
            let rightFirst = [Self.secret, Self.key("Bearer sk-test"), Self.key("Bearer nope")]
            #expect(try Self.status(rightFirst) == nil)
        }

        @Test("Nothing is checked when nothing is configured")
        func open() throws {
            #expect(
                AuthenticationMiddleware.refusal(
                    for: try Self.headers([Self.key("Bearer nope")]), originSecret: "", apiKey: "")
                    == nil)
        }

        @Test("Bytes compare equal only when they are the same, whatever their lengths")
        func constantTime() {
            #expect(ConstantTime.equal(Array("sk-test".utf8), Array("sk-test".utf8)))
            #expect(ConstantTime.equal([], []))
            #expect(!ConstantTime.equal(Array("sk-tes".utf8), Array("sk-test".utf8)))
            #expect(!ConstantTime.equal(Array("sk-testx".utf8), Array("sk-test".utf8)))
            #expect(!ConstantTime.equal(Array("sk-tesT".utf8), Array("sk-test".utf8)))
            #expect(!ConstantTime.equal([], Array("s".utf8)))
            // A length difference is a mismatch even when the shorter is padded with zeros.
            #expect(!ConstantTime.equal([0x61, 0], [0x61]))
        }

        /// Hummingbird's router skips empty path components, so every path that reaches a `/v1`
        /// route must be checked, not only those that start with `/v1/`.
        @Test("Every path the router sends to a /v1 route is authenticated")
        func paths() async throws {
            let settings = try ServerSettings(apiKey: "sk-test")
            let service = try await ServerHarness.diffusionService(settings)
            try await ServerHarness.withClient(settings: settings, service: service) { client in
                let expected: [(String, HTTPResponse.Status)] = [
                    ("/v1/models", .forbidden), ("//v1/models", .forbidden),
                    ("/v1//models", .forbidden), ("/v1/models/", .forbidden),
                    ("/v1/models?x=1", .forbidden), ("/v1/nope", .forbidden),
                    ("/v1/", .forbidden), ("/v1", .notFound), ("/V1/models", .notFound),
                    ("/%761/models", .notFound), ("/health", .ok),
                ]
                for (path, status) in expected {
                    let response = try await ServerHarness.send(client, .get, path)
                    #expect(response.status == status, "\(path)")
                    ServerHarness.expectServerHeaders(response, path)
                }
                let served = try await ServerHarness.send(
                    client, .get, "//v1/models", headers: ["authorization": "Bearer sk-test"])
                #expect(served.status == .ok)
            }
            #expect(VersionedPath.contains("/v1/systemone"))
            #expect(VersionedPath.contains("//v1/systemone"))
            #expect(!VersionedPath.contains("/v1"))
            #expect(!VersionedPath.contains("/health"))
            #expect(!VersionedPath.contains("/v10/models"))
        }

        /// Upstream's `auth_key_missing_post`: refused before the body is read.
        @Test("A request without its key is refused before its body is read")
        func refusedBeforeTheBody() async throws {
            let settings = try ServerSettings(apiKey: "sk-test")
            let service = try await ServerHarness.diffusionService(settings)
            let body = UnreadableBody()
            let response = try await ServerHarness.respond(
                settings: settings, service: service, method: .post, path: "/v1/systemone",
                headers: ["content-type": "application/json"], body: body)
            #expect(response.status == .forbidden)
            #expect(
                ServerHarness.text(response)
                    == #"{"detail":{"error_type":"authentication_error","message":"#
                    + #""Must supply an API key! Check your request and try again."}}"#)
            ServerHarness.expectServerHeaders(response, "refused before the body")
            #expect(body.reads.count == 0)
        }
    }
#endif
