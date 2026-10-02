import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

#if canImport(Glibc)
    import Glibc
#elseif canImport(Darwin)
    import Darwin
#endif

/// The live suite's own machinery, with no OpenJev server, so these run everywhere, CI included:
/// how the listing is read, that a cancelled request stops waiting, and the header parsers.
@Suite("Live suite client")
struct LiveClientTests {
    @Test(
        "Every listing upstream's server sends decodes to its names",
        .enabled(if: WireFixtures.exists("models.json"), WireFixtures.missingMessage))
    func recordedListings() throws {
        let listings = try #require(WireFixtures.load("models.json")["listings"]?.arrayValue)
        #expect(listings.count >= 6)
        for listing in listings {
            let body = try #require(listing["body_text"]?.stringValue)
            let expected = try #require(JSONParser().parse(body)["models"]?.arrayValue)
                .compactMap { $0["name"]?.stringValue }
            #expect(!expected.isEmpty)
            #expect(try ModelListing.names(inListing: Data(body.utf8)) == Set(expected))
        }
    }

    @Test(
        "A listing that is not Jev's fails the tests that need it instead of listing nothing",
        arguments: [
            #"{"models":[{"description":"d","release_date":"2026-09-22"}]}"#,
            #"{"models":[{"name":5,"description":"d","release_date":"2026-09-22"}]}"#,
            #"{"models":{}}"#, #"{}"#, #"[]"#, "Bad Gateway",
        ])
    func malformedListing(_ body: String) {
        #expect(throws: LiveFailure.self) { try ModelListing.names(inListing: Data(body.utf8)) }
    }

    @Test("A failed request in a task group cancels the others instead of waiting for them")
    func groupCancellation() async throws {
        let listener = try SilentListener()
        let settings = try LiveSettings(environment: [
            "OPENJEV_LIVE_URL": "http://127.0.0.1:\(listener.port)"
        ])
        // An exchange that outlived its task would wait for this timeout.
        let client = LiveClient(settings: settings, timeout: 60)
        let clock = ContinuousClock()
        let started = clock.now
        await #expect(throws: LiveFailure.self) {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<4 {
                    group.addTask { _ = try await client.get("/v1/models") }
                }
                group.addTask {
                    try await Task.sleep(for: .milliseconds(300))
                    throw LiveFailure("one request failed")
                }
                // As test_concurrent_reads reads its answers: the first error leaves the group,
                // which then cancels the requests still waiting. (waitForAll would wait for them.)
                while try await group.next() != nil {}
            }
        }
        let elapsed = clock.now - started
        #expect(elapsed < .seconds(30), "\(elapsed)")
        withExtendedLifetime(listener) {}
    }

    @Test(
        "A cancelled request throws CancellationError, cancelled while it waits or before it starts",
        arguments: [true, false])
    func cancellation(waitsFirst: Bool) async throws {
        let listener = try SilentListener()
        let settings = try LiveSettings(environment: [
            "OPENJEV_LIVE_URL": "http://127.0.0.1:\(listener.port)"
        ])
        let client = LiveClient(settings: settings, timeout: 60)
        let clock = ContinuousClock()
        let started = clock.now
        let exchange = Task { try await client.get("/v1/models") }
        if waitsFirst {
            try await Task.sleep(for: .milliseconds(300))
        }
        exchange.cancel()
        let outcome = await exchange.result
        let elapsed = clock.now - started
        #expect(throws: CancellationError.self) { try outcome.get() }
        #expect(elapsed < .seconds(30), "\(elapsed)")
        withExtendedLifetime(listener) {}
    }

    @Test("A request id is req_ and 32 lowercase hex characters")
    func requestIDs() {
        #expect(JevContract.isRequestID("req_" + String(repeating: "0a", count: 16)))
        #expect(!JevContract.isRequestID("req_" + String(repeating: "0A", count: 16)))
        #expect(!JevContract.isRequestID("req_" + String(repeating: "a", count: 31)))
        #expect(!JevContract.isRequestID("req_" + String(repeating: "a", count: 33)))
        #expect(!JevContract.isRequestID("rid_" + String(repeating: "a", count: 32)))
        #expect(!JevContract.isRequestID(""))
    }

    @Test("server-timing's spans are read by name, other metrics and parameters ignored")
    func serverTiming() {
        let spans = JevContract.durations(
            inServerTiming: "model;dur=41.2, server;dur=2.8, total;dur=44.0")
        #expect(spans == ["model": 41.2, "server": 2.8, "total": 44.0])
        let mixed = JevContract.durations(
            inServerTiming: "cdn;desc=HIT, total;desc=\"all\";dur=7, model;dur=0.0")
        #expect(mixed == ["total": 7, "model": 0])
        #expect(JevContract.durations(inServerTiming: "").isEmpty)
    }
}

extension WireFixtures {
    /// ``missingMessageText`` as a test comment.
    static var missingMessage: Comment { Comment(rawValue: missingMessageText) }
}

/// A TCP socket on 127.0.0.1 that listens and never accepts: the kernel completes each handshake
/// into the backlog, so a request is sent and its answer never comes.
final class SilentListener {
    /// The port it listens on.
    let port: Int
    private let descriptor: Int32

    /// Opens the socket on a free port.
    init() throws {
        #if canImport(Glibc)
            let descriptor = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #else
            let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        #endif
        guard descriptor >= 0 else { throw LiveFailure("socket failed with errno \(errno)") }
        var address = sockaddr_in()
        #if canImport(Darwin)
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        #endif
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, length)
            }
        }
        guard bound == 0, listen(descriptor, 64) == 0 else {
            let code = errno
            close(descriptor)
            throw LiveFailure("bind or listen failed with errno \(code)")
        }
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(descriptor, $0, &length)
            }
        }
        guard named == 0 else {
            let code = errno
            close(descriptor)
            throw LiveFailure("getsockname failed with errno \(code)")
        }
        self.descriptor = descriptor
        port = Int(UInt16(bigEndian: address.sin_port))
    }

    deinit {
        close(descriptor)
    }
}
