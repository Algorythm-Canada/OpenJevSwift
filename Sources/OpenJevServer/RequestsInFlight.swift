// How a shutdown tells whether it cut requests short (issue #37). Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import HummingbirdCore

    /// The requests one server is answering, so that a shutdown that has to cancel the server can
    /// tell whether it cut any of them short.
    ///
    /// A request counts from when Hummingbird hands it to the responder until its response has
    /// been written to the connection (``counting(_:)``), so a response still being written to a
    /// slow client counts too.
    final class RequestsInFlight: @unchecked Sendable {
        // Guarded by `lock`.
        private let lock = NSLock()
        private var active = 0
        private var cut = false

        /// The requests being answered now.
        var count: Int {
            lock.withLock { active }
        }

        /// Whether the server was cancelled while it was answering a request.
        var cutShort: Bool {
            lock.withLock { cut }
        }

        /// Counts a request in.
        func enter() {
            lock.withLock { active += 1 }
        }

        /// Counts a request out.
        func leave() {
            lock.withLock { active -= 1 }
        }

        /// Notes that the server is being cancelled, which cuts short the requests it is
        /// answering, if there are any.
        func serverCancelled() {
            lock.withLock {
                if active > 0 {
                    cut = true
                }
            }
        }

        /// `responder`, with each request counted while it runs: Hummingbird's responder routes
        /// the request and then writes the response, and returns once the connection has taken
        /// the last part.
        func counting(
            _ responder: @escaping HTTPChannelHandler.Responder
        ) -> HTTPChannelHandler.Responder {
            { request, writer, channel in
                self.enter()
                defer { self.leave() }
                try await responder(request, writer, channel)
            }
        }
    }
#endif
