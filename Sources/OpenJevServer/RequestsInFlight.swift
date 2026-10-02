// How a shutdown tells whether it cut requests short (issue #37, D-049). Apache-2.0. See
// THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import HummingbirdCore

    /// Whether the cancellation of one server found it answering requests, which it then cut
    /// short.
    ///
    /// Each request tells on its own task, which the cancellation marks before it stops anything
    /// the task runs: a request is cut short when its responder ends by throwing on a cancelled
    /// task (``noting(_:)``). SwiftNIO sends no write that a cancelled task makes or is still
    /// waiting to make, and throws instead, so that is every request the cancellation reaches
    /// before the connection has taken its whole answer, and no other. A request whose answer the
    /// connection took first is not cut short, however late its task ends: its client may have
    /// read all of it already.
    final class RequestsInFlight: @unchecked Sendable {
        // Guarded by `lock`.
        private let lock = NSLock()
        private var cut = false

        /// Whether the cancellation cut a request short.
        var cutShort: Bool {
            lock.withLock { cut }
        }

        /// `responder`, noting a request cut short when it throws on a cancelled task:
        /// Hummingbird's responder routes the request and then writes the answer, and returns once
        /// the connection has taken the last part.
        func noting(
            _ responder: @escaping HTTPChannelHandler.Responder
        ) -> HTTPChannelHandler.Responder {
            { request, writer, channel in
                do {
                    try await responder(request, writer, channel)
                } catch {
                    if Task.isCancelled {
                        self.lock.withLock { self.cut = true }
                    }
                    throw error
                }
            }
        }
    }
#endif
