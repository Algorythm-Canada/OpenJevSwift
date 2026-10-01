// How a shutdown tells whether it cut requests short (issue #37). Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import Hummingbird

    /// The requests one server is answering, so that a shutdown that has to cancel the server can
    /// tell whether it cut any of them short.
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
    }

    /// Counts each request in ``RequestsInFlight`` while the middlewares and the route handle it.
    /// It runs just inside ``RequestLogMiddleware``, outside every other middleware.
    struct InFlightMiddleware: RouterMiddleware {
        typealias Context = OpenJevRequestContext

        let requests: RequestsInFlight

        func handle(
            _ request: Request, context: Context,
            next: (Request, Context) async throws -> Response
        ) async throws -> Response {
            requests.enter()
            defer { requests.leave() }
            return try await next(request, context)
        }
    }
#endif
