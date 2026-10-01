// Cancelling the decision of a client that has gone away (issue #37). Upstream's uvicorn runs a
// request to its end whether or not the client is still there; this server cancels the reads,
// as structured concurrency allows. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import Foundation
    import NIOCore
    import OpenJevCore

    /// Whether the client of one connection is still there: closed once the client has shut its
    /// side, which reaches the server as the end of its input, or the connection has gone.
    ///
    /// A client that half-closes after sending its request and still waits for the answer counts
    /// as gone, as it does for most HTTP servers: an HTTP/1.1 client does not do that.
    final class ConnectionWatch: @unchecked Sendable {
        private typealias Waiter = CheckedContinuation<Void, Never>

        // Guarded by `lock`.
        private let lock = NSLock()
        private var closed = false
        private var nextID = 0
        private var waiters: [Int: Waiter] = [:]

        /// Whether the client has gone.
        var isClosed: Bool {
            lock.withLock { closed }
        }

        /// Marks the client gone and wakes every waiter. Later calls do nothing.
        func close() {
            let woken = lock.withLock { () -> [Waiter] in
                guard !closed else { return [] }
                closed = true
                defer { waiters.removeAll() }
                return Array(waiters.values)
            }
            for waiter in woken {
                waiter.resume()
            }
        }

        /// Returns once the client has gone, or once the task is cancelled.
        func waitUntilClosed() async {
            let id = lock.withLock {
                nextID += 1
                return nextID
            }
            await withTaskCancellationHandler {
                await withCheckedContinuation { (continuation: Waiter) in
                    let now = lock.withLock { () -> Bool in
                        if closed || Task.isCancelled {
                            return true
                        }
                        waiters[id] = continuation
                        return false
                    }
                    if now {
                        continuation.resume()
                    }
                }
            } onCancel: {
                let waiter = lock.withLock { waiters.removeValue(forKey: id) }
                waiter?.resume()
            }
        }
    }

    /// The watch of every open connection of one server, by channel, so that a route can find
    /// the watch of the connection its request came on. It belongs to one ``DecisionServer`` and
    /// is never global.
    final class ConnectionRegistry: @unchecked Sendable {
        // Guarded by `lock`.
        private let lock = NSLock()
        private var watches: [ObjectIdentifier: ConnectionWatch] = [:]

        /// The watch of the connection `channel` is, or `nil` for a channel this server does not
        /// know, such as a test's.
        func watch(for channel: (any Channel)?) -> ConnectionWatch? {
            guard let channel else { return nil }
            return lock.withLock { watches[ObjectIdentifier(channel)] }
        }

        /// Registers a connection's watch.
        func register(_ watch: ConnectionWatch, for channel: any Channel) {
            lock.withLock { watches[ObjectIdentifier(channel)] = watch }
        }

        /// Forgets a connection.
        func remove(_ channel: any Channel) {
            lock.withLock { _ = watches.removeValue(forKey: ObjectIdentifier(channel)) }
        }

        /// The connections open now, for tests.
        var count: Int {
            lock.withLock { watches.count }
        }
    }

    /// A channel handler that closes its connection's ``ConnectionWatch`` when the client goes:
    /// at the end of the input, which a client's close or half-close sends, or when the channel
    /// becomes inactive. It sits after the HTTP decoder and passes everything on unchanged.
    final class ClientDisconnectHandler: ChannelInboundHandler, RemovableChannelHandler {
        typealias InboundIn = NIOAny

        private let registry: ConnectionRegistry
        private let watch = ConnectionWatch()

        init(registry: ConnectionRegistry) {
            self.registry = registry
        }

        func handlerAdded(context: ChannelHandlerContext) {
            registry.register(watch, for: context.channel)
        }

        func handlerRemoved(context: ChannelHandlerContext) {
            watch.close()
            registry.remove(context.channel)
        }

        func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
            if let event = event as? ChannelEvent, event == .inputClosed {
                watch.close()
            }
            context.fireUserInboundEventTriggered(event)
        }

        func channelInactive(context: ChannelHandlerContext) {
            watch.close()
            context.fireChannelInactive()
        }
    }

    /// The client went away before its answer was ready. The server cancelled the decision and
    /// answers an empty 499, nginx's code for a client that closed its request, which the request
    /// log shows; only a client that half-closed and still reads receives it.
    struct ClientDisconnected: Error, Sendable, Equatable {}

    enum ClientConnection {
        /// Runs `body`, cancelling it when the client of `watch` goes away first. `nil` runs it
        /// as it is. The work runs as a child task, never a detached one, so cancelling it reaches
        /// the backend's reads, and `body` has finished, however it ended, before this returns.
        ///
        /// - Throws: ``ClientDisconnected`` when the client went away first, `CancellationError`
        ///   when the request itself was cancelled, and `body`'s error otherwise.
        static func cancellingOnDisconnect<T: Sendable>(
            _ watch: ConnectionWatch?, _ body: @escaping @Sendable () async throws -> T
        ) async throws -> T {
            guard let watch else {
                return try await body()
            }
            return try await withThrowingTaskGroup(of: Outcome<T>.self) { group in
                group.addTask { .finished(try await body()) }
                group.addTask {
                    await watch.waitUntilClosed()
                    return .disconnected
                }
                guard let first = try await group.next() else {
                    preconditionFailure("the group has two tasks")
                }
                group.cancelAll()
                switch first {
                case .finished(let value):
                    return value
                case .disconnected:
                    // The waiter also returns when the request itself is cancelled, which the
                    // server does when a shutdown runs out of time; it then closes the input too.
                    // That is the server stopping, not the client going away.
                    if Task.isCancelled || !watch.isClosed {
                        throw CancellationError()
                    }
                    // Let the cancelled work end; its result no longer has anywhere to go.
                    while (try? await group.next()) != nil {}
                    throw ClientDisconnected()
                }
            }
        }

        private enum Outcome<T: Sendable>: Sendable {
            case finished(T)
            case disconnected
        }
    }
#endif
