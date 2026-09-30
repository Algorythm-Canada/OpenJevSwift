// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, class
// `Upstream`, which `Engine._post` raises for a vLLM 4xx and the `systemone` route of
// `openjev/api.py` answers as a 400. Apache-2.0. See THIRD_PARTY.md.

/// The inference backend refused a request as sent, upstream's `Upstream` error: the backend's
/// own reason, answered as the 400 `the model rejected this request: <reason>`.
///
/// A backend throws it for a refusal that is the client's to fix, such as a prompt over the
/// backend's context length that the engine could not see coming. The server answers it with
/// ``WireError/modelRejected400(_:)``, keeping the first 500 characters of the reason as
/// upstream's `str(msg)[:500]` does. Anything else a backend throws is a failure of the backend,
/// not of the request, and the server answers it as the 503.
public struct BackendRefusal: Error, Sendable, Hashable, CustomStringConvertible {
    /// The backend's reason, as it gave it.
    public var reason: String

    /// Creates a refusal.
    public init(reason: String) {
        self.reason = reason
    }

    /// The reason.
    public var description: String { reason }
}
