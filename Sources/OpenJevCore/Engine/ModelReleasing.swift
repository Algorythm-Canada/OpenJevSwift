// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `close` methods of
// `openjev/engine.py`, `openjev/mlx_backend.py` and `openjev/encoders.py`, which the `lifespan` of
// `openjev/api.py` awaits once the server has stopped. Apache-2.0. See THIRD_PARTY.md.

/// A service or a backend that holds a model, or what runs one, and releases it when the server
/// stops: upstream's `close()`, which `lifespan` awaits after the last request.
///
/// Adopting it is optional. The server calls ``close()`` on its ``SystemOneService`` once its
/// in-flight requests have finished or been cancelled. ``DecisionEngine`` and
/// ``EncoderDecisionEngine`` adopt it and pass the call on to a backend that adopts it too, so a
/// backend with nothing to release does nothing.
public protocol ModelReleasing: Sendable {
    /// Releases the model and what runs it. Nothing calls the service or the backend afterwards;
    /// a type documents what a later call would do, such as loading what it needs again.
    func close() async
}

extension DecisionEngine: ModelReleasing {
    /// Releases the backend's model, when the backend adopts ``ModelReleasing``.
    public nonisolated func close() async {
        await (backend as? any ModelReleasing)?.close()
    }
}

extension EncoderDecisionEngine: ModelReleasing {
    /// Releases the backend's model, when the backend adopts ``ModelReleasing``.
    public nonisolated func close() async {
        await (backend as? any ModelReleasing)?.close()
    }
}
