// A port of upstream OpenJev (razorback16/openjev at dcd2094), `read_capped_body` in
// `openjev/api.py` and the part of the `request_id_and_auth` middleware that calls it after
// `check_auth`. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import OpenJevCore

    /// Upstream's `read_capped_body`: the body of a `POST` under `/v1/` is read up to
    /// `OPENJEV_MAX_BODY_BYTES`, after authentication, so an anonymous giant is refused before
    /// it costs any memory. Neither Hummingbird nor the routes bound a body otherwise.
    ///
    /// A `Content-Length` over the cap is the 413 without reading a byte. A body without one, or
    /// with one that does not hold, is counted as it arrives and refused as soon as it passes
    /// the cap. The route gets the whole body in one buffer. Other methods are never capped.
    struct BodyCapMiddleware: RouterMiddleware {
        typealias Context = OpenJevRequestContext

        /// The most bytes a body may have, `OPENJEV_MAX_BODY_BYTES`.
        let limit: Int

        func handle(
            _ request: Request, context: Context,
            next: (Request, Context) async throws -> Response
        ) async throws -> Response {
            guard request.method == .post, VersionedPath.contains(request.uri.path) else {
                return try await next(request, context)
            }
            var request = request
            request.body = RequestBody(buffer: try await Self.read(request, limit: limit))
            return try await next(request, context)
        }

        /// The body of `request`, read to its end.
        ///
        /// - Throws: ``WireError/bodyTooLarge413(limit:)`` when `Content-Length` declares more
        ///   than `limit` bytes, before reading, or when more than `limit` bytes arrive; and
        ///   whatever reading the body throws.
        static func read(_ request: Request, limit: Int) async throws -> ByteBuffer {
            let tooLarge = WireError.bodyTooLarge413(limit: limit)
            // upstream: int(request.headers.get("content-length", "0")) > limit, a ValueError
            // ignored
            if let declared = HeaderText(.contentLength, in: request.headers),
                ContentLength.exceeds(declared, limit: limit) == true
            {
                throw tooLarge
            }
            var body = ByteBuffer()
            for try await chunk in request.body {
                guard chunk.readableBytes <= limit - body.readableBytes else {
                    throw tooLarge
                }
                var chunk = chunk
                body.writeBuffer(&chunk)
            }
            return body
        }
    }

    /// Upstream's reading of `Content-Length`: Python's `int()` of the header's text.
    enum ContentLength {
        /// Whether the value declares more than `limit` bytes, as `int(value) > limit` decides,
        /// or `nil` when `int()` refuses the value, which upstream ignores. That includes a value
        /// of more than 4,300 digits, CPython's limit. An integer too large for `Int` is past any
        /// limit when positive and under it when negative.
        static func exceeds(_ value: HeaderText, limit: Int) -> Bool? {
            let text = String(decoding: value.utf8, as: UTF8.self)
            if let length = PythonNumber.integer(text) {
                return length > limit
            }
            guard PythonNumber.isInteger(text) else {
                return nil
            }
            return !PythonNumber.isNegative(text)
        }
    }
#endif
