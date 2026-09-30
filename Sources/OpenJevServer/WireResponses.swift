// A port of upstream OpenJev (razorback16/openjev at dcd2094), the `error` helper of
// `openjev/api.py`, and of the responses FastAPI and Starlette send for an unknown route and an
// unhandled exception. Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import OpenJevCore

    /// Builds responses with the bytes and headers upstream sends.
    enum WireResponses {
        /// A JSON body with upstream's `content-type`, `application/json` without a charset.
        static func json(
            status: HTTPResponse.Status, bytes: [UInt8], headers: [WireError.Header] = []
        ) -> Response {
            var fields: HTTPFields = [.contentType: "application/json"]
            for header in headers {
                fields[HeaderName.named(header.name)] = header.value
            }
            return Response(
                status: status, headers: fields,
                body: ResponseBody(byteBuffer: ByteBuffer(bytes: bytes)))
        }

        /// A 200 with a wire value as its body.
        ///
        /// - Throws: ``JSONWriteError`` for a value with an infinite or NaN float, which
        ///   upstream's `allow_nan=False` refuses too.
        static func ok(_ value: some WireEncodable) throws -> Response {
            json(status: .ok, bytes: try WireEncoder().bytes(value))
        }

        /// The response for an error thrown by a route: a ``WireError`` as its status, body and
        /// headers; a Hummingbird error, such as the router's 404, as FastAPI's
        /// `{"detail": "<reason phrase>"}`; anything else as Starlette's plain-text 500, logged.
        static func response(for error: any Error, context: OpenJevRequestContext) -> Response {
            switch error {
            case let wire as WireError:
                return response(for: wire, context: context)
            case let http as any HTTPResponseError:
                // A string detail has no float, so writing it cannot fail.
                let detail = PlainDetailBody(detail: http.status.reasonPhrase)
                guard let bytes = try? WireEncoder().bytes(detail) else { return internalError() }
                return json(status: http.status, bytes: bytes)
            default:
                context.logger.error("unhandled error: \(String(describing: error))")
                return internalError()
            }
        }

        /// The response for a ``WireError``, or the 500 when its body cannot be written.
        static func response(for error: WireError, context: OpenJevRequestContext) -> Response {
            do {
                return json(
                    status: HTTPResponse.Status(code: error.status),
                    bytes: try WireEncoder().bytes(error), headers: error.headers)
            } catch {
                context.logger.error(
                    "an error body could not be written: \(String(describing: error))")
                return internalError()
            }
        }

        /// Starlette's `ServerErrorMiddleware` response: 500, `Internal Server Error` as plain
        /// text.
        static func internalError() -> Response {
            Response(
                status: .internalServerError,
                headers: [.contentType: "text/plain; charset=utf-8"],
                body: ResponseBody(byteBuffer: ByteBuffer(string: "Internal Server Error")))
        }
    }
#endif
