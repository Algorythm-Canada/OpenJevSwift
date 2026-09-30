// A port of upstream OpenJev (razorback16/openjev at dcd2094), `check_auth` in `openjev/api.py`
// and the part of the `request_id_and_auth` middleware that calls it for `/v1/` paths.
// Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import OpenJevCore

    /// The paths upstream's middleware authenticates and caps.
    enum VersionedPath {
        /// Whether `path` is upstream's `request.url.path.startswith("/v1/")`, or a path the
        /// router resolves to a `/v1` route anyway. Hummingbird skips empty path components, so
        /// `//v1/models` reaches `GET /v1/models` although it does not start with `/v1/`;
        /// upstream answers it with a 404, and here it is authenticated like `/v1/models`.
        static func contains(_ path: String) -> Bool {
            if path.hasPrefix("/v1/") {
                return true
            }
            let components = path.split(separator: "/")
            return components.count > 1 && components[0] == "v1"
        }
    }

    /// Upstream's `check_auth` for every path under `/v1/`, before the body is read: the origin
    /// secret a front proxy adds, then the API key clients send.
    ///
    /// It runs inside ``ResponseHeadersMiddleware``, so a refusal carries the request ids and
    /// `server-timing`. `/health` is never checked.
    struct AuthenticationMiddleware: RouterMiddleware {
        typealias Context = OpenJevRequestContext

        /// `OPENJEV_ORIGIN_SECRET`; empty means no secret is required.
        let originSecret: String
        /// `OPENJEV_API_KEY`; empty means no key is required.
        let apiKey: String

        /// The name of the header a front proxy sets.
        static let originSecretHeader = HeaderName.named("x-origin-secret")

        func handle(
            _ request: Request, context: Context,
            next: (Request, Context) async throws -> Response
        ) async throws -> Response {
            if VersionedPath.contains(request.uri.path),
                let refusal = Self.refusal(
                    for: request.headers, originSecret: originSecret, apiKey: apiKey)
            {
                throw refusal
            }
            return try await next(request, context)
        }

        /// Upstream's `check_auth`, or `nil` when the request may go on.
        ///
        /// - With an origin secret, `X-Origin-Secret` must be it, or the 403 `permission_error`.
        /// - With an API key, a missing or empty `Authorization` is the 403
        ///   `authentication_error`, and one whose value, after a leading `Bearer ` (that exact
        ///   text) is removed and Python's whitespace is stripped, is not the key is the 401.
        ///
        /// Values are compared as upstream compares them, UTF-8 bytes in constant time; the
        /// header reads as Latin-1, so a non-ASCII value never matches an ASCII setting and never
        /// fails in any other way.
        static func refusal(
            for headers: HTTPFields, originSecret: String, apiKey: String
        ) -> WireError? {
            if !originSecret.isEmpty {
                let given = HeaderText(originSecretHeader, in: headers) ?? HeaderText(bytes: [])
                if !ConstantTime.equal(given.utf8, Array(originSecret.utf8)) {
                    return .permission403
                }
            }
            if !apiKey.isEmpty {
                let authorization =
                    HeaderText(.authorization, in: headers) ?? HeaderText(bytes: [])
                if authorization.isEmpty {
                    return .authenticationMissing403
                }
                let token = authorization.removingPrefix("Bearer ").stripped()
                if !ConstantTime.equal(token.utf8, Array(apiKey.utf8)) {
                    return .authentication401
                }
            }
            return nil
        }
    }
#endif
