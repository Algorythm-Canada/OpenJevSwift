// How upstream OpenJev (razorback16/openjev at dcd2094) gets the body of `POST /v1/systemone`:
// FastAPI's request handler (`fastapi/routing.py` in FastAPI 0.142, the version the fixtures
// record) reads it, parses it with Starlette's `Request.json()` when the content type is JSON,
// and answers a body it cannot read with `json_invalid` or "There was an error parsing the body".
// Apache-2.0. See THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import Hummingbird
    import OpenJevCore

    /// FastAPI's reading of the `POST /v1/systemone` body, up to the value pydantic validates.
    struct RequestBodyReader: Sendable {
        /// `OPENJEV_MAX_BODY_BYTES`.
        let maxBodyBytes: Int

        /// The most characters of a body's bytes repr the 422 echoes, upstream's `TRIM_CHARS`:
        /// `trim` writes a value that is neither JSON nor a string as `str(value)[:500]`.
        static let reprCharacters = 500

        /// The value pydantic validates for the body, in FastAPI's order:
        ///
        /// - `nil` for an empty body, whatever its content type (`if body_bytes:`);
        /// - for a JSON content type (``isJSON(_:)``), the parsed body;
        /// - for any other content type, or none, the body as bytes: pydantic refuses them with
        ///   one `model_attributes_type` at `body`, whose `input` upstream writes as their repr,
        ///   `b'...'`, cut at 500 characters. The repr stands in for them here, so validating it
        ///   gives that 422.
        ///
        /// - Throws: ``WireError/jsonInvalid422(message:position:)`` for a body that is not JSON,
        ///   ``WireError/unparsableBody400`` for one `json.loads` cannot read at all (see
        ///   ``parse(_:)``), and whatever reading the body throws.
        func value(of request: Request) async throws -> JSONValue? {
            // The cap middleware has read the body into one buffer already; the limit keeps this
            // route bounded should it ever be mounted without it.
            let buffer = try await request.body.collect(upTo: maxBodyBytes)
            let body = buffer.readableBytesView
            if body.isEmpty {
                return nil
            }
            guard Self.isJSON(HeaderText(.contentType, in: request.headers)) else {
                return .string(String.pythonRepr(bytes: body, maxLength: Self.reprCharacters))
            }
            return try parse(body)
        }

        /// FastAPI's test for a JSON body: the media type, read as `email.message` reads it, is
        /// `application/json` or `application/` something ending in `+json`, with any
        /// parameters. The media type is the text before the first `;`, stripped and lowercased,
        /// and anything without exactly one `/` counts as `text/plain`. A missing or empty header
        /// is not JSON (FastAPI's `strict_content_type`).
        static func isJSON(_ contentType: HeaderText?) -> Bool {
            guard let contentType, !contentType.isEmpty else {
                return false
            }
            let mediaType = HeaderText(
                bytes: Array(contentType.bytes.prefix { $0 != UInt8(ascii: ";") })
            ).stripped().asciiLowercased
            let parts = mediaType.split(
                separator: UInt8(ascii: "/"), omittingEmptySubsequences: false)
            guard parts.count == 2, parts[0].elementsEqual("application".utf8) else {
                return false
            }
            let subtype = parts[1]
            return subtype.elementsEqual("json".utf8)
                || subtype.reversed().starts(with: "nosj+".utf8)
        }

        /// Starlette's `json.loads(body)`: a UTF-8 byte order mark is dropped, the rest must be
        /// UTF-8 and one JSON value, and integers are limited to 4,300 digits.
        ///
        /// - Throws: ``WireError/jsonInvalid422(message:position:)`` with CPython's message and
        ///   character position for a body `json.loads` refuses with a `JSONDecodeError`, and
        ///   ``WireError/unparsableBody400`` for one that is not UTF-8 or holds a longer integer,
        ///   which FastAPI answers alike. A body only the stricter ``JSONParser`` refuses gets the
        ///   closest of those (``refusal(for:body:document:)``).
        func parse(_ body: some Collection<UInt8>) throws(WireError) -> JSONValue {
            let document = PythonJSONLoads.document(body)
            let parser = JSONParser(options: JSONParser.Options(maximumBytes: maxBodyBytes))
            let value: JSONValue
            do {
                value = try parser.parse(document)
            } catch {
                throw Self.refusal(for: error, body: body, document: document)
            }
            if PythonJSONLoads.hasIntegerBeyondDigitLimit(value) {
                throw WireError.unparsableBody400
            }
            return value
        }

        /// The answer for a body ``JSONParser`` refused: what `json.loads` raises for it, found
        /// by ``PythonJSONLoads/outcome(of:)``.
        ///
        /// When CPython reads the body anyway (decision D-016), the answer is the closest one
        /// CPython gives, at the place the parser stopped: "Expecting value" at `NaN`,
        /// `Infinity`, `-Infinity` or a float beyond `Double`; "Invalid \uXXXX escape" at a lone
        /// surrogate's `u`; and the 400 for nesting deeper than the parser's 1,024 levels,
        /// which is what upstream answers once CPython's stack runs out, as it does here past
        /// ``PythonJSONLoads/maximumNesting``.
        static func refusal<Body: Collection<UInt8>>(
            for error: JSONParseError, body: Body, document: Body.SubSequence
        ) -> WireError {
            switch PythonJSONLoads.outcome(of: body) {
            case .notUTF8, .integerTooLong, .nestingTooDeep:
                return .unparsableBody400
            case .decodeError(let message, let position):
                return .jsonInvalid422(message: message, position: position)
            case .accepted:
                let position = PythonJSONLoads.characterCount(of: document.prefix(error.offset))
                switch error.kind {
                case .depthExceeded:
                    return .unparsableBody400
                case .loneSurrogate:
                    // The parser stops at the backslash; CPython's \u errors point at the u.
                    return .jsonInvalid422(
                        message: "Invalid \\uXXXX escape", position: position + 1)
                case .invalidNumber:
                    // Only -Infinity gets here, refused at its I: point at the minus sign.
                    return .jsonInvalid422(
                        message: "Expecting value", position: max(0, position - 1))
                default:
                    return .jsonInvalid422(message: "Expecting value", position: position)
                }
            }
        }
    }
#endif
