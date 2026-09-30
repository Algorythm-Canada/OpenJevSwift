// How upstream OpenJev (razorback16/openjev at dcd2094) sees a request header through Starlette:
// `request.headers.get(name)` is the first field of that name, decoded as Latin-1, and
// `openjev/api.py` strips and re-encodes it with Python's `str` methods. Apache-2.0. See
// THIRD_PARTY.md.

#if canImport(Hummingbird)
    import HTTPTypes
    import OpenJevCore

    /// A header value as Starlette gives it to upstream: the bytes of the first field with the
    /// name, each byte one Latin-1 character (U+0000 to U+00FF).
    ///
    /// Keeping the bytes, rather than the field's `value` string, reproduces Python exactly for
    /// the bytes that reach the server: `str.strip()` removes the Latin-1 whitespace characters,
    /// `str.encode()` writes each byte from 0x80 up as two UTF-8 bytes, and non-ASCII values
    /// compare as upstream compares them. Behind Hummingbird's HTTP/1 server those bytes are
    /// UTF-8: NIO reads every header value as UTF-8 and turns a byte that is not UTF-8 into
    /// U+FFFD, so such a value arrives changed (decision D-031).
    struct HeaderText: Equatable {
        /// The characters, one per byte.
        var bytes: [UInt8]

        /// The value of the first field named `name`, or `nil` when there is none, as Starlette's
        /// `Headers.get` reads it.
        init?(_ name: HTTPField.Name, in headers: HTTPFields) {
            guard let field = headers[fields: name].first else { return nil }
            self.bytes = field.withUnsafeBytesOfValue { Array($0) }
        }

        /// A value from its characters.
        init(bytes: [UInt8]) {
            self.bytes = bytes
        }

        /// Whether the value is the empty string, which Python treats as false.
        var isEmpty: Bool { bytes.isEmpty }

        /// `str.removeprefix(prefix)` for an ASCII prefix.
        func removingPrefix(_ prefix: String) -> HeaderText {
            let prefixBytes = Array(prefix.utf8)
            return bytes.starts(with: prefixBytes)
                ? HeaderText(bytes: Array(bytes.dropFirst(prefixBytes.count))) : self
        }

        /// `str.strip()`: without the leading and trailing characters Python counts as
        /// whitespace, which among Latin-1 are U+0009 to U+000D, U+001C to U+0020, U+0085 and
        /// U+00A0.
        func stripped() -> HeaderText {
            let isSpace = { (byte: UInt8) in TextOf.isPythonWhitespace(Unicode.Scalar(byte)) }
            guard let start = bytes.firstIndex(where: { !isSpace($0) }),
                let end = bytes.lastIndex(where: { !isSpace($0) })
            else {
                return HeaderText(bytes: [])
            }
            return HeaderText(bytes: Array(bytes[start...end]))
        }

        /// `str.encode()`: the characters as UTF-8.
        var utf8: [UInt8] {
            var out: [UInt8] = []
            out.reserveCapacity(bytes.count)
            for byte in bytes {
                if byte < 0x80 {
                    out.append(byte)
                } else {
                    out += [0xC0 | byte >> 6, 0x80 | byte & 0x3F]
                }
            }
            return out
        }

        /// The text with ASCII letters lowercased. Python's `str.lower()` also lowercases the
        /// Latin-1 letters, which never turns them into ASCII, so comparisons with ASCII text
        /// come out the same.
        var asciiLowercased: [UInt8] {
            bytes.map { $0 >= UInt8(ascii: "A") && $0 <= UInt8(ascii: "Z") ? $0 | 0x20 : $0 }
        }
    }

    /// Byte comparison whose time does not depend on where the bytes differ, upstream's
    /// `hmac.compare_digest`.
    enum ConstantTime {
        /// Whether `given` and `expected` are the same bytes. Every position up to the longer
        /// length is compared, with no early return, and a difference in length counts as one
        /// more mismatch, so the time depends on the lengths alone.
        static func equal(_ given: [UInt8], _ expected: [UInt8]) -> Bool {
            var difference: UInt8 = given.count == expected.count ? 0 : 1
            for index in 0..<max(given.count, expected.count) {
                let left = index < given.count ? given[index] : 0
                let right = index < expected.count ? expected[index] : 0
                difference |= left ^ right
            }
            return difference == 0
        }
    }
#endif
