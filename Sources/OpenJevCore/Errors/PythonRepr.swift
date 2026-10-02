// Matches the output of CPython's `repr` for `str` (`unicode_repr` in Objects/unicodeobject.c),
// which upstream OpenJev's `{value!r}` messages rely on, for `bytes` (`PyBytes_Repr` in
// Objects/bytesobject.c), which its `trim` writes for a body FastAPI did not parse, and for
// `float` (`float_repr` in Objects/floatobject.c), which its f-strings write. Written from the
// documented behaviour.

/// Python's `repr` of a float, for messages that upstream formats with `!r`.
extension Double {
    /// The float as Python's `repr` writes it, which is also what `str` and an f-string
    /// replacement field without a format spec write: `0.0`, `1.0`, `0.1`, `1e+16`, `1e-05`.
    ///
    /// Finite values are laid out as ``PythonJSONWriter`` writes them (the shortest digits that
    /// round-trip, in CPython's fixed or exponential layout). The infinities are `inf` and `-inf`,
    /// and NaN is `nan` whatever its sign, as CPython writes them.
    public var pythonRepr: String {
        if isNaN {
            return "nan"
        }
        if isInfinite {
            return self < 0 ? "-inf" : "inf"
        }
        return pythonFloatRepr(self)
    }
}

/// Python's `repr` of a string, for messages that upstream formats with `!r`.
extension String {
    /// The string as Python's `repr` writes it, for messages that upstream formats with `!r`.
    ///
    /// The quote is `'`, or `"` when the string contains `'` and no `"`. The chosen quote and the
    /// backslash are escaped; tab, newline and carriage return become `\t`, `\n` and `\r`; other
    /// control characters and U+007F become `\xhh`. Non-ASCII scalars stay as they are when
    /// Python counts them printable, and are otherwise escaped as `\xhh`, `\uhhhh` or
    /// `\Uhhhhhhhh`. Printability uses the Unicode general category of this platform's Unicode
    /// tables, which can lag or lead the CPython build for newly assigned characters.
    public var pythonRepr: String {
        let scalars = unicodeScalars
        let quote: Unicode.Scalar = scalars.contains("'") && !scalars.contains("\"") ? "\"" : "'"
        var out = String.UnicodeScalarView()
        out.append(quote)
        for scalar in scalars {
            switch scalar {
            case quote, "\\":
                out.append("\\")
                out.append(scalar)
            case "\t":
                out.append(contentsOf: #"\t"#.unicodeScalars)
            case "\n":
                out.append(contentsOf: #"\n"#.unicodeScalars)
            case "\r":
                out.append(contentsOf: #"\r"#.unicodeScalars)
            default:
                if scalar.value < 0x20 || scalar.value == 0x7F {
                    out.append(contentsOf: Self.hexEscape(scalar.value, marker: "x", digits: 2))
                } else if scalar.value < 0x7F || Self.isPythonPrintable(scalar) {
                    out.append(scalar)
                } else if scalar.value <= 0xFF {
                    out.append(contentsOf: Self.hexEscape(scalar.value, marker: "x", digits: 2))
                } else if scalar.value <= 0xFFFF {
                    out.append(contentsOf: Self.hexEscape(scalar.value, marker: "u", digits: 4))
                } else {
                    out.append(contentsOf: Self.hexEscape(scalar.value, marker: "U", digits: 8))
                }
            }
        }
        out.append(quote)
        return String(out)
    }

    /// `\` and the marker, then the value in lowercase hex padded to `digits`.
    private static func hexEscape(
        _ value: UInt32, marker: Unicode.Scalar, digits: Int
    ) -> String.UnicodeScalarView {
        let hex = String(value, radix: 16)
        let padded = String(repeating: "0", count: Swift.max(0, digits - hex.count)) + hex
        return ("\\" + String(marker) + padded).unicodeScalars
    }

    /// Python's `str.isprintable` for one scalar: false for the "Other" and "Separator"
    /// categories (Cc, Cf, Cs, Co, Cn, Zl, Zp, Zs), true otherwise. The ASCII space is handled by
    /// the caller.
    private static func isPythonPrintable(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse, .unassigned, .lineSeparator,
            .paragraphSeparator, .spaceSeparator:
            return false
        default:
            return true
        }
    }
}

/// Python's `repr` of bytes, for the 422 of a body FastAPI did not parse.
extension String {
    /// Python's `repr` of a `bytes` object, `b'...'`, cut to its first `maxLength` characters when
    /// one is given, as `str(value)[:maxLength]` cuts it.
    ///
    /// The quote is `'`, or `"` when the bytes contain `'` and no `"`, decided over all the bytes
    /// even when the result is cut. The chosen quote and the backslash are escaped; tab, newline
    /// and carriage return become `\t`, `\n` and `\r`; every other byte below 0x20 or from 0x7F
    /// up becomes `\xhh` in lowercase hex. The result is ASCII, so characters are bytes.
    public static func pythonRepr(bytes: some Collection<UInt8>, maxLength: Int? = nil) -> String {
        let limit = Swift.max(0, maxLength ?? .max)
        let singleQuote = UInt8(ascii: "'")
        let doubleQuote = UInt8(ascii: "\"")
        let quote =
            bytes.contains(singleQuote) && !bytes.contains(doubleQuote) ? doubleQuote : singleQuote
        let hexDigits = Array("0123456789abcdef".utf8)
        var out: [UInt8] = [UInt8(ascii: "b"), quote]
        for byte in bytes {
            if out.count >= limit {
                break
            }
            switch byte {
            case quote, UInt8(ascii: "\\"):
                out += [UInt8(ascii: "\\"), byte]
            case UInt8(ascii: "\t"):
                out += [UInt8(ascii: "\\"), UInt8(ascii: "t")]
            case UInt8(ascii: "\n"):
                out += [UInt8(ascii: "\\"), UInt8(ascii: "n")]
            case UInt8(ascii: "\r"):
                out += [UInt8(ascii: "\\"), UInt8(ascii: "r")]
            case 0x20..<0x7F:
                out.append(byte)
            default:
                out += [
                    UInt8(ascii: "\\"), UInt8(ascii: "x"), hexDigits[Int(byte >> 4)],
                    hexDigits[Int(byte & 0x0F)],
                ]
            }
        }
        out.append(quote)
        return String(decoding: out.prefix(limit), as: UTF8.self)
    }
}
