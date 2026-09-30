// Matches the output of CPython's `repr` for `str` (`unicode_repr` in Objects/unicodeobject.c),
// which upstream OpenJev's `{value!r}` messages rely on. Written from the documented behaviour.

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
