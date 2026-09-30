import OpenJevCore
import Testing

/// `String.pythonRepr(bytes:maxLength:)` against the `repr` CPython 3.14 writes for the same
/// bytes.
@Suite("Python bytes repr")
struct PythonReprTests {
    @Test("Bytes are written as CPython's repr writes them")
    func bytesRepr() {
        let cases: [([UInt8], String)] = [
            ([], "b''"),
            (Array("plain".utf8), "b'plain'"),
            ([39], #"b"'""#),
            ([34], #"b'"'"#),
            ([39, 34], #"b'\'"'"#),
            (Array("it's".utf8), #"b"it's""#),
            (Array(#"say "hi""#.utf8), #"b'say "hi"'"#),
            (Array(#"back\slash"#.utf8), #"b'back\\slash'"#),
            ([9, 10, 13], #"b'\t\n\r'"#),
            ([0, 1, 31, 127, 128, 255], #"b'\x00\x01\x1f\x7f\x80\xff'"#),
            (Array("é€😀".utf8), #"b'\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80'"#),
            (Array(#"{"state":"x"}"#.utf8), #"b'{"state":"x"}'"#),
        ]
        for (bytes, expected) in cases {
            #expect(String.pythonRepr(bytes: bytes) == expected, "\(bytes)")
        }
    }

    @Test("A cut keeps the first characters, with no marker, and the quote of the whole")
    func cut() {
        let long = [UInt8](repeating: UInt8(ascii: "a"), count: 600)
        let cut = String.pythonRepr(bytes: long, maxLength: 500)
        #expect(cut == "b'" + String(repeating: "a", count: 498))
        // The only ' comes after the cut, and still decides the quote, as in str(value)[:5].
        let quoted = [UInt8](repeating: UInt8(ascii: "a"), count: 550) + [39]
        #expect(String.pythonRepr(bytes: quoted, maxLength: 5) == #"b"aaa"#)
        #expect(String.pythonRepr(bytes: Array("ab".utf8), maxLength: 500) == "b'ab'")
    }
}
