import Foundation
import OpenJevCore
import OpenJevTestSupport
import Testing

/// Loads Fixtures/seeds.json, which Tools/fixtures/upstream_tables.py writes from upstream's
/// route and CPython's `random` module:
/// - `cases`: `{name, body_text, key_text, sha256, seed, randrange_262144, same_seed_as?}`, a body
///   as sent, the bytes upstream hashes for it, their digest, the seed and the first 64
///   `randrange(262144)` draws;
/// - `upstream_only`: the same for bodies only Python's `json.loads` accepts. They are not read
///   here: `JSONParser` refuses them by design (decision D-016);
/// - `mt19937_seeds`: 10,000 rows `[seed, getrandbits(32), randrange(262144)]`, each value the
///   first draw of a fresh `random.Random(seed)`, with the column names in `mt19937_columns`;
/// - `mt19937_streams`: `{seed, call, values}`, long runs of one call from one seed.
///
/// The file is parsed once per process. Tests that need it are disabled with a message when it
/// is missing.
enum SeedFixtures {
    /// The fixture file.
    static let url = UpstreamFixtures.directory.appendingPathComponent("seeds.json")

    /// The message shown when the file is missing.
    static let missingMessage: Comment = "Fixtures/seeds.json is missing; run make fixtures"

    /// True when the file exists.
    static var exists: Bool { FileManager.default.fileExists(atPath: url.path) }

    /// The parsed file, or the error parsing it gave.
    private static let parsed: Result<JSONValue, any Error> = Result {
        try JSONParser().parse(Data(contentsOf: url))
    }

    /// The parsed file.
    static func load() throws -> JSONValue {
        try parsed.get()
    }

    /// The named array of the file.
    static func array(_ key: String) throws -> [JSONValue] {
        try #require(load()[key]?.arrayValue, "seeds.json has no array \(key)")
    }

    /// An integer field as `UInt64`; seeds in the file reach 63 bits.
    static func unsigned(_ value: JSONValue?) throws -> UInt64 {
        try #require(
            value?.integerText.flatMap { UInt64($0) },
            "not an unsigned integer: \(String(describing: value))")
    }

    /// An array of integers.
    static func integers(_ value: JSONValue?) throws -> [Int] {
        let values = try #require(
            value?.arrayValue, "not an array: \(String(describing: value))")
        return try values.map { try #require($0.intValue, "not an integer: \($0)") }
    }

    /// Lowercase hexadecimal, as Python's `hexdigest()` writes a digest.
    static func hex(_ bytes: [UInt8]) -> String {
        bytes.map { byte in
            let digits = String(byte, radix: 16)
            return byte < 16 ? "0" + digits : digits
        }
        .joined()
    }
}
