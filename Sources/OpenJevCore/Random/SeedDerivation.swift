// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/api.py` lines 262 to 263
// (the request seed) and `openjev/engine.py` lines 342 and 383 (the derived seeds). Apache-2.0.
// See THIRD_PARTY.md.

/// How upstream turns a request into the seed of its noise draws, and the seeds it derives from it.
///
/// Upstream hashes `json.dumps([state, questions] + ([image URLs] if images else []),
/// sort_keys=True)` with SHA-256 and takes the first four bytes as a big-endian integer. The same
/// request therefore always gets the same canvases and the same answers (decision D-006). The
/// model name and the extension fields are not part of the key.
public enum SeedDerivation {
    /// The value upstream serializes to seed a request: `[state, questions]`, plus the list of
    /// image data URLs when there are images.
    ///
    /// `questions` is each question as pydantic's `model_dump()` gives it, so every declared field
    /// is present and unset ones are `null`:
    /// - noul: `{"type": "noul", "instructions": ..., "criteria": null}` or, with criteria,
    ///   `"criteria": {"true": ..., "false": ...}`;
    /// - choice: `{"type": "choice", "instructions": ..., "criteria": {name: description}}`;
    /// - score: `{"type": "score", "instructions": ..., "criteria": [...]}`.
    ///
    /// An empty `images` list adds nothing, as upstream's `if images` does.
    public static func seedKey(
        state: JSONValue, questions: OrderedMap<Question>, images: [ImagePart]
    ) -> JSONValue {
        let dumped = JSONObject(
            uniqueKeysWithValues: questions.map { ($0.key, modelDump($0.value)) })
        var key: [JSONValue] = [state, .object(dumped)]
        if !images.isEmpty {
            key.append(.array(images.map { .string($0.dataURL) }))
        }
        return .array(key)
    }

    /// The bytes upstream hashes: `json.dumps(key, sort_keys=True)`, ASCII because `ensure_ascii`
    /// is on.
    ///
    /// - Throws: The writer's error for a value it cannot write. ``JSONParser`` never produces
    ///   one.
    public static func seedBytes(for key: JSONValue) throws(JSONWriteError) -> [UInt8] {
        try PythonJSONWriter.canonicalSeedBytes(key)
    }

    /// The request seed: the first four bytes of the SHA-256 digest of ``seedBytes(for:)`` as a
    /// big-endian integer.
    ///
    /// The value fits in 32 bits. It is returned as `UInt64` so the derived seeds, which can pass
    /// 2^32, need no conversion.
    public static func seed(for key: JSONValue) throws(JSONWriteError) -> UInt64 {
        let digest = SHA256.digest(try seedBytes(for: key))
        return digest.prefix(4).reduce(0) { $0 << 8 | UInt64($1) }
    }

    /// The seed of question group `k`: `base + 104729·k` (`engine.py` line 383).
    public static func groupSeed(_ base: UInt64, _ k: Int) -> UInt64 {
        base + 104_729 * UInt64(k)
    }

    /// The seed of sample or re-read `k`: `base + 7919·k` (`engine.py` line 342).
    public static func sampleSeed(_ base: UInt64, _ k: Int) -> UInt64 {
        base + 7_919 * UInt64(k)
    }

    /// A question as pydantic's `model_dump()` gives it, every declared field present.
    private static func modelDump(_ question: Question) -> JSONValue {
        let criteria: JSONValue
        switch question {
        case .noul(_, let noulCriteria):
            if let noulCriteria {
                criteria = [
                    "true": noulCriteria.whenTrue ?? .null,
                    "false": noulCriteria.whenFalse ?? .null,
                ]
            } else {
                criteria = .null
            }
        case .choice(_, let options):
            criteria = .object(options)
        case .score(_, let levels):
            criteria = .array(levels)
        }
        return [
            "type": .string(question.type),
            "instructions": question.instructions ?? .null,
            "criteria": criteria,
        ]
    }
}
