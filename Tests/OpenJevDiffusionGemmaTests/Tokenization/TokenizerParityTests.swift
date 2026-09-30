import Foundation
import OpenJevCore
import OpenJevDiffusionGemma
import Testing

/// Compares the real tokenizer, ``SwiftTransformersTokenizer`` over the pinned DiffusionGemma
/// files, with what Python's `transformers` recorded in Fixtures/tokenizer and
/// Fixtures/labels.json (spike #20, decision D-008).
///
/// Every test collects all mismatches before failing, and the failure message groups them by
/// the corpus row's categories, so one run shows the whole picture. The counts are also written
/// by ``SpikeReport`` for the decision record.
@Suite(
    "Tokenizer parity with the Python fixtures",
    .enabled(if: TokenizerFixtures.available, TokenizerFixtures.missingMessage))
struct TokenizerParityTests {
    /// One disagreement between the tokenizer and a recording.
    struct Mismatch: CustomStringConvertible {
        var field: String
        var text: String
        var expected: String
        var actual: String
        var categories: [String]

        var description: String {
            "\(field) of \(text.debugDescription): expected \(expected), got \(actual)"
        }
    }

    /// A readable account of `mismatches`: the count, the count per category, and the first
    /// few in full.
    static func summary(_ mismatches: [Mismatch], of total: Int, in file: String) -> String {
        var perCategory: [String: Int] = [:]
        for mismatch in mismatches {
            for category in mismatch.categories {
                perCategory[category, default: 0] += 1
            }
        }
        let categories = perCategory.sorted { $0.key < $1.key }
            .map { "\($0.key): \($0.value)" }.joined(separator: ", ")
        var lines = ["\(file): \(mismatches.count) mismatches over \(total) rows"]
        if !categories.isEmpty {
            lines.append("by category: \(categories)")
        }
        lines += mismatches.prefix(8).map(\.description)
        return lines.joined(separator: "\n")
    }

    @Test("The tokenizer loads from the pinned files and finds the chat template")
    func loads() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        #expect(tokenizer.hasChatTemplate)
        #expect(tokenizer.files.modelConfig != nil)
        #expect(tokenizer.chatTemplateSource.contains("<|turn>model\\n"))
        let metrics = tokenizer.loadMetrics
        #expect(metrics.wallTime > .zero)
        #expect(metrics.wallTime < .seconds(30), "loading took \(metrics.wallTime)")
        #expect(metrics.peakResidentBytes > 0)
        let megabyte = 1024.0 * 1024.0
        SpikeReport.record(
            "tokenizer-load",
            """
            load wall time: \(metrics.wallTime)
            resident before: \(String(format: "%.1f", Double(metrics.residentBytesBefore) / megabyte)) MB
            resident after: \(String(format: "%.1f", Double(metrics.residentBytesAfter) / megabyte)) MB
            resident added: \(String(format: "%.1f", Double(metrics.residentBytesAdded) / megabyte)) MB
            peak resident (ru_maxrss): \(String(format: "%.1f", Double(metrics.peakResidentBytes) / megabyte)) MB
            """)
    }

    @Test("Every corpus row encodes and decodes as recorded")
    func corpus() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let rows = try TokenizerFixtures.cases("tokenizer/corpus.json")
        #expect(rows.count == 917)
        var mismatches: [Mismatch] = []
        var mismatchedRows = 0
        for row in rows {
            let text = try #require(row["text"]?.stringValue)
            let categories = try TokenizerFixtures.strings(row["categories"])
            let ids = try TokenizerFixtures.ints(row["ids"])
            let withSpecial = try TokenizerFixtures.ints(row["ids_with_special_tokens"])
            let decoded = try #require(row["decoded"]?.stringValue)
            let decodedSkipping = try #require(row["decoded_skip_special_tokens"]?.stringValue)
            var rowMismatches: [Mismatch] = []

            let actualIDs = try tokenizer.encode(text, addSpecialTokens: false)
            if actualIDs != ids {
                rowMismatches.append(
                    Mismatch(
                        field: "ids", text: text, expected: "\(ids)", actual: "\(actualIDs)",
                        categories: categories))
            }
            let actualWithSpecial = try tokenizer.encode(text, addSpecialTokens: true)
            if actualWithSpecial != withSpecial {
                rowMismatches.append(
                    Mismatch(
                        field: "ids_with_special_tokens", text: text,
                        expected: "\(withSpecial)", actual: "\(actualWithSpecial)",
                        categories: categories))
            }
            let actualDecoded = try tokenizer.decode(ids, skipSpecialTokens: false)
            if actualDecoded != decoded {
                rowMismatches.append(
                    Mismatch(
                        field: "decoded", text: text, expected: decoded.debugDescription,
                        actual: actualDecoded.debugDescription, categories: categories))
            }
            let actualSkipping = try tokenizer.decode(ids, skipSpecialTokens: true)
            if actualSkipping != decodedSkipping {
                rowMismatches.append(
                    Mismatch(
                        field: "decoded_skip_special_tokens", text: text,
                        expected: decodedSkipping.debugDescription,
                        actual: actualSkipping.debugDescription, categories: categories))
            }
            if !rowMismatches.isEmpty {
                mismatchedRows += 1
                mismatches += rowMismatches
            }
        }
        let summary = Self.summary(mismatches, of: rows.count, in: "corpus.json")
        SpikeReport.record(
            "tokenizer-parity",
            "corpus.json: \(rows.count - mismatchedRows) rows matched, \(mismatchedRows) "
                + "mismatched (\(mismatches.count) field mismatches)\n\(summary)")
        #expect(mismatches.isEmpty, "\(summary)")
    }

    @Test("Every engine encoding matches")
    func engineEncodings() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let pairs = try TokenizerFixtures.cases("tokenizer/engine_encodings.json")
        #expect(pairs.count == 2634)
        var mismatches: [Mismatch] = []
        for pair in pairs {
            let text = try #require(pair[0]?.stringValue)
            let ids = try TokenizerFixtures.ints(pair[1])
            let actual = try tokenizer.encode(text, addSpecialTokens: false)
            if actual != ids {
                mismatches.append(
                    Mismatch(
                        field: "ids", text: text, expected: "\(ids)", actual: "\(actual)",
                        categories: ["engine_encoding"]))
            }
        }
        let summary = Self.summary(mismatches, of: pairs.count, in: "engine_encodings.json")
        SpikeReport.record(
            "tokenizer-parity",
            "engine_encodings.json: \(pairs.count - mismatches.count) pairs matched, "
                + "\(mismatches.count) mismatched")
        #expect(mismatches.isEmpty, "\(summary)")
    }

    @Test("Label discovery over the real tokenizer reproduces labels.json")
    func labels() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let recorded = try TokenizerFixtures.load("labels.json")
        let discovered = try LabelDiscovery.choiceLabels(using: tokenizer)
        let labels = try TokenizerFixtures.strings(recorded["labels"])
        let labelIDs = try TokenizerFixtures.ints(recorded["label_ids"])

        #expect(
            try tokenizer.encode("q1: A", addSpecialTokens: false)
                == TokenizerFixtures.ints(recorded["base_ids"]))
        #expect(discovered.labels == labels)
        #expect(discovered.labelIDs == labelIDs)
        #expect(discovered.labels.count == 255)

        let rejected = try #require(recorded["rejected"]?.arrayValue)
        let base = try tokenizer.encode("q1: A", addSpecialTokens: false)
        var names: [String] = []
        for row in rejected {
            let candidate = try #require(row["candidate"]?.stringValue)
            names.append(candidate)
            let ids = try tokenizer.encode("q1: " + candidate, addSpecialTokens: false)
            #expect(ids == (try TokenizerFixtures.ints(row["ids"])), "\(candidate)")
            #expect(!discovered.labels.contains(candidate), "\(candidate)")
            #expect(ids.count != base.count || ids.dropLast() != base.dropLast(), "\(candidate)")
        }
        #expect(names == ["BQ", "BZ", "FQ", "FZ", "GZ", "HZ"])

        let matched = zip(discovered.labels, labels).filter { $0 == $1 }.count
        SpikeReport.record(
            "tokenizer-parity",
            "labels.json: \(matched) of \(labels.count) labels matched, ids "
                + "\(discovered.labelIDs == labelIDs ? "matched" : "mismatched"), "
                + "\(rejected.count) rejected candidates rejected")
    }

    @Test("Special tokens resolve to the recorded ids and encode as single ids")
    func specialTokens() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        let special = try TokenizerFixtures.load("tokenizer/special_tokens.json")
        var mismatches: [String] = []

        let named = try #require(special["named"]?.arrayValue)
        #expect(named.count == 23)
        for entry in named {
            let token = try #require(entry["token"]?.stringValue)
            let id = try #require(entry["id"]?.intValue)
            if tokenizer.tokenID(of: token) != id {
                mismatches.append(
                    "\(token): id \(String(describing: tokenizer.tokenID(of: token))), "
                        + "expected \(id)")
            }
            if tokenizer.token(of: id) != token {
                mismatches.append(
                    "\(id): token \(String(describing: tokenizer.token(of: id))), "
                        + "expected \(token)")
            }
            let encoded = try tokenizer.encode(token, addSpecialTokens: false)
            if encoded != [id] {
                mismatches.append("\(token) encodes to \(encoded), expected [\(id)]")
            }
        }
        for entry in try #require(special["extra_special_tokens"]?.arrayValue) {
            let token = try #require(entry["token"]?.stringValue)
            let id = try #require(entry["id"]?.intValue)
            if try tokenizer.encode(token, addSpecialTokens: false) != [id] {
                mismatches.append("\(token) does not encode to [\(id)]")
            }
        }

        // The ids the read path depends on, by name.
        let byName = Dictionary(
            uniqueKeysWithValues: try named.map {
                (try #require($0["name"]?.stringValue), try #require($0["id"]?.intValue))
            })
        #expect(byName["bos_token"] == 2)
        #expect(byName["eos_token"] == 1)
        #expect(byName["pad_token"] == 0)
        #expect(byName["boi_token"] == 255_999)
        #expect(byName["image_token"] == 258_880)
        #expect(byName["eoi_token"] == 258_882)
        #expect(byName["soc_token"] == 100)
        #expect(byName["eoc_token"] == 101)
        #expect(byName["eot_token"] == 106)
        #expect(tokenizer.tokenID(of: "<turn|>") == EngineTokens.turnClose)
        #expect(tokenizer.tokenID(of: "<pad>") == EngineTokens.pad)

        // `<end_of_turn>` is not a token of this vocabulary.
        let endOfTurn = try #require(special["end_of_turn_text"])
        #expect(
            try tokenizer.encode(
                try #require(endOfTurn["text"]?.stringValue), addSpecialTokens: false)
                == TokenizerFixtures.ints(endOfTurn["ids"]))

        // Special tokens written inside a state text are single ids.
        let inside = try tokenizer.encode("state with <turn|> inside", addSpecialTokens: false)
        #expect(inside.contains(EngineTokens.turnClose))
        #expect(inside.filter { $0 == EngineTokens.turnClose }.count == 1)

        // Engine tokens over the real tokenizer equal the recorded engine table.
        let engine = try #require(special["engine"])
        let tokens = try EngineTokens(tokenizer: tokenizer)
        #expect(tokens.scaffold == [100, 45518, 107, 101])
        #expect(tokens.scaffold == (try TokenizerFixtures.ints(engine["scaffold"])))
        #expect(tokens.thoughtOpen == (try TokenizerFixtures.ints(engine["thought_open"])))
        #expect(tokens.thoughtClose == (try TokenizerFixtures.ints(engine["thought_close"])))
        #expect(EngineTokens.vocabularySize == special["vocabulary_size"]?.intValue)

        SpikeReport.record(
            "tokenizer-parity",
            "special_tokens.json: \(named.count) named tokens, \(mismatches.count) mismatches; "
                + "engine table \(tokens.scaffold == [100, 45518, 107, 101] ? "matched" : "mismatched")"
        )
        #expect(mismatches.isEmpty, "\(mismatches)")
    }

    @Test("The real tokenizer agrees with the replay tokenizer on every text either knows")
    func agreesWithReplay() async throws {
        let real = try await TokenizerFixtures.tokenizer()
        let replay = ReplayTokenizer.shared
        var disagreements: [String] = []

        let texts = try replay.texts
        #expect(texts.count > 3000)
        for text in texts {
            let expected = try replay.encode(text, addSpecialTokens: false)
            let actual = try real.encode(text, addSpecialTokens: false)
            if actual != expected {
                disagreements.append("encode \(text.debugDescription)")
            }
        }
        for text in try replay.textsWithSpecialTokens {
            let expected = try replay.encode(text, addSpecialTokens: true)
            let actual = try real.encode(text, addSpecialTokens: true)
            if actual != expected {
                disagreements.append("encode with special tokens \(text.debugDescription)")
            }
        }
        for ids in try replay.idsDecoded {
            for skip in [false, true] {
                let expected = try replay.decode(ids, skipSpecialTokens: skip)
                let actual = try real.decode(ids, skipSpecialTokens: skip)
                if actual != expected {
                    disagreements.append("decode \(ids.prefix(8)) skip \(skip)")
                }
            }
        }
        let prompts = try replay.chatPrompts
        #expect(prompts.count == 24)
        for prompt in prompts {
            for thinking in [false, true] {
                let expected = try replay.chatPromptIDs(
                    system: prompt.system, user: prompt.user, thinking: thinking)
                let actual = try real.chatPromptIDs(
                    system: prompt.system, user: prompt.user, thinking: thinking)
                if actual != expected {
                    disagreements.append(
                        "chat prompt \(prompt.user.prefix(40).debugDescription) thinking "
                            + "\(thinking)")
                }
            }
        }
        // The replay tokenizer refuses what was never recorded; the real one answers.
        let unrecorded = "a text no fixture recorded 7f3a"
        #expect(throws: TokenizerError.self) {
            try replay.encode(unrecorded, addSpecialTokens: false)
        }
        #expect(try !real.encode(unrecorded, addSpecialTokens: false).isEmpty)

        SpikeReport.record(
            "tokenizer-parity",
            "replay agreement: \(texts.count) texts, \(prompts.count) prompts, "
                + "\(disagreements.count) disagreements")
        #expect(disagreements.isEmpty, "\(disagreements.prefix(10))")
    }

    /// Decodes no fixture row exercises: byte tokens at the end of a sequence, mixed with special
    /// tokens, and the punctuation that `clean_up_tokenization_spaces` would rewrite. Python's
    /// results here follow from `tokenizers`' byte fallback and transformers' default of no clean
    /// up; docs/spikes/tokenizer-parity.md lists the fixture rows to add so they are recorded.
    @Test("Trailing byte tokens and spaced punctuation decode as Python does")
    func decodeDepartures() async throws {
        let tokenizer = try await TokenizerFixtures.tokenizer()
        #expect(tokenizer.specialTokenIDs.count == 24)
        #expect(tokenizer.specialTokenIDs.isSuperset(of: [0, 1, 2, 3, 4, 106, 258_880, 258_884]))

        #expect(try tokenizer.decode([238], skipSpecialTokens: false) == "\0")
        #expect(
            try tokenizer.decode([482, 381, 429, 429], skipSpecialTokens: true) == "\u{10FFFF}")
        // Bytes before, between and after special tokens; the special tokens are dropped first.
        #expect(try tokenizer.decode([2, 238, 1], skipSpecialTokens: true) == "\0")
        #expect(try tokenizer.decode([2, 238, 1], skipSpecialTokens: false) == "<bos>\0<eos>")
        #expect(
            try tokenizer.decode([482, 381, 106, 429, 429], skipSpecialTokens: true)
                == "\u{10FFFF}")
        // An incomplete sequence decodes to replacement characters, one per maximal subpart.
        #expect(try tokenizer.decode([482, 381], skipSpecialTokens: false) == "\u{FFFD}")
        for text in ["a , b .", "so ?", "no !", "it 's", "do n't", " . ", "x .y"] {
            let ids = try tokenizer.encode(text, addSpecialTokens: false)
            #expect(try tokenizer.decode(ids, skipSpecialTokens: false) == text, "\(text)")
        }
    }

    @Test("The files are validated before loading")
    func files() throws {
        let files = try TokenizerFiles(directory: TokenizerFixtures.tokenizerDirectory)
        #expect(files.tokenizerData.lastPathComponent == "tokenizer.json")
        #expect(files.chatTemplate.lastPathComponent == "chat_template.jinja")
        try files.verify(digests: try TokenizerFixtures.recordedDigests())

        #expect {
            try files.verify(digests: ["chat_template.jinja": String(repeating: "0", count: 64)])
        } throws: { error in
            if case .digestMismatch("chat_template.jinja", _, _) = error as? TokenizerFilesError {
                return true
            }
            return false
        }

        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("openjev-empty-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(TokenizerFiles.missingNames(in: empty) == TokenizerFiles.requiredNames)
        #expect(throws: TokenizerFilesError.self) {
            try TokenizerFiles(directory: empty)
        }
    }
}
