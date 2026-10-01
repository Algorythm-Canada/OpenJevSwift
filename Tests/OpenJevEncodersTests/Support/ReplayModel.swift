import Foundation
import OpenJevEncoders

/// A tokenizer that hands back the ids Fixtures/encoders recorded for each prompt.
struct ReplayTokenizer: VerdictTokenizing {
    let ids: [String: [Int]]

    init(_ reads: [VerdictFixtures.Read]) {
        var ids: [String: [Int]] = [:]
        for read in reads {
            ids[read.prompt] = read.inputIDs
        }
        self.ids = ids
    }

    init(ids: [String: [Int]]) {
        self.ids = ids
    }

    /// The recorded ids, or `[-1]`, which ``ReplayModel`` refuses, for a prompt never recorded.
    func inputIDs(for prompt: String) -> [Int] {
        ids[prompt] ?? [-1]
    }
}

/// A model that returns the logits Fixtures/encoders recorded for the rows it is handed: each
/// row's first k logits, then `filler` up to Verdict's 25, so a backend that reads past k gets
/// visibly wrong probabilities. Rows with the same ids get their recorded logits in order.
actor ReplayModel: EncoderModelRunner {
    struct UnknownRow: Error, CustomStringConvertible {
        var ids: [Int]
        var description: String { "no recorded logits for a row of \(ids.count) tokens" }
    }

    /// One call: the rows' token ids and the attention masks' sums.
    struct Call: Sendable {
        var ids: [[Int]]
        var maskSums: [Int]
    }

    private var logits: [[Int]: [[Float]]]
    private let width: Int
    private let filler: Float
    private(set) var calls: [Call] = []

    init(_ reads: [VerdictFixtures.Read], width: Int = 25, filler: Float = 50) {
        var logits: [[Int]: [[Float]]] = [:]
        for read in reads {
            logits[read.inputIDs, default: []].append(read.logits)
        }
        self.logits = logits
        self.width = width
        self.filler = filler
    }

    func run(_ rows: [[[Int32]]]) throws -> [[Float]] {
        let ids = rows.map { $0[0].map { Int($0) } }
        calls.append(Call(ids: ids, maskSums: rows.map { $0[1].reduce(0) { $0 + Int($1) } }))
        return try ids.map { row in
            guard var queue = logits[row], !queue.isEmpty else {
                throw UnknownRow(ids: row)
            }
            let recorded = queue.removeFirst()
            logits[row] = queue
            return recorded + [Float](repeating: filler, count: max(0, width - recorded.count))
        }
    }
}

/// A model that returns fixed rows whatever it is handed, and counts its calls.
actor FixedModel: EncoderModelRunner {
    let rows: [[Float]]
    private(set) var rowCounts: [Int] = []

    init(rows: [[Float]]) {
        self.rows = rows
    }

    func run(_ input: [[[Int32]]]) throws -> [[Float]] {
        rowCounts.append(input.count)
        return Array(rows.prefix(input.count))
    }
}
