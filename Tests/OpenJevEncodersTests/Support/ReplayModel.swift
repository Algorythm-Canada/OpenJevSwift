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

/// A model that returns the outputs Fixtures/encoders recorded for the rows it is handed, found
/// by the rows' token ids. Rows with the same ids get their recorded outputs in order.
///
/// For Verdict each output is the row's first k logits, then `filler` up to the head's 25, so a
/// backend that reads past k gets visibly wrong probabilities. For Laya it is a score per
/// position, the recorded scores at the markers and `filler` elsewhere
/// (``init(laya:filler:)``).
actor ReplayModel: EncoderModelRunner {
    struct UnknownRow: Error, CustomStringConvertible {
        var ids: [Int]
        var description: String { "no recorded output for a row of \(ids.count) tokens" }
    }

    /// One call: the rows' token ids, the attention masks' sums and every row's planes.
    struct Call: Sendable {
        var ids: [[Int]]
        var maskSums: [Int]
        var planes: [[[Int32]]]
    }

    private var outputs: [[Int]: [[Float]]]
    private(set) var calls: [Call] = []

    /// A model that answers each row of `ids` with its `output`.
    init(rows: [(ids: [Int], output: [Float])]) {
        var outputs: [[Int]: [[Float]]] = [:]
        for row in rows {
            outputs[row.ids, default: []].append(row.output)
        }
        self.outputs = outputs
    }

    /// Verdict's recorded logits, padded to `width` with `filler`.
    init(_ reads: [VerdictFixtures.Read], width: Int = 25, filler: Float = 50) {
        self.init(
            rows: reads.map { read in
                (
                    read.inputIDs,
                    read.logits
                        + [Float](repeating: filler, count: max(0, width - read.logits.count))
                )
            })
    }

    func run(_ rows: [[[Int32]]]) throws -> [[Float]] {
        let ids = rows.map { $0[0].map { Int($0) } }
        calls.append(
            Call(
                ids: ids, maskSums: rows.map { $0[1].reduce(0) { $0 + Int($1) } }, planes: rows))
        return try ids.map { row in
            guard var queue = outputs[row], !queue.isEmpty else {
                throw UnknownRow(ids: row)
            }
            let recorded = queue.removeFirst()
            outputs[row] = queue
            return recorded
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
