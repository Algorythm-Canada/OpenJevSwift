// A port of laya 0.3.6 (NandhaKishorM/laya), `laya/common.py`, function `build_sequence`, with
// the `max_len` and `head_max_len` that `Agent.system_one` reads from rl_agent_config.json and its
// check that every option kept its marker. Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore

/// One question's model input as laya's `build_sequence` writes it:
/// `[CLS] head [SEP] [MASK] option [MASK] option ... [SEP] state [SEP]`, and the positions of the
/// `[MASK]` markers, where the model's scores are read.
///
/// The budgets, in order:
///
/// 1. Each option keeps at most 48 of its tokens after its marker.
/// 2. The head and the options share ``LayaCalibration/headMaxLength`` (256) tokens. When the
///    options leave fewer than 16, each option is cut to `max(4, (headMaxLength - 16) / n)`
///    tokens, its marker included, for `n` options.
/// 3. The head is cut to what the options leave, but never below 8 tokens.
/// 4. The state fills what is left of ``LayaCalibration/maxLength`` (1,024) after the final
///    `[SEP]`, cut on the right.
/// 5. The sequence is cut to the maximum length, and markers past it are dropped. A question
///    that loses a marker this way overflows the head's budget, which ``LayaBackend`` refuses
///    as upstream does.
public struct LayaSequence: Sendable, Hashable {
    /// The token ids, `[CLS]` to the final `[SEP]`: the row's length is what upstream bills.
    public var ids: [Int]
    /// The position of each option's `[MASK]` marker in ``ids``, in laya's option order.
    public var markers: [Int]

    /// Creates a sequence from its ids and markers.
    public init(ids: [Int], markers: [Int]) {
        self.ids = ids
        self.markers = markers
    }

    /// The most tokens an option keeps after its marker.
    public static let optionTokenLimit = 48
    /// The fewest tokens the options may leave the head before each option is cut.
    public static let headRoomFloor = 16
    /// The fewest tokens the head keeps, however long the options.
    public static let headTokenFloor = 8
    /// The fewest tokens an option keeps, its marker included, when the options are cut.
    public static let optionTokenFloor = 4

    /// `build_sequence` over a prompt: the head and each option tokenized here, the state's ids
    /// given, so that a batch tokenizes its state once for all its questions.
    ///
    /// - Parameters:
    ///   - stateIDs: The ids of ``LayaPrompt/stateText(_:)`` of the state, uncut.
    public init(
        prompt: LayaPrompt, stateIDs: [Int], tokenizer: any LayaTokenizing, maxLength: Int,
        headMaxLength: Int
    ) {
        self = Self.build(
            head: tokenizer.encode(prompt.head),
            options: prompt.optionTexts.map(tokenizer.encode), state: stateIDs,
            special: tokenizer.specialTokens, maxLength: maxLength, headMaxLength: headMaxLength)
    }

    /// `build_sequence` over token ids: the head's, each option's (after its marker) and the
    /// state's, none of them cut yet.
    ///
    /// - Precondition: `maxLength` and `headMaxLength` are positive, as
    ///   ``LayaCalibration`` guarantees.
    public static func build(
        head: [Int], options: [[Int]], state: [Int], special: LayaSpecialTokens, maxLength: Int,
        headMaxLength: Int
    ) -> LayaSequence {
        precondition(maxLength > 0 && headMaxLength > 0, "laya's lengths are positive")
        var optionIDs = options.map { [special.mask] + $0.prefix(optionTokenLimit) }
        var budget = headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < headRoomFloor {
            let share = floorDivision(headMaxLength - headRoomFloor, max(1, optionIDs.count))
            let per = max(optionTokenFloor, share)
            optionIDs = optionIDs.map { Array($0.prefix(per)) }
            budget = headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        var ids = [special.classToken] + head.prefix(max(headTokenFloor, budget))
        ids.append(special.separator)
        var markers: [Int] = []
        markers.reserveCapacity(optionIDs.count)
        for option in optionIDs {
            markers.append(ids.count)
            ids += option
        }
        ids.append(special.separator)
        let room = max(0, maxLength - ids.count - 1)
        ids += state.prefix(room)
        ids.append(special.separator)
        return LayaSequence(
            ids: Array(ids.prefix(maxLength)), markers: markers.filter { $0 < maxLength })
    }

    /// Python's `//` on integers: the quotient rounded toward minus infinity.
    static func floorDivision(_ dividend: Int, _ divisor: Int) -> Int {
        let quotient = dividend / divisor
        return (dividend % divisor != 0 && (dividend < 0) != (divisor < 0))
            ? quotient - 1 : quotient
    }
}

extension LayaSequence {
    /// The refusal upstream's `LayaEngine.read_batch` gives when laya's `system_one` raises
    /// `ValueError` because a question lost a marker: its options do not fit in the head's
    /// budget.
    public static func overflowError(model: String, headMaxLength: Int) -> SchemaError {
        SchemaError(
            "Too many choices for \(model): a question's options must fit in "
                + "\(headMaxLength) tokens.")
    }
}
