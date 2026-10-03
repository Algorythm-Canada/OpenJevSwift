// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/encoders.py`, class
// `JevK5Engine` (`max_choices`, `load`'s letter check, `letter_logprobs`, `read_question` and
// `read`) and function `jevk5_temperature`, with the model in process instead of behind vLLM.
// Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// The model a ``JevK5Backend`` reads its answer letters from.
///
/// ``Qwen35LetterReadoutModel`` runs JevK5 on MLX; tests pass a stub. The backend calls it from
/// inside its actor, one pass at a time, so an implementation needs no locking of its own.
public protocol LetterReadoutModel: Sendable {
    /// One forward pass over `tokens`: the logits at the last position of the vocabulary entries
    /// `letterIDs`, in that order.
    func letterLogits(tokens: [Int], letterIDs: [Int]) throws -> [Float]
}

/// The tokenizer a ``JevK5Backend`` writes its prompts with.
public protocol LetterReadoutTokenizing: Sendable {
    /// The ids of `text`, with no special tokens added, as upstream asks vLLM for them
    /// (`add_special_tokens: false`). The template's `<|im_start|>` and `<|im_end|>` are in the
    /// text and come out as their own ids.
    func encode(_ text: String) -> [Int]
    /// The vocabulary's longest entry in Unicode scalars, which bounds a prompt's characters
    /// (``JevK5PromptLimit/maxCharactersPerToken``).
    var maxCharactersPerToken: Int { get }
}

/// Why JevK5 could not be loaded as the backend needs it.
public enum JevK5LoadError: Error, Sendable, Hashable, CustomStringConvertible {
    /// A file the backend reads does not exist.
    case missingFiles([String], in: URL)
    /// An answer letter is not one token: upstream's `every answer letter must be one token,
    /// got {ids}`, the ids of every letter as Python prints the list.
    case letterNotOneToken([[Int]])
    /// `jevk5_config.json` holds no usable temperature.
    case invalidTemperature(String)
    /// The checkpoint is not the text-only Qwen3.5 the backend runs.
    case unsupportedModel(String)
    /// The default checkpoint's repository has not been published yet (D-052), so there is
    /// nothing to download.
    case notPublished(repository: String)
    /// The MLX cache limit is not a finite number of GB, 0 or more.
    case invalidCacheLimit(Double)

    /// What went wrong, naming the file, the letters or the repository.
    public var description: String {
        switch self {
        case .missingFiles(let names, let directory):
            return "\(directory.path) lacks \(names.joined(separator: ", "))"
        case .letterNotOneToken(let ids):
            let list = ids.map { "[" + $0.map(String.init).joined(separator: ", ") + "]" }
            return "every answer letter must be one token, got [\(list.joined(separator: ", "))]"
        case .invalidTemperature(let message), .unsupportedModel(let message):
            return message
        case .invalidCacheLimit(let gb):
            return "the MLX cache limit is \(gb) GB (OPENJEV_MLX_CACHE_LIMIT_GB); it must be a "
                + "finite number of GB, 0 or more"
        case .notPublished(let repository):
            return "\(repository) is not published yet; convert the checkpoint with "
                + "Tools/jevk5/convert.py and set OPENJEV_JEVK5_MODEL to the folder it writes "
                + "(docs/deployment.md)"
        }
    }
}

/// The model returned something its contract forbids, such as fewer logits than letters asked.
public struct JevK5ModelError: Error, Sendable, Hashable, CustomStringConvertible {
    /// What was wrong.
    public var message: String

    /// Creates the error.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}

/// JevK5 (`jevk5-0.2`, Alibi Serikbay's Qwen3.5-4B with a merged, distilled LoRA) behind the
/// encoder backend contract, upstream's `JevK5Engine`.
///
/// Each question is read on its own, as upstream reads it: its options lettered `A` to `P` in
/// ``JevK5Prompt``'s JSON prompt, one forward pass, and a softmax over the letters' next-token
/// logits under the checkpoint's calibration temperature (``JevK5Readout/letterProbabilities(logits:temperature:)``).
/// A question with more than 16 options takes ceil(n / 16) + 1 passes, combined by
/// ``JevK5Readout/spread(_:texts:method:temperature:)`` with the knockout method. The questions
/// of a batch are read concurrently, as upstream's `fanout` pool reads a request's, and their
/// passes queue on this actor, which runs them one at a time: concurrency buys the overlap of
/// rendering and tokenizing with the model's passes, not GPU parallelism.
///
/// `usage.input_tokens` counts the prompt tokens of every pass, as upstream bills them. A pass
/// over ``JevK5PromptLimit``'s bound is refused with upstream's 400 through
/// ``/OpenJevCore/BackendRefusal``, never truncated; when several questions of a batch fail, the
/// first in question order is reported, as upstream's ordered `map` reports it.
///
/// ``load(_:cache:token:cacheLimitGB:)`` runs the model on MLX. The initializer takes any
/// ``LetterReadoutModel`` and ``LetterReadoutTokenizing``, so tests can replay recorded logits.
public actor JevK5Backend: QuestionReadBackend {
    /// ``/OpenJevCore/KnownEncoderModels/jevk5``: `jevk5-0.2` with upstream's description.
    public nonisolated let modelInfo = KnownEncoderModels.jevk5
    /// 255, Jev's limit, as upstream's encoder engines keep it.
    public nonisolated let maxChoices = 255

    /// The model the passes run through.
    public nonisolated let model: any LetterReadoutModel
    /// The tokenizer the prompts are written with.
    public nonisolated let tokenizer: any LetterReadoutTokenizing
    /// The calibration temperature, `jevk5_config.json`'s `temperature` (1.532 for v0.2).
    public nonisolated let temperature: Double
    /// The vocabulary ids of the letters `A` to `P`, in order.
    public nonisolated let letterIDs: [Int]
    /// The longest prompt a pass reads.
    public nonisolated let limit: JevK5PromptLimit
    /// How more than 16 options are combined: the knockout, upstream's.
    public nonisolated let method: JevK5Readout.Method

    /// The passes run so far.
    public private(set) var passCount = 0

    /// 16,383: a pass holds at most that many prompt tokens (``JevK5PromptLimit/maxInputTokens``).
    public nonisolated var maxPromptTokens: Int? { limit.maxInputTokens }

    /// Creates a backend over a model, a tokenizer and a temperature.
    ///
    /// - Parameter limit: the prompt bound; by default upstream's 16,384-token context with the
    ///   tokenizer's ``LetterReadoutTokenizing/maxCharactersPerToken``.
    /// - Throws: ``JevK5LoadError/letterNotOneToken(_:)`` when a letter is not one token, as
    ///   upstream's `load` checks, and ``JevK5LoadError/invalidTemperature(_:)`` for a temperature
    ///   that is not a positive finite number.
    public init(
        model: any LetterReadoutModel, tokenizer: any LetterReadoutTokenizing, temperature: Double,
        limit: JevK5PromptLimit? = nil, method: JevK5Readout.Method = .knockout
    ) throws(JevK5LoadError) {
        let ids = JevK5Prompt.letters.map { tokenizer.encode($0) }
        guard ids.allSatisfy({ $0.count == 1 }) else {
            throw .letterNotOneToken(ids)
        }
        guard temperature.isFinite, temperature > 0 else {
            throw .invalidTemperature(
                "the calibration temperature is \(temperature); it must be a positive number")
        }
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.letterIDs = ids.map { $0[0] }
        self.limit =
            limit ?? JevK5PromptLimit(maxCharactersPerToken: tokenizer.maxCharactersPerToken)
        self.method = method
    }

    /// Reads one batch, upstream's `JevK5Engine.read` over the batch's questions.
    ///
    /// - Returns: One distribution per question, in the question's option order (a noul's is
    ///   `[P(true), P(false)]`), and the prompt tokens of every pass.
    /// - Throws: The first error in question order: a ``/OpenJevCore/BackendRefusal`` for a
    ///   prompt over the limit, ``JevK5ModelError`` for a model that answered wrongly, the model's
    ///   own errors, or `CancellationError`.
    public nonisolated func readBatch(
        state: JSONValue, stateText: String, questions: [EncoderQuestion]
    ) async throws -> BatchReadResult {
        // Each question's outcome by its index: its read, or the error it ended with.
        let outcomes = await withTaskGroup(
            of: (Int, Result<QuestionRead, any Error>).self,
            returning: [Int: Result<QuestionRead, any Error>].self
        ) { group in
            for (index, question) in questions.enumerated() {
                group.addTask {
                    // The read ends before the index is paired with it. Swift 6.4 at -O returned
                    // 0 for every child's index from `do { return (index, .success(try await
                    // ...)) } catch { ... }`, so a release server lost all but one read.
                    let outcome: Result<QuestionRead, any Error>
                    do {
                        outcome = .success(try await self.read(question, state: state))
                    } catch {
                        outcome = .failure(error)
                    }
                    return (index, outcome)
                }
            }
            var outcomes: [Int: Result<QuestionRead, any Error>] = [:]
            while let next = await group.next() {
                outcomes[next.0] = next.1
            }
            return outcomes
        }
        var probabilities: [[Double]] = []
        probabilities.reserveCapacity(questions.count)
        var tokens = 0
        for (index, question) in questions.enumerated() {
            guard let outcome = outcomes[index] else {
                throw JevK5ModelError(
                    "\(modelInfo.name) has no read of question \(question.key.pythonRepr)")
            }
            let read = try outcome.get()
            probabilities.append(read.probabilities)
            tokens += read.tokens
        }
        return BatchReadResult(probabilities: probabilities, inputTokens: tokens)
    }

    /// One question's read: its distribution and the prompt tokens of its passes.
    struct QuestionRead: Sendable {
        var probabilities: [Double]
        var tokens: Int
    }

    /// One question, upstream's `read_question`: its options, then as many passes as
    /// ``JevK5Readout/spread(_:texts:method:temperature:)`` takes, and the tokens of all of them.
    nonisolated func read(_ question: EncoderQuestion, state: JSONValue) async throws
        -> QuestionRead
    {
        let texts = JevK5Option.options(for: question.question).map(\.text)
        let criterion = question.rawInstructions
        var tokens = 0
        let probabilities = try await JevK5Readout.spread(
            { texts in
                let ids = try self.promptTokens(
                    JevK5Prompt.text(state: state, criterion: criterion, options: texts))
                tokens += ids.count
                let logits = try await self.pass(ids, letters: texts.count)
                return JevK5Readout.letterProbabilities(
                    logits: logits.map(Double.init), temperature: self.temperature)
            }, texts: texts, method: method)
        return QuestionRead(probabilities: probabilities, tokens: tokens)
    }

    /// The prompt's ids, or vLLM's refusal of it: the character bound, checked before the text
    /// is tokenized, then the token bound.
    ///
    /// - Throws: ``/OpenJevCore/BackendRefusal`` with vLLM's message for a prompt over
    ///   ``limit``.
    public nonisolated func promptTokens(_ prompt: String) throws(BackendRefusal) -> [Int] {
        if let refusal = limit.characterRefusal(characters: prompt.unicodeScalars.count) {
            throw refusal
        }
        let ids = tokenizer.encode(prompt)
        if let refusal = limit.tokenRefusal(tokens: ids.count) {
            throw refusal
        }
        return ids
    }

    /// One forward pass, on the actor, so passes run one at a time. A cancelled request starts no
    /// further pass; a pass cannot be interrupted.
    func pass(_ ids: [Int], letters count: Int) throws -> [Float] {
        try Task.checkCancellation()
        let logits = try model.letterLogits(tokens: ids, letterIDs: Array(letterIDs.prefix(count)))
        passCount += 1
        guard logits.count == count else {
            throw JevK5ModelError(
                "\(modelInfo.name) returned \(logits.count) letter logits for \(count) letters")
        }
        return logits
    }
}

extension JevK5Backend: ModelReleasing {
    /// Releases the model when it adopts ``/OpenJevCore/ModelReleasing``.
    public nonisolated func close() async {
        await (model as? any ModelReleasing)?.close()
    }
}

/// `jevk5_config.json`, the calibration stored with the weights, upstream's `jevk5_temperature`.
public struct JevK5Calibration: Sendable, Hashable {
    /// The letter temperature: 1.532 for JevK5 v0.2.
    public var temperature: Double

    /// Creates a calibration.
    public init(temperature: Double) {
        self.temperature = temperature
    }

    /// Reads `temperature` from the file, as upstream's `float(json.load(f)["temperature"])`
    /// does: a JSON number. Other keys are ignored, as upstream ignores them.
    ///
    /// - Throws: ``JevK5LoadError/invalidTemperature(_:)`` when the file is not a JSON object
    ///   holding a positive finite number under `temperature`, and the file system's errors.
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let value = try? JSONParser().parse(data)
        guard case .object(let object)? = value, let number = object["temperature"]?.doubleValue
        else {
            throw JevK5LoadError.invalidTemperature(
                "\(url.path) holds no number under \"temperature\"")
        }
        guard number.isFinite, number > 0 else {
            throw JevK5LoadError.invalidTemperature(
                "\(url.path) gives the temperature \(number); it must be a positive number")
        }
        temperature = number
    }
}
