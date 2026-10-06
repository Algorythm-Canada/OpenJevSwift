import Foundation
import Observation
import OpenJevCore

/// The screen's state: the message, the request built from it, and the latest answers.
///
/// The view calls ``answer()`` from a task keyed on the text, so a new keystroke cancels the
/// pending read; the engine is an actor, so the read itself runs off the main thread.
@MainActor
@Observable
public final class TriageModel {
    /// One question's answer, ready to draw.
    public struct Row: Identifiable, Equatable, Sendable {
        public var id: String
        /// The question as the user reads it.
        public var title: String
        /// `noul`, `choice` or `score`.
        public var type: String
        /// The answer in a few words: "Yes", "outage", "annoyed (1.2)".
        public var headline: String
        /// Each outcome and its probability, in the question's order.
        public var bars: [Bar]
        public var confidence: Double
    }

    public struct Bar: Identifiable, Equatable, Sendable {
        public var label: String
        public var probability: Double
        public var id: String { label }
    }

    public var text = ""
    public var engine: EncoderDecisionEngine?

    public private(set) var rows: [Row] = []
    /// The time the last read spent in the model.
    public private(set) var modelTime: Duration?
    /// Set when Verdict refused a question type and the app dropped it.
    public private(set) var note: String?
    public private(set) var error: String?

    /// The questions Verdict answers, all three until a refusal says otherwise.
    public private(set) var supportedIDs: [String] = TriageQuestions.all.keys

    public init() {}

    /// The request for a message: upstream's README example with `state` set to the message.
    public func request(for state: String) -> SystemOneRequest {
        let questions = OrderedMap<Question>(
            uniqueKeysWithValues: TriageQuestions.all.filter { supportedIDs.contains($0.key) }
                .map { ($0.key, $0.value) })
        return SystemOneRequest(
            model: TriageQuestions.model, state: .string(state), questions: questions)
    }

    /// Answers the current text after a short pause, unless the text changes first.
    public func answer(debounce: Duration = .milliseconds(250)) async {
        let state = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let engine, !state.isEmpty else {
            if state.isEmpty {
                rows = []
                modelTime = nil
            }
            return
        }
        do {
            try await Task.sleep(for: debounce)
        } catch {
            return
        }
        do {
            let decision = try await decide(state, with: engine)
            try Task.checkCancellation()
            rows = Self.rows(from: decision)
            modelTime = decision.modelTime
            error = nil
        } catch is CancellationError {
            return
        } catch {
            self.error = String(describing: error)
        }
    }

    /// Decides the supported questions. On a refusal, asks each question alone, keeps the ones
    /// Verdict answers and says which it dropped.
    private func decide(_ state: String, with engine: EncoderDecisionEngine) async throws
        -> Decision
    {
        do {
            return try await engine.decide(request(for: state))
        } catch let refusal where refusal is SchemaError || refusal is BackendRefusal {
            var kept: [String] = []
            var dropped: [String] = []
            for (id, question) in TriageQuestions.all where supportedIDs.contains(id) {
                let single = SystemOneRequest(
                    model: TriageQuestions.model, state: .string(state), questions: [id: question])
                do {
                    _ = try await engine.decide(single)
                    kept.append(id)
                } catch is SchemaError, is BackendRefusal {
                    dropped.append("\(id) (\(question.type))")
                }
            }
            guard !kept.isEmpty else { throw refusal }
            supportedIDs = kept
            note =
                "Verdict refused \(dropped.joined(separator: ", ")), so the app asks only "
                + kept.joined(separator: ", ") + "."
            return try await engine.decide(request(for: state))
        }
    }

    /// Turns a decision into rows, in the questions' order.
    static func rows(from decision: Decision) -> [Row] {
        decision.answers.compactMap { id, answer in
            guard let question = TriageQuestions.all[id] else { return nil }
            let title = question.instructions?.stringValue ?? id
            switch answer {
            case .noul(let yes):
                return Row(
                    id: id, title: title, type: "noul", headline: yes >= 0.5 ? "Yes" : "No",
                    bars: [
                        Bar(label: "yes", probability: yes), Bar(label: "no", probability: 1 - yes),
                    ],
                    confidence: Confidence.compute([yes, 1 - yes]))
            case .choice(let choice, let probabilities, let confidence):
                return Row(
                    id: id, title: title, type: "choice", headline: choice,
                    bars: probabilities.map { Bar(label: $0.key, probability: $0.value) },
                    confidence: confidence)
            case .score(let score, let legend, let probabilities, let confidence):
                let labels = legend.map { $0.stringValue ?? "" }
                let nearest = labels[min(max(Int(score.rounded()), 0), labels.count - 1)]
                return Row(
                    id: id, title: title, type: "score",
                    headline:
                        "\(nearest) (\(score.formatted(.number.precision(.fractionLength(1)))))",
                    bars: zip(labels, probabilities).map { Bar(label: $0, probability: $1) },
                    confidence: confidence)
            }
        }
    }
}
