// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.resolve_template`. Apache-2.0. See THIRD_PARTY.md.

/// An answer template tokenized, with the position and the label ids of every question's slot.
public struct ResolvedTemplate: Sendable, Hashable {
    /// One question's place in the template.
    public struct Slot: Sendable, Hashable {
        /// The index in ``ResolvedTemplate/template`` the question's label occupies.
        public var position: Int
        /// The token id of each of the question's labels at ``position``, in label order.
        /// `labelIDs[0]` is `template[position]`.
        public var labelIDs: [Int]

        /// Creates a slot.
        public init(position: Int, labelIDs: [Int]) {
            self.position = position
            self.labelIDs = labelIDs
        }
    }

    /// The head followed by the tokens of the lead and the answer text at every question's first
    /// label.
    public var template: [Int]
    /// One slot per question, in question order.
    public var slots: [Slot]

    /// Creates a resolved template.
    public init(template: [Int], slots: [Slot]) {
        self.template = template
        self.slots = slots
    }
}

/// Tokenizes answer templates and finds each question's slot, as upstream's
/// `Engine.resolve_template` does, memoizing the results in a ``TemplateCache``.
///
/// The template is `head + enc(lead + answerText)`. For every question and every label but the
/// first, the same text is tokenized again with that label; the result must have the template's
/// length and differ from it at exactly one index, the same index for all of the question's
/// labels. That index is the slot, and the differing ids are the label ids.
///
/// The resolver is a value: its tokenizer, marker tokens and canvas size never change, and the
/// only mutable state, the cache, is a ``TemplateCache`` reference with its own lock. Copies
/// share the cache, so the engine actor of #17 can hand the resolver to its concurrent group reads
/// and to helpers outside the actor without forking the memo.
///
/// Each resolver creates its own cache and no other resolver can be given it. The cache key does
/// not name the tokenizer or the canvas, as upstream's does not, because upstream's `_templates`
/// belongs to one `Engine` with one tokenizer and one `Settings`. A cache shared across resolvers
/// with different canvases would return a template that never met the smaller canvas's check, and
/// one shared across tokenizers would return the wrong ids, so the sharing is by copying the
/// resolver only.
public struct TemplateResolver: Sendable {
    /// The tokenizer the templates are encoded with, upstream's `Engine.enc`.
    public let tokenizer: any DecisionTokenizer
    /// The marker sequences; ``EngineTokens/scaffold`` heads a plain read's template.
    public let tokens: EngineTokens
    /// The canvas length in tokens, upstream's `OPENJEV_CANVAS`. A template and its turn close
    /// must fit it.
    public let canvas: Int
    /// The memo of resolved templates, owned by this resolver and its copies.
    public let cache: TemplateCache

    /// Creates a resolver over `tokenizer` for a canvas of `canvas` tokens (upstream's default
    /// is 64), with a fresh cache that clears past `cacheLimit` entries (upstream's 4,096).
    public init(
        tokenizer: any DecisionTokenizer, tokens: EngineTokens, canvas: Int = 64,
        cacheLimit: Int = 4096
    ) {
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.canvas = canvas
        self.cache = TemplateCache(limit: cacheLimit)
    }

    /// The template and slots of `questions` in `format`.
    ///
    /// `head` is the token run the canvas starts with: `nil` means the scaffold, the empty thought
    /// block of a plain read; an empty array is a read after a thought the prompt already closes.
    /// `lead` is the text before the first answer when earlier answers are already in the prompt,
    /// the format's join in a sequential read.
    ///
    /// A cache hit returns the stored template. The key is the format, the head as used, the lead
    /// and each question's id and labels, so `nil` and the scaffold share an entry. Errors are not
    /// cached.
    ///
    /// - Precondition: Every question has at least two labels. ``QuestionSchemaBuilder``
    ///   guarantees this: a question with one answer is forced and never read.
    /// - Throws: A ``SchemaError`` when the template and its turn close exceed the canvas
    ///   (`"answer template is N tokens; the canvas holds C"`) or when a question's labels do not
    ///   change exactly one shared token (`"question 'key': labels do not share one template
    ///   slot"`, the key in Python's `repr`). The tokenizer's own error passes through.
    public func resolve(
        _ questions: [ReadQuestion], format: AnswerFormat, head: [Int]? = nil, lead: String = ""
    ) throws -> ResolvedTemplate {
        let head = head ?? tokens.scaffold
        let key = TemplateCache.Key(
            format: format, head: head, lead: lead,
            questions: questions.map { .init(id: $0.id, labels: $0.labels) })
        if let hit = cache.template(for: key) {
            return hit
        }
        let resolved = try compute(questions, format: format, head: head, lead: lead)
        cache.insert(resolved, for: key)
        return resolved
    }

    /// `resolve_template` without the cache.
    private func compute(
        _ questions: [ReadQuestion], format: AnswerFormat, head: [Int], lead: String
    ) throws -> ResolvedTemplate {
        let baseLabels = [Int](repeating: 0, count: questions.count)
        let base = head + (try encode(lead, questions, baseLabels, format))
        if base.count + 1 > canvas {
            throw SchemaError(
                "answer template is \(base.count) tokens; the canvas holds \(canvas - 1)")
        }
        var slots: [ResolvedTemplate.Slot] = []
        for (qi, question) in questions.enumerated() {
            precondition(
                question.labels.count >= 2,
                "question \(question.id) has \(question.labels.count) label(s); a read question "
                    + "has at least two")
            var position: Int? = nil
            var ids = [Int](repeating: 0, count: question.labels.count)
            for li in 1..<question.labels.count {
                var labels = baseLabels
                labels[qi] = li
                let alternative = head + (try encode(lead, questions, labels, format))
                // The indices where the alternative differs, over the shared prefix, as
                // upstream's `range(min(len(e), len(base)))` compares them.
                let differing = zip(alternative, base).enumerated()
                    .filter { $0.element.0 != $0.element.1 }
                    .map(\.offset)
                guard alternative.count == base.count, let index = differing.first,
                    differing.count == 1, position == nil || position == index
                else {
                    throw SchemaError(
                        "question \(question.key.pythonRepr): labels do not share one template slot"
                    )
                }
                position = index
                ids[li] = alternative[index]
            }
            guard let found = position else {
                preconditionFailure("question \(question.id) has no alternative label")
            }
            ids[0] = base[found]
            slots.append(ResolvedTemplate.Slot(position: found, labelIDs: ids))
        }
        return ResolvedTemplate(template: base, slots: slots)
    }

    /// `enc(lead + answer_text(qs, labels, fmt))`.
    private func encode(
        _ lead: String, _ questions: [ReadQuestion], _ labelIndices: [Int], _ format: AnswerFormat
    ) throws -> [Int] {
        try tokenizer.encode(
            lead + AnswerText.render(questions, labelIndices: labelIndices, format: format),
            addSpecialTokens: false)
    }
}
