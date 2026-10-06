import OpenJevCore

/// A backend that reads and generates text, as the DiffusionGemma runtime does through
/// ``/OpenJevCore/TextGenerator``: ``StubBackend``'s reads and ``StubTextGenerator``'s generations.
///
/// A ``/OpenJevCore/DecisionEngine`` over it serves `POST /v1/systemone` and, through
/// ``/OpenJevCore/DecisionEngine/textGenerator``, `POST /v1/chat/completions`. Both share one
/// prompt limit, upstream's `OPENJEV_MLX_MAX_PROMPT`, which is the generator's.
public final class StubGeneratingBackend: DecisionBackend, TextGenerator, ModelReleasing,
    @unchecked Sendable
{
    /// The reads.
    public let reads: StubBackend
    /// The generations.
    public let generation: StubTextGenerator

    /// Creates the backend from its two halves.
    public init(reads: StubBackend = StubBackend(), generation: StubTextGenerator) {
        self.reads = reads
        self.generation = generation
    }

    public var tokenizer: any DecisionTokenizer { reads.tokenizer }
    public var maxPromptTokens: Int { generation.maxPromptTokens }
    public var capabilities: BackendCapabilities { reads.capabilities }
    public var modelName: String { reads.modelName }
    public var thoughtChannelMarkerIDs: [Int] { generation.thoughtChannelMarkerIDs }

    public func read(_ read: CanvasRead) async throws -> ReadResult {
        try await reads.read(read)
    }

    public func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration
    {
        try await reads.think(prompt: prompt, budget: budget, stopIDs: stopIDs)
    }

    public func generationPromptIDs(messages: [JSONValue], thinking: Bool) async throws -> [Int] {
        try await generation.generationPromptIDs(messages: messages, thinking: thinking)
    }

    public func encode(_ text: String) throws -> [Int] {
        try generation.encode(text)
    }

    public func generate(
        prompt: [Int], maxTokens: Int, stopIDs: [Int], skipSpecialTokenIDs: [Int],
        emit: @Sendable (_ text: String, _ token: Int?) -> Bool
    ) async throws -> TextGeneration {
        try await generation.generate(
            prompt: prompt, maxTokens: maxTokens, stopIDs: stopIDs,
            skipSpecialTokenIDs: skipSpecialTokenIDs, emit: emit)
    }

    public func close() async {
        await reads.close()
    }
}
