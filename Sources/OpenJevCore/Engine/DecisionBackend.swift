// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the contracts
// of `Engine.one_read` and `Engine.think`, and `openjev/mlx_backend.py`, the `ImagePrompt` and the
// runtime calls of `MlxEngine.one_read` and `MlxEngine.think`. Apache-2.0. See THIRD_PARTY.md.

/// The prompt a read continues: token ids, or the pieces of an image prompt.
///
/// Upstream's engine hands a vLLM read either a `prompt` of ids (a thought or earlier answers)
/// or the system and user messages; its MLX engine prefills the chat prompt ids for a text state
/// and an `ImagePrompt(sys_text, state_text, urls)` for images, whose processor expands the images
/// into tokens itself. So an image prompt has no ids for the engine to count or cache.
public enum ReadPrompt: Sendable, Hashable {
    /// The prompt as token ids: the chat prompt of a text state, or a prefix that continues a
    /// thought or earlier answers.
    case tokens([Int])
    /// A prompt with images ahead of the state. The backend renders the chat template with
    /// `images` and then `stateText` as the user turn.
    case image(systemText: String, stateText: String, images: [ImagePart])
}

/// Everything one read needs: the prompt, the canvas and where to look on it.
///
/// This is the argument of ``DecisionBackend/read(_:)``, upstream's `one_read` after it has built
/// the canvas: the backend runs one denoise over `canvas` conditioned on `prompt` and returns the
/// log-probabilities at every slot position.
public struct CanvasRead: Sendable, Hashable {
    /// What the model conditions on.
    public var prompt: ReadPrompt
    /// The system text of the read, for backends and logs; for an image prompt it is also inside
    /// ``prompt``.
    public var systemText: String
    /// The state text of the read, the user turn.
    public var stateText: String
    /// The resolved answer template the canvas starts from.
    public var template: [Int]
    /// The slot of each question, in question order.
    public var slots: [ResolvedTemplate.Slot]
    /// The template with noise at the slots, padded to the canvas width.
    public var canvas: SeededCanvas
    /// Denoise passes; between passes the slots keep their argmax and the rest is pinned.
    public var steps: Int
    /// The seed the canvas noise was drawn from, for logs and tests.
    public var seed: UInt64

    /// Creates a read.
    public init(
        prompt: ReadPrompt, systemText: String, stateText: String, template: [Int],
        slots: [ResolvedTemplate.Slot], canvas: SeededCanvas, steps: Int, seed: UInt64
    ) {
        self.prompt = prompt
        self.systemText = systemText
        self.stateText = stateText
        self.template = template
        self.slots = slots
        self.canvas = canvas
        self.steps = steps
        self.seed = seed
    }

    /// The sorted union of the slots' label ids: the exact ids a vLLM read asks logprobs for, and
    /// the count the read limit ``DecisionEngine/maxLabelIDs`` applies to.
    public var labelIDs: [Int] {
        Array(Set(slots.lazy.flatMap(\.labelIDs))).sorted()
    }
}

/// The answer at one slot: the label distribution and the top-k entropy.
public struct SlotRead: Sendable, Hashable {
    /// One probability per label of the slot, in label order.
    public var probabilities: [Double]
    /// The entropy of the backend's top-k logprobs, which ``DecisionEngine`` compares with the
    /// re-read threshold.
    public var entropy: Double

    /// Creates a slot read.
    public init(probabilities: [Double], entropy: Double) {
        self.probabilities = probabilities
        self.entropy = entropy
    }
}

/// What a backend returns for one read.
public struct ReadResult: Sendable {
    /// One entry per slot of the read, in slot order.
    public var slots: [SlotRead]
    /// The prompt tokens the read processed, which the engine bills as input.
    public var promptTokens: Int

    /// Creates a result from ready distributions.
    public init(slots: [SlotRead], promptTokens: Int) {
        self.slots = slots
        self.promptTokens = promptTokens
    }

    /// Creates a result from raw log-probabilities, as upstream's `one_read` does with
    /// `slot_distribution`.
    ///
    /// `tops[i]` is the backend's map for slot `i`, in the backend's order (the top 20 tokens
    /// and every label, for the MLX runtime), and `labelIDs[i]` that slot's label ids. Each slot
    /// goes through ``SlotDistribution/compute(top:labelIDs:)``.
    ///
    /// - Precondition: `tops` and `labelIDs` have the same count, and no map is empty.
    public init(
        tops: [[(tokenID: Int, logprob: Double)]], labelIDs: [[Int]], promptTokens: Int
    ) {
        precondition(
            tops.count == labelIDs.count,
            "\(tops.count) logprob maps for \(labelIDs.count) slots")
        self.slots = zip(tops, labelIDs).map { top, ids in
            let distribution = SlotDistribution.compute(top: top, labelIDs: ids)
            return SlotRead(
                probabilities: distribution.probabilities, entropy: distribution.entropy)
        }
        self.promptTokens = promptTokens
    }
}

/// What a backend generated for a thought.
///
/// The backend generates after the thought-open marker and stops at the first stop id or at the
/// budget. It returns the ids as generated; ``DecisionEngine`` cuts them at the first
/// thought-close id and appends the close marker, as upstream's `think` does, so that step is
/// implemented once.
public struct ThoughtGeneration: Sendable, Hashable {
    /// The generated ids, which may end with a stop id.
    public var generated: [Int]
    /// The prompt tokens the generation processed, billed as input.
    public var promptTokens: Int

    /// Creates a generation.
    public init(generated: [Int], promptTokens: Int) {
        self.generated = generated
        self.promptTokens = promptTokens
    }
}

/// Which request options a backend can honour.
///
/// Upstream's encoder engines refuse `steps`, `samples`, `think`, `sequential` and `images` with
/// `"{model} does not support {field}"`; the diffusion engines accept them all. The engine checks
/// these before it does anything else with a request.
public struct BackendCapabilities: Sendable, Hashable {
    /// More than one denoise pass per read.
    public var steps: Bool
    /// More than one billed read per group.
    public var samples: Bool
    /// A generated thought before the read.
    public var think: Bool
    /// Groups read in series, each conditioned on the earlier answers.
    public var sequential: Bool
    /// Images ahead of the state.
    public var images: Bool

    /// Creates a capability set.
    public init(steps: Bool, samples: Bool, think: Bool, sequential: Bool, images: Bool) {
        self.steps = steps
        self.samples = samples
        self.think = think
        self.sequential = sequential
        self.images = images
    }

    /// Every option, as upstream's diffusion engines.
    public static let all = BackendCapabilities(
        steps: true, samples: true, think: true, sequential: true, images: true)

    /// Plain reads only, as upstream's encoder engines.
    public static let readsOnly = BackendCapabilities(
        steps: false, samples: false, think: false, sequential: false, images: false)
}

/// A model that reads a canvas and, optionally, writes a thought.
///
/// This is the boundary of `OpenJevCore` (decision D-005): everything up to the canvas is the
/// engine's, and everything from the prompt and canvas to the slot logprobs is the backend's,
/// upstream's `one_read` and `think` against a runtime. The MLX runtime of milestone 2 conforms;
/// the core's tests use a stub.
public protocol DecisionBackend: Sendable {
    /// The tokenizer the engine encodes markers, labels and templates with.
    var tokenizer: any DecisionTokenizer { get }
    /// The most prompt tokens one read or thought may carry, upstream's `OPENJEV_MLX_MAX_PROMPT`.
    var maxPromptTokens: Int { get }
    /// The options this backend honours.
    var capabilities: BackendCapabilities { get }
    /// The name the engine puts in a capability error, upstream's `model_name`.
    var modelName: String { get }

    /// One denoise over the canvas: the label distribution and the top-k entropy at every slot,
    /// and the prompt tokens processed.
    func read(_ read: CanvasRead) async throws -> ReadResult

    /// Generates up to `budget` tokens after `prompt`, stopping when a token of `stopIDs` is
    /// produced. Returns the tokens as generated and the prompt tokens processed.
    func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration
}
