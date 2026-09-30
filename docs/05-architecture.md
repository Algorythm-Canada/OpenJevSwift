# Proposed architecture

## Goals

1. Answer the same request the same way upstream does: same prompt bytes, same template slots,
   same canvas, same distributions within numeric tolerance, same wire shapes and errors.
2. Be a good Swift library: small dependency surface for the core, `Sendable` value types,
   structured concurrency, no global state, no environment variables read by the library itself
   (the CLI and server map `OPENJEV_*` variables onto typed configuration).
3. Keep the model-specific code behind one protocol so that the encoder models and the
   letter-readout models can share the engine and the server, as upstream does.
4. Make correctness provable without weights: everything up to the decoder pass is pure Swift
   and testable against golden fixtures from upstream.

## Package layout

```
OpenJevSwift/                          Swift package, tools 6.2, strict concurrency
  Sources/
    OpenJevCore/                       Foundation only. Builds on macOS, iOS, Linux.
      Wire/          SystemOneRequest, Question, Answer, Usage, ModelInfo, error bodies
      JSON/          Order-preserving JSONValue, parser, Python-compatible serializer
      Schema/        QuestionSchema (build_schema), forced answers, limits
      Prompt/        SystemText, AnswerTemplate (FORMATS), LabelDiscovery
      Canvas/        TemplateResolver (slots), Grouping, CanvasBuilder, CanvasWidth
      Random/        MT19937, PythonRandom.randrange, SeedDerivation (SHA-256)
      Read/          SlotDistribution, Confidence, AnswerAssembly, ReadAveraging
      Engine/        DecisionBackend protocol, CanvasRead, ReadResult, EngineConfiguration,
                     ReadOptions, DecisionEngine (auto re-read, samples, steps, think, sequential)
      Images/        Data-URL and {content_type, base64} validation (no decoding of pixels)
    OpenJevDiffusionGemma/             Apple silicon only. Depends on mlx-swift, MLXLMCommon,
                                       MLXVLM (Gemma 4 vision), swift-transformers Tokenizers.
      Model/         Configuration, TextBlock, Attention, Router, Experts, SelfConditioning,
                     Encoder, Decoder, Softcap, WeightLoading (sanitize, quantization map)
      Runtime/       DiffusionGemmaRuntime actor: prefill cache, read(), think(), generate()
      Tokenization/  Tokenizer adapter, chat prompt builder, label discovery hookup
      Vision/        Processor parity, pixel embedding, block ids (later milestone)
      Generation/    Sampler, stopping rules, block loop, streaming detokenizer (later)
    OpenJevServer/                     Hummingbird 2. Routes, validation, error contract,
                                       auth, capacity, routes forwarding, settings.
    openjev/                           CLI executable: serve, decide, models
  Tests/
    OpenJevCoreTests/                  Fixture-driven unit tests (no model)
    OpenJevDiffusionGemmaTests/        Unit tests on synthetic shapes; opt-in live tests
    OpenJevServerTests/                Contract tests with a stub backend; SDK compatibility
  Tools/
    fixtures/                          Python: generate golden fixtures from pinned upstream
    sdk-compat/                        Python and TypeScript SDK smoke tests against a server
  Fixtures/                            Checked-in JSON fixtures (small)
```

Later milestones add `OpenJevEncoders` (Verdict, Laya) and `OpenJevLetterReadout` (JevK5 style
on `MLXLLM` models) as separate targets so that iOS consumers never link the 26B model code.

## Module dependency graph

```
openjev (CLI) ──► OpenJevServer ──► OpenJevCore
                        │                ▲
                        └──► OpenJevDiffusionGemma ──► mlx-swift, mlx-swift-lm (MLXLMCommon, MLXVLM),
                                                        swift-transformers (Tokenizers)
```

`OpenJevCore` has no third-party dependencies (an `OrderedDictionary` from `swift-collections`
is acceptable if it saves a hand-rolled type). `OpenJevServer` depends on Hummingbird. Backends
depend on the core, never the reverse.

## Core types (sketch)

```swift
public struct SystemOneRequest: Sendable, Codable {
    public var model: String
    public var state: JSONValue              // string, object or array; object order preserved
    public var questions: OrderedQuestions   // ordered by request
    public var images: [ImageInput]?
    public var steps: Int?, samples: Int?, think: Int?, sequential: Bool?
}

public enum Question: Sendable {
    case noul(instructions: JSONValue?, criteria: NoulCriteria?)
    case choice(instructions: JSONValue?, criteria: OrderedDictionary<String, JSONValue?>)
    case score(instructions: JSONValue?, criteria: [JSONValue])
}

public enum Answer: Sendable, Codable {
    case noul(Double)
    case choice(choice: String, probabilities: OrderedDictionary<String, Double>, confidence: Double)
    case score(score: Double, legend: [JSONValue], probabilities: [Double], confidence: Double)
}

public struct SystemOneResponse: Sendable, Codable {
    public var model: String
    public var answers: OrderedDictionary<String, Answer>
    public var usage: Usage
}
```

The engine and backend boundary (`Sources/OpenJevCore/Engine/`, issue #17):

```swift
public protocol DecisionBackend: Sendable {
    var tokenizer: any DecisionTokenizer { get }   // encode(addSpecialTokens: false), chatPromptIDs(...)
    var maxPromptTokens: Int { get }               // OPENJEV_MLX_MAX_PROMPT
    var capabilities: BackendCapabilities { get }  // steps, samples, think, sequential, images
    var modelName: String { get }                  // for "{model} does not support {field}"
    func read(_ read: CanvasRead) async throws -> ReadResult          // one prompt, one canvas, N steps
    func think(prompt: [Int], budget: Int, stopIDs: [Int]) async throws -> ThoughtGeneration
}

public actor DecisionEngine {
    public init(backend: any DecisionBackend, configuration: EngineConfiguration = .default) throws
    public func decide(_ request: SystemOneRequest, seed: UInt64? = nil) async throws -> Decision
}

public struct Decision { answers: OrderedMap<Answer>; inputTokens: Int; outputTokens: Int; modelTime: Duration }
```

`CanvasRead` carries the prompt (`ReadPrompt.tokens(ids)` for a text state, a thought or earlier
answers, or `ReadPrompt.image(systemText:stateText:images:)`, which the backend's processor
expands), the system and state texts, the template, the slot positions and label ids, the seeded
canvas, the step count and the seed. `ReadResult` is one `SlotRead` per slot (the label
probabilities and the top-k entropy) plus the prompt tokens processed; its
`init(tops:labelIDs:promptTokens:)` takes each slot's raw map from token id to log-probability
(the top 20 tokens and every label, upstream's `MlxRuntime.read` contract) and runs
`SlotDistribution.compute`, so a real backend returns raw logprobs and the Python fixtures apply.
`ThoughtGeneration` is the ids a backend generated after the thought-open marker; the engine cuts
them at the first thought-close id and appends the close marker. `EngineConfiguration` holds
upstream's settings (canvas geometry, `autoThreshold`, `autoMax`, `maxInflight`, `maxQueue`, the
image limits, the template cache limit) and `ReadOptions` the request's `steps`, `samples`,
`think` and `sequential` with upstream's defaults. `OverloadedError` is upstream's `Overloaded`.

The in-process, public entry point for apps:

```swift
let model = try await OpenJev.DiffusionGemma.load(.fourBit)   // downloads or opens a local dir
let engine = try DecisionEngine(backend: model, configuration: .default)
let decision = try await engine.decide(SystemOneRequest(
    model: "jev-latest",
    state: "Everything is down and we have a demo at noon.",
    questions: [
        "urgent": .noul(instructions: "Does the customer need a reply within the hour?"),
        "team":   .choice(instructions: "Which team should handle it?",
                          criteria: ["outage": "service down", "billing": "charges, refunds"]),
        "tone":   .score(instructions: "How upset is the customer?", criteria: ["calm", "annoyed", "furious"]),
    ]))
decision.answers["team"]?.choice   // "outage"
```

## Engine flow (per request)

1. Validate and normalise the request (core wire types; errors carry `loc` paths).
2. Derive the seed: SHA-256 of the Python-canonical JSON of `[state, questions, images?]`, first
   four bytes big-endian.
3. Build the schema: forced answers, `q1..qN`, labels, format.
4. Split into groups that fit the canvas; render each group's system text.
5. For each group (in parallel unless `sequential`): resolve the template and slots (cached),
   build the canvas from the group seed, perform the reads according to the policy (auto
   re-read, `samples`, `steps`, `think`), average the label distributions.
6. Assemble answers in request order; compute billed tokens and thought tokens.

The engine never touches MLX. Parallel groups become concurrent `read` calls; the runtime actor
serialises them on the GPU, so concurrency buys prefill-cache reuse and overlap of CPU work, not
GPU parallelism (same as upstream's MLX backend).

## The DiffusionGemma runtime

An `actor DiffusionGemmaRuntime: DecisionBackend`:

- Owns the model, tokenizer and caches. All MLX evaluation happens inside the actor, which gives
  the single-thread discipline upstream enforces with a one-worker executor.
- Prefill cache: LRU keyed by prompt token ids (or by system text, state text and image digests),
  bounded by entries (default 12) and tokens (16,384), no exempt entry, hits move to the end.
- `read`: prefill or hit → decoder masks → one or more decoder steps → float32 log-softmax of the
  slot rows → top-20 plus labels. Between steps: argmax write-back at slot positions only,
  self-conditioning from the previous logits.
- `think` and `generate`: the diffusion generation loop (later milestone) with stop ids and
  special-token skipping.
- Memory controls: MLX GPU cache limit (`MLX.GPU.set(cacheLimit:)`), prefill cache budgets,
  prompt length cap; memory statistics exposed for diagnostics.
- Warmup: one small read after load so the first user does not pay kernel compilation.

## The server

Hummingbird 2 application with the routes in [02-jev-wire-api.md](02-jev-wire-api.md). Body
handling reads up to the cap and rejects the rest; JSON is parsed by the core's order-preserving
parser (Foundation's `JSONDecoder` cannot preserve object order). Middleware adds request ids,
authentication and the server-timing header; per-request model time is accumulated through a
task-local. Capacity is a counter plus a semaphore mirroring `max_inflight` and `max_queue`.
`OPENJEV_MODEL_ROUTES` forwarding uses `URLSession` or Hummingbird's client. Text generation
routes are added only when a generation-capable backend is loaded.

Configuration: a `ServerSettings` struct with the same names, defaults and startup validation as
upstream's `Settings`, populated from `OPENJEV_*` variables by the CLI so existing deployment
docs and compose files keep working.

## Platform support matrix

| Target | `OpenJevCore` | `OpenJevDiffusionGemma` | `OpenJevServer` | Small encoder models (later) |
|---|---|---|---|---|
| macOS 14+ Apple silicon | yes | yes (32 GB+ recommended) | yes | yes |
| iOS 17+ | yes | no (memory) | no | yes |
| Linux | yes (tests, tooling) | no | builds with a stub backend for contract tests | no |

## Deliberately not in scope for 0.1

- A client SDK for hosted Jev (see `NSStudent/JevSwiftSDK`).
- Training, fine-tuning or calibration fitting.
- Multi-GPU or distributed serving.
- Re-quantizing checkpoints; the existing `mlx-community` conversions are used as they are.
