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
                     ReadOptions, DecisionEngine (auto re-read, samples, steps, think, sequential);
                     QuestionReadBackend protocol, EncoderEngineConfiguration,
                     EncoderDecisionEngine (batched reads); SystemOneService, ServedModels
      Images/        Data-URL and {content_type, base64} validation (no decoding of pixels)
    OpenJevDiffusionGemma/             Apple silicon only. Depends on mlx-swift, MLXLMCommon,
                                       MLXVLM (Gemma 4 vision), swift-transformers Tokenizers.
      Model/         Configuration (config.json decoding, #23); Norms, Attention, DenseMLP,
                     Router, Experts, DecoderLayer, LayerCache, Softcap (text blocks, #24);
                     ModelTree (decoder, encoder scalars, root with sanitize and a one-piece
                     prefill) and WeightLoading (strict coverage, loadWeights, metrics) (#27);
                     SelfConditioning (#28); Prefill (PromptCache, prefill(promptIDs:), cache
                     digests, #25); DecoderPass (decoder masks, logits, self-conditioning
                     signal, #26 and #28); Read (SlotRequest, ReadOutput, read(), #26 and #28)
      Runtime/       DiffusionGemmaRuntime actor: prefill cache, read(), think(), generate()
      Tokenization/  Tokenizer adapter, chat prompt builder, label discovery hookup
      Vision/        Processor parity, pixel embedding, block ids (later milestone)
      Generation/    Sampler, stopping rules, block loop, streaming detokenizer (later)
    OpenJevEncoders/                   Apple platforms; its Core ML types need macOS 15 and iOS 18.
                                       Depends on Core ML and swift-transformers Tokenizers; no MLX.
      Verdict/       VerdictPrompt, VerdictTokenizer, VerdictCalibration, VerdictBackend actor
      Laya/          LayaPrompt (render_options, serialize_state), LayaSequence (build_sequence),
                     LayaTokenizer, LayaCalibration (buckets, clamp, float32 softmax, rounding),
                     LayaBackend actor, LayaPackageSet
      CoreML/        EncoderPackageSpec (one function per shape, or one program for one shape),
                     CoreMLEncoderModel (every function loaded on macOS, one on iOS),
                     CoreMLPackagesByLength (Laya's per-length packages, each kept loaded),
                     CompiledEncoderModel, EncoderComputeUnits
      Store/         EncoderPackageManifest (Verdict's, Laya's five), EncoderPackageStore
                     (download on first use, SHA-256, the tokenizer alone, held packages)
    OpenJevServer/                     Hummingbird 2. ServerSettings, BackendProvider,
                                       OpenJevApplication (routes), SystemOneHandler, request id,
                                       server-timing, request log, authentication and body cap
                                       middleware, the body reader, the refusal log, the client
                                       disconnect watch, DecisionServer (graceful shutdown) and
                                       ModelRouter (the model routes).
    OpenJevTestSupport/                Fixture loaders, FixtureTokenizer, the stub backends and
                                       ReadGate the test targets share. Foundation only; not a
                                       product.
    openjev/                           CLI executable: serve, decide, models; the backend registry
    openjev-stub-server/               The server over OpenJevTestSupport's stubs, for the SDK
                                       suite; an executable target, not a product
  Tests/
    OpenJevCoreTests/                  Fixture-driven unit tests (no model)
    OpenJevDiffusionGemmaTests/        Unit tests on synthetic shapes; opt-in live tests
    OpenJevEncodersTests/              Fixture-driven parity tests over recorded logits; opt-in
                                       tokenizer and Core ML parity tests
    OpenJevServerTests/                Contract tests with a stub backend, capacity and model time
                                       among them; disconnects and shutdown on live sockets; the
                                       model routes against an in-process routed server
    OpenJevCLITests/                   Parsing, the commands in-process with stub backends, the
                                       built binary as a child process; opt-in Verdict smoke test
  Tools/
    fixtures/                          Python: generate golden fixtures from pinned upstream
    encoders/                          Python and Swift: Verdict's and Laya's reference outputs,
                                       the Core ML converters and the package manifest
    sdk-compat/                        Python: TypeSafe's Python and TypeScript SDKs, and
                                       JevSwiftSDK, against openjev-stub-server
  Fixtures/                            Checked-in JSON fixtures (small)
```

`OpenJevEncoders` holds the encoder backends on Core ML (D-011): Verdict (#57) and Laya (#58).
A later milestone adds `OpenJevLetterReadout` (JevK5 style on `MLXLLM` models). Both are
separate targets so that iOS consumers never link the 26B model code. Both implement
`QuestionReadBackend` and run behind `EncoderDecisionEngine`, so the server holds either kind of
engine as a `SystemOneService`.

## Module dependency graph

```
openjev (CLI) ──► OpenJevServer ──► OpenJevCore
      │                                  ▲
      ├──► OpenJevEncoders (macOS) ──────┤──► Core ML, swift-transformers (Tokenizers)
      └──► OpenJevDiffusionGemma (#29) ──┘──► mlx-swift, mlx-swift-lm (MLXLMCommon, MLXVLM),
                                              swift-transformers (Tokenizers)
```

`OpenJevCore` has no third-party dependencies (an `OrderedDictionary` from `swift-collections`
is acceptable if it saves a hand-rolled type). `OpenJevServer` depends on Hummingbird,
swift-http-types, swift-log, swift-nio's `NIOCore`, `NIOPosix` and `NIOHTTP1`,
swift-service-lifecycle and AsyncHTTPClient, never on a backend. The CLI picks the backend and
links it: `OpenJevEncoders` and `OpenJevDiffusionGemma` on macOS. Backends depend on the core,
never the reverse.

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

The encoder-style boundary (`Sources/OpenJevCore/Engine/`, issue #67), the sibling protocol of
decision D-005 that Verdict, Laya, JevK5 and CLM implement, upstream's `EncoderEngine` contract:

```swift
public protocol QuestionReadBackend: Sendable {
    var modelInfo: ModelInfo { get }       // served name, description and release date, upstream's ENCODER_MODELS
    var maxChoices: Int { get }            // 24 for Verdict, 255 otherwise
    var maxPromptTokens: Int? { get }      // nil when the backend truncates instead of refusing
    func readBatch(state: JSONValue, stateText: String, questions: [EncoderQuestion]) async throws -> BatchReadResult
}

public actor EncoderDecisionEngine {
    public init(backend: any QuestionReadBackend, configuration: EncoderEngineConfiguration = .default)
    public func decide(_ request: SystemOneRequest) async throws -> Decision
    public func warmUp() async throws        // upstream's WARMUP_QUESTIONS, called by the CLI and server after load
}
```

`EncoderDecisionEngine` owns what upstream's `EncoderEngine` owns apart from the model: the
`EncoderQuestionSchemaBuilder` over the backend's `maxChoices` (the same forced answers and
limits as the diffusion engine), the refusal of `images`, `steps > 1`, `samples > 1`, `think` and
`sequential` before any read (`"{model} does not support {field}"`), the queue bound
(`"{model} is at capacity. Retry shortly."`), reads in batches of `batchSize` (16) in request
order under a `maxInflight` semaphore (1, upstream's one model thread), the billing and
`Answer.make`. A backend returns one distribution per question in the caller's option order
(noul is `[P(true), 1 - P(true)]`); the engine checks the count, the `[0, 1]` range and the sum
of every distribution and throws `BackendContractError` otherwise, a backend bug rather than a
client error. Reads are deterministic, so the request seed is not used and `outputTokens` is 0.
`EncoderEngineConfiguration` carries `batchSize` (`OPENJEV_ENCODER_BATCH`), `maxQueue`
(`OPENJEV_MAX_QUEUE`), `maxInflight` and `warmUp` (`OPENJEV_WARMUP`).

The two engines share the option refusal, the queue counter and the answer reordering
(`RequestAdmission.swift`), and both conform to the protocol the server holds:

```swift
public protocol SystemOneService: Sendable {
    var servedModels: ServedModels { get }   // upstream's config.served_models(backend)
    func decide(_ request: SystemOneRequest) async throws -> Decision
}

public struct ServedModels { version: String; acceptedNames: Set<String>; listing: [ModelInfo] }
// .diffusionGemma: "openjev-0.1", {openjev-latest, openjev-0.1, jev-latest, jev-preview}, upstream's MODELS
// .encoder(modelInfo): modelInfo.name, {modelInfo.name, jev-latest, jev-preview}, [modelInfo]
```

`KnownEncoderModels` holds upstream's `ENCODER_MODELS` metadata (`laya-1.0`, `verdict-1.4`,
`clm-v0.1`, `jevk5-0.2`) for the backends and for the routes listing; `Fixtures/wire/models.json`
is the oracle for the listings and the accepted names.

The in-process, public entry point for apps:

```swift
let runtime = try await DiffusionGemmaRuntime.load(.fourBit)   // downloads or opens the cache
let engine = try DecisionEngine(backend: runtime, configuration: .default)
let request = try SystemOneRequest(json: JSONParser().parse("""
    {"model": "jev-latest",
     "state": "Everything is down and we have a demo with our biggest client at noon.",
     "questions": {
       "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
       "team": {"type": "choice", "instructions": "Which team should handle it?",
                "criteria": {"outage": "service down", "billing": "charges, refunds",
                             "feature": "requests, how-to"}},
       "tone": {"type": "score", "instructions": "How upset is the customer?",
                "criteria": ["calm", "annoyed", "furious"]}}}
    """))
let decision = try await engine.decide(request)
decision.answers["team"]   // .choice(choice: "outage", ...), as measured on the 4-bit checkpoint
```

`DiffusionGemmaRuntime.load(.directory(url))` opens a local checkpoint without network access.

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

`public actor DiffusionGemmaRuntime: DecisionBackend` (`Sources/OpenJevDiffusionGemma/Runtime/`,
issue #29), upstream's `MlxRuntime` and `MlxEngine.one_read` in one actor:

- Owns the model, the tokenizer and the prefill cache. Every MLX evaluation runs inside the
  actor, one at a time, which gives the single-thread discipline upstream enforces with a
  one-worker executor (R14). Only values cross it: `CanvasRead` in, `ReadResult` out.
- Prefill cache: `PrefillCache<Value>`, generic so its eviction rule is tested without MLX, which
  the runtime instantiates with `PromptCache`. Ordered and keyed by the prompt token ids (an image
  key arrives with the vision milestone), bounded by entries (`promptCacheEntries`, default 12)
  and tokens (`promptCacheTokens`, 16,384) with a running token total, no exempt entry (insert,
  then evict oldest first while either budget is exceeded; the caller keeps what it was handed),
  hits moved to the end, zero entries meaning no caching.
- `read`: a token prompt longer than `maxPromptTokens` (32,768) is refused with upstream's
  `SchemaError("the request is {n} tokens; the limit is {max}")` before anything runs, and an image
  prompt with `DiffusionGemmaRuntimeError.unsupported("images")`; then the prefill or a cached one,
  `model.read` over the canvas with the slots as `SlotRequest`s, `steps` passes and the top 20,
  and `ReadOutput.readResult(for:)` with the prompt cache's token count.
- Capabilities: steps, samples and sequential; not `think` (milestone 5) nor images (vision
  milestone), so the engine answers `"openjev-0.1 does not support think"` and its image refusal.
  `think(prompt:budget:stopIDs:)` throws `unsupported("think")`. `modelName` is `openjev-0.1`.
- Memory controls: `Configuration.cacheLimitGB` (nil leaves MLX alone, 0 disables MLX's buffer
  pool, otherwise `Memory.cacheLimit` in bytes, applied inside the actor at load, or later with
  `setCacheLimit(gb:)`), the prefill cache budgets and the prompt cap. `memoryReport()` gives
  MLX's active, cache and peak bytes and the process's resident bytes; `statistics()` the reads,
  the prefill hits and misses and the model time.
- Warm-up: `warmUp()` runs one small read directly on the model (one noul question over upstream's
  warm-up state, its prompt from the chat template, its canvas from `CanvasBuilder` with seed 0),
  so the first user does not pay kernel compilation; loading runs it when
  `Configuration.warmUp` is on.
- Loading: `DiffusionGemmaRuntime.load(_:configuration:cache:token:resolver:progress:)` resolves a
  `ModelSource`, loads the tokenizer (`TokenizerFiles`) and the weights
  (`DiffusionGemmaModel.load`), applies the cache limit and warms up, reporting one `LoadStage`
  progression and keeping a `LoadReport` (resolve time, downloaded bytes, tokenizer and weight
  metrics, warm-up time, memory).

Model resolution and download (`Sources/OpenJevDiffusionGemma/Download/`, issue #30):
`ModelSource` is `.directory(URL)` or `.hub(repository:revision:)`, with the presets `.fourBit`
(the pinned `a7a81407`), `.eightBit` and `.bf16` (pinned in THIRD_PARTY.md), and
`ModelSource(setting:)` reads `OPENJEV_MLX_MODEL`'s value. `ModelResolver` uses a directory as is
once `config.json` and the shard index (and the shards it names) are there; a Hub source is
resolved to a commit (a 40-hex revision as is, a branch or tag through the Hub API, writing
`refs/<name>`), and every file of the tree is looked for in
`<hub>/models--{org}--{repo}/snapshots/{commit}/` and downloaded when missing: into
`blobs/<id>.incomplete` with HTTP Range resume, checked (SHA-256 for LFS files, size and git blob
SHA-1 for the others), renamed to `blobs/<id>` and linked from the snapshot with a relative
symlink. That is huggingface_hub's layout, so upstream, mlx-vlm and this library share one copy.
`HubCacheLocation(environment:)` follows `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` and
`~/.cache/huggingface/hub` in the environment the caller passes, and
`HubCacheLocation.token(environment:)` reads `HF_TOKEN`, an empty value counting as absent; the
library never reads the process environment (D-013).

## The server

Hummingbird 2 application with the routes in [02-jev-wire-api.md](02-jev-wire-api.md). A
`BackendProvider` loads the `SystemOneService` once, before `OPENJEV_HOST` and `OPENJEV_PORT` are
bound, as upstream's `lifespan` loads its engine. `DecisionBackendProvider` wraps a
`DecisionBackend` in a `DecisionEngine`, and `QuestionReadBackendProvider` wraps a
`QuestionReadBackend` in an `EncoderDecisionEngine`, each configured from the settings, and runs
the warm-up read when `OPENJEV_WARMUP` asks. Tests hand the router a stub-backed service; the CLI
hands it the DiffusionGemma runtime or an encoder. `GET /v1/models` lists the service's
`ServedModels`.

`DecisionServer` is the server over a loaded service, a swift-service-lifecycle `Service`:
`OpenJevApplication.application(settings:service:logger:onServerRunning:)` binds the settings'
address, and on a graceful shutdown the listening socket closes, idle connections close, the
requests in flight finish, and the service's `close()` releases the model (`ModelReleasing`,
upstream's `await app.state.engine.close()`). The service group that runs it bounds the shutdown
with its `maximumGracefulShutdownDuration`; past it the requests in flight are cancelled, the
model is still released, and `run()` throws `ShutdownInterrupted`. The engines pass `close()` on
to their backend, and `VerdictBackend` releases its loaded Core ML functions.

`RequestLogMiddleware` runs first, outside everything else, and writes one info line per request:
`{method} {path} {status} {ms}ms {request id}`, never a body, a header value or a query string.
`ResponseHeadersMiddleware` runs in front of every route. It turns a thrown `WireError` into its
response, answers an unknown route as FastAPI's 404, and adds `x-typesafe-request-id`,
`x-request-id` and `server-timing` to every response, errors included. Inside it, as upstream's
`request_id_and_auth` does for `/v1/` paths, `AuthenticationMiddleware` checks the origin secret
and the API key (`check_auth`) and `BodyCapMiddleware` reads a `POST` body up to the cap, the 413
before a byte is read when `Content-Length` declares more (`read_capped_body`). `/health` is
never authenticated.

`POST /v1/systemone` reads the body as FastAPI does (`RequestBodyReader`): only a JSON content
type is parsed, by the core's order-preserving parser (Foundation's `JSONDecoder` cannot preserve
object order), and a body it refuses gets CPython's `json_invalid` message and position from
`PythonJSONLoads`. `SystemOneHandler` does the rest, and `openjev decide` calls it too, so the
command prints the bytes the server sends: the body is checked by `RequestValidator`, then the
model name and the questions cap are checked before `SystemOneService.decide`. `SchemaError`,
`OverloadedError` and `BackendRefusal` become upstream's 400, 529 and 400 `the model rejected this
request`; any other error of the service is the 503 naming its type. `RefusalLog` logs the
refusals upstream logs, where and why, never the body.

Capacity is the engines' own: the queue counter (`OPENJEV_MAX_QUEUE`, upstream's
`waiting >= max_queue`, so 0 refuses every request) and the in-flight semaphore
(`OPENJEV_MAX_INFLIGHT`; one model call at a time for an encoder), whose cancelled waiters leave
without taking or returning a permit. Model time is upstream's `model_ns`: the headers middleware
installs a task-local `ModelTimeRecorder` (OpenJevCore) per request, the engines add each backend
call to it when the call ends, wait for a permit included, whether it returned, threw or was
cancelled, and the middleware reports the sum as `server-timing`'s `model`. Concurrent reads sum,
so `model` can exceed `total`, and a request refused after a read still reports the read. A
request forwarded to another server reports the whole exchange, network included.

Every connection carries a `ClientDisconnectHandler`, which sees the end of the client's input.
The route runs the decision in a child task beside a watch of its connection: a client that goes
away cancels the decision, which reaches the reads through task cancellation, and the request log
shows 499. Request handling creates no unstructured or detached task. Decisions D-030, D-031 and
D-038 record where the server differs from upstream. Text generation routes are added only when a
generation-capable backend is loaded.

`ModelRouter` is `OPENJEV_MODEL_ROUTES`, upstream's `forward`. The route asks it once the body has
passed validation, before the model name is checked: a model with a route that the service does
not accept is sent to `{url}/v1/systemone` as the bytes the client sent, with the client's
`authorization`, `x-origin-secret` and `content-type`, through an AsyncHTTPClient made for that
request on swift-nio's shared event loops and shut down before the route returns. The routed
status and body come back with only `content-type` and `retry-after`; a transport failure is the
503 naming httpx's error for it (`ForwardingFailure`), logged with the model's name. The exchange
is model time, and a client that goes away cancels it, as it cancels a decision. `GET /v1/models`
lists the routed names after the service's own (`ServedModels.listing(routedNames:)`), without
asking the routed servers. D-040 records the choices.

`openjev-stub-server`, an executable target that is not a product, runs a `DecisionServer` over
OpenJevTestSupport's stubs with the `OPENJEV_*` settings and prints the port it bound. The SDK
compatibility suite in `Tools/sdk-compat` starts it, on Linux in CI, where Core ML does not exist.

Configuration: a `ServerSettings` struct with the same names, defaults and startup validation as
upstream's `Settings`, populated from `OPENJEV_*` variables by the CLI so existing deployment
docs and compose files keep working.

## The command line tool

`openjev` (`Sources/openjev`, swift-argument-parser) has three subcommands. Each reads
`ServerSettings(environment:)` from the process environment, with its flags written over their
variables first, so a flag's value is checked and refused as its variable's is.

- `openjev serve` loads the backend `OPENJEV_BACKEND` names, logs its phases (the settings
  without secrets, `loading`, `warming up`, `serving on host:port`) to standard error, and runs a
  `DecisionServer` in a service group that starts the graceful shutdown on SIGINT and SIGTERM,
  bounded by `--shutdown-timeout` (30 seconds).
- `openjev decide` answers one request from a file or standard input through
  `SystemOneHandler`, without the warm-up read, and prints the server's bytes; a refusal prints
  the error body to standard error and exits 4.
- `openjev models` prints the backend's `GET /v1/models` body without loading a model.

The exit statuses are 0, 1 for any other failure, 2 for invalid settings or command line, 3 for
a backend this build lacks or that failed to load, and 4 for a request `decide` was refused
([deployment.md](deployment.md)). The commands read a task-local `CommandContext` (environment,
streams, backends, loggers, shutdown signals), which tests replace to run them in-process with
stub backends.

`BackendRegistry` lists the backends: any name it does not list is upstream's invalid-setting
error, and each backend is one line. For `OPENJEV_BACKEND=mlx`, with `environment` the process
environment the CLI also hands to `ServerSettings(environment:)`:

```swift
DecisionBackendProvider { settings in
    try await DiffusionGemmaRuntime.load(
        ModelSource(setting: settings.mlxModel),
        configuration: .init(
            maxPromptTokens: settings.mlxMaxPrompt, promptCacheEntries: settings.mlxPromptCache,
            cacheLimitGB: settings.mlxCacheLimitGB, warmUp: settings.warmup),
        cache: HubCacheLocation(environment: environment),
        token: HubCacheLocation.token(environment: environment))
}
```

The default `OPENJEV_MLX_MODEL`, `mlx-community/diffusiongemma-26B-A4B-it-4bit`, maps to
`ModelSource.fourBit` at its pinned revision; a path opens a local directory; `repo@revision`
picks a revision. The runtime warms itself up when `OPENJEV_WARMUP` asks, and its `.warmingUp`
stage prints the `warming up` phase. On Linux, where `OpenJevDiffusionGemma` does not exist, `mlx`
is a known backend that exits 3. For `OPENJEV_BACKEND=verdict` and `OPENJEV_BACKEND=laya`:

```swift
QuestionReadBackendProvider { settings in
    try await VerdictBackend.load(
        from: EncoderPackageStore(environment: environment),
        functionCapacity: settings.encoderFunctions)
}
QuestionReadBackendProvider { settings in
    try await LayaBackend.load(
        from: EncoderPackageStore(environment: environment),
        functionCapacity: settings.encoderFunctions)
}
```

On Linux, where `OpenJevEncoders` does not exist, `verdict` and `laya` are known backends that
exit 3.

`EncoderPackageStore(environment:)` uses the folder `OPENJEV_ENCODER_MODELS` names when it is set,
as the converters in `Tools/encoders` write it. Otherwise it downloads the model's package,
tokenizer and calibration file to Application Support on first use and checks every file's SHA-256
against the manifest the library embeds (D-033); a manifest whose downloads are off, as a new
package's is before its release exists, is refused with
`EncoderPackageError.packageDownloadsUnavailable`. On a Mac, Laya's loader gets its multifunction
package; on an iPhone it gets only the tokenizer and rl_agent_config.json, reads each question
through the smallest per-length package the device holds, and leaves fetching them to the app
(`LayaBackend.prefetch(lengths:)`, D-037). `load(configuration:)` takes the locations directly. The
loaders build for the package's macOS 14 floor and throw on an OS older than macOS 15 or iOS 18,
which the Core ML packages need (D-034). The provider builds the
`EncoderDecisionEngine` from `OPENJEV_ENCODER_BATCH`, `OPENJEV_MAX_QUEUE` and `OPENJEV_WARMUP`,
and the warm-up read loads the first Core ML function. On a Mac each function the reads need then
stays loaded, up to the package's six or eight, unless `OPENJEV_ENCODER_FUNCTIONS` sets fewer
(`ServerSettings.encoderFunctions`, this port's setting, D-042).

## Platform support matrix

| Target | `OpenJevCore` | `OpenJevDiffusionGemma` | `OpenJevServer` | `OpenJevEncoders` |
|---|---|---|---|---|
| macOS 14+ Apple silicon | yes | yes (32 GB+ recommended) | yes | yes, from macOS 15 (Core ML's multifunction packages) |
| iOS 17+ | yes | no (memory) | no | yes, from iOS 18 |
| Linux | yes (tests, tooling) | no | builds with a stub backend for contract tests and the SDK suite's stub server | no (Core ML) |

## Deliberately not in scope for 0.1

- A client SDK for hosted Jev (see `NSStudent/JevSwiftSDK`).
- Training, fine-tuning or calibration fitting.
- Multi-GPU or distributed serving.
- Re-quantizing checkpoints; the existing `mlx-community` conversions are used as they are.
