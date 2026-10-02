# Making decisions in an app

Load a model backend, wrap it in its engine and answer Jev requests inside your own process.

## Overview

An app needs no server to make decisions. It links the library, loads a backend once, keeps the
engine for the life of the process and calls it with a ``SystemOneRequest``. The engine applies
upstream's rules to the questions, reads the model and returns a ``Decision``: one ``Answer`` per
question, in the request's order, and the tokens the request cost.

Three backends exist today. Each module's own documentation covers its loading options, its
downloads and its memory.

| Backend | Served as | Module | Engine | Runs on |
|---|---|---|---|---|
| `VerdictBackend` | `verdict-1.4` | OpenJevEncoders | ``EncoderDecisionEngine`` | macOS 15 and iOS 18 or later, on Core ML |
| `LayaBackend` | `laya-1.0` | OpenJevEncoders | ``EncoderDecisionEngine`` | macOS 15 and iOS 18 or later, on Core ML |
| `DiffusionGemmaRuntime` | `openjev-0.1` | OpenJevDiffusionGemma | ``DecisionEngine`` | Apple silicon Macs; the 4-bit weights take about 16 GB |

### Add the package

OpenJevSwift has no release yet, so depend on a branch or a revision:

```swift
dependencies: [
    .package(
        url: "https://github.com/Algorythm-Canada/OpenJevSwift.git", branch: "main"),
],
targets: [
    .target(
        name: "MyApp",
        dependencies: [
            .product(name: "OpenJevCore", package: "OpenJevSwift"),
            .product(name: "OpenJevEncoders", package: "OpenJevSwift"),
        ]),
]
```

A version requirement cannot work yet: the package pins mlx-swift-lm by revision, and SwiftPM
refuses a revision-pinned dependency inside a package that another package requires by version.
Release 0.1.0 (issue #65) waits for an mlx-swift-lm release that holds the pinned commit. Add
`OpenJevDiffusionGemma` instead of, or beside, `OpenJevEncoders` for DiffusionGemma.

### Load a backend and its engine

Verdict, on an iPhone or a Mac:

```swift
import OpenJevCore
import OpenJevEncoders

let store = try EncoderPackageStore(environment: [:])
let verdict = try await VerdictBackend.load(from: store)
let engine = EncoderDecisionEngine(backend: verdict)
try await engine.warmUp()
```

The first load downloads Verdict's converted Core ML package (306 MB) from the project's model
releases, and its tokenizer and calibrator from Hugging Face, into Application Support. It checks
every file's size and SHA-256 against the manifest the library embeds and compiles the package
once; later loads find the files and the compiled model in place. ``EncoderDecisionEngine/warmUp()``
runs upstream's warm-up read, three short questions that load a first Core ML function before any
request arrives. The store takes the environment as a dictionary because the library never reads the
process environment: `OPENJEV_ENCODER_MODELS` in it names a folder of locally converted packages,
and an app passes `[:]`.

DiffusionGemma, on a Mac:

```swift
import OpenJevCore
import OpenJevDiffusionGemma

let runtime = try await DiffusionGemmaRuntime.load(.fourBit)
let engine = try DecisionEngine(backend: runtime)
```

The first load downloads the pinned 4-bit checkpoint (13 files, 16.58 GB) into the Hugging Face
cache, `~/.cache/huggingface/hub` by default, where upstream and mlx-vlm find it too. Loading
warms the model up unless its configuration says otherwise.
``DecisionEngine/init(backend:configuration:)`` discovers the single-token choice labels with the
backend's tokenizer, so it throws when the tokenizer fails.

### Build a request

From JSON, as a client sends it:

```swift
let request = try SystemOneRequest(json: JSONParser().parse("""
    {"model": "jev-latest",
     "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
     "questions": {
       "department": {"type": "choice", "instructions": "Which team should handle this",
                      "criteria": {"billing": "Payment or subscription issues",
                                   "technical": "Bugs or integration problems",
                                   "sales": "Pricing or account questions"}},
       "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                       "criteria": ["Calm, just stating facts", "Frustrated but civil",
                                    "Very angry, strong language"]},
       "is_urgent": {"type": "noul",
                     "instructions": "The message conveys urgency or time-sensitivity"}}}
    """))
```

Or in Swift, where the literals keep their order as the JSON does:

```swift
let request = SystemOneRequest(
    model: "jev-latest",
    state: "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
    questions: [
        "department": .choice(
            instructions: "Which team should handle this",
            criteria: [
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            ]),
        "frustration": .score(
            instructions: "How frustrated the customer appears",
            criteria: [
                "Calm, just stating facts", "Frustrated but civil", "Very angry, strong language",
            ]),
        "is_urgent": .noul(
            instructions: "The message conveys urgency or time-sensitivity", criteria: nil),
    ])
```

An engine answers whatever model the request names. The server checks the name first against
the engine's ``ServedModels``, which accept the SDK aliases `jev-latest` and `jev-preview` on every
backend, as upstream does, so that TypeSafe's SDKs work unchanged. <doc:RequestsAndAnswers>
describes every field.

### Decide and read the answers

```swift
let decision = try await engine.decide(request)
for (id, answer) in decision.answers {
    switch answer {
    case .noul(let yes):
        print(id, "yes:", yes)
    case .choice(let choice, let probabilities, let confidence):
        print(id, choice, probabilities.values, confidence)
    case .score(let score, _, let probabilities, let confidence):
        print(id, score, probabilities, confidence)
    }
}
print(decision.inputTokens, "input tokens")
```

To send the answer on, or to store it, write it as the server writes it:

```swift
let response = SystemOneResponse(
    model: engine.servedModels.version, answers: decision.answers,
    usage: Usage(inputTokens: decision.inputTokens, outputTokens: decision.outputTokens))
let body = try WireEncoder().string(response)
```

The bytes are upstream's: the same key order, Python's float formatting and compact separators.

### Handle refusals

- ``SchemaError``: the model cannot answer the request as asked, for example a choice with more
  options than the backend takes, or an option the backend does not honour, such as `steps` on an
  encoder (`verdict-1.4 does not support steps`). The server answers a 400 with its message.
- ``OverloadedError``: the engine already holds its queue bound of requests, 512 by default.
  Retry shortly; the server answers a 529.
- ``JSONParseError``, from ``JSONParser/parse(_:)-(String)``: the text is not JSON.
- ``WireError``, from ``SystemOneRequest/init(json:)``: the JSON does not have the request's
  shape. It carries the 422 or 400 body upstream sends.
- Anything else, ``BackendRefusal`` and ``BackendContractError`` among them, is a failure of the
  model, which the server answers with a 400 for a refusal and a 503 otherwise.

### Concurrency and lifetime

Both engines are actors and take requests from many tasks at once. ``DecisionEngine`` runs up to
``EngineConfiguration/maxInflight`` reads at once, 64 by default, and the DiffusionGemma runtime
runs them on the GPU one at a time. ``EncoderDecisionEngine`` makes one backend call at a time, in
batches of ``EncoderEngineConfiguration/batchSize`` questions, 16 by default. Cancelling the task
that waits for a decision stops its reads that are still waiting for their turn under those
limits; a read that has its turn runs to its end, and with DiffusionGemma that includes a read
queued for the GPU behind another.

Keep one engine per model: loading is the expensive part. When the app is done with an encoder,
``ModelReleasing/close()`` on its engine releases the Core ML functions Verdict or Laya loaded; a
later read loads them again. The DiffusionGemma runtime does not adopt ``ModelReleasing``, so
`close()` leaves it loaded; its model is released with the runtime itself.

### On an iPhone

Verdict reads one question per Core ML call on the Neural Engine. Laya on an iPhone runs one
package per sequence length, each about 845 MB: `LayaBackend.load(from:)` fetches only its
tokenizer and configuration, and the app downloads the packages it wants with
`prefetch(lengths:)`, at least the 128-token one before the warm-up read. A question longer than
every package the device holds throws `EncoderLoadError.noPackage`, which an app can answer by
sending that request to an OpenJev server over the same wire API instead.

## See Also

- <doc:RequestsAndAnswers>
- <doc:ImplementingABackend>
