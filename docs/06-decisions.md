# Decision records

Status values: **Accepted for planning** means the plan assumes it; a spike may still overturn it.
**Open** means an issue must record the outcome before dependent code merges.

## D-001 Upstream is razorback16/openjev, pinned at `dcd2094`

Context. Several projects are or were called OpenJev. The most-starred one renamed itself SemIf
and has no Jev wire API. razorback16/openjev is the one still named OpenJev, is Jev-compatible,
serves the widest model set, ships an MLX backend and is described by third parties as the most
complete implementation.

Decision. Compatibility target is razorback16/openjev at commit `dcd2094` (v0.5.0). SemIf is
related work; its readout is used through JevK5.

Consequences. Every fixture and every "matches upstream" claim refers to this commit. Moving the
pin is an explicit task (see the upstream tracking issue).

Status. Accepted for planning.

## D-002 Project name stays OpenJevSwift

Context. The SemIf rename made the name look stale. It is not: the upstream being ported is still
called OpenJev. "Jev" is TypeSafe's mark; upstream carries a non-affiliation disclaimer and so does
this repository.

Decision. Keep OpenJevSwift. Carry the disclaimer in README, NOTICE and `/v1/models`
descriptions. Revisit only if TypeSafe objects or upstream renames; GitHub redirects renamed
repositories, so the cost of a later rename is low.

Status. Accepted for planning.

## D-003 Inference runs on MLX Swift, not Core ML or llama.cpp

Context. See [04-swift-inference-landscape.md](04-swift-inference-landscape.md). Core ML cannot
express a 25B MoE encoder/decoder diffusion pass with dynamic shapes; llama.cpp's DiffusionGemma
support is an open draft without Metal; the Layr-Labs fork proves MLX Swift runs this model.

Decision. `mlx-swift` plus `mlx-swift-lm`'s `MLXLMCommon`/`MLXVLM` primitives are the base for
DiffusionGemma. Core ML remains a candidate for the small encoder models only (D-011).

Status. Accepted for planning.

## D-004 Port DiffusionGemma into this repository; do not depend on the Layr-Labs fork

Context. The fork has a working Swift DiffusionGemma but is pinned to an old upstream base, will
not merge upstream, carries product-specific engine code and environment switches, and pulls a
server framework into any consumer.

Decision. Implement `OpenJevDiffusionGemma` on upstream `mlx-swift-lm` primitives, following
mlx-vlm's implementation (the same reference upstream OpenJev runs) and cross-checking against
the fork's code and numbers. Attribute both (MIT). Offer the model port upstream to ml-explore
once stable.

Alternatives rejected. (a) Depend on the fork: fastest start, worst long-term position. (b) Wait
for upstream mlx-swift-lm to add the model: no signal it will. (c) Python interop (PythonKit to
mlx-vlm): not native, not shippable in apps.

Consequences. The first hardware spike validates feasibility and tolerances against the fork and
mlx-vlm before the port begins.

Status. Accepted for planning; confirmed or revised by the backend spike.

## D-005 Backend-agnostic core behind a `DecisionBackend` protocol

Context. Upstream shares the API layer, the schema builder, the answer shapes and the error
contract across five model families; only the read differs.

Decision. `OpenJevCore` owns everything up to "give me logprobs at these slot positions for this
prompt and canvas". Backends implement that. Encoder and letter-readout models implement a
sibling protocol that skips canvases (upstream's `EncoderEngine` contract).

Status. Accepted for planning.

## D-006 Byte-exact prompt, template and canvas construction, including upstream's seeds

Context. Upstream's answers are deterministic for a request: SHA-256 of the canonical JSON of the
request selects a seed; Python's Mersenne Twister draws the noise tokens; the MLX and vLLM
backends share this so their answers can be compared. If OpenJevSwift reproduces the same
canvases, its slot logprobs can be compared with the Python MLX backend on the same weights and
kernels within a small tolerance, which is the only strong end-to-end conformance test available.

Decision. Implement MT19937 with Python's `seed(int)` initialisation and `randrange` rejection
sampling, and a Python-`json.dumps`-compatible canonical serialiser (`sort_keys=True`,
`ensure_ascii=True`, default separators, Python float repr). Verify both against CPython outputs
in fixtures. Also implement `ensure_ascii=False` rendering for the text that reaches the model.

Fallback. If Python float repr cannot be matched for some inputs, document that seeds match
upstream only for states without floating-point numbers, and keep everything else identical.

Status. Accepted for planning; feasibility confirmed by a spike in milestone 0.

## D-007 Order-preserving JSON model with its own parser

Context. Jev semantics depend on the order of `questions` and of choice `criteria`; Python
preserves insertion order and upstream relies on it. Foundation's `JSONDecoder` and
`JSONSerialization` return unordered dictionaries.

Decision. `OpenJevCore` ships a `JSONValue` type whose objects are ordered, a strict RFC 8259
parser, and a serialiser with Python-compatible options. `Codable` bridging exists for
convenience where order does not matter.

Status. Accepted for planning.

## D-008 Tokenizer via swift-transformers `Tokenizers`, parity proven by fixtures

Context. mlx-swift-lm integrates `Tokenizers` by default; it maps `GemmaTokenizer` to BPE and
renders chat templates with swift-jinja. Parity with Hugging Face `tokenizers` is required for
slot resolution to work at all.

Decision. Use `Tokenizers` behind the core's `DecisionTokenizer` protocol. Prove parity with a
fixture corpus generated by Python. Render the chat prompt with the shipped Jinja template if the
spike shows identical ids for `[system, user]` with the generation prompt and `enable_thinking`
on and off; otherwise implement a hand-rolled Gemma 4 prompt builder verified against the same
fixtures and record which one is used.

Status. Accepted for planning; two spikes in milestone 2 decide the details.

Outcome of the spikes (#20 and #21, 2026-09-30; details in
[spikes/tokenizer-parity.md](spikes/tokenizer-parity.md) and
[spikes/chat-template.md](spikes/chat-template.md)).

Parity, `SwiftTransformersTokenizer` against the Python fixtures, all run under Xcode's Test
action with the pinned tokenizer files (digests checked against `special_tokens.json`):

| Fixture | Matched | Mismatched |
|---|---|---|
| `tokenizer/corpus.json`, 917 rows: `ids`, `ids_with_special_tokens`, `decoded`, `decoded_skip_special_tokens` | 917 | 0 |
| `tokenizer/engine_encodings.json`, 2,634 pairs | 2,634 | 0 |
| `labels.json`: 255 labels and ids, 6 rejected candidates | 261 | 0 |
| `tokenizer/special_tokens.json`: 23 named ids, `<\|video\|>`, `<end_of_turn>`, `engine` table (scaffold `[100, 45518, 107, 101]`) | 26 | 0 |
| `chat-prompts/prompts.json`, 24 rows, thinking off and on: rendered text | 48 | 0 |
| `chat-prompts/prompts.json`, 24 rows, thinking off and on: ids | 48 | 0 |
| Replay tokenizer agreement: 3,252 texts, 917 decodes, 24 prompts | 4,241 | 0 |

Three corpus decodes (`"\0"`, `"\u{2028}"`, `"\u{10FFFF}"`, each only byte tokens) mismatched
before the adapter's decode fix, none after. Encoding never mismatched.

Load: 3.6 s wall time for the 32 MB `tokenizer.json`, 204 MB of resident memory added, 386 MB
peak resident for a process that had loaded nothing else (Apple silicon, macOS 27). Paid once
per process.

Tokenizer entry point: `SwiftTransformersTokenizer.load(from:)` builds
`PreTrainedTokenizer(tokenizerConfig:tokenizerData:)` from
`LanguageModelConfigurationFromHub(modelFolder:)`, the two steps of
`Tokenizers.AutoTokenizer.from(modelFolder:)`, and sets `clean_up_tokenization_spaces` to
false between them when the checkpoint does not set it, which is transformers' default since
4.45 and not swift-transformers'. mlx-swift-lm's `#huggingFaceTokenizerLoader()` was loaded
once and gives the same ids; it is not used because it hides the upstream tokenizer, brings
`MLXHuggingFace`, the macro plugin and `MLXFoundationModels` into the model target, and adds
nothing over the direct call.

Chat template path: the engine. swift-transformers 1.3.4 reads `chat_template.jinja` on its
own when loading from a folder and merges it into the configuration; `chatPromptIDs` calls
`applyChatTemplate` with `addGenerationPrompt: true` and `additionalContext:
["enable_thinking": thinking]`, and `chatPromptText` renders the same file with swift-jinja
(`lstripBlocks`, `trimBlocks`) and the same context, because swift-transformers returns only
ids. Every recorded text and id sequence matches, so no hand-rolled builder is written. The
mlx-vlm image message shape (`content` as image parts then a text part) renders through the
same template to one `<|image|>` (258880) per image directly before the text.

Two decode departures in swift-transformers 1.3.4 are handled generally in the adapter and
pinned by a test over the unmodified path: byte tokens at the end of a sequence are dropped by
its `ByteFallbackDecoder`, and `clean_up_tokenization_spaces` defaults to true. Neither needs a
custom loader; follow-up issues A to D in spikes/tokenizer-parity.md cover the upstream reports
and the fixture rows that would record them.

Holds for swift-transformers 1.3.4, swift-jinja 2.5.1, swift-huggingface 0.11.0, mlx-swift-lm
`c043fb3`, tokenizer revision `a7a81407`, fixtures from transformers 5.17.0 and tokenizers
0.23.2. `Tests/OpenJevDiffusionGemmaTests/Tokenization` is the permanent regression suite; it
skips with a message when the tokenizer files are absent (`OPENJEV_TEST_TOKENIZER` or the
Hugging Face cache).

Status. Accepted: swift-transformers for the tokenizer, the shipped template through
swift-jinja for prompts. Decided by spikes #20 and #21.

## D-009 Hummingbird 2 for the server

Context. Structured-concurrency native, Swift 6 clean, small, streaming request bodies; Vapor 5
is unreleased and will sit on the same HTTP server.

Decision. Hummingbird 2.23 or later. The server target is separate from the core and from the
model target so library consumers never link it.

Status. Accepted for planning.

## D-010 Apache-2.0 license, third-party notices for MIT ports

Decision. Same license as upstream OpenJev. Ported code from mlx-vlm and the Layr-Labs fork (MIT)
keeps its copyright headers and is listed in `THIRD_PARTY.md`. Weights are never redistributed by
this repository; Gemma Terms of Use apply to the DiffusionGemma weights and are the user's
responsibility.

Status. Accepted.

## D-011 Encoder models: Core ML or MLX decided by a spike; JevK5 first among the extra models

Context. Verdict (151M) and Laya (421M) are ModernBERT encoders with custom heads; Core ML would
reach iOS and the Neural Engine, while an MLX port shares the codebase. JevK5 is Qwen3.5-4B with a
merged LoRA and a next-token letter readout, which `MLXLLM`'s existing Qwen3.5 implementation
can run today with no new model code.

Decision. Implement JevK5 first among the additional models. Run one spike converting Verdict to
Core ML and measuring accuracy parity and latency on macOS and iOS, and one comparing with an MLX
ModernBERT port; choose per model. CLM (Qwen3-8B embeddings plus heads) is deferred until the
others exist.

Status. Open (spike).

## D-012 Reads come first; text generation and `think` come later

Context. Upstream's value is the read path. `think` and `/v1/chat/completions` need the full
diffusion generation loop (samplers, stopping, block commits, streaming), which is a large port
with its own parity problems.

Decision. Milestones 0 to 4 deliver reads with all read-only extensions (`steps`, `samples`,
`sequential`, `images`); milestone 5 adds generation, `think` and chat. A server without a
generation-capable backend omits the chat routes, as upstream's encoder containers do.

Status. Accepted for planning.

## D-013 Configuration names follow upstream's `OPENJEV_*` variables

Decision. The CLI and server accept upstream's variable names, defaults and validation rules
(and refuse the same invalid values at startup) so deployment documentation and compose files
transfer. Library APIs take typed configuration and never read the environment.

Status. Accepted for planning.

## D-014 Answers are compared with tolerances, never bit for bit

Context. Upstream itself observes that BF16 execution paths changed 5 to 6 of 777 argmaxes and
that vLLM and Transformers kernels differ by up to 0.055 in probability for JevK5. MLX Swift
kernels will not match Python MLX bit for bit either.

Decision. Conformance is defined as: identical prompt ids, templates, slots and canvases (exact);
slot log-probabilities within a tolerance to be fixed by the parity spike (proposal: 0.02 in
probability per label, top label agreement at 99% or better on the fixture set); identical wire
shapes and errors (exact). See [09-conformance-and-testing.md](09-conformance-and-testing.md).

Status. Accepted for planning; tolerance value open until the spike reports.

## D-015 Slot-only output projection is allowed

Context. A read needs logits only at slot positions; the full canvas logits are used only for
self-conditioning between steps.

Decision. The single-step read projects only the slot rows through the tied output head. For
`steps > 1`, the implementation may compute full-row logits (needed for self-conditioning) or
prove that slot-only self-conditioning is numerically equivalent to upstream; equivalence must be
shown by the parity tests before the shortcut is used.

Status. Accepted for planning.

## D-016 The JSON parser is stricter than Python's `json.loads`

Context. Upstream reads request bodies with Starlette's `request.json()`, which is Python's
`json.loads`. That parser accepts `NaN`, `Infinity` and `-Infinity`, turns a float that overflows
into infinity (`1e400`), accepts lone surrogate escapes such as `"\ud83d"`, and rejects integers of
more than 4,300 digits (`sys.int_max_str_digits`). Issue #3 asks for a strict RFC 8259 parser.

Decision. `JSONParser` follows RFC 8259: it rejects `NaN`, `Infinity`, overflowing floats and lone
surrogates, and it accepts integers of any length. A request that upstream would accept with one
of these values gets a parse error from OpenJevSwift instead. The writer matches CPython exactly
for every value the parser can produce, including `ensure_ascii` escaping U+007F as `\u007f`.

Status. Proposed with issue #3. The HTTP layer (the 422 body for invalid JSON) decides whether any
of these cases must instead reproduce upstream's behaviour.

## D-017 Wire types follow the recorded oracle, not the issue text, where they differ

Context. Issue #5 describes the wire types. Recording upstream's FastAPI app
(`Tools/fixtures/wire_tables.py`, fastapi 0.142.1, pydantic 2.13.5, Python 3.14.7) showed places
where the issue text, or the brief it was worked from, does not match what upstream does.

Decision.

1. Wire types encode and decode through `JSONValue` (`init(json:)`, `var json`, `WireEncoder`),
   not `Codable`. `JSONEncoder` loses key order and writes `1.0` as `1`. `Usage` and `ModelInfo`
   also conform to `Codable` as a convenience; wire output never uses it.
2. `steps`, `samples`, `think` and `sequential` are coerced as pydantic's lax mode does, not
   strictly: `"3"`, `" 3 "`, `"1_0"`, `"3.00"`, `true` and `2.0` are integers, and `"yes"`, `"off"`,
   `1` and `1.0` are Booleans. Rejecting them would turn upstream's 200 into a 422.
3. Names: the error enum is `WireError` (a struct with a status, a body and headers) rather than
   `OpenJevError`; the bodies are `TypedErrorBody` and `PlainDetailBody`. `NoulCriteria` exposes
   `whenTrue` and `whenFalse` because `true` and `false` are Swift keywords; the wire keys are
   unchanged.
4. The recordings live in `Fixtures/wire/` (`cases.json`, `answers.json`, `requests.json`,
   `models.json`) rather than `Fixtures/requests/` and `Fixtures/errors/`. Issue #6 can fold them
   into its layout.
5. A decode followed by an encode reproduces pydantic's `model_dump(exclude_unset=True)` except
   that an extension field or a noul `criteria` sent as `null` is dropped. Upstream treats `null`
   and absent the same, so no behaviour changes; forwarding sends the original bytes.
6. `RequestValidator` checks shape only. The empty choice, more than 255 options, more than 10
   levels and more than 256 questions stay semantic 400s from the schema builder (#10); images
   are #18's; the unknown model is the server's (#38); the `json_invalid` 422, with Python's
   decoder messages and character offsets, is the HTTP layer's (#35).
7. The contract is pinned to the pydantic version in the fixture headers. Upstream only requires
   `fastapi>=0.115`, so a deployment with another pydantic may word messages differently;
   regenerate and diff when the pin moves.

Status. Proposed with issue #5.

## D-018 Image checks: where the port goes beyond or differs from the issue text

Context. Issue #18 ports upstream's `image_parts` and asks for a `SchemaError` type shared with
the schema builder (#10) and the engine (#17). Porting it exactly needed a few choices the issue
does not spell out.

Decision.

1. `SchemaError` is `{message, loc}` with `loc` defaulting to `["body"]`, as upstream's does, and
   conforms to `CustomStringConvertible` (`body.images.3: message`) for logs.
   `WireError.semantic400(_ error: SchemaError)` sends only the message; the `loc` is never sent.
2. `ImagePart`'s fields are `let` and its initialiser builds `dataURL` from the content type and
   the base64 text, so the data URL cannot drift from the parts that enter the seed key.
3. The strict base64 check computes the decoded length without allocating the decoded bytes. It
   accepts exactly what CPython 3.14's `base64.b64decode(data, validate=True)` accepts, checked
   with `python3`: the standard alphabet only, a length that is a multiple of four, at most two
   `=` and only at the end, non-zero trailing bits allowed (`QR==` is valid), no whitespace, no
   non-ASCII. Whoever feeds the image to a vision encoder decodes it then.
4. The data URL split, the `data:` and `;base64` tests, and the content type comparison work on
   Unicode scalars, as Python's `str` operations do, not on Swift's `Character` with canonical
   equivalence. The length bound counts scalars, which is Python's `len`.
5. `{t!r}` in the unsupported type message is reproduced by an internal `String.pythonRepr`
   (CPython's `unicode_repr` rules). Whether a non-ASCII character is printable comes from this
   platform's Unicode tables, which can differ from the CPython build's for newly assigned
   characters.
6. The test that the 8 MB payload never reaches decoding measures time (under 500 ms) and also
   sends a payload of the same length whose last character is invalid: it still gets the size
   message, so the bound runs first. No test hook was added to the production type.
7. An empty image list returns no parts. Upstream never calls `image_parts` for an empty or absent
   list; the result is the same.

Status. Proposed with issue #18.

## D-019 Sums follow CPython 3.12 and later: compensated, not a running total

Context. Issue #16 asks for ports of `slot_distribution`, `confidence`, `to_answer` and the read
averaging, and for the recorded answers to be reproduced byte for byte. Every one of those uses
Python's built-in `sum` over floats. Since Python 3.12, `sum` uses Neumaier's compensated
summation, so `sum([0.1] * 10)` is `1.0`, not `0.9999999999999999`. The fixtures were recorded
with Python 3.14.7, and upstream's image and development setups run 3.12 or later.

Decision. `OpenJevCore` has an internal `pythonSum` that reproduces CPython's algorithm, including
adding the compensation only when it is non-zero and finite. It was checked against CPython
3.14's `sum` on 20,000 random vectors with no difference. The expected score, the entropies, the
softmax denominator and the averaged probabilities all use it. A deployment of upstream on Python
3.11 or earlier would give different last bits; that is not a supported target.

Status. Proposed with issue #16.

## D-020 Slot distribution API and the expected distributions fixture layout

Context. Issue #16 gives `SlotDistribution.compute(top: [Int: Double], labelIDs:)`. The entropy is
a compensated sum over `top` in the order the backend returned it, and a compensated sum can
differ in its last bit between orders. A Swift dictionary has no stable order.

Decision.

1. `SlotDistribution.compute(top: [(tokenID: Int, logprob: Double)], labelIDs:)` is the primary
   form and keeps the backend's order; the dictionary form from the issue forwards to it. The
   probabilities do not depend on the order.
2. An empty `top`, an empty `labelIDs` or a token id repeated in `top` stops with a precondition
   failure rather than throwing: a backend always returns at least one token and a question
   always has a label, so these are programming errors.
3. `Confidence.compute` returns 1.0 for `K <= 1`, where upstream's formula divides by zero. Upstream
   never gets there because forced answers set 1.0 directly, and `Answer.make` with `[1.0]` for a
   single-option choice or single-level score gives exactly upstream's forced answer.
4. `ReadAveraging` lives in its own file, `Read/ReadAveraging.swift`.
5. The tests for `Fixtures/distributions/distributions.json` (issue #6) read its
   `slot_distribution` rows (`{name, top, label_ids, result: {probs, entropy}}`, with `top` as
   `[token id, logprob]` pairs in the backend's order) and its `confidence` rows
   (`{name, p, result}`). Rows that record an upstream exception are skipped: the empty map is
   item 2's precondition, and one option gives 1.0 by item 3. Both compare exactly, because
   these are pure functions of the same doubles (D-014's tolerances apply to model logprobs,
   not to this arithmetic). The loader was first written against a guessed layout, before #6
   landed, and was brought in line with the real file on the branch for #10 and #11.

Status. Proposed with issue #16.

## D-021 Question schema: where the port goes beyond or differs from the issue text

Context. Issue #10 ports `Engine.build_schema`, `text_of` and `FORMATS` from `engine.py` and the
encoder backends' `build_schema` from `encoders.py`. A few choices were needed that the issue does
not spell out.

Decision.

1. `TextOf.render` does not throw. A value that `PythonJSONWriter` cannot write (an infinite or
   NaN float, or integer text that is not normalized digits) stops with a precondition failure.
   `JSONParser` never produces one (D-016), so only a hand-built value can get there. Upstream's
   `json.dumps` would write `NaN`; the writer does not support `allow_nan`.
2. The question type is a public `QuestionKind` enum (`noul`, `choice`, `score`) shared by
   `ReadQuestion` and `EncoderQuestion`. `ReadQuestion.choices` is the tuple array the issue
   gives, so `ReadQuestion` and `QuestionSchema` are `Sendable` but not `Equatable`.
3. The score limit message interpolates `maxScoreLevels`; at the default of 10 it is upstream's
   text. The choice limit is `min(maxChoices, choiceLabels.count)`, and the message names that
   number, as upstream's names `len(choice_labels)`.
4. `AnswerFormat` also has `afterID` (the lead without the id), `lead(id:)` (Python's
   `lead.format(id=...)`) and `forReadCount(_:)` (the 10-question rule), so later issues read the
   format rules from one place. Its raw values are `lines` and `indexed`, the fixture names.
5. `EncoderQuestion` keeps the question as sent (`question: Question`) rather than separate raw
   instructions and criteria; `rawInstructions` reads it. Upstream's encoders rebuild
   `{type, instructions, criteria}` from those two fields, which is that question. For a noul
   sent without criteria upstream keeps `{}`, this port keeps `nil`; both mean no descriptions.
   `EncoderQuestionSchemaBuilder` keeps upstream's fixed limit of 10 score levels.
6. Both builders go through one internal `SchemaRules.entry(key:question:)`, which holds the
   limits, the forced answers and the rendered answers; only the labels, ids and format are
   the engine builder's own.
7. A score with no levels cannot pass `RequestValidator`. Built by hand, it is read with no
   labels, as upstream would.
8. The fixture test checks the two `api_reachable: false` rows by asserting that
   `RequestValidator` refuses them: there is no `Question` to build. The loader for the engine
   fixtures, `Tests/OpenJevCoreTests/Schema/UpstreamFixtures.swift`, is shared with the prompt
   tests of #11.

Status. Proposed with issue #10.

## D-022 Prompt text: where the port goes beyond or differs from the issue text

Context. Issue #11 ports `Engine.system_text`, `Engine.answer_text` and the state text of
`Engine.decide`. A few choices were needed that the issue does not spell out.

Decision.

1. `SystemText` exposes its fixed strings as `opening`, `defaultInstructions` and
   `chunkedSentence`, so tests and later issues (#13, #17) read them from one place, as
   `AnswerFormat` does for the format strings (D-021).
2. `SystemText.render` pairs a question's choices and labels with `zip`, as upstream does, so a
   hand-built question with unequal counts lists the shorter one. The schema builder always makes
   them equal.
3. `AnswerText.render` stops with a precondition failure when the question and index counts
   differ, as the issue asks, and also when an index is outside a question's labels. Upstream's
   `zip` would drop the extra entries and its indexing would raise `IndexError`.
4. `StateText.render` shares `TextOf`'s precondition for values `PythonJSONWriter` cannot write
   (D-021, item 1). A string state is not stripped.
5. `Fixtures/tokenizer/corpus.json` records the `json_state` texts but not the states they came
   from. The test renders the state of every request in `schemas/`, `system-texts/` and
   `templates/` and requires each of the three `json_state` texts to be one of those renderings.

Status. Proposed with issue #11.

## D-023 Random numbers and seeds: where the port goes beyond or differs from the issue text

Context. Issue #15 ports CPython's `random.Random(seed)` seeding, `getrandbits` and `randrange`,
and upstream's request seed (`api.py` lines 262 to 263) and derived seeds (`engine.py` lines 342
and 383). A few choices were needed that the issue does not spell out.

Decision.

1. Seeds are `UInt64`, not arbitrary integers. `MT19937(seed:)` builds a one-word key below 2^32
   and a two-word key above, as CPython does. Every seed upstream makes fits: the request seed is
   32 bits and the derived seeds add `104729·k` or `7919·k` for small `k`. `MT19937(key:)` takes
   the words directly for anything else.
2. `SeedDerivation.seed(for:)` returns the 32-bit seed as `UInt64` so that `groupSeed` and
   `sampleSeed` need no conversion. The API is `seedKey(state:questions:images:)`, which takes
   the parts `ImageValidation` returned, then `seedBytes(for:)` and `seed(for:)`, each taking the
   key.
3. `PythonRandom.getrandbits` supports 1 to 64 bits and stops with a precondition failure
   outside that range; `randrange` requires `n > 0`, where Python raises `ValueError`. Upstream
   only calls `randrange(262144)`.
4. SHA-256 is a pure Swift `OpenJevCore.SHA256` so the core stays Foundation-only on Linux. It is
   public, so a file that also imports CryptoKit must qualify the name.
5. The rows of `seeds.json`'s `upstream_only` are not tested: their bodies need Python's lenient
   `json.loads`, which `JSONParser` does not reproduce (D-016).
6. CPython (PSF-2.0) and the MT19937 reference code (BSD-3-Clause) it is built on are listed in
   `THIRD_PARTY.md`, and `MT19937.swift` keeps the reference code's notice.

Status. Proposed with issue #15.

## D-024 Tokenizer protocol and labels: where the port goes beyond or differs from the issue text

Context. Issue #12 defines the core's tokenizer protocol, ports `Engine._single_token_labels` and
exposes the engine's marker sequences, with a replay tokenizer for the tests. A few choices were
needed that the issue does not spell out, and the protocol differs from the issue's sketch.

Decision.

1. `DecisionTokenizer` is `encode(_:addSpecialTokens:) throws`, `decode(_:skipSpecialTokens:)
   throws` and `chatPromptIDs(system:user:thinking:) throws`. The issue's `encode` and `decode`
   do not throw and `decode` has no option; every method throws here so a replay tokenizer can
   refuse an unrecorded input instead of returning wrong ids, and `skipSpecialTokens` mirrors the
   corpus's two recorded decodes. `IDs` follows Swift's acronym casing. Errors are a
   `TokenizerError` struct with a message.
2. `LabelDiscovery.choiceLabels(using:)` returns a `LabelSet` with the labels and the single id
   of each, which the template and read code need. `LabelDiscovery` also holds `prefix`,
   `candidates`, `maxChoices` (255), `noulLabels` and `scoreLabels`. The schema builder's noul
   labels now read `noulLabels`; its score labels stay `String(i)` because `maxScoreLevels` is
   configurable, and at the default of 10 they equal `scoreLabels`.
3. `EngineTokens.turnClose` is 106 and is documented as `<turn|>`, which it is in this
   vocabulary; upstream's docs name it after `<end_of_turn>`, which is not a token here
   (`Fixtures/tokenizer/special_tokens.json`). The marker texts are exposed as constants beside
   the encoded sequences.
4. The test-only `FixtureTokenizer` compares texts by their UTF-8 bytes, so NFC and NFD spellings
   stay distinct as in Python. `encode` without special tokens tries `engine_encodings.json`
   before `corpus.json`; with special tokens and for `decode` only the corpus is recorded.
5. The schema fixture test builds with the labels discovered through `FixtureTokenizer` and
   checks they equal `labels.json`, so the schemas tests exercise discovery end to end.

Status. Proposed with issue #12.

## D-025 Template slots and cache: where the port goes beyond or differs from the issue text

Context. Issue #13 ports `Engine.resolve_template` and the `_templates` cache. A few choices were
needed that the issue does not spell out, and two signatures differ from the issue's sketch.

Decision.

1. `TemplateResolver.resolve` throws untyped, not `throws(SchemaError)`. The tokenizer's
   `encode` throws (D-024) and its failure is not a request problem: the caller must be able to
   tell a `SchemaError` (a 400) from a tokenizer failure (a 500), so the tokenizer's error passes
   through unchanged. Upstream's `enc` never raises for a `str`, so this path is new.
2. The slot type is nested as `ResolvedTemplate.Slot`, with `position` and `labelIDs`, rather
   than a top-level `Slot`, to keep the core's namespace to types that stand alone.
3. The cache is `TemplateCache`, a final class with an `NSLock`, and `TemplateResolver` is a
   `Sendable` struct holding it by reference. Copies of the resolver share one memo, so the engine
   actor of #17, its concurrent group reads and any caller outside the actor use the same cache
   without a mutable property. The key is a `Hashable` struct (format, head as used, lead, and
   each question's id and labels), the same fields as upstream's `json.dumps` key, so `nil` and
   the scaffold share an entry. Errors are not cached. The limit is upstream's 4,096 by default
   and configurable through the resolver's `cacheLimit`, and `count` is exposed for tests and
   diagnostics. The resolver always creates its own cache; a cache cannot be injected, because
   the key names neither the canvas nor the tokenizer (upstream's does not either, its cache
   belonging to one engine), and a cache shared across resolvers with different canvases would
   return a template that never met the smaller canvas's check.
4. Every question must have at least two labels, which the schema builder guarantees. A
   hand-built question with fewer stops with a precondition failure at that question, where
   upstream would raise a `TypeError` from `base[None]`.
5. The hand-written tests use `WordTokenizer`, a test-only stand-in that splits letters, digits
   and other characters and hashes each piece, because the fixtures do not record the texts of
   upstream's `test_indexed_format_with_mixed_types` (40-option choices) nor the 4,098 templates
   the cache-limit test needs. Only token counts and the one-slot property matter to those tests.
   In #13 the mixed schema is resolved as one group; #14 adds the grouping.
6. `label_ids_over_read_limit` in `Fixtures/templates/errors.json` is skipped with a comment: the
   512 label-id check belongs to `Engine.one_read` and issue #17.

Status. Proposed with issue #13.

## D-026 Grouping and canvases: where the port goes beyond or differs from the issue text

Context. Issue #14 ports `Engine.groups`, `Engine.canvas_width` and `Engine.build_canvas`, and
the `canvas` and `canvas_step` settings they read. A few choices were needed that the issue does
not spell out, and one return type differs from the issue's sketch.

Decision.

1. The vocabulary size, turn close and pad stay on `EngineTokens`, where #12 put them; no second
   constants type.
2. `CanvasGeometry(canvas:step:)` throws `CanvasGeometryError` for a value below 1 rather than
   trapping. Upstream's `Settings` raises at startup for the same values, so a thrown error is the
   closer port and the test can check it. `width(templateCount:)` computes the ceiling as
   `(need + step - 1) / step`, which equals Python's `-(-need // step)` for positive operands; the
   doc comment says so.
3. `CanvasBuilder.build` returns a `SeededCanvas` with `tokens` and `noise` (the draws in slot
   order) rather than a bare `[Int]`, so diagnostics need no second call and no second generator.
   It stops with a precondition failure when the template does not fit the width or a slot is
   outside the template. Upstream pads with `[PAD] * negative`, an empty list, and would send a
   canvas longer than its declared width; `TemplateResolver` has already refused such a template.
4. `ReadGrouping.rows(of:)` is public, so the fixture test compares the number `groups()`
   compares with the canvas and later issues can log it.
5. `ReadGrouping.groups` returns `[]` for no questions, as the issue asks; upstream returns
   `[[]]` but only calls `groups()` when there is something to read.
6. The fixture test reads `settings.step` as well as `settings.canvas`, defaulting to 16 and 64,
   so a future fixture at another step needs no test change. The hand-written boundary tests
   (a template one token under and one over the canvas, 30 nouls at canvas 32) use the test-only
   `WordTokenizer` of D-025, because the fixtures do not record those trial texts.

Status. Proposed with issue #14.

## D-027 Decision engine: where the port goes beyond or differs from the issue text

Context. Issue #17 ports `Engine.decide`, `read_group`, `_sequential` and the contracts of
`one_read` and `think` into `DecisionEngine` behind the `DecisionBackend` protocol, with a stub
backend that reproduces the reads `Fixtures/policies/` was recorded with. A few choices were needed
that the issue does not spell out, and some signatures differ from the issue's sketch.

Decision.

1. `ReadPrompt` is an enum with `.tokens([Int])` and `.image(systemText:stateText:images:)`, not
   a struct. The two shapes are exclusive, which is what an enum says.
2. `DecisionBackend.think(prompt:budget:stopIDs:)` returns `ThoughtGeneration`, the ids as
   generated plus the prompt tokens, rather than the issue's `Thought` (prefix, thought tokens,
   prompt tokens). The engine cuts the ids at the first thought-close id and appends the close
   marker, `Engine.think` lines 317 to 330, so that step is implemented and tested once.
   `read(_:)` returns `ReadResult`, one `SlotRead` distribution per slot plus the prompt tokens;
   `ReadResult(tops:labelIDs:promptTokens:)` takes the raw maps and runs
   `SlotDistribution.compute`, so a real backend follows upstream's `one_read`. The protocol also
   has `capabilities` and `modelName`, which the capability check of upstream's encoder engines
   needs (`"{model} does not support {field}"`).
3. `decide(_:seed:)` takes an optional request seed. Upstream's `Engine.decide` receives the seed
   from the route, which derives it (`SeedDerivation`); `nil` does the same here. The `read_group`
   rows of `Fixtures/distributions/distributions.json` were recorded at seed 1000 and are replayed
   through it.
4. `decide` returns `Decision` with `modelTime: Duration`, the time inside backend calls
   including the wait for a permit, summed over the calls, rather than a task-local. Reads of one
   request run at once, so it can exceed the request's wall time, as upstream's Server-Timing
   `model` value does.
5. The queue bound is upstream's `waiting >= max_queue` as written: `maxQueue` 1 admits one
   request and refuses a second concurrent one, and `maxQueue` 0 refuses every request. The
   issue's test line (`maxQueue 0` refusing the second concurrent request) describes `maxQueue`
   1 under that rule; the test uses 1 and also checks that 0 refuses the first.
6. The 512 label-id limit is `DecisionEngine.checkLabelLimit(_:)`, public and static, called
   before every read. It cannot be reached through `decide` with the real labels, whose whole
   union is at most 267 ids, so the test drives it with the slots of
   `label_ids_over_read_limit` in `Fixtures/templates/errors.json`, which D-025 item 6 left to
   this issue.
7. The thought prompt and every token read prompt, the prefix included, are checked against
   `maxPromptTokens` with `MlxEngine`'s message (`"the request is {n} tokens; the limit is
   {max}"`), as `MlxEngine.one_read` and `MlxEngine.think` check them. Upstream's vLLM engine
   has no such check; a backend with no limit sets `maxPromptTokens` to `Int.max`.
8. `think` is used when it is not 0 and `samples` when it is above 0, Python's truthiness of
   `opts["think"]` and `opts["samples"]`; `RequestValidator` keeps both in range anyway. A
   backend that returns a different number of slot reads than slots is a precondition failure,
   as D-020 treats contract violations.
9. The concurrent work runs outside the actor: an internal `GroupReader` holds the read
   policies, the backend and the `maxInflight` semaphore (an internal `AsyncSemaphore`, no
   dependency), and the actor holds only `waiting`. Results are placed by group and read index,
   so the answer order never depends on scheduling. The fixture tests match reads to the
   recording by seed, which is unique within a case, because the recording's order across
   concurrent calls is asyncio's.
10. `CanvasRead` is `Hashable`, so tests compare recorded calls; `CanvasGeometry.standard` is
    upstream's default geometry, since `CanvasGeometry.init` throws. `EngineConfiguration` keeps
    `servedModelVersion` (`openjev-0.1`) for building a response outside the server.
11. `StubBackend` lives in the test target. It reproduces upstream's `fake_read` and
    `fake_think`, records every call, and can answer named seeds from recorded raw maps, take a
    delay, refuse capabilities and cap the prompt length.

Status. Proposed with issue #17.
