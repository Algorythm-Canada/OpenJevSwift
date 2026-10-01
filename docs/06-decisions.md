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

Outcome of spike #22 (2026-09-30; details in
[spikes/backend-validation.md](spikes/backend-validation.md)). Confirmed, with one refinement.
mlx-vlm is the only numeric oracle; the fork is a reference for code structure only.

| Implementation of the read, against mlx-vlm 0.6.15 on 27 fixture reads | Bit-identical reads | Top label agreement (156 slots) | Max label probability difference |
|---|---|---|---|
| mlx-vlm's read path written out in Swift on ml-explore/mlx-swift 0.32.2 and mlx-swift-lm `c043fb3`, with the mlx-metal wheel's `mlx.metallib` and mlx-vlm's RoPE table | 27 | 156 | 0 |
| The same code on mlx-swift's own kernels | 0 | 150 | 0.374 |
| Layr-Labs/mlx-swift-lm `eeba2af` (`encode` then `denoise`) | 0 | 147 | 0.451 |
| mlx-vlm itself with a 64-token chunked prefill | 0 | 148 | 0.618 |

- **A port on upstream primitives can match mlx-vlm bit for bit.** With the same kernels, 321
  intermediate tensors of the prefill, all 27 reads, the argmaxes of steps 2 and 3 and the
  prefill caches of prompts up to 2,939 tokens are identical.
- **The upstream pieces used:** `SwitchLinear`, `gatherSort` and `scatterUnsort`, `loadWeights`
  with `BaseConfiguration.perLayerQuantization`, `MLXFast.RoPE` with explicit frequencies,
  `MLXFast.rmsNorm`, `MLXFast.scaledDotProductAttention` and `QuantizedEmbedding.asLinear`.
- **The pieces not used:** upstream's Gemma 4 router, whose rounding differs from mlx-vlm's, and
  its `SwitchGLU` experts. DiffusionGemma's experts are a fused, quantized `gate_up_proj`.
- **The fork cannot be made exact.** It runs on its own MLX fork (Layr-Labs/mlx-swift `0f4fe403`,
  MLX 0.32.2 with kernel patches), and its differences from mlx-vlm are the size of mlx-vlm's own
  under a chunked prefill. Depending on it would not improve parity.

Status. Accepted: port mlx-vlm's DiffusionGemma onto upstream mlx-swift-lm primitives, with mlx-vlm
as the numeric oracle. Decided by spike #22.

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
peak resident for a process that had loaded nothing else (Apple silicon, macOS 27); the
mlx-swift-lm macro path loaded cold the same way costs 3.8 s, 236 MB added and the same peak.
Paid once per process.

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
custom loader; follow-up issues A to E in spikes/tokenizer-parity.md cover the upstream reports,
the fixture rows that would record them, and running the suite in CI.

Holds for swift-transformers 1.3.4, swift-jinja 2.5.1, swift-huggingface 0.11.0, mlx-swift-lm
`c043fb3`, tokenizer revision `a7a81407`, fixtures from transformers 5.17.0 and tokenizers
0.23.2. `Tests/OpenJevDiffusionGemmaTests/Tokenization` is the permanent regression suite. It
needs the checkpoint's tokenizer files (`OPENJEV_TEST_TOKENIZER`, `OPENJEV_TEST_MODEL` or the
Hugging Face cache) and otherwise skips as a model opt-in test, naming `OPENJEV_TEST_MODEL`, so
hosted CI builds it but does not run it until follow-up E in spikes/tokenizer-parity.md.

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

## D-011 Encoder models: Core ML for Verdict and Laya; JevK5 first among the extra models

Context. Verdict (151M) and Laya (421M) are ModernBERT encoders with custom heads; Core ML would
reach iOS and the Neural Engine, while an MLX port shares the codebase. JevK5 is Qwen3.5-4B with a
merged LoRA and a next-token letter readout, which `MLXLLM`'s existing Qwen3.5 implementation
can run today with no new model code. Spike #56 converted both encoders to Core ML and measured
them on an M3 Max and an iPhone 13 Pro Max (A15, iOS 27.0); the evidence is in
[spikes/encoder-runtime.md](spikes/encoder-runtime.md).

Decision.

1. **Verdict runs on Core ML** on iOS and macOS, from the float16 package that
   `Tools/encoders/convert_verdict.py` converts from the PyTorch checkpoint: one function per
   input shape (batch 1 and 16 by 128, 256 and 512 tokens), weights stored once, 306 MB. On an
   iPhone it uses `.cpuAndNeuralEngine` and reads one question per call: 9.7 ms at 128 tokens,
   17 ms at 256 and 46 ms at 512, in 106 MB. On a Mac it uses `.cpuAndGPU` (7.5 to 20 ms a
   question) and may read up to 16 questions per call.
2. **Laya runs on Core ML** (`convert_laya.py`, 128 to 1,024 tokens), with the marker gather, the
   temperatures, the clamp and the 4-decimal rounding in Swift. On an iPhone it uses
   `.cpuAndNeuralEngine` with one package per sequence length, each one program for one shape (845
   MB each), because Core ML does not load the multifunction package for the Neural Engine: 27.9 ms
   at 128 tokens, 137 ms at 512 and 513 ms at 1,024, in 106 MB or less, loading included. That is
   2.6 to 2.9 times faster than the iPhone's GPU (82 ms to 1.35 s, 855 MB). An app downloads the
   package for a length when a state first needs it. On a Mac it uses the multifunction package (849
   MB) with `.cpuAndGPU` (15 to 82 ms).
3. **No MLX port of ModernBERT.** It would take 5 to 6 days and about 700 lines to maintain, and it
   could not use the Neural Engine, which ran Verdict about 2.5 times faster than the iPhone's GPU
   with a third of the memory, and Laya 2.6 to 2.9 times faster in an eighth of it.
4. **iOS 18 and macOS 15.** The multifunction packages need them. The iOS 17 alternative, one
   program with enumerated input shapes, crashes Core ML's CPU backend on macOS 27.0.1 and iOS 27.0.
   iOS 18 runs on the same iPhones as iOS 17 (XS, XR and later), so no device is lost; the library
   keeps its iOS 17 platform and the encoder backends are available from iOS 18.
5. **Packages are downloaded on first use,** verified by SHA-256 and compiled on the device, not
   bundled in apps. On iOS, Verdict's package needs only its batch-1 functions, and each of Laya's
   per-length packages is downloaded when a state first needs that length.
6. **Fallback.** When a device cannot meet an app's time budget (Laya at 1,024 tokens takes 0.51 s
   on an A15's Neural Engine), or has not downloaded the package a read needs, the app sends the
   read to an OpenJev server over the same wire API
   (`/v1/systemone` with `verdict-1.4` or `laya-1.0`).
7. Implement JevK5 first among the additional models; on iPhones it waits for a measurement on an
   8 GB device (issue #55). CLM (Qwen3-8B embeddings plus heads) is deferred until the others
   exist.

Alternatives rejected. (a) An MLX ModernBERT port: see 3. (b) iOS 17 packages with enumerated
shapes: see 4. (c) ONNX Runtime with its Core ML execution provider: a large binary dependency for
two small models, and coremltools 9.0 cannot read the shipped ONNX export. (d) Server-only
encoders: the iPhone numbers show on-device reads are affordable. (e) Laya on the iPhone's GPU from
the multifunction package, one 849 MB download: 2.6 to 2.9 times slower than the Neural Engine,
with 855 MB in use.

Consequences. Issues #57 and #58 build on the spike's harness (tokenization, the prompt and
sequence builders, calibration, the Core ML runner) and on Fixtures/encoders; the report lists their
scope. The converted packages need a home and a release step. Core ML's own defects (the crash
with enumerated shapes, a misleading `functionName` load error, silent fallbacks to the CPU, and
Laya's multifunction package failing to load for the Neural Engine, which costs an iPhone one
download per length) enter the backends' test matrix on every OS release.

Status. Decided by spike #56 on 2026-09-30.

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

## D-014 Model parity: bit for bit under the oracle's kernels, bounded aggregates otherwise

Context. Upstream itself observes that BF16 execution paths changed 5 to 6 of 777 argmaxes and
that vLLM and Transformers kernels differ by up to 0.055 in probability for JevK5. MLX Swift
kernels will not match Python MLX bit for bit either.

Decision. Conformance is defined as: identical prompt ids, templates, slots and canvases (exact);
slot log-probabilities within a tolerance to be fixed by the parity spike (proposal: 0.02 in
probability per label, top label agreement at 99% or better on the fixture set); identical wire
shapes and errors (exact). See [09-conformance-and-testing.md](09-conformance-and-testing.md).

Outcome of spike #22 (2026-09-30; numbers and method in
[spikes/backend-validation.md](spikes/backend-validation.md)). The proposal is replaced.

The read is chaotic in bfloat16: last-bit differences in a few kernels move label probabilities
by up to 0.62. The same changes move top labels even where the oracle's top-two margin is 0.68.
A chunked prefill in mlx-vlm itself, exact in real arithmetic, fails 0.02 on 67 of 156 slots.
The fork fails it on 70 and a faithful Swift port on mlx-swift's kernels on 64. No per-label
tolerance can be both met and useful. Model parity against `Fixtures/oracle/reads.json` therefore
has two tiers.

1. **Exact tier, the conformance test.** This runs on the reference machine, or any machine that
   regenerates the oracle. The port runs with `MLX.GPU.metallib` set to the pinned mlx-metal
   wheel's `mlx.metallib`, SHA-256 in the oracle's generator. Its full-attention layers use the
   RoPE frequency table recorded in the oracle's `rope`. Under those conditions every read must
   be identical: every `[token id, logprob]` pair, `slot_distribution`'s probabilities and
   entropies, the argmaxes written by steps 2 and 3, the prompt token counts, and the prefill
   cache digests of layers 0 and 29. Spike #22's transliteration meets this on all 27 reads.
   Any deviation from mlx-vlm's operations fails it, including one that is exact in real
   arithmetic, such as upstream's Gemma 4 router.
2. **Native tier, the production check.** With mlx-swift's own kernels, the 27 reads must meet
   every bound below. The table gives the legitimate range (mlx-vlm chunked or unsorted, the fork,
   the Swift port on native kernels) and the planted bugs (self-conditioning skipped on step 1,
   canvas RoPE from position 0, no sliding window for the canvas).

| Aggregate over the fixture reads | Bound | Legitimate range | Planted bugs |
|---|---|---|---|
| Mean absolute label probability difference, all 1,763 labels | ≤ 0.02 | 0.0011 to 0.0100 | 0.0159 to 0.0562 |
| The same over the 50 slots of the prompts over 1,024 tokens | ≤ 0.01 | 0.0000 to 0.0069 | 0.0142 to 0.0320 |
| Mean absolute entropy difference, all 156 slots | ≤ 0.2 | 0.016 to 0.102 | 0.133 to 0.765 |
| The same over the prompts over 1,024 tokens | ≤ 0.2 | 0.000 to 0.161 | 0.275 to 0.926 |
| Top label agreement, all slots | ≥ 90% | 94.2% to 100% | 67.3% to 91.7% |
| Top label agreement where the oracle's top-two margin is at least 0.5 | ≥ 97% | 99.2% to 100% | 76.7% to 98.3% |

Each planted bug breaks at least one bound, and every legitimate implementation passes all of
them. The window bug shows only in the long-prompt rows. Per-read and per-slot maxima are
reported, never bounded: the legitimate maximum is 0.62. The argmaxes written between steps are
compared only in the exact tier; under native kernels they flipped on 2 of 6 multi-step reads.
Entropy never changed upstream's re-read decision: every fixture read has a slot entropy above
0.1. Prompt ids, templates, slots, canvases, wire shapes and errors stay exact. Synthetic MLX tests
(D-028) keep their CPU-reference tolerances. The bounds are calibrated on 27 reads; issue #31
should widen the fixture set and recompute them with `Tools/oracle/tolerance_stats.py`.

Status. Accepted: two tiers, with the bounds above. Decided by spike #22.

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

## D-028 MLX synthetic tests run in CI on the hosted runner's GPU

Context. Issue #8 asked whether GitHub-hosted macOS runners execute mlx-swift's Metal kernels well
enough for unit tests on small synthetic shapes, or whether every MLX test must stay opt-in on
developer machines. `mlx-probe.yml` ran mlx-swift 0.32.2 on `macos-15`, `macos-26` and `xcode-27`
on 2026-09-30. The findings are under R13 in [07-risks-and-unknowns.md](07-risks-and-unknowns.md):
the runners' paravirtual GPU runs every kernel the probe tried, including attention, 4-bit and
expert-gathered 4-bit matmuls and RoPE, but only in a build that carries MLX's Metal library, and
inside `swift test` only once MLX is pointed at it.

Decision.

1. MLX synthetic tests (small shapes, random or recorded weights, no checkpoint) run in CI, on the
   GPU of the `macos-26` runner, in the `OpenJevDiffusionGemma` test target with every other test.
2. The macOS job builds and tests with `--build-system swiftbuild`. Xcode 26.6 defaults to the
   native build system, whose products carry no Metal library, and MLX cannot run without one, on
   the GPU or the CPU.
3. Every MLX test calls a helper that sets `GPU.metallib` to the copy of the library in the test
   bundle before its first MLX call. Swift Build copies the library there, but MLX finds it only
   through a `Bundle` object for the test bundle, and the Swift Testing runner creates none. The
   first MLX test adds the helper from [development.md](development.md) ("MLX in tests").
4. MLX tests compare with tolerances (D-014), never bit for bit: GPU results differ from the CPU's
   in the last digits.
5. Tests that need the weights stay opt-in behind `OPENJEV_TEST_MODEL`, and CI fails if any other
   test skips. A synthetic test must fit the runner: 7 GB of memory, a 4.7 GB Metal working set
   and three CPU cores.

Alternatives rejected. (a) Run MLX tests on the CPU only: MLX loads its Metal library as soon as
it creates a stream on a Mac, so a CPU-only test needs the same build and library and saves
nothing, and the GPU is the path the model takes. (b) Keep every MLX test opt-in: the hosted GPU
runs them, and the block, mask and cache code would go untested until someone ran it by hand.
(c) Test with `xcodebuild test`, as mlx-swift's own CI does: the XCTest runner creates the bundle
object and MLX finds its library unaided, but the job would trade `swift test` and its log for an
Xcode scheme and a result bundle to save one line of test setup.

Consequences. The macOS job builds the way Xcode 27, the reference toolchain, builds by default.
The runners' GPU reports no Apple GPU family, so a kernel that behaves differently on a real Apple
GPU would not show it in CI; the model's parity tests on developer machines remain the check for
that. Run `mlx-probe.yml` again when mlx-swift, Xcode or the runner image changes.

Status. Proposed with issues #7 and #8.

## D-029 Encoder engine and served models: where the port goes beyond or differs from the issue text

Context. Issue #67 ports upstream's `EncoderEngine` contract (`encoders.py` lines 50 to 158) and
`config.served_models` into `OpenJevCore` as `QuestionReadBackend`, `EncoderDecisionEngine`,
`SystemOneService` and `ServedModels`. The issue's sketch predates the work on #10 and #17, and
a few points needed choices it does not spell out.

Decision.

1. `QuestionReadBackend` exposes `modelInfo: ModelInfo` rather than the sketch's `modelName`:
   the served name, description and release date travel together, so `ServedModels.encoder(_:)`
   and the `/v1/models` listing come from the backend itself. `KnownEncoderModels` holds
   upstream's `ENCODER_MODELS` texts word for word; `Fixtures/wire/models.json` is the oracle.
2. `batchSize` is a setting of `EncoderEngineConfiguration` (`OPENJEV_ENCODER_BATCH`, 16), not a
   property of the backend as the sketch had it: upstream reads it from `Settings`, and a
   deployment tunes it per machine, not per model. The configuration also carries `maxQueue`
   (512), `maxInflight` (1, upstream's one model thread; CLM and JevK5 may raise it) and `warmUp`
   (true).
3. `readBatch(state:stateText:questions:)` receives both the raw state and its `StateText`
   rendering and returns a `BatchReadResult` struct rather than a tuple. Verdict reads the text;
   Laya and CLM render the raw value themselves; JevK5 embeds it in its JSON prompt.
4. `maxPromptTokens` is an optional the engine does not enforce. It has no tokenizer, and upstream's
   encoder engines count tokens inside their own reads; Verdict and Laya truncate (`nil`), CLM and
   JevK5 refuse. The property documents the contract for the backends and the routes listing.
5. Batches of one request run in order, one backend call each, under the `maxInflight`
   semaphore, as upstream's `read` loop does on its one thread. `modelTime` is the wall time inside
   those calls, wait for a permit included, so the Server-Timing `model` value counts the read as
   `test_server_timing_counts_the_read` expects. Nothing is read when every question is forced:
   no backend call, `inputTokens` 0, `modelTime` zero.
6. Every distribution a backend returns is checked (one per question, one value per option, every
   value in `[0, 1]`, which also rules out NaN and the infinities, sum within 1e-6 of 1) and a
   violation throws `BackendContractError`, a new error type
   that is neither a `SchemaError` nor an `OverloadedError`: a backend bug is not a client error,
   and `Answer.make`'s preconditions would otherwise crash the process on a bad backend. Two rows
   of `Fixtures/wire/answers.json` (`choice_layout`, `score_layout`) probe number rendering with
   vectors that are not distributions; the encoder engine test checks that they are refused, and
   compares the other thirteen byte for byte.
7. The queue bound is upstream's `waiting >= max_queue` as written, as D-027 item 5 decided for
   the diffusion engine: the issue's test line (`maxQueue 0` refusing the second concurrent
   request) is tested with `maxQueue` 1, and `maxQueue` 0 is checked to refuse the first. The
   message names the model: `"{model} is at capacity. Retry shortly."`.
8. `warmUp()` is a method the CLI and server call after load, never run by `init`: an actor's
   initializer cannot await the read, and a library user may not want it. Its questions are
   `EncoderDecisionEngine.warmUpQuestions`, upstream's `WARMUP_QUESTIONS`, against the state
   `warmup`, and the read does not count against the queue bound.
9. The option refusal (`UnsupportedOptions.check`), the queue counter (`RequestQueue`) and the
   answer reordering (`OrderedMap<Answer>.ordered(as:)`) were lifted out of `DecisionEngine` into
   `RequestAdmission.swift` and are shared by both engines; `DecisionEngine`'s behaviour and its
   `Fixtures/policies` tests are unchanged. The diffusion engine checks the options against its
   backend's `BackendCapabilities`, the encoder engine against `.readsOnly`.
10. `DecisionEngine` gains `decide(_:)` without a seed to satisfy `SystemOneService`; it forwards
    to `decide(_:seed:)` with `nil`, so the route's seed derivation applies. Its `servedModels` is
    `.diffusionGemma`. `EncoderEngineConfiguration` has no `servedModelVersion`: the encoder's
    version is its backend's model name.

Status. Proposed with issue #67.

## D-030 Server skeleton: where the port goes beyond or differs from the issue text

Context. Issue #34 builds the Hummingbird application: `ServerSettings`, the three routes, the
request id and `server-timing` headers and a `BackendProvider`. The error contract (#35),
authentication (#36), capacity and shutdown (#37) and model routes (#38) have their own issues,
and a few points needed choices the issue does not spell out.

Decision.

1. The fixture loaders, `FixtureTokenizer` and the two stub backends moved from
   `OpenJevCoreTests` into a new library target, `OpenJevTestSupport`, because test targets
   cannot import each other and the server tests need them. It is not a product. It does not
   import Testing: the loaders throw `FixtureError` where they used `#require`, and each test
   target turns `missingMessageText` into its own `Comment`. The iOS scheme builds it.
2. The issue's `ModelRegistry` is the core's `ServedModels` (D-029): `GET /v1/models` lists the
   service's `servedModels.listing`, and the unknown-model 400 uses its `accepts(_:)`. Routed
   models are not listed and not forwarded yet; `OPENJEV_MODEL_ROUTES` is parsed and validated
   at startup, and #38 uses it.
3. `ServerSettings(environment:)` reads the environment as upstream does, with one
   improvement. A missing variable is the default. An empty string is kept for a string
   setting, is the default for `_env_num`'s two MLX cache settings, and is refused for every
   other number, as Python's `int("")` is. Numbers parse as Python's `int` and `float` parse
   them (whitespace, sign, `_` between digits, `inf` and `nan`; hex floats refused). Where
   upstream raises a bare `ValueError` naming only the text, this port names the variable, with
   `_env_num`'s `{NAME}={raw!r} is not a int` wording. An integer beyond `Int` is refused as not
   an int, where Python would accept it. `OPENJEV_LOG_LEVEL` accepts uvicorn's level names plus
   swift-log's `notice`, case-sensitively. The default backend is `mlx`, not upstream's `vllm`,
   because this port has no vLLM backend. Settings that exist only for vLLM, CLM and JevK5 are
   left out.
4. `String.pythonRepr`, which `ImageValidation` already used, is public so the server's messages
   format `{value!r}` the same way.
5. The route applies upstream's order: body, shape, model name, questions cap, engine.
   `SchemaError` becomes the plain-detail 400 and `OverloadedError` the 529 with
   `retry-after: 1`. Any other error, which includes a backend failure until #35 maps
   it to the 503, is logged and answered as Starlette's plain-text 500 `Internal Server Error`.
   An unknown route is FastAPI's `{"detail":"Not Found"}` 404.
6. The body is read up to `OPENJEV_MAX_BODY_BYTES` in the route, not in a middleware ahead of
   authentication, and the 413 carries `server-timing`, which upstream's middleware answer does
   not. A body that is not JSON gets FastAPI's `json_invalid` 422 shape with the parser's byte
   offset and description; matching Python's character offset and `json` message, the
   content-type rules and the rest of `Fixtures/wire/cases.json`'s `body_*` rows is #35's.
7. `server-timing`'s `model` is `Decision.modelTime`, added to a task-local `ModelTimeRecorder`
   that the headers middleware reads. A request the engine refuses reports `model;dur=0.0`,
   where upstream would count any backend time it spent before refusing. `server` is clamped at
   `0.0` without producing `-0.0`, and each value is written with `%.1f`, which rounds the
   binary value as Python's `{:.1f}` does.
8. The server target declares `swift-http-types` directly, pinned to the version Hummingbird
   already resolved, because it names header fields. The Hummingbird files are wrapped in
   `#if canImport(Hummingbird)`, so `OpenJevServer` still compiles for iOS, where the manifest
   leaves Hummingbird out.

Status. Proposed with issue #34. Items 5 and 6 are completed by D-031 (issues #35 and #36): the
503 for a backend failure, the body cap in a middleware after authentication, and the
`json_invalid` messages, positions and content-type rules.

## D-031 Error contract and authentication: where the port goes beyond or differs from the issue text

Context. Issue #35 maps every failure to upstream's status, body and headers, and issue #36 ports
`check_auth`. Upstream answers through FastAPI, Starlette and CPython, whose behaviour the
fixtures record (`Fixtures/wire/cases.json`, `Fixtures/errors/cases.json`), and several points
needed choices the issues do not spell out.

Decision.

1. A body `JSONParser` refuses is scanned again by `PythonJSONLoads`, a port of CPython 3.14's
   scanner (`Modules/_json.c`), which reports only how `json.loads` ends: the first
   `JSONDecodeError` with its message and position, a `UnicodeDecodeError`, the `ValueError` of
   `int()` past 4,300 digits, or the `RecursionError` of nesting past the stack (item 2). The parser's own errors cannot give CPython's answer: the same
   offset can be "Expecting value" or "Expecting property name enclosed in double quotes"
   depending on the container, "Unterminated string starting at" points at the opening quote,
   and CPython reads `01`, `1.` and `1e` in part and fails on what follows. Positions count
   characters (Unicode scalars) after any byte order mark. `Fixtures/python-json/decode_errors.json`,
   written by `Tools/fixtures/python_json_tables.py`, records CPython's answer for 728 documents
   (128 chosen, 600 seeded mutations), and every one matches; a fuzz of 197,000 mutated documents
   against CPython 3.14.7 found no difference before the table was committed.
2. Where CPython accepts what the stricter parser refuses (D-016), the answer is the closest one
   CPython gives, at the place the parser stopped: "Expecting value" at `NaN`, `Infinity`,
   `-Infinity` or a float beyond `Double`, "Invalid \uXXXX escape" at a lone surrogate's `u`, and
   the 400 "There was an error parsing the body" for nesting deeper than the parser's 1,024
   levels. Upstream parses those bodies and answers from the value: a 422, then a 500 when the
   value cannot be written as JSON (`NaN` in `state`), or a read. An integer of more than 4,300
   digits is the 400, as upstream. CPython itself gives up on deep nesting with a
   `RecursionError`, which FastAPI answers with the same 400, at a level that depends on the
   version and the stack: 1,497 levels with CPython 3.12, which upstream's container runs, 9,998
   with 3.13, and with 3.14.7, which recorded the fixtures, where the thread's stack runs out:
   58,081 levels on an 8 MiB stack, the size of the Linux main thread uvicorn reads a body on,
   and 116,208 on macOS's 16 MiB main thread. `PythonJSONLoads` stops at 58,000 levels
   (`maximumNesting`), so a malformed body nested deeper is the 400, as upstream answers it
   there, and one nested between 1,024 and 58,000 levels gets CPython's message.
3. The body is read as FastAPI reads it. An empty body is no body, whatever its content type. A
   JSON content type is `application/json` or `application/` ending in `+json`, the media type
   read as `email.message` reads it: the text before the first `;`, stripped of Python's
   whitespace and lowercased, with exactly one `/` (a lone 0x85 or 0xA0 byte arrives as U+FFFD,
   item 6, and is not stripped). Any other content type, or none, hands pydantic
   the bytes: the 422 `model_attributes_type` whose `input` is Python's bytes repr cut at 500
   characters without a marker, as `trim` writes `str(value)[:500]`. `String.pythonRepr(bytes:)`
   writes it. A UTF-8 byte order mark is dropped, as `json.loads` decodes `utf-8-sig`.
   `json.loads` also reads UTF-16 and UTF-32, recognized by a byte order mark or by NUL bytes
   among the first four, and UTF-8 with encoded surrogates (`surrogatepass`); this port reads
   UTF-8 only and answers such bodies from their UTF-8 reading.
4. `AuthenticationMiddleware` and then `BodyCapMiddleware` run inside
   `ResponseHeadersMiddleware`, upstream's `check_auth(...) or await read_capped_body(...)`, so a
   refusal carries both request ids and `server-timing`. Upstream's middleware answers the 401,
   403 and 413 without `server-timing`, and `Fixtures/wire/cases.json` records
   `server_timing_present` false for exactly those rows (in `Fixtures/errors/cases.json` only
   upstream's two bare 500s lack it too); this server sets it on every response, as D-030 item 6
   did for the 413.
   The cap applies to `POST` only. `Content-Length` is read as Python's `int()` reads it
   (whitespace, a sign, `_` between digits, at most 4,300 digits counting leading zeros): past
   the cap is the 413 before a byte is read, a value `int()` refuses is ignored and the body
   counted, and an integer too large for `Int` is past the cap when positive. A body is counted as it arrives and refused at the first chunk
   that passes the cap. The route reads the collected buffer; `collect(upTo:)` keeps it bounded
   should it ever be mounted without the middleware.
5. Upstream authenticates and caps paths that start with `/v1/`. Hummingbird's router skips
   empty path components, so `//v1/models` reaches `GET /v1/models` without starting with
   `/v1/`; the middlewares also cover a path whose first component is `v1` with more after it.
   Upstream answers `//v1/models` with a 404; here it needs the key. Hummingbird does not
   percent-decode a path, so `/%761/models` is a 404 here, where Starlette decodes it to
   `/v1/models` and upstream authenticates it; neither serves it without the key.
6. A header is read as Starlette reads it: the first field of the name, one Latin-1 character per
   byte of the value that reaches the server. `removeprefix("Bearer ")` is the exact text, `strip()`
   removes Python's whitespace among the Latin-1 characters (U+0009 to U+000D, U+001C to U+0020,
   U+0085, U+00A0), `encode()` writes UTF-8, and the comparison runs over every byte up to the
   longer length with a length difference counted as one more mismatch, never returning early. What
   reaches the server differs from what reaches Starlette in two ways. NIO's HTTP/1 decoder reads
   each header value as UTF-8 and turns a byte that is not UTF-8 into U+FFFD, so a lone 0x85, 0xA0
   or Latin-1 letter arrives changed: a key padded with 0xA0 is refused where upstream strips the
   byte, and a non-ASCII setting can never be matched, where upstream accepts its Latin-1 bytes.
   Such a request is refused, never failed another way. A value in UTF-8, as in every recorded case,
   compares as upstream compares it. swift-http-types also drops the whitespace around a field
   value, as h11 does in front of upstream, so `" s3"` arrives as `s3` and a key of spaces as the
   missing key's 403; upstream's own tests, which call `check_auth` with such strings directly,
   would give 403 and 401 there.
7. A backend refuses a request with `BackendRefusal(reason:)`, next to `BackendContractError`,
   upstream's `Upstream`. The server answers `WireError.modelRejected400` with the reason's first
   500 Unicode scalars, as `str(msg)[:500]` keeps them. Any other error a service throws is the
   503 naming its Swift type, `String(describing: type(of: error))` for upstream's
   `type(e).__name__`, with `retry-after: 2`: that includes `BackendContractError`, a tokenizer's
   error and `CancellationError`. Upstream answers the 503 only for httpx's errors and lets
   anything else an engine raises become a 500; this port has no HTTP backend, and every
   failure of the model is the backend's. How a vLLM response becomes a refusal or a failure is
   not ported. The unknown route's 404 and the plain-text 500 for an error outside the service,
   such as a body that cannot be written, stay as D-030 has them. `WireError` gains
   `jsonInvalid422(message:position:)` and `unparsableBody400`.
8. `RefusalLog` writes upstream's `log_invalid` line, `{status} {request_id} {problems}` with the
   problems joined by `; `, at warning level, for the refusals upstream logs: each 422 (`loc:
   type` per item, `body.24: json_invalid` for a malformed body), the invalid-request 400 (every
   problem, each `union_tag_invalid` included, from `RequestValidator.problems(_:)`) and each
   plain-detail 400 (`loc: reason`, with `SchemaError.loc`, `body.questions` for the questions
   cap and `body` for a refusal). The body, the state and the instructions are never written.
   Upstream logs the invalid-request 400 with the status 422, its handler's default argument;
   here the line carries the 400 the client got. Authentication, the cap, a body FastAPI cannot
   read, an unknown model and a full queue are not logged, as upstream does not log them. A
   backend failure, which upstream does not log either, is logged at error level with the
   message of its 503, the type name only, since an error's description can hold request text.
9. The engine read a request's groups concurrently and threw whichever group's refusal came
   first. Upstream's `asyncio.gather` starts the groups in order and each resolves its template
   and prompt before its first `await`, so it always reports the first group's error; at canvas 8
   the quickstart was refused with either the first group's 8 tokens or the second's 9. Without
   a thought, every group is now prepared (template, prompt, label limit), concurrently, before
   any read, and the first group's error is thrown. With a thought, upstream's order depends on
   timing too, and nothing changed.
10. The server test target declares swift-log, for a capturing `LogHandler` given to
    `Application(logger:)`, and swift-nio, whose `NIOAsyncTestingChannel` lets a test send a
    request built by hand straight to the responder: without `Content-Length`, or with a body
    that fails the test if it is read. Both are pinned to the versions Hummingbird already
    resolved, so `Package.resolved` is unchanged. The recorded requests that pass validation are
    sent to a stub whose reads throw `ConnectError`, a test type named after httpx's error, so
    the 503 comes back byte for byte; the tokenizer fixtures never rendered some of those
    prompts, so the tests use `AnyPromptTokenizer`, which gives stand-in ids for those alone.

Status. Proposed with issues #35 and #36.

## D-032 DiffusionGemma configuration: where the port goes beyond or differs from the issue text

Context. Issue #23 decodes the checkpoint's `config.json` into
`DiffusionGemmaConfiguration` (`Sources/OpenJevDiffusionGemma/Model/Configuration.swift`), the
first production file of the port D-004 and D-014 confirmed. Every later file under `Model/`
reads it. Its defaults are mlx-vlm 0.6.15's `mlx_vlm/models/diffusion_gemma/config.py`, and a
few points needed choices the issue does not spell out.

Decision.

1. The type covers the whole file: the top level (with mlx-vlm's `ModelConfig` defaults, plus
   `tie_word_embeddings` true, `vision_soft_tokens_per_image` 280 and `initializer_range`
   0.02), `text_config` with every `TextConfig` field and default, `quantization`,
   `generation_config` and `vision_config`. `layer_types` and `rope_parameters` are derived as
   config.py lines 40 to 59 derive them when absent. A layer type other than
   `sliding_attention` and `full_attention`, in either key, is an error. The accessors the
   model files need (`fullAttentionLayers`, `headDim(for:)`, `keyValueHeads(for:)`,
   `ropeParameters(for:)`) follow language.py's `Attention`, including its fallbacks: a null
   `num_global_key_value_heads` gives `num_key_value_heads`, and a layer type missing from
   `rope_parameters` gives the default RoPE with theta 10,000.
2. Unknown keys are ignored at every level, as mlx-vlm's `_config_kwargs` drops them. The file
   carries `quantization_config` beside `quantization` with the same content; only
   `quantization` is read, because that is the key mlx-vlm and MLXLMCommon's
   `BaseConfiguration` read. It is decoded as `BaseConfiguration` decodes it: scalar keys are
   the default, object keys are per-module overrides, and a module set to `false` stays
   unquantized. An override without `mode` is `affine`, not the default's mode, as both
   MLXLMCommon and mlx-vlm (which hands the object to `to_quantized`) read it; issue #23's brief
   said it inherits, which would load such a module with the wrong mode under a non-affine
   default. `perLayerQuantization` gives MLXLMCommon's type for `loadWeights`.
3. `text_config` is required. mlx-vlm leaves a missing one `None` and fails later, far from the
   cause; here the error names `text_config`. Every error is a
   `DiffusionGemmaConfigurationError` whose description starts with the JSON key path, for
   example `text_config.hidden_size: expected a number`. `num_hidden_layers` below 1 is an
   error, where config.py would raise an `IndexError`, and so is a `layer_types` list whose
   length differs from `num_hidden_layers`: language.py builds layer i from `layer_types[i]`, and
   Transformers refuses such a list.
4. `vision_config` reuses mlx-swift-lm's public `Gemma4VisionConfiguration`: its keys and
   defaults match this checkpoint. It is not Equatable, so the configuration holds it in a
   private wrapper that compares its fields, which keeps equality synthesized.
5. mlx-vlm keeps `generation_config` as an untyped dict with no defaults, so every field of
   `DiffusionGemmaGenerationConfiguration` is optional. `load(from:)` also merges a
   non-empty `generation_config.json` as mlx-vlm's `load_config` does: it replaces
   `generation`, and its `eos_token_id` replaces the top-level one. For the pinned checkpoint
   both files hold the same object.
6. A verbatim copy of `config.json` cannot be a fixture, because every file under `Fixtures/`
   starts with a `generator` object. `Tools/fixtures/checkpoint_tables.py` writes
   `Fixtures/model/config.json` (`files` with the digests of the three files it reads, `config`
   and `generation_config` verbatim) and `Fixtures/model/weight_map.json` (`total_size`,
   `shards`, `weight_map`, for #27). Their generator records `model_repo` and `model_revision`,
   the names `Fixtures/oracle/reads.json` already uses for the checkpoint, and no upstream
   pins, because no upstream code is involved. `FixturePinTests` checks `model/` that way.

Status. Proposed with issue #23.

## D-033 Encoder packages: GitHub Releases, one asset per package file, checked by SHA-256

Context. D-011 item 5 decided that the converted Core ML packages are downloaded on first use,
checked by SHA-256 and compiled on the device, not bundled in apps, and spike #56's report names
the organisation's Hugging Face account as one place to publish them
([spikes/encoder-runtime.md](spikes/encoder-runtime.md), "Packaging"). Issue #57 needs a host, a
URL for every file of Verdict's package `verdict-m18-fp16` (306 MB in three files), and a
manifest the library embeds.

Decision.

1. **Host: GitHub Releases** of a dedicated public repository, `Algorythm-Canada/openjev-models`,
   one release per package version (`verdict-m18-fp16-v1`). It needs no new account and no Git
   LFS, and a release asset may be up to 2 GB. The manifest does not depend on the host: moving
   to the organisation's Hugging Face account later changes only the URLs that
   `Tools/encoders/manifest.py` writes.
2. **One asset per package file, not an archive.** `Manifest.json`,
   `Data/com.apple.CoreML/model.mlmodel` and `Data/com.apple.CoreML/weights/weight.bin` are
   uploaded under their paths with `/` replaced by `--`, since an asset name cannot hold a folder.
   An iPhone then needs no unzip and no second copy on disk, and a failed download is retried per
   file.
3. **The tokenizer and the calibrator come from the checkpoint**: tokenizer.json,
   tokenizer_config.json and calibrator.json of `heman10x/rlcd-modernbert-151m` on Hugging Face at
   the pinned revision `8af2496`, the files upstream reads. They are not re-hosted.
4. **The manifest is code.** `EncoderPackageManifest.verdict` lists every file with its URL, size
   and SHA-256. `Tools/encoders/manifest.py` writes it from the converted package and the Hugging
   Face cache, `--check` reports a stale one, and the script prints the `gh release create` and
   `gh release upload` commands (for `ghp` by default) that publish exactly those files. The
   embedded digests describe the package converted for spike #56 on the reference Mac.
5. **Storage.** `EncoderPackageStore` keeps the files in
   `Application Support/OpenJevSwift/encoders/{package}/`, excluded from backups: the package under
   `{package}.mlpackage/`, the tokenizer and the calibrator under `tokenizer/`. Each download goes
   to a temporary file and is moved into place only when its size and SHA-256 match the manifest. A
   mismatch is refused with an error naming the file, its URL and both digests. Every destination
   must resolve inside the package's folder with no symbolic link on the way, so a link left in the
   folder cannot redirect a checked download. `verified.json` records the digest each file was
   checked against, so a later launch downloads only what is missing or what a newer manifest
   changed, without hashing 300 MB at every launch. A package that needs a newer OS than the device
   runs is refused before anything is downloaded. `CompiledEncoderModel` compiles the package once
   with `MLModel.compileModel(at:)` and keeps `{package}.mlmodelc` beside it, compiling again when
   the package's files change.
6. **Local packages.** When `OPENJEV_ENCODER_MODELS` names a folder, the store downloads and checks
   nothing: the package is `{folder}/{package}.mlpackage`, as the converters write it, and the
   tokenizer and the calibrator are read from `{folder}/{package}/tokenizer/`, else from the
   checkpoint's snapshot in the Hugging Face cache. The OS check applies there too. The library
   never reads the environment itself: `EncoderPackageStore(environment:)` takes the one the CLI
   passes.

Publishing is a manual step after review: `python3 Tools/encoders/manifest.py` rewrites the manifest
(unchanged for the spike's package) and prints the commands. They create the repository once, with
the Apache-2.0 license as its first commit, since a release needs a commit to tag; then they copy
the three files under their asset names, create the release and upload them. Once the uploaded
assets match the manifest, `PACKAGE_DOWNLOADS_ENABLED` in the script turns the downloads on, and the
script runs again.

Alternatives rejected. (a) The organisation's Hugging Face account, the spike report's suggestion:
it needs a new account, where GitHub needs none; it stays open for a later move, since only the URLs
change. (b) An archive per package: an unzip on iOS and a second copy on disk while it expands. (c)
Git LFS in this repository: the repository never holds weights (CONTRIBUTING.md). (d) The package in
the app bundle: rejected by D-011.

Consequences. Until the release exists, the embedded manifest has `packageDownloadsEnabled` false,
and the store refuses with `EncoderPackageError.packageDownloadsUnavailable` instead of requesting
assets that are not there; a deployment sets `OPENJEV_ENCODER_MODELS` in the meantime. A changed
package needs a new release tag and a new manifest, and a published asset is never replaced.
`openjev-models` should carry the Apache-2.0 license and a notice crediting Heman10x's checkpoint,
and Laya's authors once #58 publishes its packages the same way.

Status. Proposed with issue #57; the host needs the maintainers' confirmation.

## D-034 Verdict backend: where the port goes beyond or differs from the issue text

Context. Issue #57 predates spike #56 and D-011, and describes a backend that batches like
upstream's PyTorch one. A few points follow the spike instead, and a few needed choices the issue
does not spell out.

Decision.

1. **No CLI registration yet.** The issue asks for `OPENJEV_BACKEND=verdict` in the CLI, and
   `openjev serve` is issue #40. This issue exposes `VerdictBackend.load(from:)` and
   `load(configuration:)`, and 05-architecture.md documents the one-line
   `QuestionReadBackendProvider` registration.
2. **Rows per Core ML call follow D-011, not a fixed 16.** `EncoderDecisionEngine` still hands the
   backend batches of `OPENJEV_ENCODER_BATCH` questions. The backend runs them one question per
   call on iOS, through the batch-1 functions on the Neural Engine, and up to 16 per call on
   macOS, through the batch-16 functions on the GPU, in the smallest function that holds the rows
   and the longest row. Rows are padded to that function's length (128, 256 or 512 tokens), where
   upstream pads to the longest row; the billing, the rows' unpadded lengths, is the same.
3. **Where the files come from.** The issue's download from `heman10x/rlcd-modernbert-151m` holds
   for the tokenizer and the calibrator; the weights are the Core ML package of D-033.
4. **Compute units.** `EncoderComputeUnits` has no `.all`, so D-011's rule against it holds by
   type; `.cpuOnly` remains for tests.
5. **Tolerance.** The calibration runs in float32, as upstream's does, and widens only the result to
   double, so it underflows where upstream's does. At temperature 2, the logits `[0, 1, 400]` fall
   back to the uniform answer and `[0, 1, 200]` keep their subnormal exponentials, and both answers
   equal upstream's PyTorch; double arithmetic answers neither. The uniform answer is upstream's
   Python double, `1.0 / (k - 1)`. The Core ML acceptance bounds are the spike's scope notes rather
   than the issue's 0.007 (bfloat16 moved probabilities by up to 0.0115 on this corpus): the largest
   difference at most 0.02, the mean at most 0.003, and the top answer unchanged wherever the
   reference's top two are at least 0.01 apart.
6. **A model's wrong output is an error.** A model that returns the wrong number of rows, or fewer
   logits than a question's labels, throws `EncoderModelError`, which the server answers as a
   backend failure, where upstream would raise from the same place or broadcast silently. A score
   with no levels, which only a request built in code can hold, gets an empty distribution, and
   the engine refuses it with `BackendContractError` rather than stopping the process.
7. **The core gains `Double.pythonRepr`**, CPython's `repr(float)`, for the score labels'
   `float(i)`. It lays out finite values as the JSON writer does and is checked against the whole
   `python-json/float_repr.json` table.
8. **The loaders check the OS when they run.** The Core ML types are
   `@available(macOS 15, iOS 18, *)`, as D-011 item 4 says, but
   `VerdictBackend.load(configuration:)` and `load(from:)` are not: they throw
   `EncoderLoadError.unsupportedOperatingSystem` on an older OS. The CLI and the server build for
   the package's macOS 14 floor, and this keeps their registration one line.

Status. Proposed with issue #57.

## D-035 DiffusionGemma text blocks and weight loading: where the port goes beyond or differs from the issue text

Context. Issues #24 and #27 port mlx-vlm 0.6.15's text blocks (`language.py` lines 23 to 330)
and its weight loading (`diffusion_gemma.py` lines 346 to 401) into
`Sources/OpenJevDiffusionGemma/Model/`, on the configuration of D-032. The reference is spike
#22's transliteration (`Tools/oracle/UpstreamProbe/Sources/Transliteration/Model.swift`), which
matched mlx-vlm bit for bit; the library keeps its operations, order, shapes and dtypes, under the
checkpoint's module names.

Decision.

1. **What the blocks reuse from mlx-swift-lm (D-004), and what they do not.** Reused: MLXNN's
   `Linear`, `Embedding`, `RMSNorm` and their quantized forms; `MLXFast.RoPE` with explicit
   frequencies, `MLXFast.rmsNorm` with `MLXArray.mlxNone` and `scaledDotProductAttention`;
   mlx-swift-lm's `SwitchLinear`, `QuantizedSwitchLinear`, `gatherSort` and `scatterUnsort`
   (the experts sort at 64 assignments or more, as `switch_layers.py`); and
   `loadWeights(modelDirectory:model:perLayerQuantization:)` with `BaseConfiguration`'s per-layer
   map. Not reused: `Gemma4TextRouter` (it folds the scale into the norm weight and uses a plain
   softmax, which rounds differently), `Gemma4TextExperts` and `SwitchGLU` (the checkpoint fuses
   gate and up into one 1,408-output `gate_up_proj`), `ProportionalRoPE` (it rotates a slice;
   mlx-vlm rotates the whole head with infinite frequencies), and Gemma 4's fused norms and
   compiled expert sum. The GeGLU and the float32 softcap are compiled shapeless, as mlx-vlm
   compiles them.
2. **Strict loading.** `DiffusionGemmaModel.load(from:)` first reads only the shard headers:
   it applies `sanitizedName(_:)`, quantizes the lazy tree where the checkpoint (or its index)
   has a module's `.scales`, as `loadWeights` decides, and requires the tree's parameters and the
   tensors to be the same names with the same shapes. A mismatch is a
   `WeightLoadingError.coverage` whose description names the first missing tensor with the shard
   the index places it in, the first unexpected tensor with its shard, and the first shape
   mismatch. Only then does `loadWeights` read the 16.5 GB, update with `verify: .all` and
   evaluate. A shard the index names but the directory lacks makes its tensors missing;
   `loadWeights` alone would have fallen back to other files.
3. **Sanitize is text-only.** The tree has no vision tower, so `model.encoder.vision_tower.*`
   and `model.encoder.embed_vision.*` are dropped with `rotary_emb`, `lm_head.weight` and the
   encoder's non-scalar text weights; the bare-expert rename is kept, a no-op for this
   checkpoint. Of the 1,647 tensors, 358 are dropped (355 vision tower, 3 embedder) and 1,289
   load. The vision path keeps mlx-vlm's `.linear.` names when it arrives. The text tree has 299
   quantized modules; the checkpoint has 300 `.scales`, the 300th being the embedder's.
4. **The lazy coverage check works.** The real-size tree built from Fixtures/model/config.json
   and quantized adds 26 MB of resident memory, because MLX arrays are lazy, so the model-free
   test compares its parameter names and shapes with the sanitized weight map directly. No
   fallback from the configuration was needed.
5. **The exact-tier hook.** `Attention.fullAttentionFrequencies` exposes the full-attention RoPE
   table, computed with MLX `pow` at init as mlx-vlm computes it at load time; the exact tier
   installs the oracle's `rope` table there. The test helper `MetalLibrary.configure()` takes
   `OPENJEV_MLX_METALLIB` when set, and inside Xcode's test host it takes effect: with the wheel's
   metallib and the oracle's table, layers 0 to 5 of a one-piece prefill are bit-identical to
   mlx-vlm on all 37 recorded stages of both stage dumps (quickstart/g0, 182 tokens;
   indexed_12_mixed/g0, 1,572 tokens with the window mask). Under native kernels the stage
   bounds are twice the measured differences, which grow with depth as last-bit router changes
   flip experts.
6. **The self-conditioning placeholder.** `model.decoder.self_conditioning` holds its four
   modules' parameters so that the tree's key set is the checkpoint's; its forward pass is #28's.
7. **The one-piece prefill** (`prefill(_:)` and `prefill(embeddings:layers:stages:)`) runs
   every layer in encoder mode at offset 0 with the encoder's scalars and `.causal` masks, or the
   boolean band for sliding layers past the window, and returns one `LayerCache` per layer. The
   stage observer, a closure passed in, replaces the transliteration's global recorder. #25 owns
   the prefill API.
8. **Concurrency.** The new value types (errors, metrics, load stages, the checkpoint header
   table) are Sendable. The module tree, the caches and `LoadedModel` hold MLX arrays and are not;
   their callers serialise them, as the transliteration's did. The MLX test suites are nested in
   one serialized parent suite.

Status. Proposed with issues #24 and #27.
