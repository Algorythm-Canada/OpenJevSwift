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
`json_invalid` messages, positions and content-type rules. Item 7's zero for a refused request is
replaced by D-038 item 7 (issue #37): the engines record the time they spent. Item 2's routed
models are forwarded and listed by D-040 (issue #38).

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

Consequences. The release `verdict-m18-fp16-v1` was published on 2026-10-01 and its three assets
match the embedded manifest, so `packageDownloadsEnabled` is true and the store downloads the
package on first use; a deployment without network sets `OPENJEV_ENCODER_MODELS` instead. A changed
package needs a new release tag and a new manifest, and a published asset is never replaced.
`openjev-models` carries the Apache-2.0 license and a NOTICE crediting Heman10x's checkpoint and
Laya's authors.

Laya (#58, D-037) adds five releases of the same shape: `laya-m18-fp16-v1`, the Mac's package
(849 MB), and `laya-f18-b1s128-fp16-v1`, `laya-f18-b1s256-fp16-v1`, `laya-f18-b1s512-fp16-v1` and
`laya-f18-b1s1024-fp16-v1`, the iPhone's (843 to 845 MB each). Their tokenizer and
rl_agent_config.json are those of `convaiinnovations/laya-typed-decisions` at `1a793eb`, the
tokenizer under the checkpoint's `tokenizer/`, and are not re-hosted. `EncoderPackageManifest.laya`
and `layaByLength` describe them; `Tools/encoders/manifest.py --model laya` writes both and prints
each release's commands. The five releases were published on 2026-10-01, each asset verified by
download against its manifest, and every Laya manifest has its downloads on. The repository's
NOTICE credits Heman10x's Verdict checkpoint and Laya by Nandakishor M / Convai Innovations
(github.com/NandhaKishorM/laya).

Status. Accepted on 2026-10-01: GitHub Releases of `Algorythm-Canada/openjev-models`, confirmed by
the maintainers, with `verdict-m18-fp16-v1` and Laya's five releases published and their assets
verified by download.

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

## D-036 DiffusionGemma prefill, decoder read pass and multi-step reads: where the port goes beyond or differs from the issue text

Context. Issues #25, #26 and #28 port the read path of mlx-vlm 0.6.15's DiffusionGemma onto the
blocks of D-035: the one-piece text prefill (`language.py` lines 555 to 771 and the Backbone's
`diffusion_prefill_cache`), the decoder pass with its masks and self-conditioning (`language.py`
lines 351 to 553, `diffusion_gemma.py` lines 183 to 202), and upstream's `MlxRuntime.read`
(`mlx_backend.py` lines 175 to 208), into `Sources/OpenJevDiffusionGemma/Model/`
(`SelfConditioning.swift`, `Prefill.swift`, `DecoderPass.swift`, `Read.swift`). The reference is
spike #22's transliteration, bit-identical with mlx-vlm on all 27 oracle reads.

Decision.

1. **The prefill is one piece, and reads never chunk.** `prefill(promptIDs:)` runs the encoder
   over the whole prompt, as `MlxRuntime._prefill` calls `diffusion_prefill_cache` without
   `chunk_prefill`, evaluates the caches and returns a `PromptCache` (the per-layer
   `LayerCache`s, the offset, the prompt token count) that every read of the prompt shares and
   none writes. A chunked prefill is exact in real arithmetic, but in bfloat16 it moves read
   probabilities by up to 0.62 and fails 0.02 on 67 of 156 slots (spike #22), so the issue's
   chunked prefill and its "chunked and unchunked give the same logits" criterion are not
   implemented. The prefill refuses an empty prompt and ids outside the vocabulary with
   `ReadInputError`; the read refuses an empty canvas, canvas or label ids outside the
   vocabulary, no slots, a slot outside the canvas or without labels, `steps` below 1, `topK`
   outside `1 ..< vocab` and a cache from a model with another layer count. Upstream sends none
   of these, so `read` throws, a departure from the signature in the brief.
2. **The issue's other hooks are documented, not built.** Image placeholders replaced by `pad`
   with the vision features scattered after the embedding, and the bidirectional overlay over
   each image's block, belong to the vision milestone. Chunked prefill for prompts that do not fit
   one pass and `diffusion_update_cache` (appending committed tokens, which needs a cache that
   takes several updates and a sliding cache that rotates in place) belong to generation,
   milestone 5. The issue's `KVCacheSimple` and `RotatingKVCache(maxSize: 1024)` are D-035's
   `LayerCache`, which keeps every position after a one-piece prefill, as mlx-vlm's caches do;
   no static prefix cache is built. `decoder_attention_mask` (padded prompts) is not ported:
   reads run one unpadded prompt.
3. **What `post_norm` is.** The checkpoint has `self_conditioning.pre_norm.weight` and no
   `post_norm` tensor because `post_norm` is `RMSNormNoScale`, a norm without a weight. The
   module is `post_norm(embeddings + down_proj(geglu(gate_proj(pre_norm(signal)),
   up_proj(pre_norm(signal)))))`. On the first step the signal is zeros and the module runs, as
   mlx-vlm runs it; skipping it is a measured bug (spike #22).
4. **Self-conditioning for the quantized embedding.** `decoderLogits(canvas:cache:conditioning:masks:)`
   takes nil on the first step and the previous step's full float32 logits afterwards. With the
   checkpoint's 8-bit `QuantizedEmbedding` (mlx-vlm's `prefers_logits_self_conditioning`), the
   signal is `softmax(logits, precise: true)` cast to the embedding dtype, `quantizedMM` against
   the embedding's weight, scales and biases with `transpose: false` and its group size, bits and
   mode, cast to the embedding dtype, times the embedding scale. A dense embedding takes
   `diffusion_self_conditioning`'s path (the logits cast to the weight's dtype, the precise
   softmax, `probs @ weight`, the scale), which only the tiny synthetic model exercises. The
   masks are made once per read and the canvas's RoPE offset is the prompt length. Between steps
   only the slot positions take their argmax; the template is never overwritten.
5. **The slot-only projection (D-015) failed its condition, so reads project every row.**
   `decoderSlotLogits` takes the slot rows of `norm(h)` through the tied head. The smaller
   quantized matmul rounds differently: on the 27 oracle reads its final-step maps were identical
   to the full projection's on 29 of 156 slots in the exact tier and 40 under native kernels,
   with logprobs up to 0.16 apart. `read` therefore defaults to `SlotProjection.full`;
   `.slotsOnly` stays for measurement. D-015 remains allowed in principle but is unproven for
   this checkpoint.
6. **The read output and #29.** `read(canvas:slots:cache:steps:topK:projection:)` returns
   `ReadOutput`: per slot the `(token id, logprob)` pairs of the top 20 and every label, sorted by
   token id, each logprob the float32 log-softmax value widened to Double (what `MlxRuntime.read`
   returns and Python's `tolist()` gives); the argmaxes written between steps; and the prompt
   token count, which steps do not change. `ReadOutput.readResult(for:)` hands the maps and the
   slots' label ids to `ReadResult(tops:labelIDs:promptTokens:)`, which applies
   `SlotDistribution.compute` as upstream's `one_read` applies `slot_distribution`; #29's
   runtime returns that to the engine.
7. **Measured on 2026-10-01 (Apple silicon Mac, the pinned checkpoint).** Exact tier (the
   wheel's `mlx.metallib`, SHA-256 as in the oracle's generator, through a temporary Xcode scheme
   setting `OPENJEV_MLX_METALLIB`, and the oracle's RoPE table): 27 of 27 reads bit-identical,
   all 72 cache digests equal (layers 0 and 29 of the 12 prompts, and the sliding layer's
   decoder view), the written argmaxes equal on 6 of 6 multi-step reads, every prompt token
   count equal. Native kernels: mean absolute label probability difference 0.0084 over 1,763
   labels (bound 0.02) and 0.0054 over the 50 long-prompt slots (bound 0.01); mean absolute
   entropy difference 0.086 over 156 slots and 0.131 over the long prompts (bound 0.2 each); top
   label 150 of 156 (96.2%, bound 90%) and 120 of 120 where the oracle's margin is at least 0.5
   (bound 97%); written argmaxes equal on 4 of 6. These are the spike transliteration's figures
   to the reported precision. A cached and a cold prefill give bit-identical reads in one process.
8. **Concurrency.** `PromptCache` and the model hold MLX arrays and are not Sendable; the caller
   serialises them, as before. `SlotRequest`, `ReadOutput`, `TensorDigest` and `ReadInputError`
   are Sendable values. The live suites share one model load through `LiveCheckpoint` in the
   test support.

Status. Proposed with issues #25, #26 and #28.

## D-037 Laya backend: where the port goes beyond or differs from the issue text

Context. Issue #58 predates spike #56 and D-011, and describes the forward pass and the download
as upstream's PyTorch `LayaEngine` has them. The runtime, the packages and the bounds follow the
spike instead, in the shape D-033 and D-034 gave Verdict, and a few points needed choices the issue
does not spell out.

Decision.

1. **No CLI registration yet,** as for Verdict (D-034 item 1): `OPENJEV_BACKEND=laya` is issue
   #40. This issue exposes `LayaBackend.load(from:packageSet:)` and `load(configuration:)`, and
   05-architecture.md documents the one-line registration beside Verdict's.
2. **The forward pass is the Core ML package's.** The encoder, the question-type embedding, the
   two-layer head and the scorer run inside the converted package (`convert_laya.py`), which
   returns the scorer at every position; Swift reads it at the markers and applies the bucket's
   temperature, the clamp, the softmax, the 4-decimal rounding and the renormalisation. The action
   head is not converted: upstream drops its output.
3. **Packages and batches follow D-011.** On macOS the backend runs `laya-m18-fp16` on the GPU,
   up to 16 questions per call in the smallest function that holds them, with two functions
   loaded. On iOS it runs `laya-f18-b1s128-fp16` to `laya-f18-b1s1024-fp16` on the Neural Engine,
   one question per call, each through the smallest package the device holds that takes its
   sequence. A package loads when a read first needs its length and stays loaded, because the
   first load of a Laya package took 33 to 56 s on an A15. A sequence longer than every package
   the device holds is `EncoderLoadError.noPackage`, which names the package to fetch; the app
   sends that read to a server meanwhile (D-011 item 6) and fetches the package with
   `prefetch(lengths:)`, which downloads, checks and compiles it; a read that needs a package a
   prefetch is compiling waits for that compile. `LayaBackend.Configuration`'s defaults follow the
   package set rather than the platform: the GPU for the multifunction package, which Core ML does
   not load for the Neural Engine, and the Neural Engine with one question per call for the
   per-length packages. Rows are padded to the function's or the package's length, where laya pads
   to the longest row; the billing, the rows' unpadded lengths, is the same.
4. **Where the files come from.** The issue's download from
   `convaiinnovations/laya-typed-decisions` holds for the tokenizer, under the checkpoint's
   `tokenizer/` (`EncoderPackageManifest.checkpointTokenizerFolder`), and for
   rl_agent_config.json, the calibration file; the weights are the Core ML packages of D-033.
   `EncoderPackageStore.tokenizerLocations(for:)` fetches the tokenizer and the configuration file
   without a package, also while the packages are unpublished, so an iPhone can build sequences
   before it holds any package, and `heldPackageDirectory(for:)` tells which packages it holds
   without downloading anything. Each per-length package keeps its own copy of the tokenizer
   (3.6 MB) in the store's layout of one folder per package.
5. **laya's arithmetic, not an approximation of it.** laya computes the softmax in float32 with
   numpy, and the port does the same in numpy's order: the division by the temperature, the
   maximum subtracted, `exp`, and numpy's pairwise sum. On the recorded scores it reproduces all
   200 of laya's unrounded distributions bit for bit, so the rounded answers are laya's own. The
   rounding is CPython's `round(x, 4)` done with integers (the exact binary value rounded half to
   even), and the renormalisation divides by Python's compensated sum: `pythonSum` (D-019) becomes
   public in `OpenJevCore`.
6. **Tolerance.** The issue's 0.021 (upstream's bfloat16) gives way to the spike's bounds, as for
   Verdict (D-034 item 5): the largest difference at most 0.02, the mean at most 0.003, and the
   top answer unchanged wherever the reference's top two are at least 0.01 apart, on the
   probabilities before rounding. Float16 moves them by up to 0.004 on the Mac's GPU and 0.015 on
   its Neural Engine (0.018 on an A15's, spike #56), so the rounded answers differ from PyTorch's
   in most questions while the top answers hold.
7. **The configuration file is checked at load.** `temperature` must hold a value per question
   type (laya would fail at read time with an `IndexError`), and a `max_len` longer than the
   packages take (1,024) is refused. `clamp_temperature` gives 1 for a string, where Python's
   `float` would parse one that holds a number.
8. **One tokenization of the state per batch.** laya tokenizes the state again for every
   question; the ids are the same, and on an iPhone tokenizing costs 9 ms per question at the
   median (spike #56).
9. **The overflow refusal comes before any read.** A question whose options lose a marker gets
   upstream's `"Too many choices for laya-1.0: a question's options must fit in 256 tokens."`,
   located at `["body"]`, before the batch is read, as laya builds every sequence of a call before
   its forward pass. A model's wrong output is `EncoderModelError` and a score with no levels gets
   an empty distribution that the engine refuses, as D-034 item 6 decided for Verdict. The loaders
   check the OS when they run, as D-034 item 8 has it, and `load(from:)` does so before it asks the
   store, so an older OS gets `EncoderLoadError.unsupportedOperatingSystem` rather than the store's
   `EncoderPackageError`; `VerdictBackend.load(from:)` gains the same check.
10. **"[MASK]" is matched as Python matches it,** code point by code point: a combining mark
    after the `]` does not hide the token, as Swift's character comparison would.

Status. Proposed with issue #58. Item 3's two functions loaded on macOS are replaced by D-042:
every function stays loaded, and `OPENJEV_ENCODER_FUNCTIONS` caps them.

## D-038 The openjev CLI, capacity and shutdown: where the port goes beyond or differs from the issue text

Context. Issue #40 builds the `openjev` command line tool and issue #37 finishes the server's
capacity, model-time and shutdown work. Both predate the Verdict backend (#57) and the
DiffusionGemma runtime (#29), which has not landed, and several points needed choices the issues
do not spell out.

Decision.

1. **Backends.** `OPENJEV_BACKEND` selects from the CLI's `BackendRegistry`: `verdict` and
   `laya` on Core ML (macOS), Laya registered once #58 merged into this branch, and `mlx`, which
   upstream has and this build does not yet: it exits 3 with a message naming the variable and
   issue #29. Every other name, upstream's `vllm`, `clm` and `jevk5` included, is upstream's
   `create_app` error with this port's list and the variable named as `__post_init__` names it:
   `unknown backend 'vllm'; use one of mlx, laya, verdict (OPENJEV_BACKEND)`, exit 2. The check
   is the CLI's rather than `ServerSettings`', so tests register stub backends in the registry.
   On Linux `verdict` and `laya` are known and exit 3. The issue's `mlx` acceptance criterion,
   the 4-bit checkpoint serving the README example, waits for #29; the live smoke test serves
   Verdict, and no test loads Laya, whose package is 810 MB.
2. **Flags.** `serve` takes `--host`, `--port`, `--backend`, `--log-level` and `--no-warmup`,
   each written over its variable before `ServerSettings(environment:)` reads it, so a flag's
   value is checked and refused with the variable's message (`OPENJEV_PORT='abc' is not a int`).
   `--shutdown-timeout` (30 seconds) has no variable, since upstream has none. `decide` and
   `models` take `--backend`; the issue's `decide --model <source>` is `--backend`, because the
   files a backend loads come from its own variables (`OPENJEV_ENCODER_MODELS`,
   `OPENJEV_MLX_MODEL`).
3. **Exit statuses.** 0 for success and, for `serve`, a clean shutdown; 1 for any other failure
   (an unreadable request file, a backend that fails during `decide`, an address in use, a
   shutdown that had to cancel requests); 2 for invalid settings and for a command line the
   parser refuses, whose status 64 becomes 2, as Python's argparse exits; 3 for a backend this
   build lacks or that failed to load, with the error's description; 4 for a request `decide`
   was refused, which the server answers with a 4xx or the 529. Messages go to standard error
   prefixed `openjev: `. launchd and docs/deployment.md read these.
4. **decide and models.** `decide` prints the 200's body exactly, without a trailing newline,
   through `SystemOneHandler`, which the route calls too. The body is read as a JSON body, under
   the body cap's 413. An error prints the server's error body to standard error, exactly.
   `decide` skips the warm-up read, which only delays its one read. `models` prints the listing
   of every known backend without loading a model, `mlx` included; a backend whose
   listing only the loaded model knows is loaded, as the test registry's stub encoder is.
5. **Phases and the request log.** `serve` logs to standard error through swift-log: the
   settings line, with the API key and the origin secret as `set` or `unset` and the model routes
   by name; `loading {model}`; `warming up`, from a new callback of `QuestionReadBackendProvider`;
   `serving on host:port`, with the bound port, so port 0 works; `shutting down`,
   `released {model}` and `stopped`. Hummingbird's own `Server started and listening` line stays.
   `RequestLogMiddleware` runs outside every other middleware and writes
   `{method} {path} {status} {ms}ms {request id}` at info for every request, refusals included,
   in the place of uvicorn's access log, which also writes the client's address and the query
   string; this one writes neither, nor a body or a header value.
6. **The queue bound of 0.** The issue's acceptance line ("with `maxQueue` 0 and a slow stub, a
   second concurrent request gets 529") is `maxQueue` 1 under upstream's
   `waiting >= max_queue`, as D-027 item 5 found. The server test holds the first request at a
   gate with `maxQueue` 1, expects the 529 with `retry-after: 1` for the second and a 200 for the
   first, and checks that 0 refuses the first request too.
7. **Model time.** `ModelTimeRecorder` moves from the server to `OpenJevCore`, a task-local
   reference as upstream's `model_ns` is a context variable holding a list. The engines add each
   backend call, the wait for a permit included, when the call ends, whether it returned, threw
   or was cancelled, as upstream's `_post` adds in its `finally`. A request refused after a
   thought or a read, or failed after a batch, now reports that time, where D-030 item 7 reported
   zero, and the route no longer records `Decision.modelTime`, which stays for library callers.
   A forwarded request will add the other server's time when #38 forwards it.
8. **Cancellation.** A client that closes its connection cancels its decision. A channel handler
   on each connection sees the end of the client's input, and the route runs the decision in a
   child task beside a watch of the connection; the request log shows 499, nginx's code, the
   answer is an empty 499 that only a client that half-closed and still reads receives, and
   nothing is logged as a failure. A client that half-closes and still waits counts
   as gone, as for most HTTP servers. Upstream runs a request to its end. The encoder engine
   starts no batch for a cancelled request and Verdict no further Core ML call; a call in
   progress finishes. A decision the server cancels while stopping is the 503 naming
   `CancellationError`, D-031 item 7's mapping. Request handling creates no unstructured task.
9. **Shutdown and release.** `DecisionServer` is a swift-service-lifecycle `Service`; the CLI
   runs it in a service group with SIGINT and SIGTERM as its graceful shutdown signals and
   `--shutdown-timeout` as the group's `maximumGracefulShutdownDuration`. A second signal does
   not cut the wait short. The model is released afterwards through `ModelReleasing.close()`,
   upstream's `close()`, a protocol a service or backend adopts when it has something to release:
   the two engines pass it on, `VerdictBackend` and `LayaBackend` pass it to their Core ML
   runner, and `CoreMLEncoderModel` and `CoreMLPackagesByLength` drop what they have loaded. A server cancelled while it is answering requests throws
   `ShutdownInterrupted`, exit 1; one cancelled with nothing in flight stops cleanly, so a
   `--shutdown-timeout` of 0 exits 0 when no request is running. The timeout is at most a day,
   since a larger `Duration` overflows. `OpenJevApplication.make(settings:provider:)` is replaced by
   `application(settings:service:logger:onServerRunning:)` and `DecisionServer`.
10. **Tests.** The CLI's code stays in the executable target, and `OpenJevCLITests` imports it
    with `@testable import openjev`, which the native build system and Swift Build both build.
    The commands read a task-local `CommandContext` (environment, streams, backends, loggers,
    shutdown signals), so tests run them in-process with stub backends. Child-process tests of the
    built binary check the exit statuses and `models` in CI. A stub backend compiled into the
    shipped binary would have let a child-process test run `decide` without a model; instead the
    opt-in smoke test serves Verdict from the binary, and `decide` against a stub is tested
    in-process. `ReadGate` in `OpenJevTestSupport` holds a stub's calls until a test opens it,
    and the stubs record their own call times, count `close()` and can fail after a number of
    calls.

Alternatives rejected. (a) Checking `OPENJEV_BACKEND` in `ServerSettings`: the library would
have to know every backend name, a test could not register a stub, and upstream checks it in
`create_app` too. (b) A library target for the CLI's code: the executable target tests under both
build systems, and the issue places the code in `Sources/openjev`. (c) Leaving a client that goes
away unobserved, as uvicorn does: issue #37 asks that its reads be cancelled. (d) Keeping
ArgumentParser's 64 for a refused command line: one status for every invalid input keeps the
table short, and it is what upstream's Python tooling exits with.

Consequences. launchd's `ExitTimeOut`, whose default is system-defined, must exceed
`--shutdown-timeout` or launchd may kill the server before its requests finish;
docs/deployment.md sets it. The
DiffusionGemma backend registers with one entry in `BackendRegistry.standard` and drops its
placeholder, as Laya's did.

Status. Proposed with issues #40 and #37. Item 7's forwarded request reports the routed server's
time as model time since D-040 (issue #38).

## D-039 DiffusionGemma runtime and model download: where the port goes beyond or differs from the issue text

Context. Issue #29 ports upstream's `MlxRuntime` and `MlxEngine` (`mlx_backend.py` lines 76 to
303) as `DiffusionGemmaRuntime`, and issue #30 adds model resolution and download, into
`Sources/OpenJevDiffusionGemma/Runtime/` (`DiffusionGemmaRuntime.swift`, `PrefillCache.swift`,
`RuntimeConfiguration.swift`) and `Sources/OpenJevDiffusionGemma/Download/` (`ModelSource.swift`,
`ModelResolver.swift`), on the model of D-035 and D-036.

Decision.

1. **One actor is the MLX boundary.** `DiffusionGemmaRuntime` owns the loaded model, the
   `SwiftTransformersTokenizer` and the prefill cache, and runs every MLX evaluation inside
   itself, one at a time, as upstream's one-worker executor does. It runs on the default actor
   executor rather than a dedicated thread; R14 asks for one execution context, which a serial
   actor is. Its `DecisionBackend` properties are `nonisolated`, and only values cross it. The
   model is reached through two stored closures (prefill and read), so the model-free tests bind
   them to a stub whose prefill is a `PromptCache` without layers.
2. **The prefill cache is generic.** `PrefillCache<Value>` is upstream's `OrderedDict` with its
   two budgets and running token total; the runtime instantiates it with `PromptCache`, and the
   eleven cache and settings tests of `tests/test_mlx_backend.py` run over it and a stub runtime
   without weights or MLX. Its key, `PrefillKey`, has one case, `.tokens([Int])`; the image key
   (system text, state text, image digests) arrives with the vision milestone. Upstream's two
   module constants are `PrefillCacheDefaults`. The token budget is a configuration value too,
   which upstream does not expose.
3. **Settings are typed.** `DiffusionGemmaRuntime.Configuration` holds `maxPromptTokens`
   (32,768), `promptCacheEntries` (12), `promptCacheTokens` (16,384), `cacheLimitGB` (nil leaves
   MLX alone, 0 disables the pool, otherwise `Memory.cacheLimit` set to `gb × 1024³` bytes, the
   non-deprecated form of `GPU.set(cacheLimit:)`); `nan`, `inf`, a negative value or one too large for an `Int` of
   bytes is refused with `DiffusionGemmaRuntimeError.invalidCacheLimit` before anything loads,
   since `OPENJEV_MLX_CACHE_LIMIT_GB` accepts `nan` as upstream's does and `warmUp` (on). The module cannot import
   `OpenJevServer`, so the CLI maps `ServerSettings` onto the memberwise initializer (05, "The
   server"). The runtime checks the prompt cap itself, as `MlxEngine.one_read` does, although the
   engine checks it first too.
4. **The think stub.** `capabilities` is steps, samples and sequential; `think` and images are
   off until milestones 5 and the vision milestone, so the engine refuses them with upstream's
   messages before any read. `think(prompt:budget:stopIDs:)` and an image prompt throw
   `DiffusionGemmaRuntimeError.unsupported`. `generate` is not declared: no protocol asks for it
   yet.
5. **Warm-up is one read on the model, outside the cache.** One noul question (the first of
   upstream's `warmup.questions`) over upstream's warm-up state, its prompt from the chat
   template, its canvas from `CanvasBuilder` with seed 0, prefilled and read without touching the
   prefill cache, so no engine is needed and the first request is not served from the warm-up's
   prefill.
6. **The downloader is the module's own, not `HubApi`.** swift-transformers' `HubApi` downloads
   into `downloadBase/models/{org}/{repo}/`, so it cannot share the 16.5 GB that upstream and
   mlx-vlm keep in huggingface_hub's layout; it was the reference for the metadata and the resume.
   `ModelResolver` reads the Hub API's revision and tree JSON (following `Link: rel="next"`),
   streams each missing file through a URLSession data delegate into
   `blobs/<id>.incomplete` with `Range: bytes=N-`, checks it (SHA-256 for LFS files, size and the
   git blob SHA-1 of `blob <size>\0` and the content for the others), renames it to `blobs/<id>`
   and links `snapshots/<commit>/<path>` to it with a relative symlink; a branch or tag writes
   `refs/<name>`. A transport error is retried from the bytes on disk, 5 attempts in all; a server that
   answers a range with the whole file or a 416 restarts the file; a digest mismatch removes the
   download and names the file and both digests; 401 and 403 say the repository is gated and name
   `HF_TOKEN`; a 404 names the revision. The token is dropped when a download redirects to another
   host (the Hub's CDN), as huggingface_hub does. A tree whose `Link` next page is on another origin is refused, so
   the token goes only to the endpoint. Shard names from the index must be relative paths without
   `.` or `..`. Cancelling the task that loads stops the transfer in flight, keeps the partial
   file for the next run and throws `CancellationError`, never the offline fallback. When the Hub cannot be reached, a commit (given, or read
   from `refs/<name>`) whose snapshot is complete is used offline. The issue's "all files except
   README" is not followed: the whole tree is fetched (13 files, README and `.gitattributes`
   included), so the snapshot is the one huggingface_hub writes. No lock files are written, so
   two processes downloading the same blob at once is unsupported.
7. **The cache location and the token come from the caller.** `HubCacheLocation(environment:)`
   follows huggingface_hub: `HF_HUB_CACHE`, then `HF_HOME/hub`, then
   `XDG_CACHE_HOME/huggingface/hub` (beyond the issue text, as `EncoderPackageStore` already
   does), then `~/.cache/huggingface/hub`. `HubCacheLocation.token(environment:)` reads `HF_TOKEN`
   and treats an empty value as absent, the bug upstream's `LayaEngine.load` works around; the
   resolver never sends an empty token. The library never reads the process environment (D-013).
8. **The pins.** `.fourBit` is `mlx-community/diffusiongemma-26B-A4B-it-4bit` at
   `a7a81407613811e8ba63af92ac0d852b809e191f`. `.eightBit` is `-8bit` at
   `7b95e3887078ba56283c24f2578d6e5a06b9d7e8` and `.bf16` is `-bf16` at
   `2cd36f950eb065c96c80810fb6b859b114cd052d`, each `main`'s commit on 2026-10-01 through the
   Hub API (both last modified 2026-07-15); no fixture covers either. `ModelSource(setting:)`
   maps a preset's bare repository to its pin, a path to `.directory` and `repo@revision` to that
   revision.
9. **The CLI registers `mlx`.** The CLI (D-038) merged before this branch, so
   `BackendRegistry` now loads `mlx` instead of exiting 3: a `DecisionBackendProvider` whose load
   is `DiffusionGemmaRuntime.load(ModelSource(setting: mlxModel), configuration: .init(
   maxPromptTokens: mlxMaxPrompt, promptCacheEntries: mlxPromptCache, cacheLimitGB:
   mlxCacheLimitGB, warmUp: warmup), cache: HubCacheLocation(environment:), token:
   HubCacheLocation.token(environment:))`, whose `.warmingUp` stage prints the `warming up` phase;
   `openjev` and its tests link `OpenJevDiffusionGemma` on macOS, and on Linux `mlx` is a known
   backend that exits 3. The CLI does not print the `LoadReport` yet: its providers get no logger,
   and changing that is the CLI's design to make. A directory that is not a checkpoint exits 3
   naming what it lacks; an opt-in smoke test (`OPENJEV_TEST_MODEL`) serves the quickstart from
   the built binary.
10. **Tests.** `LiveCheckpoint` now loads through `DiffusionGemmaRuntime.load(.directory(...))`,
    so the model-level suites and the runtime suites share one 16 GB load; the runtime exposes
    its loaded model to the tests only (`sharedLoadedModel`, internal). The resolver's tests run a
    local HTTP server (Network framework) with the Hub API's JSON shapes; one opt-in test
    (`OPENJEV_TEST_DOWNLOAD=1`) downloads two small files of the pinned revision from the real Hub.
    The issue's "downloading the 4-bit checkpoint to an empty cache completes" (16.58 GB) is not
    run as a test; its parts (cold download, resume, digest check, layout) are tested on the fake
    repository and on the two real files.
11. **Measured on 2026-10-01 (M3 Max, the pinned checkpoint, native kernels).** Through the
    runtime, the 27 oracle reads meet D-014 with the figures D-036 records (mean label
    probability difference 0.0084, long prompts 0.0054; entropy 0.086 and 0.131; top label 150 of
    156 and 120 of 120), and the runtime's maps are bit-identical to `DiffusionGemmaModel.read`'s.
    D-014's bounds are aggregates: the quickstart's five reads alone have a mean of 0.045, from its
    `is_urgent` slot (the oracle reads 0.57, 0.94 and 0.52 for yes), so per-request bounds are not
    asserted. Load over four test runs (warm file cache): tokenizer 4.9 to 6.3 s, weights 0.9 to 4.1 s,
    warm-up 0.30 to 0.83 s. The built `openjev serve --backend mlx` was serving 12.2 s after it
    started (warm-up about 4 s) and answered the quickstart in 751 ms. Memory over 100 unique short
    prompts: R4.

Status. Proposed with issues #29 and #30.

## D-040 Model routes and the SDK compatibility suite: where the port goes beyond or differs from the issue text

Context. Issue #38 forwards a request for a routed model and lists the routed models, upstream's
`forward`, `parse_routes` and `models_list`, and issue #39 runs TypeSafe's official SDKs against the
Swift server in CI. Upstream forwards through httpx, which a Swift server does not have, and the
SDK suite needs a server that answers without a model on Linux. Several points needed choices the
issues do not spell out.

Decision.

1. **The forwarding client is AsyncHTTPClient.** It runs on swift-nio's sockets on macOS and
   Linux alike: the router hands it NIOPosix's shared event loops rather than letting it pick
   Network.framework on macOS. It was already resolved, as a dependency of Hummingbird's
   HummingbirdTesting, so `Package.resolved` is unchanged. Its options map onto httpx's one for
   one: 5 seconds to connect and `OPENJEV_FORWARD_TIMEOUT` for each read and write, upstream's
   `httpx.Timeout(forward_timeout, connect=5.0)`; no redirect followed; HTTP/1.1; `gzip` and
   `deflate` asked for and decoded, as httpx asks for and decodes them; and a refused connection
   reported at once rather than retried until the connect timeout. Its errors become httpx's
   exception names, which the 503 repeats as upstream's `type(e).__name__` does: `ConnectError`,
   `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `ReadError`, `WriteError`,
   `RemoteProtocolError`, `DecodingError` and `UnsupportedProtocol` (`ForwardingFailure`). Any other error is the 503 naming its Swift type, as a backend's is
   (D-031 item 7). The cost is the binary: `openjev` now links AsyncHTTPClient and swift-nio-ssl
   with BoringSSL, which the server did not use before. A stripped release build for Linux grows
   from 11.2 MB to 17.4 MB.
2. **One client per forwarded request.** The client is made when a request is forwarded and shut
   down before the route returns, so nothing outlives the request and the server has no client to
   shut down when it stops. A routed server that closes an idle keep-alive connection cannot fail
   the next forwarded request, a race httpx's connection pool has. The cost is a new connection
   for each forwarded request, a fraction of a millisecond on the same Mac or the LAN, against
   reads of milliseconds; an `https` route pays a TLS handshake each time.
3. **What is forwarded and what comes back,** as upstream: after the body has passed validation
   and before the model name is checked, a request whose model has a route and is not accepted
   here, so the SDK aliases and the served names stay local; the bytes the client sent; the first
   field of `authorization`, `x-origin-secret` and `content-type`, as Starlette reads a header; and
   the routed status and body unchanged, with only `content-type` and `retry-after`, several
   fields joined with `, ` as httpx joins them, and no `content-type` when the routed server sent
   none. This server's request ids and `server-timing` replace the routed server's. The questions
   cap is the routed server's to apply, since upstream forwards before it. A route URL with
   `user:password@` sends them as `Authorization: Basic` in place of the client's header, as httpx
   0.28.1 does (recorded with it). A `gzip` or `deflate` answer comes back decoded, without its
   `content-encoding`, as httpx's `r.content` is, and one in another encoding as it was sent, as
   httpx gives it without brotli or zstandard installed. Departures: the client adds `host`,
   `content-length` and `accept-encoding: deflate, gzip`, where httpx adds `accept`,
   `accept-encoding: gzip, deflate`, `connection` and `user-agent`; it ignores `HTTP_PROXY` and
   the other proxy variables httpx reads; and a route URL Foundation cannot parse is the 503
   `InvalidURL`, where httpx's `InvalidURL` is not an `HTTPError` and upstream answers Starlette's
   bare 500. The routed answer is read and decoded whole without a bound, as httpx's `r.content`
   is: the routed server is the operator's own.
4. **Model time, cancellation and the log.** The time from sending the request to reading the
   whole answer is added to the request's `ModelTimeRecorder` when the exchange ends, failed or
   not, as upstream's `finally` adds it to `model_ns`, so `server-timing`'s `model` reports it;
   this is D-038 item 7's follow-up. A client that goes away cancels the exchange, which closes the
   connection to the routed server, as it cancels a decision (D-038 item 8); upstream runs the
   forward to its end. A failed exchange is logged at error level with its 503's message and the
   routed model's name, as a backend failure is (D-031 item 8), never the URL.
5. **The listing.** `GET /v1/models` and `openjev models` list the backend's models, then each
   routed name the backend does not accept, in the routes' order: `KnownEncoderModels`' entry for
   a name an encoder backend serves, and an empty description and release date for any other.
   Upstream's `known` holds only `ENCODER_MODELS`, so a route named `openjev-0.1` is listed with
   empty texts too, and `diffusiongemma-26b`, which the DiffusionGemma backend lists without
   accepting it for System One, is listed twice when routed, as upstream lists it. The routed
   servers are never asked, so the listing holds while one is down. `openjev decide` never
   forwards: it answers with the loaded model, so a model only a route serves is the
   unknown-model 400 there.
6. **The routes in the log.** `serve` logs `forwarding {name} to {url}` for each route after the
   settings line, the URL without the `user:password@` it may hold, a scheme-relative `//` URL's
   and a scheme-less one's included; the settings line keeps naming the routes alone.
7. **Parsing.** `parse_routes` is read as Python reads it, code point by code point, so a `,` or
   an `=` followed by a combining mark still separates, and trimmed of the characters
   `str.strip()` removes, which include U+001C to U+001F that Foundation's whitespace set leaves.
   A URL is not checked at startup, as upstream checks only the `name=url` shape; an unusable one
   is the 503 of item 3 at request time.
8. **The stub server.** `openjev-stub-server` is an executable target that is not a product and
   never ships: the real application, `DecisionServer` and the `OPENJEV_*` settings over
   OpenJevTestSupport's stubs, `StubBackend` behind the DiffusionGemma engine for `mlx`, the
   default, and `StubQuestionReadBackend` for `laya` and `verdict`, so the suite runs on Linux,
   where Core ML does not exist. It prints the port it bound alone on a line, which is how a
   caller learns it when `OPENJEV_PORT` is 0, logs to standard error, and exits 0 on SIGTERM.
   Prompts the tokenizer fixtures never recorded get stand-in ids from `AnyPromptTokenizer`, which
   moves from the server tests into OpenJevTestSupport; the stubs never read them.
9. **The suite.** `Tools/sdk-compat/run.py`, standard library only, starts three stub servers
   with `OPENJEV_API_KEY=sk-test`: the DiffusionGemma stub with `laya-1.0` routed to the second, a
   Laya stub, and one with `OPENJEV_MAX_QUEUE=0`, which refuses every request with the 529. Each
   SDK reaches them through a recording proxy and runs each scenario in a process of its own,
   configured from `TYPESAFE_BASE_URL` and `TYPESAFE_API_KEY`; the runner checks what the SDK
   observed and the exchanges the proxy recorded, prints every exchange of a failed check and
   exits 1. The scenarios: Jev's quickstart decodes, with the SDK's default model `jev-latest`
   (`SystemOneResponse` with `.choices`, `.scores` and `.nouls`); the listing, routed model
   included; a wrong key raises the authentication error with the request id and is not retried;
   the 529 is retried twice, each time after the one second `retry-after` asks for, with
   `X-TypeSafe-Retry-Count`, then raised as `TypeSafeInternalServerError`, a `TypeSafeAPIError`;
   `samples: 33` raises the 422 joined as `samples: Input should be less than or equal to 32`; and
   a routed model's answers come back through the forwarding, which exercises issue #38 with real
   SDKs. The TypeScript SDK runs the same six under Node.js. The pins are typesafe-sdk 0.7.2
   (2026-09-26) with its dependencies in `requirements.txt`, a complete lock for CPython 3.12, and
   `@typesafe-ai/sdk` 0.6.0 (2026-09-15) in `package.json` and `package-lock.json`, the newest
   releases on PyPI and npm on 2026-10-01.
10. **JevSwiftSDK.** It takes a base URL, as `JevConfiguration(baseURL:)` and from
    `TYPESAFE_BASE_URL`, so the runner's `--swift-sdk` builds it at 0.1.0 (commit `ce35d20`) in a
    package of its own, `Tools/sdk-compat/swift`, and CI runs it: the listing, the wrong key, the
    529 and the routed answers. It sends only state, model and questions, so `samples: 33` cannot
    reach the server. It writes the questions and the choice criteria from Swift dictionaries,
    whose order changes from run to run, and the DiffusionGemma stub answers from tokenizations
    upstream's tests recorded in their order, so its quickstart answers are read through the Laya
    stub, in the routed scenario. Jev defines the labels and the answer order by that order, so
    against a real model the SDK's answers to one request change between runs.

Alternatives rejected. (a) URLSession: on Linux it is FoundationNetworking over libcurl, which
behaves unlike macOS's CFNetwork; it has one timeout for the connection and the reads, so a route
whose host drops the connection attempt waits out `OPENJEV_FORWARD_TIMEOUT` and is a `ReadTimeout`
where upstream answers a `ConnectTimeout` after 5 seconds, and its `URLError.timedOut` cannot tell
the two apart; and it follows redirects and adds `User-Agent`, `Accept-Language` and
`Accept-Encoding` unless a delegate stops it. (b) One long-lived client shut down with the server:
`OpenJevApplication.router(settings:service:)` has no end to shut it down at, and AsyncHTTPClient
stops a debug build whose client is released without a shutdown. (c) A hidden `--backend stub` in
`openjev`, enabled by a variable: D-038 item 10 kept stubs out of the shipped binary, and the
binary would link the fixture loaders. (d) Recording exchanges through each SDK's hooks, httpx2's
event hooks and a custom `fetch`: a proxy records the three SDKs alike, as the server saw them.

Consequences. D-030 item 2's routes and D-038 item 7's forwarded model time are done. The suite
runs Node.js 20, the TypeScript SDK's floor, which reached its end of life in April 2026. When an
SDK publishes a release, update its pin and the lock with it; the CI job then says whether this
server still satisfies it.

Status. Proposed with issues #38 and #39.

## D-041 JevBench harness: where the port goes beyond or differs from the issue text

Context. Issue #61 asks for a runner that submits JevBench v1's public items to `/v1/systemone`,
scores them with the benchmark's own scoring where it exists, and runs it against the Swift server
on DiffusionGemma 4-bit and upstream's Python MLX server on the same machine and weights; then the
SemIf/TypeSafe public-evaluation subset if its artifacts are still available. Verdict (#57) and
Laya (#58) are served today; the DiffusionGemma backend reached main with #29 and #30 (D-039) while
this work was under way, and its parity with mlx-vlm (#31) has not been shown, so the runs this
issue records are the two encoder models'. Several points needed choices the issue does not spell
out.

Decision.

1. **The dataset pin.** JevBench is `fstandhartinger/jevbench` at `bb05a33` (2026-09-29), MIT. Its
   231 public items are `datasets/public/easy.jsonl` (48), `original.jsonl` (72, the benchmark's
   "standard" tier) and `hard.jsonl` (111); their SHA-256 match the benchmark's own
   `datasets/manifest.json`. `Tools/jevbench/harness.py` downloads them, and the published results
   it compares with, into a cache outside the repository and checks every file's size and SHA-256.
   The benchmark's held-out and imported items (24, 24, 109 and 146) are not public, and its
   published tier accuracies include them, so only its public accuracy and its per-item outcomes
   (`results/v1.2/jevbench-v1.2-per-task.json`, the v1.3.0 board) compare item for item.
2. **The scoring source is the benchmark's code, unchanged.** `scoring.score_task` (exact label
   set, a sum within 1e-3 or rescaled inside the 2e-2 rounding band, argmax with the smallest
   label on a tie), `summarize.metric` and `summarize.summarize` (accuracy, the multi-class Brier
   sum, the ECE over 10 equal-width bins of top-label confidence, ordinal MAE, paraphrase
   consistency), with `tasks.py` and `metrics.py`, are vendored byte for byte under
   `Tools/jevbench/vendor/` with the MIT license and pinned by SHA-256: a run refuses a changed
   copy, `harness.py fetch` compares each with its commit, and the smoke test checks the pins
   offline. Nothing is re-implemented. The benchmark's `Runner` (a spending ledger, raw evidence
   kept outside its repository) is not used; its per-item record is reproduced, and its stop rule
   except for refusals (item 4).
3. **Item shapes map onto questions through JevBench's own adapter.** Each item is one request
   built by the vendored `adapters/typesafe.py`: question id `decision`, `{type, instructions,
   criteria}` with `criteria` left out when null, the state as the item holds it, every key order
   kept; the answer is read as that adapter reads it (a noul as `{"yes": p, "no": 1 - p}`, a
   choice's `choice` required to be one of the labels). The harness replaces only the adapter's
   HTTP call, with one that sends the same bytes and also keeps the response's `server-timing` and
   `x-request-id`.
4. **What is skipped, and what counts as wrong.** A question the model cannot take is skipped,
   never sent, and counted: more than 24 options for `verdict-1.4`, more than 255 options or 10
   levels for any model. A skipped item lowers the coverage and stays out of the accuracy, as an
   unattempted item does in the benchmark. Neither dataset has such an item (JevBench's widest
   choice has 6 options, the TypeSafe rows' 8), so nothing was skipped. A 4xx other than 401, 403
   or 429 is a refusal, recorded with its detail and counted wrong, as the benchmark counts a failed
   decision; the benchmark's stop rule exempts only a 422, while here every refusal is exempt,
   because Jev's contract answers a question it cannot ask with a 400.
5. **The second dataset is SemIf's TypeSafe subset.** Upstream reports on neither TypeSafe's
   evaluations nor SemIf's subset: no revision of its README, none of its branches and none of its
   five issues and five pull requests (2026-10-01) mention them. SemIf does
   (`TheoLeeCJ/SemIf-OpenJev`, its "TypeSafe subset agreement": Jev 0.883, Qwen3.5-4B 0.845).
   SemIf's selection at `23cf1f3` names 102 rows of 20 cases and pins the parsed payload of four
   case snapshots that evals.typesafe.ai still served on 2026-10-01 with those hashes. The harness
   downloads them (1.4 MB), rebuilds the rows with SemIf's own `build_typesafe.py`, pins the rows it
   writes by SHA-256 too, and scores them with SemIf's own `evaluate_external.type_safe` (equal-case
   modal agreement and total variation), both vendored unchanged, and with JevBench's metrics. A
   request carries TypeSafe's own question and document, as TypeSafe asked Jev, not SemIf's prompt
   rendering of them. TypeSafe's snapshots carry no license grant, so a result file stores ids,
   digests and the servers' answers only, and the TypeSafe scores are computed from the cache.
6. **The servers.** `servers.py` runs the Swift release build (`openjev serve`, the float16
   multifunction package on the GPU, D-034 and D-037), from the converted packages' folder, whose
   bytes `Tools/encoders/manifest.py --check` shows to be the published ones (a package that fails
   the check stops the run before the server starts), and upstream's server (`python -m openjev` at
   `dcd2094`, PyTorch float32 on the CPU, which upstream picks without CUDA), each on 127.0.0.1 with
   warm-up on, one request at a time. Upstream reads its checkpoints from the pinned snapshots
   (`OPENJEV_VERDICT_MODEL`, `OPENJEV_LAYA_MODEL`, `HF_HUB_OFFLINE=1`) rather than the Hub's current
   revision, from the environment `Tools/jevbench/requirements-upstream.txt` locks, with the pinned
   checkout on `PYTHONPATH` rather than an installed copy, so a moved pin cannot run stale code.
   Every result file records its server's versions and the hardware.
7. **The published rows are other setups.** JevBench's `openjev-verdict-1.4` row ran the same
   weights through the author's v1.4 engine and the benchmark's `verdict_local` adapter, which adds
   a noul's criteria to its proposition where upstream ignores them; its `laya` row ran another
   checkpoint, `convaiinnovations/laya`, with a 512-token budget. Upstream issue #6 reports only
   the DiffusionGemma rows (81.8% on the public items, 28.6% sealed). The harness compares the
   Verdict and Laya runs with their rows item by item and says how each row was produced; the
   DiffusionGemma rows wait for the `mlx` runs.
8. **What compare measures.** Two runs of one model, the upstream run as the reference: top-answer
   agreement under JevBench's argmax; the mean and largest absolute difference over every label
   probability of every item both answered, per question type, as spike #56 measured Core ML
   against PyTorch; identical answers; correctness flips with an exact McNemar test; every
   disagreement; and the items where the reference's top two are less than 0.01 apart, where the
   parity bound of D-034 and D-037 allows a changed top answer.
9. **Result files are committed, trimmed.** One per run, under `Tools/jevbench/results/`, at most
   about 400 KB: a JevBench item keeps its question whole (MIT) and its state as a SHA-256 and a
   length, since the hard tier's states alone are 480 KB.
10. **CI.** The harness's smoke test (a fake server inside the process, no model, no download)
    runs as one more step of the macOS job, with the image's `python3`; the harness needs only
    the standard library, so no job and no install step were added.

Alternatives rejected. (a) Re-implementing JevBench's scoring: its code is small and available, and
a byte-for-byte copy cannot drift from the published numbers. (b) Downloading the scoring code at
run time: the smoke test and CI would need the network. (c) A Swift harness: the benchmarks' code
is Python, and Tools/README.md keeps Swift code in the package. (d) Reusing `Tools/encoders/.venv`
for upstream's server: its lock is complete for the encoder scripts, and adding FastAPI and uvicorn
would make it wrong. (e) SemIf's own rendering of the TypeSafe rows (the document as indented JSON
text, options as `id: description`): that is SemIf's input to its scorer, not a Jev request.

Consequences. On both datasets the Swift server and upstream's give the same top answer on every
item, for both models, the same accuracy, and Brier scores and ECEs within 0.0003
([quality.md](quality.md)); the agreement figures become a release metric for 0.1. The
DiffusionGemma comparison the issue asks for, the Swift server against upstream's MLX server on the
same machine and weights, is the one remaining piece: `servers.py --backend mlx` runs both sides
today and is to be recorded once #31 shows the backend's parity. The runs also show a latency cost
the answers do not: on a Mac the encoder keeps two Core ML functions loaded, one per input shape, so
one-question requests of mixed lengths load functions again (Laya reloaded on 15 of its 333
requests, each reload taking several times an ordinary read), and a function's first load, which the
warm-up's three questions do not cover, took 0.2 to 1.9 s across the runs.

Status. Proposed with issue #61. The two Core ML functions its runs had loaded on a Mac are
every function since D-042. The DiffusionGemma runs it left for after #31 were recorded with issue
#62, both servers capping MLX's buffer pool at 4 GB ([quality.md](quality.md#diffusiongemma),
D-046).

## D-042 Encoder functions on a Mac: every function stays loaded, `OPENJEV_ENCODER_FUNCTIONS` caps them

Context. On a Mac both encoder backends run a multifunction package with one Core ML function per
input shape: Verdict's six (batch 1 and 16 by 128, 256 and 512 tokens) and Laya's eight (to 1,024).
`CoreMLEncoderModel` kept the two used most recently (`functionCapacity`, D-037 item 3), because
spike #56 found that each loaded function holds its own copy of the weights. Issue #61's JevBench
runs showed the cost of two: one-question requests whose lengths moved among three or four shapes
loaded a function again, 15 times in Laya's 333 requests, 0.46 to 0.78 s each, where a read took a
median of 20 ms at 128 tokens and 163 ms at 1,024. The measurements below are in
[spikes/encoder-function-capacity.md](spikes/encoder-function-capacity.md), taken on 2026-10-01 on
the M3 Max of spike #56 (128 GB, macOS 27.0.1); MB and GB there and here are 2^20 and 2^30 bytes,
vmmap's units.

Decision.

1. **On macOS every function stays loaded once a read has needed it.**
   `VerdictBackend.Configuration.defaultFunctionCapacity` is 6 and
   `LayaBackend.Configuration.defaultFunctionCapacity` 8, the packages' function counts
   (`EncoderPackageSpec.functions`); iOS keeps 1. A function still loads when a read first needs it,
   so the memory follows the shapes the requests take, and no request waits for a function to load a
   second time.
2. **Why not a larger fixed number.** Replaying the run's requests through the cache predicts 19
   loads, 4 first loads and 15 reloads: in a rerun, exactly the 19 requests whose model time was
   over 300 ms. The benchmark's order, its short tiers first, hides most reloads: in random orders
   of the same requests, two functions reload 36% of Laya's requests and 13% of Verdict's. Served
   shuffled, 102 of Laya's 333 requests waited 439 to 633 ms (median 497) for a function that, kept,
   answered them in a median of 50 ms; the run's 95th percentile was 568 ms against 206 and its
   model time 72.3 s against 27.5. Verdict's 39 reloads took 219 to 515 ms (median 241) against 15
   ms, and 23.8 s of model time against 15.3. Once one-question requests and batches mix (2 to 16
   questions run through a batch-16 function), any capacity below the package's function count
   reloads, because a cache smaller than the shapes in use evicts the one the next request needs:
   Laya at 4 reloads 30 to 37% of requests, at 6 10 to 14%, at 8 none; Verdict at 4 12 to 15%, at 6
   none.
3. **What a loaded function costs.** On the GPU each loaded function maps its own copy of the
   weights from files of Core ML's: 805 MB for Laya and 289 MB for Verdict, about the size of the
   packages' weight files (806.7 and 289.8 MB). The process's physical footprint does not count
   these file-backed pages; the resident memory is the footprint plus the copies. Laya holds 1.03 GB
   with `b1_s128` alone, 2.05 with its two longest batch-1 functions, 3.68 with all four, 4.65 with
   what one-question traffic loads (those and the warm-up's `b16_s128`) and 8.86 with all eight,
   9.73 counting the footprint at its peak during a batch of 16 at 1,024 tokens; its two largest
   functions alone hold 3.91 GB, 4.35 counting the peak. Verdict holds 0.85 GB with its two longest
   batch-1 functions, 1.57 with what one-question traffic loads and 2.76 with all six, 2.90 counting
   the peak. After their 333 reads the servers of item 2 held 1.99 to 2.03 GB with two functions and
   4.44 to 4.60 GB with every function kept (Verdict 0.85 to 0.88 and 1.52 to 1.54 GB). Loading a
   Laya function took 0.45 to 1.85 s, 0.14 to 0.25 s of it for `MLModel` to initialise and the rest
   for its first prediction, the longest for `b16_s1024`; loading a Verdict function took 0.22 to
   0.69 s.
4. **`OPENJEV_ENCODER_FUNCTIONS` caps it,** this port's variable, read and checked as upstream's
   `_env_num` settings are: unset or empty keeps every function; an integer of at least 1 is the
   most functions loaded at once, and 0 is `OPENJEV_ENCODER_FUNCTIONS=0 is below the minimum of 1`,
   exit 2. `ServerSettings.encoderFunctions` carries it to
   `VerdictBackend.load(from:functionCapacity:)` and
   `LayaBackend.load(from:packageSet:functionCapacity:)`, and `serve`'s settings line prints
   `encoder_functions=all` or the number. For a Mac with 8 GB serving Laya, 2, the earlier default,
   is the setting these figures suggest; no Mac that small was measured.
5. **The warm-up stays upstream's.** Its three questions load `b16_s128`, which one-question traffic
   never uses again. Reading each length alone too would spare the first request of each length its
   first load, once per process (0.49 to 0.84 s in issue #61's recorded run, 1.6 to 1.7 s in a
   rerun with a new binary), at the price of loading functions the traffic may never use, and with
   two functions they would have been released again.
6. **Answers do not change.** A call runs through the function its shape picks, whatever else is
   loaded: served in both orders with two functions and with every function, all 333 answers were
   identical, for both models. The live tests read the corpus with the new default and, one question
   per call, with two functions, so the path that loads functions again stays tested.

Alternatives rejected. (a) A larger fixed default, the number of batch-1 functions (3 for Verdict, 4
for Laya): one-question traffic stops reloading, but traffic that mixes one-question requests and
batches reloads 30 to 37% of Laya's requests at 4 (item 2). (b) A default scaled to the Mac's
memory: it would spare an 8 GB Mac the variable, at the cost of behaviour that changes with the
machine, and the variable already covers that Mac. (c) Warming every function at startup: see item
5; it would also hold all 8.86 GB from the start. (d) Running a request through a loaded function of
a longer length or a larger batch instead of loading its own: the padding changes the float16
numbers, so an answer would depend on what was loaded.

Consequences. A Mac serving Laya holds up to 8.86 GB (9.73 counting the peak footprint) with all
eight functions in use, where two held at most 3.91 GB (4.35); docs/deployment.md gives the figures
and the variable. Spike #56's peaks on the GPU, 994 MB for Verdict and 2,856 MB for Laya, were the
footprint alone, without the weight copies. Core ML also keeps each function it has compiled, with
its own copy of the weights, in `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache` (`openjev`
for the server), whatever the capacity: 6.3 GB once all eight Laya functions have run and 1.7 GB for
Verdict's six. The iPhone keeps one function: Verdict there loads one again whenever the length
changes (0.2 s once Core ML has cached it, spike #56), which stays to be measured with three loaded
on the Neural Engine.

Status. Proposed on 2026-10-01, from issue #61's runs.

## D-043 Live end-to-end suite: where the port goes beyond or differs from the issue text

Context. Issue #41 asks for an opt-in suite (`OPENJEV_LIVE_URL`, optional `OPENJEV_LIVE_KEY`,
`OPENJEV_ORIGIN_SECRET`, and `OPENJEV_LIVE_GATEWAY=1` for a gateway that strips `server-timing`)
that mirrors upstream's `tests/test_live.py`: the README example in Jev's shapes with sensible
values, a 255-option choice, many questions chunked and answered in order, an unknown model's 400
`api_usage_error`, 16 concurrent reads, the read options, images, chat and stream as their
milestones land, the encoder models `/v1/models` lists, and the `server-timing` and request-id
headers. It is to pass against `openjev serve` with the 4-bit checkpoint on the reference machine
and against upstream's Python server, which shows that the suite itself is neutral. Several points
needed choices the issue does not spell out.

Decision.

1. **A Swift Testing target over URLSession.** `OpenJevLiveTests` depends on `OpenJevCore`, for
   `JSONValue`, `JSONParser` and `PythonJSONWriter`, which keep key order (the order of the
   questions and of a choice's options is part of the contract), and `ModelsResponse`; and on
   `OpenJevTestSupport`, for the listings `Fixtures/wire` recorded from upstream. It sends its
   requests with Foundation's URLSession (FoundationNetworking on Linux), so it builds and runs on
   macOS and Linux and links no server and no backend. It stays out of the iOS scheme, although it
   would compile there: it tests a server, not the iOS build.
2. **The variables.** `OPENJEV_LIVE_URL` names the server; unset or empty, every live test skips
   with a comment naming it, which CI's test log check accepts, and a value that is not an `http` or
   `https` URL with a host fails the tests instead. The bearer key is `OPENJEV_LIVE_KEY`, the
   issue's name, else `OPENJEV_API_KEY`, upstream's, so a shell that configured a server with its
   key runs either suite unchanged. `OPENJEV_ORIGIN_SECRET` goes as `X-Origin-Secret`, and
   `OPENJEV_LIVE_GATEWAY=1`, exactly `1` as upstream compares it, drops the `server-timing` check
   and nothing else.
3. **Upstream's names, in upstream's order.** Each test's display name is the name of the upstream
   test it ports, and the suite runs them one at a time, as pytest does. `test_read_options` is one
   test with three arguments, `steps 4`, `samples 4` and `sequential true`. `test_encoder` is four
   tests, `test_encoder[laya-1.0]`, `[verdict-1.4]`, `[clm-v0.1]` and `[jevk5-0.2]`, upstream's
   four models rather than the two this port serves: a Swift Testing condition decides for a whole
   test, and Swift 6.2, the Linux job's toolchain, has no `Test.cancel` to skip one argument.
4. **What decides a skip.** As upstream's fixtures do, the server's `/v1/models`, asked once per
   process, decides: the DiffusionGemma tests run when it lists `openjev-latest`, an encoder's test
   when it lists that model, and `test_unknown_model` always. A listing that cannot be read, or
   that is not Jev's (`ModelsResponse`: `{"models": [...]}` whose entries are exactly `name`,
   `description` and `release_date`), fails those tests rather than skipping them, as a failed
   fixture errors them in pytest. `test_image`,
   `test_think`, `test_chat` and `test_chat_stream` are disabled with comments naming #48, #52 and
   #53. Their bodies are upstream's checks, ready to enable once the features land, and they passed
   against upstream's server (Consequences); `test_image` reads `hotdog.jpg` from the pinned
   checkout, since no image of its 13 KB is committed here. Every skip comment names
   `OPENJEV_LIVE_URL`.
5. **Checks beyond upstream's.** On every response to `POST /v1/systemone`, errors included,
   `server-timing` must time the `model`, `server` and `total` spans, where upstream only asks that
   it be there, and `x-request-id` must be `req_` and 32 lowercase hex characters and equal
   `x-typesafe-request-id` ([02-jev-wire-api.md](02-jev-wire-api.md)), behind a gateway too: a
   gateway that rewrites the request id fails the suite, and none was run here. Every answer must have Jev's
   shape for its question: the answers in the questions' order; a noul `{type, noul}`, a choice
   `{type, choice, probabilities, confidence}` over its options in their order and a score
   `{type, score, legend, probabilities, confidence}` with its levels as the legend; every
   probability, noul and confidence in [0, 1] and a score in [0, n - 1]; `model` a name the listing
   holds and never the `openjev-latest` alias; `usage.input_tokens` above 0 and `output_tokens` 0
   unless the request thought. The issue's "in order" for the chunked questions is that order
   check. `test_unknown_model` also checks the body's `detail.error_type`, `api_usage_error`, and
   its message, as `Fixtures/wire` records them, and `test_concurrent_reads` the headers and shapes
   of all 64 answers and that their request ids differ.
6. **The concurrency is upstream's, not the issue's.** The issue says 16 concurrent reads; upstream
   sends 64 requests with at most 32 in flight, and so does this suite. URLSession opens at most 6
   connections to one host by default, which would cap the requests in flight, so the client allows
   32. Cancelling a request's task cancels its exchange, so when one of the 64 fails, the group
   cancels the requests still waiting instead of waiting for their answers.
7. **httpx's timeouts.** Upstream's client sets `timeout=300`: 300 s to connect and for each read
   and write, with no deadline for the whole exchange. URLSession's request timeout is that kind of
   limit and is set to 300 s; its resource timeout keeps its default. One difference remains:
   URLSession follows a redirect, which httpx does not by default. Neither server redirects.
8. **The runs are recorded in the pull request.** Like the model tests, the live tests never run
   on hosted CI. The pull request records each run (server, backend, model, tests passed and
   skipped, wall time) and the runs of upstream's own file, with pytest and httpx, against the
   Swift servers.
9. **What runs in CI.** Without a server, the suite's own machinery is tested on every platform:
   its settings (the URLs accepted and refused, the key's precedence, empty values, the gateway
   flag), the decoding of every listing `Fixtures/wire/models.json` recorded from upstream and of
   malformed ones, the cancellation of requests to a socket that never answers, alone and in a
   task group, and the header parsers.

Alternatives rejected. (a) AsyncHTTPClient, which the server already links: the suite would share
the server's HTTP stack instead of a plain client, and URLSession needs no package. (b) Upstream's
pytest file as the port: `swift test` runs this suite, and the checks beyond upstream's would have
had to be patched into upstream's file. (c) Committing `hotdog.jpg`: the fixtures keep images to a
few kilobytes, and the test waits for #48 anyway. (d) `Test.cancel` for the listing-dependent
skips: Swift 6.2 lacks it, and a skip decided before the test starts reads better in the log.

Consequences. On an M3 Max with macOS 27.0.1 and Xcode 27.0, the suite passed against the Swift
server's release build on the three backends it serves, Verdict, Laya and the DiffusionGemma 4-bit
checkpoint, and unchanged against upstream's Python server at `dcd2094` on the same three, with an
API key and an origin secret required as well as without: the suite does not depend on the
implementation it tests. With its four waiting tests enabled in a local build, it passed against
upstream's MLX server too, so their bodies are ready for #48, #52 and #53. Upstream's own file
passed against the Swift encoder servers. Against the Swift DiffusionGemma server it failed only
those four: images and `think` get a 400 (`openjev-0.1 does not support images`, and `think`)
until #48 and #52, and both chat tests a 404 until #53. The chat tests run there instead of
skipping because the Swift listing names `diffusiongemma-26b`, as upstream's does; the listing
stays upstream's (`Fixtures/wire/models.json`), so that difference lasts until #53.

Status. Proposed with issue #41.

## D-044 Read parity tests and the performance baseline: where the port goes beyond or differs from the issue text

Context. Issue #31 asks for the live tests that prove the DiffusionGemma port upstream-compatible on
real weights: oracle parity with failures that print both distributions, the read cases of
upstream's `tests/test_mlx_model.py` through `DecisionEngine`, and a regression file of the port's
own answers. Issue #32 asks for `openjev-bench` and a documented baseline: read latency, throughput,
memory, prefill, a profile of where the time goes, and upstream's Python MLX backend on the same
machine. Both close milestone 2. ReadOracleTests, RuntimeLiveTests and CheckpointTests already ran
the oracle reads (D-036, D-039); this work extends them.

Decision.

1. **Per-read figures are reported, the bounds stay aggregate.** Both oracle tests
   (`ReadOracleTests` through the model, `RuntimeLiveTests` through the runtime) print, for each read
   with a slot whose top label differs from the oracle's or whose largest |dp| exceeds 0.05, the
   read's id, prompt key, width and steps and both distributions with their label ids, to four
   decimals (`ReadDivergence`). 17 of the 27 reads have such a slot under native kernels; D-014's
   six aggregate bounds remain the only assertions, as D-014 decided. `RuntimeLiveTests` also prints
   each read's latency and whether it prefilled.
2. **Upstream's remaining read cases, under upstream's names.** `UpstreamReadCaseTests` ports
   `test_many_questions_chunk_and_run_in_sequence`, `test_more_steps_still_answer_and_cost_no_more_prompt`,
   `test_steps_hold_the_template_and_reuse_one_prefill` and `test_the_prompt_cache_is_bounded_in_tokens`
   with upstream's thresholds; where upstream counts `len(rt.prefills)` or reads the cache's token
   total, the port asserts on `ReadStatistics` (prefill misses and hits, cached prefills and tokens).
   The "old single pass" of the steps case is rebuilt from the model's public decoder pass and must
   be bit-identical to a one-step read. The determinism check compares answers and usage, not the
   whole `Decision`, whose model time differs between calls (upstream compares response bodies,
   which carry no time). `test_readme_example_and_friends` is not duplicated: `RuntimeLiveTests`
   keeps the README example with `test_live.py`'s bounds, and the steps case asserts upstream's
   thresholds on all three of its states at `steps` 4. The image cases (#48), `think` (#52) and chat
   (#53) are disabled tests under upstream's names; their comments name the issue and
   `OPENJEV_TEST_MODEL`, which `check-test-log.sh` requires of every skip.
3. **The regression file is the port's own output, compared exactly.** `Fixtures/regression/reads.json`
   holds, for the 27 oracle reads (through `DiffusionGemmaRuntime.read`) and the wire quickstart
   and upstream's README example (through `DecisionEngine`, every read recorded by a pass-through
   backend and sorted by seed, steps and canvas), each slot's probabilities, entropy and top label,
   the prompt tokens, and for the engine requests the billed tokens and answers. Its pins are a
   `generator` object, as every fixture file has (`FixturePinTests` checks `regression/` for the
   test that wrote it, the checkpoint repository and revision, the mlx-swift version from
   Package.resolved, macOS, the GPU's name and the date). The tolerance is 0: seven runs in separate
   processes on the M3 Max reproduced all 180 slots bit for bit, as D-014's exact-tier
   determinism and the cached-against-cold tests lead one to expect, and any change in the last
   bit is a change in what the port computes, which is what the file guards. When the pins do not
   match, the kernels round differently, so the test applies D-014's aggregate bounds instead
   (mean |dp| at most 0.02, the top label on at least 90% of slots) and says to record that
   machine's own file. `OPENJEV_RECORD_REGRESSION=1` records it.
4. **`openjev-bench` is an executable target without a product.** It sits in the MLX block of
   Package.swift beside `OpenJevDiffusionGemma`, on ArgumentParser, with `OpenJevBenchTests`
   testing its model-free code (percentiles, `server-timing`, the request sets, the tables, the
   result file, the model directory, the command lines) through `@testable import`. Its modes are
   the issue's (`reads`, `concurrency`, `memory`, `prefill`, `--url` for `reads` and `concurrency`)
   and one more, `profile`, which times each stage of a read through the model's public stage
   observer and reports a correction for the round trip each evaluation point costs (about 0.5 ms;
   the corrected stages add up to the unstaged pass within 2%). `memory` measures one cache-limit
   setting per process (`--cache-limit-gb`), so the two runs do not share a pool. `--metallib`
   points MLX at another Metal library, to time the port on the kernels upstream runs. p50 and p95
   are NumPy's linear percentiles. `--json` appends to `Tools/bench/results/<date>-<machine>.json`;
   the machine and macOS are in the file, and each run records the power source and the thermal
   state at its start and end.
5. **Upstream is measured the same way.** `openjev-bench --url` times any `/v1/systemone` server;
   `Tools/jevbench/servers.py` gained `--command`, which starts either server as the JevBench runs
   do and runs a command against it with `{url}` replaced. `Tools/jevbench/requirements-upstream.txt`
   gained upstream's `mlx` extra (mlx-vlm 0.6.15, MLX 0.32.2 and their dependencies at the versions
   `Tools/oracle/requirements.txt` locks), which the encoder runs had not needed. Upstream's MLX
   engine writes `model;dur=0.0`, so only its HTTP time compares. `Tools/bench/upstream_stages.py`
   times upstream's own runtime prefill and read the way `prefill` and `profile` time the port's.
6. **A measurement protocol, because the laptop's heat dominated.** The first runs, taken back to
   back, disagreed by up to 80% (a three-question read 302 ms cool, 380 to 539 ms after
   10,000-token prefills) and swapped the order of the two servers. Every reported run therefore
   started on AC power after two minutes idle at the nominal thermal state, with no other build,
   test, Docker job or model server running (another Claude Code worktree was serving the model on
   and off during the day); a watcher discarded and repeated any run during which one appeared or
   the Mac went on battery. The discarded runs are not reported.
7. **Instruments could not record.** `xctrace record` 27.0 (27A266a) on macOS 27.0.1 stops with an
   assertion in `XRAugmentationManager` for every template and target, `/bin/sleep` included,
   inside and outside the sandbox. The issue's "profile the read with Instruments" is answered by
   the stage profile on the GPU side; a host-side call tree with macOS's `sample` was queued but
   not taken, because the Mac ran on battery for the rest of the session ([benchmarks.md](benchmarks.md),
   "Not measured").
8. **Not measured: a 32 GB or 48 GB Mac.** None was available. benchmarks.md has the row, marked
   not measured, and R4 says so; the issue's acceptance criterion for that machine is not met.
9. **Measured on 2026-10-01 (M3 Max, 128 GB, macOS 27.0.1).** Parity, from the live tests, native
   tier: the top label on 150 of 156 slots and 120 of 120 where the oracle's margin is at least 0.5,
   mean label probability difference 0.0084 (0.0054 past 1,024 tokens), mean entropy difference
   0.086 (0.131), the largest single difference 0.374 (quickstart's `is_urgent`, as D-039 found).
   The rest from the release build: reads in process 210, 302 and 518 ms p50 for 1, 3 and 12
   questions; over HTTP the Swift server and upstream's within 4% at every size (three questions:
   Swift 290 and 312 ms in two rounds, upstream 302 ms); 3.4 to 3.0 requests/s from 1 to 16
   concurrent callers; prefill 873, 1,301 and 757 tokens/s at 180, about 1,000 and about 10,000
   tokens against upstream's 905, 1,281 and 708; MLX active 14.83 GiB and pool 2.48 GiB after 200
   unique prompts with or without a 4 GB limit, resident 15.7 GiB. The expert matmuls are about 61%
   of both passes; the output projection 5% of a decoder pass. The wheel's `mlx.metallib` runs at
   the package library's speed.

Alternatives rejected. (a) Per-read bounds: D-014 rejected them, and the quickstart's `is_urgent`
slot alone (the oracle near 0.5) would fail any useful one. (b) A tolerance above 0 for the
regression file: nothing measured needs one, and a loose one would let a refactor's last-bit
change through unseen. (c) Recording the regression file from upstream: that is the oracle's job;
this file guards the port against itself. (d) Running both cache-limit settings in one process:
the second would inherit the first's pool. (e) Reporting the stage profile uncorrected: the round
trips double the decoder pass and inflate every small stage, the router most.

Consequences. Milestone 2's parity is shown on the reference machine and recorded in docs/09
layer 2, and the baseline in [benchmarks.md](benchmarks.md) answers R5: the port reads as fast as
upstream's Python MLX backend on the same Mac. The follow-ups with more than 10% headroom are filed
for milestone 7 (#100, #101, #102); the 32 or 48 GB measurement waits for such a Mac. The DiffusionGemma JevBench runs
that D-041 left for after #31 can now be recorded.

Status. Proposed with issues #31 and #32.

## D-045 Read extensions on the checkpoint: where the port goes beyond or differs from the issue text

Context. Issues #43, #44 and #45 ask for `steps`, `samples` with the automatic re-read policy, and
`sequential` to be verified end to end on the DiffusionGemma backend. The engine had implemented all
three against the stub since milestone 1, and the runtime had declared them since #29, so this work
is live tests and an instrumented runtime rather than new behaviour. No answer changed, so
`Fixtures/regression/reads.json` is untouched.

Decision.

1. **The instrument is the runtime's own test seam.** The read-policy tests build a
   `DiffusionGemmaRuntime` through the internal `init(tokenizer:configuration:calls:setCacheLimit:)`,
   whose `ModelCalls` call the shared checkpoint's model and record each read with the token ids of
   the prompt its cache came from (prompt caches are looked up by identity and kept alive for the
   test). Each test gets a fresh runtime, so a fresh prefill cache and fresh `ReadStatistics`, over
   the one loaded 16 GB model. Nothing is added to the shipping runtime.
2. **"Template positions hold" is shown by replay.** The model does not expose the canvas between
   steps. The steps 8 test therefore replays the step loop with the model's own decoder passes,
   checks that each step changes only slot positions, and requires the runtime's eight-step read to
   equal the replay bit for bit, written argmaxes included. Exposing a per-step hook in `read` was
   rejected because it would change a hot path for a test.
3. **sequential is shown on 40 questions, not 24.** The 24-noul request of upstream's
   `test_many_questions_chunk_and_run_in_sequence` chunks into two groups (17 and 7), and the issue's
   check needs a group k > 1. 40 nouls give three groups (17, 14, 9). That request is sent with
   `samples 1` so that the read count equals the group count whatever the entropies. Under the
   automatic policy the re-reads would add reads but no billing. The 24-noul figures (657 tokens
   plain, 1,147 sequential) are asserted in the upstream case under the default policy.
4. **"Per group" is shown with an uncertain question among certain ones.** Only the 24-noul
   request was tried, and both its groups are re-read. In the group of 17, 2 questions are above
   the 0.1 threshold and 15 below, and all 17 are re-read 4 times, which is the behaviour
   `read_group` specifies. The test asserts, for every group, 4 reads when any question of its
   first read is above the threshold and 1 otherwise, each re-read covering the whole group.
5. **`OPENJEV_AUTO_MAX=1` is shown at the settings level.** The engine's `auto_max 1` case was
   already in `Fixtures/policies/auto_rereads.json`. The new test feeds `ServerSettings(environment:)`
   through `DecisionBackendProvider` to a stub whose every read is uncertain: 4 reads by default, 1
   with the variable. A live server run would add only the model.
6. **Already covered, cited.** The 422 for `steps` 0 and 9 is replayed byte for byte from upstream's
   recordings by `ApplicationTests`'s recorded refusals. Images with `sequential` or `think` are
   refused before the backend by `DecisionEngineTests`. No new test duplicates either.
7. **The oracle breakdown by steps is printed by the existing test.** `ReadOracleTests` reports mean
   |dp|, top-label agreement and equal written argmaxes per steps value beside D-014's aggregates;
   the bounds stay aggregate. Native tier, 2026-10-02: steps 1, 0.0086 and 118 of 124; steps 2,
   0.0089, 16 of 16, written equal on 2 of 3; steps 3, 0.0003, 16 of 16, 2 of 3. The two that differ
   are the quickstart's, whose `is_urgent` is near 0.5 in the oracle.

Alternatives rejected. (a) A recording `DecisionBackend` wrapped around the runtime: it would see
each `CanvasRead` and its prompt, but not the model calls under the prefill cache, which the
`ModelCalls` seam already exposes. (b) Lowering `autoThreshold` to force a one-read group beside a
four-read one: the issue asks for the defaults. (c) Asserting quickstart thresholds at `steps 8`:
upstream sets none, and the read sharpens (`is_urgent` 0.0006 against 0.444 at `steps 1`), which is
the model's behaviour, not a port defect.

Consequences. The text part of milestone 4 is verified on the real model and recorded in docs/09
layer 2. The image issues (#46 to #48) and `think` with `sequential` (#52) remain; the `think`
attribution to the first group is checked on the stub by the recorded `sequential_think` case
of Fixtures/policies until then.

Status. Proposed with issues #43, #44 and #45.

## D-046 DiffusionGemma's probabilities stay upstream's: no temperature-scaling hook

Context. Issue #62 asks for the ECE and Brier score of DiffusionGemma's raw reads on the public
labelled sets the harness uses, for how the answers' `confidence` relates to observed accuracy per
question type and option count, and for a decision: offer an optional, clearly non-Jev,
per-deployment temperature-scaling hook, fitted offline by the user and applied to label logits
before the softmax, or stay strictly upstream-compatible. Upstream applies no calibration to
DiffusionGemma. Each read's label distribution is a softmax over the label tokens' log-probabilities
at the answer's slot (`slot_distribution` in `engine.py` at `dcd2094`), and an answer is the
arithmetic mean of the reads: the first, and three more with other noise when a slot's top-k entropy
exceeds `OPENJEV_AUTO_THRESHOLD`, 0.1 (`read_group`). `confidence` is `1 - H(p)/ln K`, how peaked an
answer is. The method is SemIf's (`docs/CALIBRATION.md` and `benchmarks/calibrate.py` at
`23cf1f3`): one scalar T per workload, fitted by NLL, with the ECE out of fold under group-disjoint
5-fold cross-validation and bootstrap intervals over the groups. `Tools/jevbench/calibration.py`
computes it from the DiffusionGemma runs this issue recorded with the harness of issue #61 (D-041),
on both servers, and [quality.md](quality.md) holds the tables.

Decision.

1. **No hook: the server returns upstream's probabilities, and only those.** No setting, request
   field or library option rescales an answer. A deployment that wants calibrated probabilities
   rescales the answers it receives, with a T fitted on its own labelled answers (item 5).
2. **The measurements** (2026-10-02, M3 Max; the Swift server's figure first, upstream's second).
   On JevBench's 231 items the reads are right 81.4% and 82.3% of the time at a mean top
   probability of 0.92: ECE 0.106 and 0.097, Brier 0.245 and 0.239, NLL 0.566 and 0.563. The answers
   at 0.9 or more are right 94.5% and 94.9% of the time at a mean of 0.99; those below 0.9 are right
   33% and 43% of the time at a mean of 0.65 and 0.68. The overconfidence is nearly all the hard
   tier's: easy items are right 100% of the time at 0.999, standard items 98.6% at 0.99 (ECE 0.016
   and 0.018), hard items 62% and 64% at 0.84 (ECE 0.22 and 0.20). On the 102 TypeSafe rows the
   reads agree with the reference's top option 89.2% and 90.2% of the time at 0.97 (ECE 0.092 and
   0.096). `confidence` ranks answers about as well as the top probability does (AUROC 0.88 and 0.87
   on JevBench, 0.76 and 0.72 on TypeSafe): at 0.8 or more, 89.5% of JevBench's nouls, 97% to 98%
   of its choices and 13 of 13 scores are right, and below 0.6 about 30% of its choices are. It is
   not a probability of being right, and upstream does not present it as one.
3. **What temperature scaling would buy.** One T fitted on JevBench is 2.01 and 2.00 (1.85 to 2.18
   over the folds). Out of fold it lowers the ECE from 0.106 to 0.063 and from 0.097 to 0.058, the
   Brier score from 0.245 to 0.227 and from 0.239 to 0.225, and the NLL from 0.566 and 0.563 to
   0.446. The paired 95% interval of the NLL's change excludes zero on both runs; those of the ECE
   and the Brier score exclude it on the Swift run and just include it on upstream's (ECE -0.057 to
   0.010, Brier -0.029 to 0.002); SemIf's own test, the two ECE intervals not overlapping, fails on
   both. Applied to the TypeSafe rows, JevBench's T lowers their ECE from 0.092 to 0.037 and from
   0.096 to 0.035. So a scalar T helps on both public sets, by about 0.04 to 0.06 of ECE, and the
   sets are too small to call the ECE gain more than likely.
4. **Why the server does not apply it.**
   - **The hook as the issue words it changes answers, and cannot be fitted offline.** A T applied
     to each read's label logits before the softmax acts before the mean of the reads, and a mean of
     tempered softmaxes need not keep the order of the labels: reads of P(yes) 0.67, 0.67, 0.67 and
     0.001 average to 0.503, a yes, and at T = 2 to 0.448, a no. Such a hook would move accuracy,
     which temperature scaling is chosen for not doing, and the per-read distributions it would be
     fitted on never reach the wire.
   - **Applied to the averaged answer, it is the client's own arithmetic.** softmax(ln p / T) is
     `q_k = p_k^(1/T) / sum_j p_j^(1/T)`, which keeps the argmax and needs only what the answer
     carries: a noul's `noul`, a choice's or score's `probabilities`, from which a client recomputes
     `confidence` and a score's expected level. A server hook would add no capability.
   - **T belongs to a workload, not to a model.** The tiers alone would be fitted to T = 0.8
     (standard) and 2.9 (hard), and every easy answer is right; the question types to about 3.7
     (noul), 1.6 (choice) and 1.4 to 1.5 (score), a difference these 74, 139 and 18 items cannot
     separate from one T. Request options change the average a T would be fitted on: `samples`
     replaces the automatic re-reads, and `steps` and `think` change each read. One server-wide T
     would be wrong for most of the traffic of a server that answers more than one workload.
   - **Upstream compatibility is the port's contract** (D-001). With a hook set, every parity claim
     of D-014 and D-041 would need a caveat, for arithmetic clients can do themselves.
5. **What the port offers instead.** quality.md publishes the measurements and the formula, and
   `Tools/jevbench/calibration.py` fits a T by SemIf's method on any result file with hard labels: a
   deployment runs its own labelled items through `harness.py run --items`, reads the T and its
   out-of-fold effect from `harness.py calibration --results DIR --model NAME`, and applies
   `p^(1/T)` to the answers it receives.
6. **What would justify a hook later.** A deployment's own labelled set of about 1,000 items or
   more on which SemIf's unpaired test separates; per-read label distributions, which only a
   server-side trace can give, to measure whether scaling before the mean does better than scaling
   after it; a T that holds across `samples`, `steps` and `think`; and a client that cannot
   post-process an answer. None of these exists today.
7. **Departures from SemIf's method,** both forced by the harness's standard library: the folds and
   the bootstrap draw from Python's `random.Random(217)` rather than NumPy's `default_rng(217)`, so
   the groups fall into other folds than SemIf's script would put them in; and a T divides the
   logarithms of an answer's probabilities, since an answer carries no logits. Beside SemIf's test
   the tables give the paired interval of each change over the same resampled groups. SemIf leaves
   rows with a reference distribution, as TypeSafe's are, out of its fits; JevBench's T is applied
   to them instead.

Alternatives rejected. (a) The hook as worded, T on each read's label logits: it changes answers
and cannot be fitted from them. (b) A server-side T on the averaged answer: the same arithmetic any
client can do, at the price of a non-upstream server mode. (c) Shipping a default T, such as
JevBench's 2.0: the tiers alone want 0.8 to 2.9, and a T fitted on a benchmark would be wrong for an
easier workload. (d) A T per question type or option count, as Laya's checkpoint carries
(`rl_agent_config.json`, [10-other-models.md](10-other-models.md)): the per-type control does not
separate it from one T at these sizes, and it would still be fitted on a benchmark. (e) Replacing
`confidence` with a calibrated probability: it is upstream's and Jev's wire value, and changing its
meaning would break every client that reads it.

Consequences. The port stays strictly upstream-compatible, and quality.md says with the numbers
that DiffusionGemma's probabilities are overconfident on hard questions. A deployment that needs
calibrated probabilities fits a T on its own labelled answers with the harness and rescales them
client-side. The conditions of item 6 are what would reopen the question.

Status. Proposed with issue #62.

## D-047 API documentation, guides and compatibility matrix: where the port goes beyond or differs from the issue text

Context. Issue #64 asks for DocC catalogs for `OpenJevCore`, `OpenJevDiffusionGemma` and
`OpenJevServer` (getting started in a Mac app and with the server, the request and answer types,
the backend protocol, the `OPENJEV_*` configuration reference), a compatibility page, a credits
page and a README for the implemented state, with `swift package generate-documentation`
succeeding and the documentation published through the Swift Package Index or GitHub Pages.
`OpenJevEncoders` was written after the issue, and a few points needed choices the issue does not
spell out.

Decision.

1. **Four catalogs.** `OpenJevEncoders`, Verdict and Laya on Core ML and the path to iOS, gets a
   catalog too. Each catalog's root page curates its module's symbols. The articles live where the
   symbols they link live: making decisions in an app, the request and answer types and
   implementing a backend in `OpenJevCore`; running the server and the configuration reference in
   `OpenJevServer`; one reading article in each backend module. `OpenJevCore` depends on no other
   module, so its getting-started article shows the backends in code samples and names their
   types in code voice, and the backend modules' articles link back to it.
2. **One site from one build.** `Tools/docs/build-site.sh`, which `make docs` and the Documentation
   workflow run, builds the four archives in one `generate-documentation` call with
   `--enable-experimental-combined-documentation`: each archive with
   `--transform-for-static-hosting --hosting-base-path OpenJevSwift` (the issue asks only for the
   Swift Package Index or GitHub Pages), then `docc merge` into one site with one sidebar and
   DocC's own landing page of the four modules at `documentation/`, and `Tools/docs/index.html` as
   the site's front page. Built that way, each module's archive is converted with its
   dependencies' archives, so a server or backend page can link to a core symbol.
   Such a link is absolute, ``` ``/OpenJevCore/DecisionEngine`` ```. `OpenJevServer` extends two
   core types, which gives it a page of its own named `OpenJevCore`: a relative `OpenJevCore/...`
   link resolves against that page and fails, and Swift 6.2's DocC did the same with every absolute
   form, where 6.4's falls back to the dependency. The build therefore passes
   `--exclude-extended-types`, so the two `init(_:)` the server adds to `EngineConfiguration` and
   `EncoderEngineConfiguration` are documented in the source but not on the site. Under Xcode 27's
   Swift Build the symbol graphs keep the extensions anyway, and 6.4's DocC resolves the links
   regardless. On Linux, where SwiftPM builds with the native build system, the default of the
   workflow's Xcode 26.6 too, the flag removed the page.
3. **Every warning is an error.** The build passes `--warnings-as-errors`, so a symbol link that
   does not resolve, or a parameter documented under the wrong name, fails it. Before this change
   DocC warned 66 times: 2 in `OpenJevCore`, 12 in `OpenJevServer`, 26 in `OpenJevEncoders` and 26
   in `OpenJevDiffusionGemma`, nearly all links into other modules or to internal types, and
   parameter lists that missed parameters. 93 public symbols had no doc comment: 81 in DiffusionGemma's model
   tree, configuration and loaders, the 7 log levels of `ServerSettings.LogLevel` and 5 extension
   blocks. Both are fixed in the doc comments alone, so no behaviour changes; one comment was wrong,
   `EncoderDecisionEngine` calling a `BackendContractError` a 500, where the server answers the 503
   of D-031 item 7.
4. **GitHub Pages, deployed only once enabled.** `.github/workflows/docs.yml` builds the site on
   `macos-26` with the macOS CI job's Xcode 26.6 for every pull request and push to `main` that
   changes the sources or the site's files, uploads it with `actions/upload-pages-artifact` and, on
   `main`, deploys it with `actions/deploy-pages`. A job asks the Pages API with the workflow's
   token whether Pages publishes from GitHub Actions; until it does, the deploy job is skipped and
   the run carries a notice, instead of failing. Enabling Pages is a repository setting the
   maintainers make. The Swift Package Index is left to release 0.1.0 (#65).
5. **The configuration reference is one table, checked by a test.** It lists the 32 variables the
   server and the tool read, the 31 of `ServerSettings(environment:)` and `OPENJEV_ENCODER_MODELS`,
   which the encoder store reads, with the default, the backends each applies to and what it does.
   It says what no decision recorded: `OPENJEV_VERDICT_MODEL`, `OPENJEV_LAYA_MODEL` and
   `OPENJEV_DEVICE` are read as upstream reads them but have no effect, since the
   encoders load the packages and checkpoint revisions their manifests pin (D-033) on the compute
   units D-011 chose; `OPENJEV_GEN_MAX_INFLIGHT`, `OPENJEV_GEN_MAX_QUEUE` and
   `OPENJEV_GEN_MAX_TOKENS` have no effect until generation (#53); and `OPENJEV_MAX_IMAGES` and
   `OPENJEV_MAX_IMAGE_BYTES` none until images (#48), since every backend refuses images first.
   `ConfigurationReferenceTests` (`OpenJevServerTests`) parses the table: its names must be exactly
   the `env.<reader>("OPENJEV_...")` reads of `ServerSettings.swift` and the store's variable, each
   documented default must leave `ServerSettings` as an unset variable does (and for the three
   numbers an empty value leaves at their default, the setting must hold the documented value or
   none), and every variable of [deployment.md](deployment.md)'s settings table must appear with
   the same default. Changing a default, adding a read or dropping a row fails it. CI's path
   filters, which skipped every Markdown-only change, now let a change to the reference or to
   deployment.md through, and one to `THIRD_PARTY.md`, which `FixturePinTests` already read.
6. **The compatibility page.** [compatibility.md](compatibility.md) has the three tables the issue
   names, each difference with its decision number, and the matrix of three platforms by four kinds
   of backend by seven features, each cell "yes", "no" with an issue, or "n/a". The matrix follows
   from `BackendCapabilities`, the CLI's `BackendRegistry` and the platform matrix of
   [05-architecture.md](05-architecture.md). No test reads the matrix itself: the capabilities it
   is derived from are tested where they are declared (`RuntimeTests` on a DiffusionGemma runtime
   without weights, the engines' refusal tests for the encoders), so a change there fails a test
   and this page is updated by hand.
7. **Credits beyond the served models.** [credits.md](credits.md) credits the three models served,
   the two upstream serves that this port does not yet, where each model's weights come from (the
   `Algorythm-Canada/openjev-models` releases for Verdict and Laya, the Hugging Face Hub for
   DiffusionGemma), the upstream projects, and the license of each of the 35 packages
   `Package.resolved` pins, read from their license files.
8. **The README's commands were run.** On 2026-10-02 on the reference Mac: the release build,
   `openjev serve --backend verdict` from an empty Application Support (it downloaded about 310 MB
   and served after 37 s), the `curl`, `openjev decide` and the graceful stop. The answers the
   README and the requests article show are the ones it gave. The articles' code samples were
   compiled against the package, in a test file deleted before the commit.
9. **swift-docc-plugin on every host.** The plugin is declared for Linux too, where
   `generate-documentation` builds `OpenJevCore` and `OpenJevServer`; no Linux job builds
   documentation, since the site needs the two Apple-only modules.

Alternatives rejected. (a) Four separate archives copied into one folder: their navigator indexes
and root files collide, so the sidebar would show one module, and no link could cross modules. (b)
A hosting base path per module: four sites with four sidebars and no links between them, none
at the repository's Pages path, `/OpenJevSwift/`. (c) A documentation-only umbrella target linking
every module: a target for documentation alone, which Linux could not build. (d) Generating the
configuration table from `ServerSettings`: DocC has no build step for it, and the test keeps a
written table honest at a fraction of the cost.

Consequences. A pull request that changes only DocC Markdown starts no CI run, unless it changes the
configuration reference, but starts the Documentation workflow, which builds the site. The workflow
has no cache, so each run compiles the dependencies on the macOS runner. The site is published at
<https://algorythm-canada.github.io/OpenJevSwift/> once GitHub Pages is enabled with GitHub Actions
as its source; until then the deploy job is skipped. A new public symbol needs a doc comment, and a
new setting a row in the reference.

Status. Proposed with issue #64.
