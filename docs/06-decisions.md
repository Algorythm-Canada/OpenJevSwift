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
