# Risks and unknowns

Each row names the spike or task that resolves it. Likelihood and impact are the author's
estimates on 2026-09-29.

| # | Risk or unknown | Likelihood | Impact | Mitigation | Resolved by |
|---|---|---|---|---|---|
| R1 | **Tokenizer parity.** swift-transformers' BPE differs from Hugging Face `tokenizers` for Gemma 4 on some inputs (special tokens, whitespace, non-ASCII, byte fallback). Any difference breaks template slot resolution or shifts positions. | Medium | Critical | Fixture corpus from Python covering labels, scaffold, markers, JSON states, CJK and emoji; fall back to a custom tokenizer loader if needed. | Tokenizer parity spike (milestone 2) |
| R2 | **Chat template parity.** swift-jinja renders Gemma 4's 300-line template differently (whitespace control, `enable_thinking`, tools macros), or the template is not picked up from `chat_template.jinja`. | Medium | High | Compare rendered ids for `[system, user]` with generation prompt, thinking on and off; hand-rolled builder as fallback. | Chat template spike (milestone 2) |
| R3 | **Numeric parity of the model port.** BF16 accumulation, quantized matmul kernels, expert gather order, precise softmax and float32 softcap differ between MLX Swift and Python MLX; slot probabilities drift. | High (small drift) / Low (large drift) | High | Oracle logprobs from mlx-vlm on fixture canvases; tolerance-based conformance (D-014); test long prompts separately. | Backend validation spike; numeric parity tests |
| R4 | **Memory.** 17 GB of weights plus prefill caches plus MLX's buffer pool (36 GB observed upstream) exceed 32 GB machines. | High | High | Bounded prefill cache (entries and tokens), `MLX.GPU.set(cacheLimit:)`, slot-only logits, document machine requirements; measure on 32/48/64 GB machines. | Runtime actor task; performance baseline task |
| R5 | **Performance.** Swift port slower than Python MLX (0.2 to 0.4 s per 3-question read) because of expert gather paths or missing compiled kernels. | Medium | Medium | Profile; reuse `SwitchGLU` quantized paths; slot-only projection; optional `compile` of softcap and sampler. | Performance baseline task |
| R6 | **mlx-swift-lm API churn.** 0.32.x breaks APIs (tokenizer decoupling already happened once); the model port depends on internal primitives. | High | Medium | Pin exact versions; keep the model port self-contained; track upstream in a scheduled task. | Package skeleton task; upstream tracking task |
| R7 | **Python `json.dumps` compatibility for seeds and model text.** Float repr, `ensure_ascii` escaping, `sort_keys` with mixed key types, and separators must match CPython or seeds and prompts differ. | Medium | Medium (conformance only) | Implement and fixture-test; documented fallback for states with floats (D-006). | Canonical JSON spike (milestone 0) |
| R8 | **Order preservation.** Foundation JSON loses object order; a missed path (for example error `input` echo or `legend`) silently reorders criteria. | Medium | High | Own parser and serialiser; property tests round-tripping order; server never touches Foundation JSON for request bodies. | Ordered JSON task |
| R9 | **Vision preprocessing parity.** Gemma 4 processor: resize policy, normalisation, token budget selection, placeholder expansion, `mm_token_type_ids`, bidirectional block overlay. Upstream needed a vLLM patch for the overlay. | High | Medium (images are an extension) | Reuse `MLXVLM` Gemma 4 processor if public; oracle `pixel_values` and expanded ids from mlx-vlm; hotdog fixture. | Vision processor task; vision live test |
| R10 | **Generation loop parity.** Entropy-bound sampler, temperature schedule, self-conditioning, stopping, block commits. Affects `think` and chat quality, not reads. | High | Medium | Port from mlx-vlm with the Layr fork's Swift sampler as a second reference; live tests compare greedy outputs. | Generation milestone tasks |
| R11 | **Encoder models on Apple.** Laya's `DecisionModel` head and Verdict's GLiClass head must be reproduced exactly, including per-option-count temperatures and abstention handling; ModernBERT has no MLX Swift implementation; Core ML conversion of custom heads may need tracing work. | Medium | Medium | Spike converting Verdict; parity against PyTorch outputs recorded as fixtures. | Encoder spike (milestone 6) |
| R12 | **Upstream drift.** Upstream is 11 days old, adds a model a week, and has open PRs (ForJev backend). The pin will age. | Certain | Medium | Pinned commit in `THIRD_PARTY.md`; a recurring review task; fixtures regenerated per pin move. | Upstream tracking task |
| R13 | **Hosted CI cannot run the model.** GitHub-hosted macOS runners have no room for 17 GB weights; MLX unit tests may or may not run on their GPUs. | High | Low (tests are opt-in) | Core tests on Linux and macOS runners; model tests opt-in behind an environment variable; consider a self-hosted Apple silicon runner later. | CI task and CI spike |
| R14 | **Concurrency correctness.** MLX arrays are not `Sendable`; GPU work must stay on one execution context; Swift 6 strict concurrency will fight the natural design. | Medium | Medium | One runtime actor owns all MLX state; only value types cross its boundary; structured concurrency for group fan-out. | Runtime actor task |
| R15 | **Sliding-window decoder masks for long prompts.** The rotating cache's physical order differs from temporal order; masks and the "last 1023 positions" slice are easy to get subtly wrong for prompts over 1024 tokens. | Medium | High | Dedicated parity test with a 3,000-token state against the oracle. | Decoder pass task; numeric parity tests |
| R16 | **Licensing and trademarks.** "Jev" is TypeSafe's mark; Gemma Terms of Use govern the weights; ported MIT code needs attribution. | Low | Medium | Disclaimers (D-002), notices (D-010), no weights in the repository. | Done in docs; kept current |
| R17 | **Quantized embedding self-conditioning.** With 8-bit `embed_tokens`, mlx-vlm switches to logits-based self-conditioning and the fork projects through packed weights; getting this wrong only shows with `steps > 1`. | Medium | Low (extension) | Oracle test for `steps` 2 and 3 against mlx-vlm. | Multi-step read task |
| R18 | **Swift package size and build time.** `MLXVLM` (Gemma 4, 3,400 lines) and vendored xgrammar in mlx-swift-lm inflate builds for a library consumer who only wants reads. | Medium | Low | Depend on `MLXLMCommon` and only the pieces needed; copy the Gemma 4 vision tower into the vision target if `MLXVLM` proves too heavy. | Package skeleton task; vision task |
| R19 | **Canvas geometry mismatch.** OpenJev's vLLM path pads the canvas to a multiple of 16 up to 64; the model's native canvas is 256. RoPE offsets, masks and the `<end_of_turn>` slot must match the Python MLX backend exactly. | Low | High | Fixtures include canvases and widths; parity test per width (16, 32, 48, 64). | Canvas task; parity tests |
| R20 | **Gemma 4 MoE loading in mlx-swift-lm.** Issue #282 says the 26B-A4B MoE checkpoint does not load upstream; if the expert tensor layout handling is missing, the port must implement it. | Medium | Medium | Verify against the actual checkpoint in the backend spike; implement `SwitchLinear` loading with mlx-vlm's `sanitize` renames if needed. | Backend validation spike; weight loading task |

## Unknowns that need experiments, not reading

1. Does swift-transformers tokenize Gemma 4 identically to Python for the fixture corpus, and how
   long does the 32 MB `tokenizer.json` take to load? (R1)
2. Does swift-jinja render the shipped chat template identically? (R2)
3. What tolerance do slot logprobs need between MLX Swift and Python MLX on the same 4-bit
   weights, and does the top label ever change on the fixture set? (R3)
4. What is the resident memory of a Swift process serving reads on a 32 GB and a 48 GB machine
   with the default cache budgets, and what cache limit keeps it under 24 GB? (R4)
5. Do GitHub-hosted macOS runners run MLX Metal kernels for small synthetic tests? (R13)
6. Can CPython's float repr be reproduced exactly in Swift for the values that appear in JSON
   states? (R7)
7. Does Core ML run Verdict's GLiClass model with identical probabilities after temperature
   scaling, and at what latency on an iPhone? (R11)
