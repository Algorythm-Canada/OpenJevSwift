# Risks and unknowns

Each row names the spike or task that resolves it. Likelihood and impact are the author's
estimates on 2026-09-29.

| # | Risk or unknown | Likelihood | Impact | Mitigation | Resolved by |
|---|---|---|---|---|---|
| R1 | **Tokenizer parity.** swift-transformers' BPE differs from Hugging Face `tokenizers` for Gemma 4 on some inputs (special tokens, whitespace, non-ASCII, byte fallback). Any difference breaks template slot resolution or shifts positions. | Resolved 2026-09-30 | Critical | Encoding matched on every fixture text (917 corpus rows, 2,634 engine encodings, 255 labels, special tokens). Two decode departures in swift-transformers 1.3.4 (trailing byte tokens dropped, `clean_up_tokenization_spaces` defaulting to true) are handled generally in `SwiftTransformersTokenizer` and pinned by tests; no custom loader. Load 3.6 s, 204 MB. | Spike #20 (D-008, [spikes/tokenizer-parity.md](spikes/tokenizer-parity.md)); follow-ups A to C and E there |
| R2 | **Chat template parity.** swift-jinja renders Gemma 4's 300-line template differently (whitespace control, `enable_thinking`, tools macros), or the template is not picked up from `chat_template.jinja`. | Resolved 2026-09-30 | High | swift-transformers 1.3.4 reads `chat_template.jinja` itself; swift-jinja 2.5.1 rendered all 24 fixture prompts, thinking off and on, to the recorded text and ids. The image message shape renders through the same path. No hand-rolled builder. | Spike #21 (D-008, [spikes/chat-template.md](spikes/chat-template.md)); follow-up D there |
| R3 | **Numeric parity of the model port.** BF16 accumulation, quantized matmul kernels, expert gather order, precise softmax and float32 softcap differ between MLX Swift and Python MLX; slot probabilities drift. | Measured 2026-09-30: certain drift on native kernels, none on the oracle's | High | Spike #22 wrote mlx-vlm's read in Swift on ml-explore/mlx-swift 0.32.2 and mlx-swift-lm `c043fb3`. With the mlx-metal wheel's `mlx.metallib` and mlx-vlm's RoPE table it matched all 27 oracle reads bit for bit, prefill caches included. On mlx-swift's own kernels it drifts up to 0.374 in label probability, 0.0084 on average, with the top label on 150 of 156 slots. The cause is the JIT-compiled `pow` and the SwiftPM-compiled RoPE, which round in the last bit; the read amplifies that. The Layr-Labs fork drifts 0.451 (147 of 156), and mlx-vlm drifts 0.618 (148 of 156) from itself under a chunked prefill. Conformance is two-tier (D-014): exact under the oracle's kernels, aggregate bounds on native ones. | Spike #22 (D-004, D-014, [spikes/backend-validation.md](spikes/backend-validation.md)); numeric parity tests #31 |
| R4 | **Memory.** 17 GB of weights plus prefill caches plus MLX's buffer pool (36 GB observed upstream) exceed 32 GB machines. | High (measured 2026-09-30) | High | Measured on an M3 Max with 12 prompts' prefills cached and 54 reads run. The physical footprint is 15.4 to 16.6 GiB after load and 25.2 to 28.1 GiB after the reads, of which MLX's buffer pool is 6.0 to 8.7 GiB. A 4 GB cache limit gives 21.0 to 24.3 GiB with identical results: Python byte-identical to the oracle, Swift bit-identical to its own unlimited run. A cached prefill costs 0.22 MB per prompt token when every position is kept, as mlx-vlm does (2.30 GiB for 10,786 tokens). A 1,024-slot ring, exact for reads, caps the sliding layers at 0.21 GB per prompt. The library's loader (#27, `DiffusionGemmaModel.load(from:)`) took 1.0 to 3.6 s over three warm runs on 2026-10-01 (15.41 GiB of shards); MLX holds 14.35 GiB after load, and the resident size after load was 15.50 and 15.92 GiB in two runs, 9.46 GiB in the first, whose weights were not yet all resident. The runtime (#29, `DiffusionGemmaRuntime`) measured on the same M3 Max on 2026-10-01 over 100 unique short prompts through the engine (upstream's README questions, 355 reads with `cacheLimitGB` 4 and 301 without, the prefill cache at its 12 entries): MLX active 14.35 GiB before and 14.80 GiB after, MLX's pool 0.65 to 0.66 GiB with the limit and the same without, MLX's peak 15.09 GiB in both runs, resident size unchanged by the reads (15.08 to 15.49 GiB over three runs, the same before and after within 0.04 GiB); the live test bounds it at 20 GiB. Short prompts do not grow the pool, so the limit changes nothing there; the growth the spike saw comes from long prompts' prefills. Mitigations in place: the bounded prefill cache (12 entries, 16,384 tokens, no exempt entry), `cacheLimitGB` (`Memory.cacheLimit`, nil leaves MLX alone, 0 disables the pool), the 32,768-token prompt cap and `memoryReport()`. Still to do: slot-only logits (D-015, unproven), document machine requirements, measure long-prompt workloads and 32/48/64 GB machines. | Spike #22 (numbers above); #29 (runtime figures); performance baseline task |
| R5 | **Performance.** Swift port slower than Python MLX (0.2 to 0.4 s per 3-question read) because of expert gather paths or missing compiled kernels. | Medium | Medium | Profile; reuse `SwitchGLU` quantized paths; slot-only projection; optional `compile` of softcap and sampler. | Performance baseline task |
| R6 | **mlx-swift-lm API churn.** 0.32.x breaks APIs (tokenizer decoupling already happened once); the model port depends on internal primitives. | High | Medium | Pin exact versions; keep the model port self-contained; track upstream in a scheduled task. | Package skeleton task; upstream tracking task |
| R7 | **Python `json.dumps` compatibility for seeds and model text.** Float repr, `ensure_ascii` escaping, `sort_keys` with mixed key types, and separators must match CPython or seeds and prompts differ. | Medium | Medium (conformance only) | Implement and fixture-test; documented fallback for states with floats (D-006). | Canonical JSON spike (milestone 0) |
| R8 | **Order preservation.** Foundation JSON loses object order; a missed path (for example error `input` echo or `legend`) silently reorders criteria. | Medium | High | Own parser and serialiser; property tests round-tripping order; server never touches Foundation JSON for request bodies. | Ordered JSON task |
| R9 | **Vision preprocessing parity.** Gemma 4 processor: resize policy, normalisation, token budget selection, placeholder expansion, `mm_token_type_ids`, bidirectional block overlay. Upstream needed a vLLM patch for the overlay. | High | Medium (images are an extension) | Reuse `MLXVLM` Gemma 4 processor if public; oracle `pixel_values` and expanded ids from mlx-vlm; hotdog fixture. | Vision processor task; vision live test |
| R10 | **Generation loop parity.** Entropy-bound sampler, temperature schedule, self-conditioning, stopping, block commits. Affects `think` and chat quality, not reads. | High | Medium | Port from mlx-vlm with the Layr fork's Swift sampler as a second reference; live tests compare greedy outputs. | Generation milestone tasks |
| R11 | **Encoder models on Apple.** Laya's `DecisionModel` head and Verdict's GLiClass head must be reproduced exactly, including per-option-count temperatures and abstention handling; ModernBERT has no MLX Swift implementation; Core ML conversion of custom heads may need tracing work. Now: both convert and match upstream within its own bfloat16 variation (findings below); what remains is Core ML's own defects (a CPU-backend crash with enumerated shapes, a misleading multifunction load error, silent CPU fallbacks) and Laya's download on iPhones (one 845 MB package per length while its multifunction package does not load for the Neural Engine). | Low | Medium | Spike #56: Core ML packages with iOS 18 multifunction shapes, parity fixtures in Fixtures/encoders, the Swift harness in Tools/encoders; test the shipped configurations on each OS release; a server read over the wire API as fallback; the MLX port (5 to 6 days) if Core ML regresses. | Spike #56 (2026-09-30); issues #57 and #58 |
| R12 | **Upstream drift.** Upstream is 11 days old, adds a model a week, and has open PRs (ForJev backend). The pin will age. | Certain | Medium | Pinned commit in `THIRD_PARTY.md`; a recurring review task; fixtures regenerated per pin move. | Upstream tracking task |
| R13 | **Hosted CI cannot run the model.** GitHub-hosted macOS runners have no room for 17 GB weights. Their virtual GPU does run MLX's kernels for small synthetic shapes, once the build includes MLX's Metal library (findings below). | Certain for the model | Low (model tests are opt-in) | Core tests on Linux and macOS runners; MLX synthetic tests on the `macos-26` GPU, built with Swift Build (D-028); model tests opt-in behind `OPENJEV_TEST_MODEL`; consider a self-hosted Apple silicon runner later. | CI task #7 and CI spike #8, both 2026-09-30 |
| R14 | **Concurrency correctness.** MLX arrays are not `Sendable`; GPU work must stay on one execution context; Swift 6 strict concurrency will fight the natural design. | Resolved (2026-10-01) | Medium | Resolved by #29: `DiffusionGemmaRuntime` is an actor that owns the model, the tokenizer and the prefill cache, and runs every MLX evaluation inside itself, one at a time; only `CanvasRead`, `ReadResult` and other values cross it, in Swift 6 mode with strict concurrency. 20 concurrent `decide` calls over 20 different requests gave the same answers as the same requests one at a time, and a cold and a cached prefill read bit-identical maps. | #29 |
| R15 | **Sliding-window decoder masks for long prompts.** The rotating cache's physical order differs from temporal order; masks and the "last 1023 positions" slice are easy to get subtly wrong for prompts over 1024 tokens. | Resolved 2026-10-01 | High | Spike #22: a 1,024-slot ring is exact for reads (the cross-feed control), and the transliteration's reads of the four prompts over the window are bit-exact against the oracle in the exact tier. The library's encoder pass over indexed_12_mixed/g0 (1,572 tokens, the boolean window mask) is bit-identical to mlx-vlm on every recorded stage of layers 0 to 5 in the exact tier (#24). The decoder pass (#26) keeps the window cut and its oracle reads. | Spike #22; issue #24 |
| R16 | **Licensing and trademarks.** "Jev" is TypeSafe's mark; Gemma Terms of Use govern the weights; ported MIT code needs attribution. | Low | Medium | Disclaimers (D-002), notices (D-010), no weights in the repository. | Done in docs; kept current |
| R17 | **Quantized embedding self-conditioning.** With 8-bit `embed_tokens`, mlx-vlm switches to logits-based self-conditioning and the fork projects through packed weights; getting this wrong only shows with `steps > 1`. | Resolved 2026-10-01 | Low (extension) | The read (#28) follows mlx-vlm's `_embed_canvas`: the precise softmax of the previous float32 logits, `quantizedMM` against the packed embedding with `transpose: false`, the embedding scale, then the module. In D-014's exact tier the six `steps` 2 and 3 reads (quickstart, lines_10_mixed, long_state) are bit-identical to the oracle, written argmaxes included (6 of 6); under native kernels the written argmaxes match on 4 of 6, as the spike measured, and the reads stay within D-014's bounds. | Issues #26, #28; D-036 |
| R18 | **Swift package size and build time.** `MLXVLM` (Gemma 4, 3,400 lines) and vendored xgrammar in mlx-swift-lm inflate builds for a library consumer who only wants reads. | Medium | Low | Depend on `MLXLMCommon` and only the pieces needed; copy the Gemma 4 vision tower into the vision target if `MLXVLM` proves too heavy. | Package skeleton task; vision task |
| R19 | **Canvas geometry mismatch.** OpenJev's vLLM path pads the canvas to a multiple of 16 up to 64; the model's native canvas is 256. RoPE offsets, masks and the `<end_of_turn>` slot must match the Python MLX backend exactly. | Resolved 2026-10-01 | High | The read takes the canvas as upstream builds it, at RoPE offset = the prompt length. The 27 oracle reads cover widths 16, 32, 48 and 64 and prompts up to 2,939 tokens: all 27 are bit-identical in D-014's exact tier, and under native kernels the mean label probability difference is 0.0084 (0.0054 on the long prompts) with the top label on 150 of 156 slots and 120 of 120 where the oracle's margin is at least 0.5. | Issues #26, #28; D-036 |
| R20 | **Gemma 4 MoE loading in mlx-swift-lm.** Issue #282 says the 26B-A4B MoE checkpoint does not load upstream; if the expert tensor layout handling is missing, the port must implement it. | Resolved 2026-09-30 | Low | Upstream `c043fb3` `LLMModelFactory` loads `mlx-community/gemma-4-26B-A4B-it-4bit` (`0d77464e`): 270 expert tensors, as 90 quantized `experts.switch_glu` projections, and a greedy "Paris". DiffusionGemma's fused, quantized `experts.gate_up_proj` has no upstream path, because `Gemma4Text`'s sanitize splits only unquantized tensors. The port loads it into upstream's `SwitchLinear` with 1,408 outputs through `loadWeights` and `perLayerQuantization`, bit-exact in spike #22. | Spike #22 ([spikes/backend-validation.md](spikes/backend-validation.md)); weight loading #27 |

## Unknowns that need experiments, not reading

1. Does swift-transformers tokenize Gemma 4 identically to Python for the fixture corpus, and how
   long does the 32 MB `tokenizer.json` take to load? (R1) Answered by spike #20: encoding is
   identical on every fixture text; decoding needed two general fixes in the adapter; loading
   takes 3.6 s and 204 MB.
2. Does swift-jinja render the shipped chat template identically? (R2) Answered by spike #21:
   yes, text and ids, for every fixture prompt with thinking off and on.
3. What tolerance do slot logprobs need between MLX Swift and Python MLX on the same 4-bit
   weights, and does the top label ever change on the fixture set? (R3)
4. What is the resident memory of a Swift process serving reads on a 32 GB and a 48 GB machine
   with the default cache budgets, and what cache limit keeps it under 24 GB? (R4)
5. Do GitHub-hosted macOS runners run MLX Metal kernels for small synthetic tests? (R13; answered
   below: yes, with a build that carries MLX's Metal library)
6. Can CPython's float repr be reproduced exactly in Swift for the values that appear in JSON
   states? (R7)
7. Does Core ML run Verdict's GLiClass model with identical probabilities after temperature
   scaling, and at what latency on an iPhone? (R11; answered below: within upstream's own bfloat16
   variation, and 10 to 46 ms a question on an A15's Neural Engine)

## Findings for R13: MLX on hosted runners

Issue #8 asked whether GitHub-hosted Apple silicon runners execute mlx-swift's Metal kernels well
enough for unit tests on small synthetic shapes. [mlx-probe.yml](../.github/workflows/mlx-probe.yml)
answered it on 2026-09-30 in runs [36732539925](https://github.com/Algorythm-Canada/OpenJevSwift/actions/runs/36732539925)
and [36734546044](https://github.com/Algorythm-Canada/OpenJevSwift/actions/runs/36734546044). The
probe is a throwaway package on mlx-swift 0.32.2. It runs a float32 matmul (64, 256 and 1,024
square), an MLXNN block (`Linear`, GELU, `RMSNorm` and `MultiHeadAttention` with a causal mask),
and the kernels DiffusionGemma relies on (a bfloat16 matmul, a 4-bit matmul, an expert-gathered
4-bit matmul and RoPE). The executable runs every check on the CPU and GPU once per build system.
With Swift Build, `swift test` separately runs the matmul and attention checks on both devices, with
and without setting `GPU.metallib`. Python MLX 0.32.2, the MLX core that mlx-swift 0.32.2 vendors,
checks the GPU independently of how SwiftPM builds the shaders.

| Label | Image | Host | Xcode | mlx-swift 0.32.2 | Python MLX on the GPU |
|---|---|---|---|---|---|
| `macos-15` | 20260907.0337.1 | macOS 15.7.9 | 26.3, the newest on the image: Swift 6.2.4 | Does not resolve: the manifest needs tools 6.3 | Runs |
| `macos-26` | 20260907.0351.1 | macOS 26.6.2 | 26.6: Swift 6.3.3, Metal Toolchain installed | Runs with Swift Build; the native build cannot run MLX | Runs |
| `xcode-27` (preview) | 20260921.0210.1 | macOS 27.0 | 27.0: Swift 6.4, Metal Toolchain downloaded (839 MB in about 20 s) | Runs with Swift Build; the native build cannot run MLX | Runs |

Every host is an Apple M1 virtual machine with 3 CPUs and 7 GB of memory.

1. **Metal is a paravirtual GPU, and MLX's kernels run on it.** Metal reports
   `Apple Paravirtual device`, architecture `air64_v27`, a recommended working set of 4,778 MiB,
   and none of the Apple GPU families 7 to 9 or Metal 3. Every probe kernel ran on it, and the GPU
   results agree with the CPU's: float32 matmuls within 4.6e-5, the attention block within 9.5e-7,
   the 4-bit and expert-gathered 4-bit matmuls within 1.2e-6, RoPE within 1.1e-5, and the bfloat16
   matmul exactly. The GPU's 4-bit matmul is within 1.2e-6 of a float32 matmul on the dequantized
   weights. Python MLX ran its matmul on the GPU of all three images, `macos-15` included.
2. **The build system decides whether MLX runs at all.** Xcode 26.6's `swift build` uses the native
   build system, which leaves out mlx-swift's Metal shaders. Without that Metal library MLX cannot
   create a stream, so the CPU runs fail as well as the GPU runs, with
   `Failed to load the default metallib`. Swift Build compiles the shaders, and every executable
   run of that build passed. `macos-26` has the Metal Toolchain Swift Build needs, installed with the image.
3. **Inside `swift test`, MLX must be pointed at the library.** Swift Build copies the library
   into the test bundle, but MLX looks for it through `Bundle` objects and the Swift Testing runner
   creates none for the test bundle. The probe's test failed that way on `macos-26` and `xcode-27`,
   and passed on both, on the CPU and the GPU, once it set `GPU.metallib` to the copy in the test
   bundle. [development.md](development.md) ("MLX in tests") has the helper.
4. **CPU fallback saves nothing.** A CPU-only test needs the same build and the same Metal library,
   because MLX loads the library as soon as it creates a stream on a Mac. CPU results match the CPU
   reference exactly, GPU results within the tolerances above, and the model runs on the GPU.
5. **Time and cost.** On `macos-26` the probe built in 115 to 160 s with either build system. Its
   steady GPU times were 3.6 ms for a 1,024-square matmul (CPU 8.0 ms) and 2.8 ms for the attention
   block (CPU 47 ms). The first GPU call of a kernel compiles its pipeline, about 0.3 s for a matmul
   and 1.2 to 1.6 s for the whole attention block, once per test process. The repository's macOS
   job, which builds everything with Swift Build, took 8 minutes with an empty cache, 270 s of it
   building, and 3 minutes with a warm one. Synthetic tests on small shapes add seconds to that, well
   inside the 5-minute target. The standard runners cost nothing for this public repository.
   `macos-26-xlarge` (M2 Pro, which GitHub describes as GPU accelerated) is billed per minute and
   was not probed; the workflow's `include_xlarge` input adds it.

D-028 records the decision that follows: MLX synthetic tests run in CI on the `macos-26` GPU.

## Findings for R11: the encoder models on Core ML

Spike #56 answered unknown 7 on 2026-09-30, with the evidence in
[spikes/encoder-runtime.md](spikes/encoder-runtime.md):

1. **Parity.** Converted to float16 Core ML packages, both models stay inside the variation
   upstream's own serving carries. Against PyTorch float32 on 200 questions, Verdict's calibrated
   probabilities moved by at most 0.0018 on the GPU and 0.0082 on the iPhone's Neural Engine, and
   Laya's by at most 0.0039 on the GPU and 0.0181 on the Neural Engine; upstream's bfloat16 serving
   moves them by 0.0115 (Verdict) and 0.0191 (Laya). The top answer changed only on near ties: 1 to
   3 of 200 for Verdict and 1 of 200 for Laya on the Neural Engine, each where the reference's top
   two options were within 0.0021 of each other. Tokenization matches exactly, and calibration
   within 1e-6.
2. **Latency on an iPhone 13 Pro Max.** Verdict answers a question in 9.7, 17 and 46 ms at 128, 256
   and 512 tokens on the Neural Engine, in 106 MB; Laya in 27.9, 57.1, 137 and 513 ms at 128 to
   1,024 tokens on the Neural Engine, in 106 MB or less, loading included (82 to 1,346 ms and 855 MB
   on the GPU). Every run longer than two minutes took the phone to the serious thermal state.
3. **Core ML's CPU backend crashes on enumerated input shapes** on macOS 27.0.1 and iOS 27.0
   (`BNNSGraphContextExecute_v2`), which rules out iOS 17 packages; iOS 18 multifunction packages
   and one-shape packages avoid it, and iOS 18 costs no devices.
4. **Laya needed two graph rewrites** before Core ML would plan it on the Neural Engine at all (a
   one-hot type embedding, a rank-4 head), and its multifunction package still does not load for the
   Neural Engine on either machine. One package per length, holding one fixed shape, does, at 845 MB
   each.
5. **The residual risk** is Core ML itself: the crash, the misleading multifunction load error,
   silent CPU fallbacks, and Laya's multifunction package that does not load for the Neural Engine,
   which makes an iPhone download one 845 MB package per length. The backends test the
   configurations they ship on each OS release, and the MLX estimate (5 to 6 days) stays the
   fallback.
