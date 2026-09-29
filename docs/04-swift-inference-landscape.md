# Swift and Apple-platform inference: what exists and what does not

Evaluated on 2026-09-29 against the requirement: run DiffusionGemma 26B-A4B's encoder prefill
and one bidirectional decoder pass over a seeded canvas, in process, from Swift, with exact
tokenization; later, run the full generation loop; later still, run 150M to 420M ModernBERT
encoders, ideally on iOS.

## 1. MLX Swift (`ml-explore/mlx-swift` 0.32.2 and `ml-explore/mlx-swift-lm`)

`mlx-swift-lm` at `c043fb3` (2026-09-28): Swift tools 6.2, platforms macOS 14, iOS 17, tvOS 17,
visionOS 1. Products: `MLXLLM`, `MLXVLM`, `MLXLMCommon`, `MLXEmbedders`, `MLXRerankers`,
`MLXHuggingFace`, `MLXFoundationModels`, `MLXGuidedGeneration`. Its only package dependencies are
`mlx-swift` and `swift-syntax`; tokenizers and downloaders are integrated through protocols
(`Tokenizer`, `TokenizerLoader`, `Downloader`), with the `MLXHuggingFace` macros wrapping
`swift-transformers`' `Tokenizers` and `swift-huggingface`'s hub client for convenience.

What it already provides that a DiffusionGemma port needs:

| Need | Present in upstream mlx-swift-lm |
|---|---|
| Quantized MoE experts | `SwitchGLU`, `SwitchLinear` (quantizable) in `SwitchLayers.swift`; `MoERouterTopK.swift`; quantized gather kernels via mlx-swift |
| Gemma 4 text blocks | `Gemma4Text.swift` (978 lines): `RMSNormNoScale`, `v_norm` without scale, no-`v_proj` fallback (`vNorm(kRaw)`), proportional RoPE with partial rotation, sliding/full layer types, `layer_scalar`, MoE configuration fields (`enable_moe_block`, `num_experts`, `top_k_experts`, `moe_intermediate_size`) |
| Gemma 4 vision | `MLXVLM/Models/Gemma4.swift` (3,388 lines) with the vision tower, embedder and processor |
| KV caches | `KVCacheSimple`, `RotatingKVCache`, `QuantizedKVCache`, `ChunkedKVCache` |
| RoPE | `RoPEUtils.swift` (`initializeRope` with scaling configs) |
| Bidirectional attention masks | `BidirectionalMasks.swift` |
| A diffusion LM precedent | `NemotronLabsDiffusion.swift`: bidirectional block denoising exposed as model-level methods that take token arrays directly, kept off the autoregressive `TokenIterator` path |
| Weight loading | `ModelFactory`, safetensors shards, per-module quantization from `config.json`, `sanitize` hooks |
| Embedders | `Bert`, `NomicBert`, `Gemma3`, `Qwen3`, `LFM2` (no ModernBERT) |
| Constrained decoding | `MLXGuidedGeneration` (xgrammar), not needed for reads |

What it does not provide: a `diffusion_gemma` model type. The documentation lists "text-to-image or
diffusion models" as usually out of scope, but the Nemotron diffusion LM shows the maintainers
accept diffusion language models. Open issue #282 reports Gemma 4 loader gaps (MoE tensor keys,
an assistant drafter type, a SIGTRAP in the 31B dense model); the MoE gap may be stale given the
configuration fields now present, and must be verified against the actual `gemma-4-26B-A4B`
checkpoint before relying on `Gemma4Text` code paths for expert weights (a spike item).

Verdict: **the right base.** Maintained by Apple's MLX team, pinned to the current mlx-swift,
already has 80% of the building blocks, and its protocols keep tokenizer and downloader choices
open.

## 2. The Layr-Labs fork of mlx-swift-lm

`Layr-Labs/mlx-swift-lm` at `eeba2af` (2026-09-29), MIT. A fork with a fixed base commit
(`7e2b710`) and an explicit policy of **not merging upstream**; needed upstream fixes are ported
by hand. It exists to serve Layr's "DarkBloom" private inference network (`d-inference`), whose
provider CLI runs MLX in process.

It contains a complete native DiffusionGemma implementation (PR #157, merged 2026-09-26):

- `Libraries/MLXLLM/Models/DiffusionGemma*.swift`: configuration, blocks (dense MLP, router,
  experts, attention, text block, self-conditioning), the text decoder with `encode(...)` and
  `denoise(canvasIds:cache:selfConditioningLogits:) → logits` (float32, soft-capped), a
  request-owned cache with sliding-window ring ordering, prefix checkpoints, soft embedding for
  quantized embeddings, expert reduction.
- `Libraries/MLXVLM/DiffusionGemma*.swift` and `Models/DiffusionGemma.swift`: the model with
  vision (`encode(tokenIds:cache:pixelValues:visualOutputLengths:visualBlockIds:)`), model
  factory and container, processor, engine, generation session, media geometry, video frames.
- `Libraries/MLXLMCommon/DiffusionGemma{Sampler,CompiledSampler,GenerationConfiguration,PrefillGeometry,TokenizerConfiguration}.swift`:
  the entropy-bound sampler, stopping rules, an experimental compiled sampler.
- About 4,450 lines of library code and 38 test files (oracle tests against pinned mlx-vlm
  `e79b0e0` and Transformers `c587bc8` outputs, live tests, tokenizer parity with 126
  cross-language cases).
- Loads `mlx-community/diffusiongemma-26B-A4B-it-4bit`. Reported on an M3 Ultra (256 GiB):
  1,658 tok/s prefill and 48.5 tok/s decode at 10k context, 25 GiB peak.

For a read, `encode` then `denoise` is exactly the pair of calls needed, and the decoder's per
position logits are returned in full. So the fork proves feasibility and offers a numeric oracle.

Reasons not to depend on it directly:

- Governance: a product fork frozen at an old upstream base, with `DARKBLOOM_*` environment
  switches and a ContinuousBatching engine designed for their server. Upstream mlx-swift-lm has
  moved on (mlx-swift 0.32.2, tokenizer decoupling, new models) and the fork will not follow.
- Dependency surface: it pulls Hummingbird, swift-huggingface, swift-transformers and swift-jinja
  into any consumer, and ships an OpenAI-compatible server of its own.
- Public library hygiene: OpenJevSwift's consumers should get a small, upstream-tracking
  dependency graph.

Verdict: **reference and oracle, not dependency.** Port the DiffusionGemma model into
OpenJevSwift on top of upstream `MLXLMCommon` primitives, following mlx-vlm and cross-checking
with the fork, with attribution (both MIT). Offer the port upstream to ml-explore later.

## 3. swift-transformers (`huggingface/swift-transformers` 1.x)

At `af520cf` (2026-09-23): Swift tools 5.9, iOS 16 and macOS 13, depends on `swift-jinja` 2.4.2
and `swift-huggingface` 0.8.1. `Tokenizers` maps `GemmaTokenizer` to its BPE tokenizer and
renders chat templates with Jinja. This is the tokenizer stack `mlx-swift-lm` integrates by
default.

Unknowns that need a spike: load time and memory for a 32 MB `tokenizer.json`; exact parity with
Hugging Face `tokenizers` for Gemma 4's byte-level BPE, added special tokens (`<|channel>`,
`<channel|>`, `<end_of_turn>`, `<|image|>`) and whitespace handling; rendering the 300-line
Gemma 4 chat template with `add_generation_prompt` and the `enable_thinking` keyword; whether
the template is read from `chat_template.jinja` when `tokenizer_config.json` has none. Upstream
OpenJev's slot resolution assumes that every label changes exactly one token in the answer
template; a tokenizer that disagrees with Python by one merge breaks every read.

## 4. Core ML

Not viable for DiffusionGemma: 25B parameters with 4-bit weights (16.6 GB), 128-expert MoE
routing, an encoder/decoder pass with dynamic canvas widths and a 262k-wide output; no existing
conversion; coremltools has no path for the MoE gather or the diffusion loop; Core ML's
compile-time shape specialisation fights the variable prompt lengths.

Viable for the small encoder models: ModernBERT conversions to Core ML exist
(`finnvoorhees/ModernBERT-CoreML`, base and large, including 4-bit; a Japanese ModernBERT
running on the Neural Engine with multiple sequence-length functions). Verdict's GLiClass head
and Laya's transformer head plus scorer are plain PyTorch modules that `coremltools` can trace.
Core ML is also the only route to the Neural Engine and the most battery-friendly route on iOS.
Whether to use Core ML or an MLX Swift ModernBERT port for Verdict and Laya is a spike.

## 5. llama.cpp

DiffusionGemma support is an open draft PR (`ggml-org/llama.cpp#24423`): CPU, CUDA and ROCm
work, Metal is unconfirmed, and a maintainer has asked for a general diffusion server design
first, with no ETA. A community fork (`cappuch/openjev.cpp`) exists. Not a base for a Swift
library today; revisit if it lands with Metal support and per-position logprobs.

## 6. Other runtimes

- Apple's Foundation Models framework: a fixed on-device model with guided generation; it cannot
  run these weights or expose logits. `mlx-swift-lm` has an adapter the other way (serving MLX
  models through `LanguageModelSession`), irrelevant for reads.
- ONNX Runtime: Verdict ships `model.onnx` (fp32 and fp16). ONNX Runtime has an iOS/macOS
  package with a Core ML execution provider, but it adds a large binary dependency; Core ML or
  MLX is preferable for two small models.
- PyTorch/ExecuTorch: no advantage over MLX or Core ML here.

## 7. HTTP server frameworks

- Hummingbird 2.x (2.23 current): rebuilt on Swift structured concurrency, Swift 6 strict
  concurrency clean, HTTP/1.1 and HTTP/2, small dependency tree, used by the Layr fork's own
  OpenAI-compatible server. Its request body streaming is what the 64 MiB cap and the "count as
  it arrives" 413 rule need.
- Vapor 5: in development; will use Hummingbird's HTTP server underneath. Vapor 4 is
  `EventLoopFuture`-based and heavier.

Verdict: **Hummingbird 2.**

## 8. Hardware and platform constraints

| Target | DiffusionGemma reads | Small encoder models | Letter readout on small Qwen3.5 |
|---|---|---|---|
| Apple silicon Mac, 32 GB+ unified memory | Yes (17 GB weights, cache growth to control) | Yes | Yes |
| Apple silicon Mac, 16 to 24 GB | Marginal: 4-bit weights load at 16 GB; expect swapping | Yes | Yes |
| Intel Mac | No MLX Metal backend | Core ML yes | No |
| iPhone / iPad | No (memory) | Yes (150M to 420M) | Small models only (0.6B to 4B, memory-dependent) |
| Linux server | No MLX; core library and tests compile | No | No |

Reference development machine used for this research: Apple M3 Max, 128 GB, macOS 27.0,
Xcode 27.0, Swift 6.4.

## 9. Existing Swift work in the Jev ecosystem

`NSStudent/JevSwiftSDK` is an independent client SDK for TypeSafe's Jev (async/await, batching,
retries, SPM). It is a client, not an inference implementation. OpenJevSwift does not need to
duplicate it; the core wire types are designed so a client can reuse them, and the server will be
tested against the official Python and TypeScript SDKs.
