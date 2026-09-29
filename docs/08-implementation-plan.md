# Implementation plan

Eight milestones, in dependency order. Each milestone has a tracking issue that lists its
issues as sub-issues. The issue index at the bottom is generated from GitHub after the issues were
created and links every issue.

## Ordering rationale

1. **Foundations first (milestone 0).** Order-preserving JSON, wire types and the fixture
   tooling are used by every later test. The package skeleton fixes the Swift tools version,
   platforms and dependency pins that the model code will build against. Two spikes here retire
   cheap unknowns early: Python-compatible canonical JSON, and whether hosted CI can run MLX.
2. **The engine core without a model (milestone 1).** Upstream's algorithm is mostly text and
   integer manipulation. Porting it first, against fixtures, means the model work later has a
   correct harness to plug into, and that most of the project can be developed and reviewed on
   any Mac or Linux box.
3. **The model (milestone 2).** Starts with three spikes (tokenizer parity, chat template parity,
   backend validation with the fork and mlx-vlm as oracles), then the port in the order the data
   flows: configuration, blocks, encoder, decoder read, weights, multi-step, runtime actor,
   download, parity tests, warmup, performance baseline.
4. **The server (milestone 3).** Only needs the core and a stub backend to be complete and
   testable; it is ordered after the model so that live tests can run as soon as it lands, but it
   can be developed in parallel with milestone 2 by a second engineer.
5. **Read extensions and images (milestone 4).** `steps`, `samples` and `sequential` are engine
   plumbing over the read pass; images need the Gemma 4 vision tower and processor.
6. **Generation (milestone 5).** The diffusion generation loop, then `think`, then chat.
7. **Other models (milestone 6).** JevK5 first (no new model code), then Verdict and Laya after
   the Core ML versus MLX spike, then a CLM decision.
8. **Quality and release (milestone 7).** JevBench harness, calibration report, benchmarks,
   DocC, release 0.1.0, upstream tracking.

## Dependency graph (milestones)

```
M0 Foundations
 └─► M1 Engine core ──► M2 DiffusionGemma reads ──► M4 Extensions and images ──► M5 Generation
                   └──► M3 Server (needs only M1 + a stub backend; live tests need M2)
                                                          M6 Other models (needs M1, M3; MLXLLM for JevK5)
                                                          M7 Quality and release (needs M2, M3)
```

## Recommended first issue

[#2 Package skeleton: targets, platforms, strict concurrency, dependency pins](https://github.com/Algorythm-Canada/OpenJevSwift/issues/2), because every other issue creates files
inside the targets it defines, and because resolving `mlx-swift-lm` at a pinned commit with Swift
tools 6.2 and strict concurrency on both macOS and Linux is the first thing that can fail.
Immediately after it, in parallel: the order-preserving JSON model and the fixture generation
tooling.

## Parallel tracks

Once milestone 0 is done, three tracks can proceed independently:

- Track A (any machine): milestone 1, then the server (milestone 3) against the stub backend.
- Track B (Apple silicon with 32 GB or more and the weights): milestone 2 spikes, then the port.
- Track C (any machine, later): JevK5 prompt and readout plumbing; encoder spike preparation.

## Sizing

Rough effort at a senior Swift engineer's pace, excluding review:

| Milestone | Estimate |
|---|---|
| 0 Foundations | 1.5 weeks |
| 1 Engine core | 2 weeks |
| 2 DiffusionGemma reads | 4 to 6 weeks (dominated by parity debugging) |
| 3 Server | 2 weeks |
| 4 Extensions and images | 3 weeks (images are two thirds of it) |
| 5 Generation | 3 to 4 weeks |
| 6 Other models | 3 to 5 weeks depending on the Core ML spike |
| 7 Quality and release | 2 weeks |

## Issue index

See the GitHub milestones: https://github.com/Algorythm-Canada/OpenJevSwift/milestones.
The table below is maintained by hand when issues are added or closed.

<!-- ISSUE_INDEX_START -->
### 0. Foundations

| Issue | Title |
|---|---|
| [#1](https://github.com/Algorythm-Canada/OpenJevSwift/issues/1) | [Tracking] Milestone 0: Foundations (tracking) |
| [#2](https://github.com/Algorythm-Canada/OpenJevSwift/issues/2) | Package skeleton: targets, platforms, strict concurrency, dependency pins |
| [#3](https://github.com/Algorythm-Canada/OpenJevSwift/issues/3) | Order-preserving JSON value model, strict parser and Python-compatible serializer |
| [#4](https://github.com/Algorythm-Canada/OpenJevSwift/issues/4) | Spike: byte-exact CPython json.dumps compatibility (float repr, escaping, key sorting) |
| [#5](https://github.com/Algorythm-Canada/OpenJevSwift/issues/5) | Wire types: request, questions, answers, usage, models and error bodies |
| [#6](https://github.com/Algorythm-Canada/OpenJevSwift/issues/6) | Golden fixture generation from the pinned upstream commit |
| [#7](https://github.com/Algorythm-Canada/OpenJevSwift/issues/7) | Continuous integration: Linux and macOS builds, fixture tests, formatting |
| [#8](https://github.com/Algorythm-Canada/OpenJevSwift/issues/8) | Spike: can GitHub-hosted macOS runners execute MLX Metal kernels for small tests |

### 1. Decision engine core

| Issue | Title |
|---|---|
| [#9](https://github.com/Algorythm-Canada/OpenJevSwift/issues/9) | [Tracking] Milestone 1: Decision engine core (tracking) |
| [#10](https://github.com/Algorythm-Canada/OpenJevSwift/issues/10) | Question schema builder: forced answers, limits, q-ids, answer format |
| [#11](https://github.com/Algorythm-Canada/OpenJevSwift/issues/11) | System text and answer template rendering (lines and indexed formats) |
| [#12](https://github.com/Algorythm-Canada/OpenJevSwift/issues/12) | Tokenizer protocol for the core and single-token label discovery |
| [#13](https://github.com/Algorythm-Canada/OpenJevSwift/issues/13) | Answer template slot resolution and template cache |
| [#14](https://github.com/Algorythm-Canada/OpenJevSwift/issues/14) | Question grouping, canvas width and canvas construction |
| [#15](https://github.com/Algorythm-Canada/OpenJevSwift/issues/15) | Python-compatible RNG (MT19937, randrange) and request seed derivation |
| [#16](https://github.com/Algorythm-Canada/OpenJevSwift/issues/16) | Slot distributions, entropy, confidence and answer assembly |
| [#17](https://github.com/Algorythm-Canada/OpenJevSwift/issues/17) | DecisionEngine, DecisionBackend protocol, read policies and a stub backend |
| [#18](https://github.com/Algorythm-Canada/OpenJevSwift/issues/18) | Image input validation: data URLs, objects, types, size bound before decoding |

### 2. DiffusionGemma reads on MLX

| Issue | Title |
|---|---|
| [#19](https://github.com/Algorythm-Canada/OpenJevSwift/issues/19) | [Tracking] Milestone 2: DiffusionGemma reads on MLX (tracking) |
| [#20](https://github.com/Algorythm-Canada/OpenJevSwift/issues/20) | Spike: Gemma 4 tokenizer parity and load time with swift-transformers |
| [#21](https://github.com/Algorythm-Canada/OpenJevSwift/issues/21) | Spike: chat template rendering parity (swift-jinja) versus a hand-rolled Gemma 4 prompt builder |
| [#22](https://github.com/Algorythm-Canada/OpenJevSwift/issues/22) | Spike: validate the read-only pass in Swift with the Layr-Labs fork and mlx-vlm as oracles |
| [#23](https://github.com/Algorythm-Canada/OpenJevSwift/issues/23) | DiffusionGemma configuration decoding |
| [#24](https://github.com/Algorythm-Canada/OpenJevSwift/issues/24) | Text blocks: attention, dense GeGLU MLP, router, quantized experts, layer scalar |
| [#25](https://github.com/Algorythm-Canada/OpenJevSwift/issues/25) | Encoder prefill: embeddings, causal and sliding masks, KV caches, layer scalars |
| [#26](https://github.com/Algorythm-Canada/OpenJevSwift/issues/26) | Decoder read pass: canvas embedding, decoder masks, encoder-plus-canvas attention, slot logits |
| [#27](https://github.com/Algorythm-Canada/OpenJevSwift/issues/27) | Weight loading: safetensors shards, sanitize, per-module quantization, strict coverage |
| [#28](https://github.com/Algorythm-Canada/OpenJevSwift/issues/28) | Self-conditioning module and multi-step reads (steps 2 to 8) |
| [#29](https://github.com/Algorythm-Canada/OpenJevSwift/issues/29) | DiffusionGemmaRuntime actor: prefill cache, read API, prompt cap, memory controls, warmup |
| [#30](https://github.com/Algorythm-Canada/OpenJevSwift/issues/30) | Model resolution and download (Hugging Face Hub or local directory) |
| [#31](https://github.com/Algorythm-Canada/OpenJevSwift/issues/31) | Numeric parity and live model tests for DiffusionGemma reads |
| [#32](https://github.com/Algorythm-Canada/OpenJevSwift/issues/32) | Performance and memory baseline for reads |

### 3. Jev-compatible HTTP server

| Issue | Title |
|---|---|
| [#33](https://github.com/Algorythm-Canada/OpenJevSwift/issues/33) | [Tracking] Milestone 3: Jev-compatible HTTP server (tracking) |
| [#34](https://github.com/Algorythm-Canada/OpenJevSwift/issues/34) | Hummingbird application: routes, settings, request ids, server-timing header |
| [#35](https://github.com/Algorythm-Canada/OpenJevSwift/issues/35) | Validation and error contract: 400, 413, 422, 503, 529 bodies, trimming and logging |
| [#36](https://github.com/Algorythm-Canada/OpenJevSwift/issues/36) | Authentication: API key bearer and origin secret |
| [#37](https://github.com/Algorythm-Canada/OpenJevSwift/issues/37) | Capacity limits, model-time accounting and graceful shutdown |
| [#38](https://github.com/Algorythm-Canada/OpenJevSwift/issues/38) | Model routes forwarding and model listing |
| [#39](https://github.com/Algorythm-Canada/OpenJevSwift/issues/39) | SDK compatibility suite: official Python and TypeScript SDKs against the Swift server |
| [#40](https://github.com/Algorythm-Canada/OpenJevSwift/issues/40) | The openjev CLI: serve, decide, models, with backend selection |
| [#41](https://github.com/Algorythm-Canada/OpenJevSwift/issues/41) | Live end-to-end tests against a running server (port of test_live.py) |

### 4. Read extensions and images

| Issue | Title |
|---|---|
| [#42](https://github.com/Algorythm-Canada/OpenJevSwift/issues/42) | [Tracking] Milestone 4: Read extensions and images (tracking) |
| [#43](https://github.com/Algorythm-Canada/OpenJevSwift/issues/43) | steps (1 to 8) end to end on the DiffusionGemma backend |
| [#44](https://github.com/Algorythm-Canada/OpenJevSwift/issues/44) | samples and the automatic re-read policy end to end |
| [#45](https://github.com/Algorythm-Canada/OpenJevSwift/issues/45) | sequential mode end to end: prefix continuation and earlier answers in the prompt |
| [#46](https://github.com/Algorythm-Canada/OpenJevSwift/issues/46) | Gemma 4 image preprocessing parity for DiffusionGemma prompts |
| [#47](https://github.com/Algorythm-Canada/OpenJevSwift/issues/47) | Vision tower, multimodal embedder and the encoder's bidirectional image-block overlay |
| [#48](https://github.com/Algorythm-Canada/OpenJevSwift/issues/48) | images request field end to end |

### 5. Text generation and think

| Issue | Title |
|---|---|
| [#49](https://github.com/Algorythm-Canada/OpenJevSwift/issues/49) | [Tracking] Milestone 5: Text generation and think (tracking) |
| [#50](https://github.com/Algorythm-Canada/OpenJevSwift/issues/50) | Diffusion sampler: entropy-bound canvas update, temperature schedule, stable-and-confident stopping |
| [#51](https://github.com/Algorythm-Canada/OpenJevSwift/issues/51) | Block generation loop: canvas sizing, cache commits, streaming detokenizer, stop ids |
| [#52](https://github.com/Algorythm-Canada/OpenJevSwift/issues/52) | The think option: thought generation before a read, prefix continuation, billing |
| [#53](https://github.com/Algorythm-Canada/OpenJevSwift/issues/53) | /v1/chat/completions: OpenAI-compatible generation with streaming, JSON mode and cancellation |

### 6. Additional System One models

| Issue | Title |
|---|---|
| [#54](https://github.com/Algorythm-Canada/OpenJevSwift/issues/54) | [Tracking] Milestone 6: Additional System One models (tracking) |
| [#55](https://github.com/Algorythm-Canada/OpenJevSwift/issues/55) | JevK5 letter-readout backend on Qwen3.5 via MLXLLM, with checkpoint conversion |
| [#56](https://github.com/Algorythm-Canada/OpenJevSwift/issues/56) | Spike: Core ML versus MLX for the ModernBERT encoder models (Verdict, Laya), including iOS |
| [#57](https://github.com/Algorythm-Canada/OpenJevSwift/issues/57) | Verdict backend (verdict-1.4): prompt contract, calibration, abstention |
| [#58](https://github.com/Algorythm-Canada/OpenJevSwift/issues/58) | Laya backend (laya-1.0): sequence builder, decision head, temperature buckets |
| [#59](https://github.com/Algorythm-Canada/OpenJevSwift/issues/59) | Decision: implement the CLM backend (Qwen3-8B embeddings plus contrastive heads) or defer |
| [#67](https://github.com/Algorythm-Canada/OpenJevSwift/issues/67) | Encoder-style backend contract in OpenJevCore: batched question reads, text-only refusals, shared answer path |

### 7. Quality, benchmarks, release 0.1

| Issue | Title |
|---|---|
| [#60](https://github.com/Algorythm-Canada/OpenJevSwift/issues/60) | [Tracking] Milestone 7: Quality, benchmarks, release 0.1 (tracking) |
| [#61](https://github.com/Algorythm-Canada/OpenJevSwift/issues/61) | JevBench harness and agreement report against upstream OpenJev |
| [#62](https://github.com/Algorythm-Canada/OpenJevSwift/issues/62) | Calibration report for DiffusionGemma reads (and decision on optional temperature scaling) |
| [#63](https://github.com/Algorythm-Canada/OpenJevSwift/issues/63) | Benchmarks: machines, memory, Swift versus Python MLX backend |
| [#64](https://github.com/Algorythm-Canada/OpenJevSwift/issues/64) | Public API documentation (DocC), guides and compatibility matrix |
| [#65](https://github.com/Algorythm-Canada/OpenJevSwift/issues/65) | Release 0.1.0: versioning, tag, Swift Package Index, changelog, security notes |
| [#66](https://github.com/Algorythm-Canada/OpenJevSwift/issues/66) | Upstream tracking process and upstreaming the DiffusionGemma port to mlx-swift-lm |

Total: 67 issues across 8 milestones (8 tracking, 8 spikes or decisions, 51 implementation tasks).
<!-- ISSUE_INDEX_END -->
