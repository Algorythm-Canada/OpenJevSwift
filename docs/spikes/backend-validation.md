# Spike #22: the read-only pass in Swift, validated against mlx-vlm and the Layr-Labs fork

Question. Before the DiffusionGemma port begins (decision D-004): does a Swift `encode` then
`denoise` on the 4-bit checkpoint reproduce mlx-vlm's slot log-probabilities for the fixture
canvases, what tolerance and top-label agreement belong in D-014, what do memory and latency look
like on the reference machine, and does upstream `mlx-swift-lm` load the Gemma 4 MoE expert
tensors (risk R20)?

Answer.

1. A Swift transliteration of mlx-vlm's read path on upstream primitives (ml-explore/mlx-swift
   0.32.2, mlx-swift-lm `c043fb3`) reproduces all 27 oracle reads bit for bit, and so do its
   prefill caches. Two conditions apply. It must load the Metal library that the mlx-metal wheel
   ships, and it must use mlx-vlm's RoPE frequency table for the full-attention layers. With
   mlx-swift's own kernels, the same code agrees only statistically: top label on 150 of 156
   slots, label probabilities within 0.37 at worst and 0.008 on average. D-004 is confirmed:
   port mlx-vlm onto upstream primitives.
2. The Layr-Labs fork at `eeba2af` agrees with mlx-vlm to the same statistical degree (147 of
   156, 0.45 at worst, 0.010 on average) and cannot be made exact. It runs on its own MLX fork,
   so it is not a parity oracle. It remains a useful reference for code structure.
3. mlx-vlm itself moves by up to 0.62 in label probability when its prefill is merely chunked. The
   read amplifies last-bit differences, so the proposed 0.02 per-label tolerance cannot be met by
   any implementation that is not bit-exact. D-014 is revised to two tiers: bit-exact under the
   oracle's kernels, and aggregate bounds under native kernels. The bounds are calibrated against
   planted bugs.
4. Upstream mlx-swift-lm loads `mlx-community/gemma-4-26B-A4B-it-4bit`, all 270 expert tensors,
   and answers "Paris" greedily. R20 is resolved for that checkpoint. DiffusionGemma's fused,
   quantized `experts.gate_up_proj` has no path in upstream's `Gemma4Text`, and upstream's
   Gemma 4 router is not mlx-vlm's arithmetic, so the port writes its own blocks on the shared
   primitives.
5. On an Apple M3 Max, all three implementations use about 15.5 to 16.6 GiB after loading and
   25 to 28 GiB after 54 reads, 21 to 24 GiB with a 4 GB cache limit. A cached 3-question read
   takes about 0.1 s and a cold one 0.4 to 0.5 s.

Versions the results hold for: upstream OpenJev `dcd2094`; mlx-vlm 0.6.15 on MLX 0.32.2
(mlx-metal 0.32.2, `mlx.metallib` SHA-256 `dc59d1cc…`), transformers 5.17.0, tokenizers 0.23.2,
CPython 3.14.7; checkpoint `mlx-community/diffusiongemma-26B-A4B-it-4bit` at
`a7a81407613811e8ba63af92ac0d852b809e191f`; Layr-Labs/mlx-swift-lm `eeba2af` with
Layr-Labs/mlx-swift `0f4fe403`; ml-explore/mlx-swift 0.32.2, which vendors MLX v0.32.2
(`1f8e74e`); mlx-swift-lm `c043fb3`; swift-transformers 1.3.4. Run on 2026-09-30 on an Apple M3
Max (GPU `applegpu_g15s`), 128 GB, macOS 27, Xcode 27 (Swift 6.4). Another spike was converting
Core ML models on the same machine throughout, with load averages of 20 to 90. The latency
figures carry that noise; the numerical results do not depend on it.

## Method

| Part | What ran | Code |
|---|---|---|
| Checkpoint | `huggingface_hub.snapshot_download` at the pinned revision into the Hugging Face cache, 501 s, one attempt | `Tools/oracle/fetch_checkpoint.py` |
| Oracle | Upstream's own `MlxRuntime.read` on 27 reads over 12 prompts, twice. Inputs are built by upstream's `Engine` and checked against the committed fixtures. | `Tools/fixtures/mlx_vlm_oracle.py`, output `Fixtures/oracle/reads.json` |
| Fork probe | `DiffusionGemmaModelFactory` load, then `encode(tokenIds:cache:)` and `denoise(canvasIds:cache:selfConditioningLogits:)` per oracle read, with upstream's slot extraction and OpenJevCore's `SlotDistribution`, twice | `Tools/oracle/Probe` |
| Sensitivity | The oracle's read with one change that is exact in real arithmetic: a 64-token chunked prefill, unsorted decoder expert gathers, or explicit all-true masks | `Tools/oracle/sensitivity.py` |
| Cross-feed | mlx-vlm's decoder run on the fork's dumped prefill caches, plus a control on mlx-vlm's own caches cut to 1,024 sliding positions | `Probe --dump`, `Tools/oracle/crossfeed.py` |
| Transliteration | mlx-vlm's read path written out on upstream primitives. Optionally it loads the wheel's metallib and the oracle's RoPE table. It can plant bugs to calibrate D-014. | `Tools/oracle/UpstreamProbe` target `Transliteration` |
| Stage bisection | mlx-vlm's module outputs, recorded by wrapping its classes, against the same points in the transliteration | `Tools/oracle/stage_dump.py`, `Transliteration --stages` |
| Tokenizer | Every oracle prompt rendered by the main package's `SwiftTransformersTokenizer` | target `TokenizerCheck` |
| R20 | `LLMModelFactory` load of the Gemma 4 text checkpoint, a greedy answer | target `Gemma4Load` |

The 27 reads cover:

- **Widths:** 16, 32, 48 and 64.
- **Question counts:** 1, 3, 6, 8, 10, 12 and 24 questions, the last split into two chunked groups.
- **Option counts:** the 255-option choice (`widest_schema`) and the 100/100/55-option choices (`many_choices`).
- **Long prompts:** four over 1,024 tokens, of 1,572, 2,076, 2,442 and 2,939 tokens. The last, `long_state`, is the quickstart's questions over an 84-line support thread. In these the decoder's sliding layers read only the last 1,023 encoder positions.
- **Steps:** 2 and 3 for `quickstart`, `lines_10_mixed` and `long_state`.

Before any read runs, the oracle checks 23 canvases against `groups_and_canvases.json`, 13 prompt
id sequences against `chat-prompts/prompts.json`, and 8 prompt-and-canvas pairs against reads
recorded in `policies/policies.json`. It records per slot the `{token id: logprob}` map exactly
as `MlxRuntime.read` returns it, `slot_distribution`'s probabilities and entropy, the argmaxes
written between steps, the prompt token count, and digests of the prefill cache for layers 0 and
29. The written argmaxes come from a pass-through spy on the decoder call, which sees each
canvas.

The probe cannot be one package with the main package's `OpenJevDiffusionGemma`, as the brief
asked. The fork has the same package identities as the main package's dependencies
(`mlx-swift-lm`, and `Layr-Labs/mlx-swift` as `mlx-swift`) at other revisions. It also pins
swift-jinja to exactly 2.3.6 and defines the same module names. SwiftPM refuses:

```text
error: mlx-swift-lm is required using two different revision-based requirements
(eeba2afaf059a153ff909c9e01aa5e65b7bcad67 and c043fb3b1ccf00f54ef8882a1e8da45c6e32e6f8),
which is not supported
```

So the fork probe compiles OpenJevCore from the main package's sources through a symlink. The
factory gets a stub tokenizer, since a read never tokenizes. A second scratch package,
`UpstreamProbe`, depends on the main package by path together with its own upstream pins.
Everything that needs `SwiftTransformersTokenizer` runs there.

## The checkpoint

| File | Bytes |
|---|---|
| `model-00001-of-00004.safetensors` | 5,215,121,220 |
| `model-00002-of-00004.safetensors` | 5,358,706,305 |
| `model-00003-of-00004.safetensors` | 5,367,044,182 |
| `model-00004-of-00004.safetensors` | 602,183,698 |
| `tokenizer.json` | 32,169,626 |
| `model.safetensors.index.json` | 165,362 |
| `config.json` | 58,854 |
| `chat_template.jinja` | 17,336 |
| `tokenizer_config.json` | 2,749 |
| `.gitattributes` | 1,570 |
| `processor_config.json` | 911 |
| `README.md` | 779 |
| `generation_config.json` | 357 |
| Total, 13 files | 16,575,472,949 |

The shard SHA-256s are the blob names in the cache (`4e9e90d1…`, `67508ed6…`, `a3606518…`,
`ead1033d…`). The SHA-256 of `config.json` is `b41320c9…cac5`, the value the fork's live tests
require, so the fork was qualified on this exact revision. The index names 1,647 tensors:

- **Quantization:** 300 modules are quantized. Of the vision tensors, only `embed_vision.embedding_projection` is, at 4 bits; the vision tower stays bfloat16.
- **Experts:** they are already named `experts.gate_up_proj.weight`.
- **Vision:** 358 vision tensors, with no clipping calibration tensors.
- **Encoder:** the only encoder text weights are the 30 `layer_scalar`s.

## The oracle

Determinism holds at every level:

- **Within a run:** every read runs twice in one process. The second pass empties the prefill cache and goes in reverse order, so a read cached in the first pass is computed from scratch in the second, and the other way round. Both passes agree bit for bit, cache digests included, so cached and uncached prefills also give identical reads.
- **Across runs:** running the generator again in a new process rewrites `reads.json` byte for byte (`git diff --exit-code` is clean).
- **Cache limit:** with `--cache-limit-gb 4` (upstream's `OPENJEV_MLX_CACHE_LIMIT_GB`), `--check` finds no difference in any top-level key of the committed file: pins, settings, fixture checks, RoPE table, prompts and reads.

`Fixtures/oracle/reads.json` is 330,596 bytes. It also records the RoPE table and the metallib hash
that the exact tier below needs.

Every one of the 21 single-step reads has a maximum slot entropy above upstream's re-read threshold
of 0.1. So with default settings upstream re-reads every one of them. No implementation below
changed that decision for any read.

## The fork against the oracle

| Metric over 27 reads and 156 slots | Fork | Fork, `DARKBLOOM_DIFFUSION_EXPERT_UNSORT=0` |
|---|---|---|
| Max label probability difference | 0.451 | 0.451 |
| Mean label probability difference | 0.0098 | same |
| Slots with every label within 0.02 | 86 | same |
| Top label agreement | 147 (94.2%) | same |
| Max entropy difference | 0.537 | same |
| Reads with steps 2 or 3 whose written argmaxes match | 4 of 6 | same |
| Bit-identical reads | 0 | 0 |

The fork is deterministic across two passes and with a 4 GB cache limit. Its fused MoE output
reduction, on by default, gives exactly the numbers of the legacy graph, so it is not the
difference. Its prefill caches compare with mlx-vlm's as follows:

- **Layer 0 values:** bit-identical for every prompt.
- **Layer 0 keys:** differ, with the sum of squares off by at most 1.6e-7 relative.
- **Layer 29 keys and values:** off by 1e-6 to 4e-5 relative.

## mlx-vlm against itself

| Variant of the oracle's read | Bit-identical reads | Max label probability difference | Top label agreement |
|---|---|---|---|
| None (the procedure, replayed without the prefill cache) | 27 of 27 | 0 | 156 |
| Explicit all-true decoder masks where mlx-vlm passes none | 27 of 27 | 0 | 156 |
| Decoder expert gathers without the sort | 22 of 27 | 0.162 | 156 |
| Prefill in 64-token chunks (mlx-vlm's `chunk_prefill`) | 0 of 27 | 0.618 | 148 |

A chunked prefill is the same computation in exact arithmetic. It moves mlx-vlm further from
mlx-vlm than the fork does. The read is chaotic in bfloat16. Tiny differences in the prefill
cache change expert selection (top 8 of 128) and attention weights for the noise tokens in the
slots, and some slots flip even at a top-two margin of 0.68.

## Cross-feed: whose half differs

| mlx-vlm's decoder run on | Compared with | Bit-identical reads | Max label probability difference | Top label agreement |
|---|---|---|---|---|
| mlx-vlm's own prefill, cut to the last 1,024 sliding positions and rebuilt | The oracle | 27 of 27 | 0 | 156 |
| The fork's prefill | The fork's decoder on the same cache | 0 of 27 | 0.296 | 152 |
| The fork's prefill | The oracle | 0 of 27 | 0.273 | 149 |

The control shows that a 1,024-slot ring for the sliding layers is exact for reads, including the
2,939-token prompt, because the decoder reads only the last 1,023 positions (R15). Both halves
of the fork's pass contribute differences of the same size as the chunked prefill.

## The transliteration: bit parity is reachable

`Tools/oracle/UpstreamProbe/Sources/Transliteration/Model.swift` writes mlx-vlm's operations out in
their order, shapes and dtypes. It is about 600 lines, text only. It is built on MLXNN,
`MLXFast.RoPE`, `MLXFast.scaledDotProductAttention`, upstream's `SwitchLinear`, `gatherSort`,
`scatterUnsort` and `loadWeights` with `BaseConfiguration.perLayerQuantization`. Upstream's
loader fills the tree with strict verification.

| Configuration | Bit-identical reads | Cache digests equal | Max label probability difference | Top label agreement |
|---|---|---|---|---|
| mlx-swift's own kernels | 0 of 27 | 0 of 24 | 0.374 | 150 |
| The same with a 4 GB cache limit | 0 of 27 | 0 of 24 | 0.374 | 150 |
| `MLX.GPU.metallib` set to the wheel's `mlx.metallib` | 0 of 27 | 12 of 24, layer 0 all equal | 0.349 | 148 |
| That, and the full-attention RoPE table from the oracle | 27 of 27 | 24 of 24 | 0 | 156 |

The stage bisection located both differences.

- **Kernel builds.** mlx-swift compiles MLX in JIT mode: its manifest excludes `nojit_kernels.cpp`, so elementwise kernels are compiled at run time, while SwiftPM compiles the rest with Xcode 27's Metal compiler. The wheel ships one metallib with every kernel precompiled. Transcendental functions round differently in the last bit. The first visible effect is in layer 0. With mlx-swift's kernels, its three projections, both weighted RMSNorms and the value norm are bit-identical to mlx-vlm's, and only the two RoPE applications (sine and cosine) differ. With the wheel's metallib loaded, layer 0's RoPE is identical too, because that kernel comes from the metallib.
- **The RoPE frequency table.** With the wheel's metallib loaded, layers 0 to 4, the sliding layers, match at every recorded point: attention, MLP, router indices and weights, experts and layer output. The first mismatch is layer 5's queries after the proportional RoPE. Its input and the `q_proj` and `q_norm` outputs are equal. The cause is the 64 finite entries of the frequency table, `1.0 * pow(1e6, arange(0, 128, 2) / 512)`: 23 of them differ from mlx-vlm's by one or two float32 steps, the same 23 with or without the wheel's metallib, because `pow` runs as a JIT kernel in Swift. With the oracle's table injected, all 321 recorded tensors of the prefill are equal. So are all 27 reads, including the written argmaxes of steps 2 and 3, the four long prompts and the 1,024-slot sliding cache.

Everything else in the port-to-be is therefore exact against mlx-vlm on the same machine:

- quantized matmuls at 4 and 8 bits, the quantized embedding and its `asLinear`;
- RMSNorm with and without scale, SDPA with causal, array and no masks, argpartition routing, the precise softmax;
- sorted and unsorted gathered expert matmuls, the compiled GeGLU and softcap, the scalar weak typing;
- the self-conditioning projection.

The oracle now records the table (`rope` in `reads.json`) and the metallib's SHA-256, so the exact
tier needs nothing beyond the committed files and the pinned wheel.

## Planted bugs: what the native-kernel tier can catch

With mlx-swift's own kernels, which production will use, only statistics compare. To see where
legitimate variation ends, the transliteration was rerun with one deliberate porting mistake at a
time (`TRANSLITERATION_BUG`). The table reports all 27 reads. The last two columns cover the 50
slots of the four prompts over 1,024 tokens.

| Implementation | Mean \|Δp\| | p90 \|Δp\| | Top label | Top label, oracle margin ≥ 0.5 | Mean \|ΔH\| | Long: mean \|Δp\| | Long: mean \|ΔH\| |
|---|---|---|---|---|---|---|---|
| mlx-vlm, chunked prefill | 0.0100 | 0.0138 | 148/156 | 119/120 | 0.102 | 0.0064 | 0.161 |
| mlx-vlm, unsorted decoder experts | 0.0011 | 0.0000 | 156/156 | 120/120 | 0.016 | 0.0000 | 0.000 |
| Layr-Labs fork | 0.0098 | 0.0146 | 147/156 | 119/120 | 0.087 | 0.0069 | 0.133 |
| Transliteration, mlx-swift kernels | 0.0084 | 0.0118 | 150/156 | 120/120 | 0.086 | 0.0054 | 0.131 |
| Upstream Gemma 4 router arithmetic (exact in real numbers) | 0.0091 | 0.0136 | 146/156 | 118/120 | 0.089 | 0.0060 | 0.158 |
| Bug: self-conditioning module skipped on step 1 | 0.0427 | 0.1380 | 129/156 | 103/120 | 0.594 | 0.0165 | 0.642 |
| Bug: canvas RoPE from position 0 | 0.0562 | 0.1487 | 105/156 | 92/120 | 0.765 | 0.0320 | 0.926 |
| Bug: no sliding window for the canvas | 0.0159 | 0.0277 | 143/156 | 118/120 | 0.133 | 0.0142 | 0.275 |

Δp is over all 1,763 label probabilities and ΔH over 156 slot entropies, against the oracle.
`Tools/oracle/tolerance_stats.py` computes the table. A per-label tolerance of 0.02 would fail 64
to 70 of the 156 slots for every legitimate implementation; only 86 to 92 slots have every label
within 0.02. The aggregates separate the two gross bugs by a factor of four to seven. The window
bug shows only on the long prompts, and there by a factor of two. D-014 records the resulting
bounds.

## Tokenizer

`SwiftTransformersTokenizer`, from the main package through `UpstreamProbe`, renders all 12
oracle prompts to the recorded ids. That includes the five prompts that are not in
`chat-prompts/prompts.json`, and the 2,939-token `long_state` among them.

## Memory and latency

Memory is `proc_pid_rusage` physical footprint, the figure Activity Monitor shows, and MLX's
allocator counters. Each process loaded the model and ran the 27 reads twice. Loads came from a
warm page cache, so disk reads are not in the load times.

| Run | Load s | Footprint after load GiB | After 2 × 27 reads GiB | MLX cache pool after reads GiB | MLX peak GiB |
|---|---|---|---|---|---|
| Python oracle | 2.7 to 5.4 | 16.56 | 25.24 | 6.00 | 18.65 |
| Python oracle, 4 GB limit | 5.6 | 16.56 | 23.25 | 4.01 | 18.68 |
| Fork probe | 3.8 | 15.60 | 28.13 | 7.79 | 21.06 |
| Fork probe, 4 GB limit | 1.3 | 15.60 | 24.34 | 4.00 | 21.06 |
| Transliteration | 1.7 | 15.44 | 25.62 | 8.69 | 17.89 |
| Transliteration, 4 GB limit | 1.8 | 15.44 | 21.02 | 4.00 | 17.89 |

The 12 prompts' prefill caches were all held at once. The Python runtime held 13 entries for the
digests, against upstream's default of 12.

Read latency in milliseconds, median of the samples. Cold includes the prompt's prefill; cached
reuses it and runs one decoder pass and the slot log-softmax.

| Run | 1 question, 78 tokens | 3 questions, 182 tokens | 12 questions, 1,572 tokens | 3 questions, 2,939 tokens |
|---|---|---|---|---|
| Python oracle | 215 / 80 | 389 / 109 | 1,792 / 185 | 3,295 / 140 |
| Python oracle, 4 GB limit | 433 / 178 | 528 / 216 | 2,304 / 512 | 3,932 / 183 |
| Fork probe | 190 / 94 | 504 / 78 | 1,564 / 140 | 3,057 / 98 |
| Fork probe, 4 GB limit | 303 / 100 | 453 / 130 | 2,501 / 339 | 4,597 / 151 |
| Transliteration | 307 / 95 | 462 / 122 | 2,280 / 257 | 4,208 / 142 |
| Transliteration, 4 GB limit | 252 / 81 | 557 / 117 | 2,250 / 226 | 4,166 / 146 |

Each cell is cold / cached. Two cached steps cost about twice one step, and three about three
times: 220 and 367 ms for the quickstart in Python, 290 and 521 ms in the transliteration. Prefill
runs at about 700 to 1,300 tokens per second in all three, and a cached decoder pass takes about
0.1 s. These figures come from a machine shared with another heavy job, and repeated runs of the
same configuration varied by up to a factor of two. They agree with upstream's 0.2 to 0.4 s per
cold 3-question read. They are not a basis for comparing the implementations with each other.
The performance baseline task (R5) should measure again on an idle machine.

## Upstream mlx-swift-lm and the Gemma 4 MoE (R20)

`LLMModelFactory` at `c043fb3` loads `mlx-community/gemma-4-26B-A4B-it-4bit`. That is revision
`0d77464e`, 15,373,588,575 bytes, `model_type` `gemma4`. The load took 3.1 s from a warm page cache.
It yields 1,339 parameters, 270 of them experts. Those are 90 quantized
`experts.switch_glu.{gate,up,down}_proj` projections: gate and up are `[128, 704, 352]` uint32
with bfloat16 scales and biases `[128, 704, 44]`, and down is `[128, 2816, 88]`. A greedy answer to
"What is the capital of France?" is `Paris` (token 50429). Issue #282's MoE loading gap does not
reproduce with this checkpoint at this commit.

| Upstream piece | Use for the DiffusionGemma port |
|---|---|
| `SwitchLinear`, `QuantizedSwitchLinear`, `gatherSort`, `scatterUnsort` (`SwitchLayers.swift`) | Use. Same operations as mlx-vlm's `switch_layers.py`; bit-exact in the transliteration with the fused 1,408-row `gate_up_proj`. |
| `loadWeights(modelDirectory:model:perLayerQuantization:)`, `BaseConfiguration.perLayerQuantization` | Use. Loads the checkpoint's 8-bit and 4-bit modules into an mlx-vlm-shaped tree, with strict verification. |
| `MLXFast.RoPE` with `freqs`, `MLXFast.rmsNorm` with `MLXArray.mlxNone`, `QuantizedEmbedding.asLinear`, `quantizedMM` | Use. |
| `Gemma4TextExperts` (`SwitchGLU`, separate gate and up) and its `sanitize` | Do not use. The checkpoint's experts are fused and quantized, and the sanitize only splits unquantized `gate_up_proj` tensors. |
| `Gemma4TextRouter` | Do not use. It folds `scale × hidden^-0.5` into the norm weight and uses a plain softmax. Exact in real numbers, but not mlx-vlm's rounding: 146 of 156 top labels. |
| `ProportionalRoPE` (`RoPEUtils.swift`) | Not used. It rotates a split and re-concatenated slice, mlx-vlm rotates the whole head with infinite frequencies. Not verified bit-equal. |
| `Gemma4` fused `_addRMSNorm`, compiled `weightedExpertSum` | Not used; mlx-vlm computes these as separate operations. |

## Porting notes

### For #24 (text blocks)

- The transliteration's `Model.swift` is a working, exact reference for every block. Port it
  with the module names of the checkpoint. Its decoder layer is `language.py:290-320` operation for
  operation.
- Router: `rmsNorm(x, weight: .mlxNone)`, then `x * scale * hidden^-0.5` as two bfloat16 multiplies,
  `proj`, `argPartition(scores, kth: -8)`, `takeAlong`, `softmax(precise: true)`, then
  `* perExpertScale[indices]`. Do not reuse upstream's `Gemma4TextRouter`.
- Experts: one `SwitchLinear` `gate_up_proj` with 1,408 outputs, sliced at 704, and `down_proj`.
  Sort when `indices.size >= 64`, with upstream's `gatherSort` and `scatterUnsort`. After the unsort,
  compute `(y * weights[..., None]).sum(axis: -2)`.
- GeGLU: `compile(shapeless: true) { gate, x in geluApproximate(gate) * x }`, compiled as mlx-vlm
  compiles it. `MLXNN.geluApproximate` is the same expression as `mlx.nn.gelu_approx`.
- Full-attention layers: no `v_proj`. The values are the raw keys through
  `rmsNorm(weight: .mlxNone)`. RoPE is `MLXFast.RoPE(x, dimensions: 512, base: nil, freqs:)` with
  mlx-vlm's 256-entry table, 64 finite entries then 192 infinite ones. Compute the table at load
  time with MLX `pow` as mlx-vlm does. On mlx-swift's kernels 23 of the 64 entries differ in the
  last bit, which is unavoidable with the production kernels. Give the exact-tier test a hook to
  install the oracle's `rope` table.
- Sliding layers: `MLXFast.RoPE(x, dimensions: 256, base: 10000, scale: 1)`.
- Encoder masks on an empty cache are `.causal`. The exception is the sliding layers for prompts
  over 1,024 tokens, which need the boolean array `rows >= cols && rows < cols + 1024`.
- Full layers' encoder cache: mlx-vlm's `KVCache` writes into a zero buffer rounded up to 256
  positions and hands SDPA a view. The transliteration mirrors that. Whether a contiguous array
  gives the same bits was not tested.
- mlx-swift turns Float scalars into the array's dtype as MLX Python does: the embedding scale
  √2816 and the router's `hidden^-0.5` round to bfloat16. The exact result confirms it.
- For the acceptance test "a single layer matches mlx-vlm", use `Tools/oracle/stage_dump.py` and
  `Transliteration --stages`. Under the oracle's kernels, require exact equality, not a bfloat16
  tolerance.

### For #26 (decoder read pass)

- On step 1, mlx-vlm passes no self-conditioning. It still runs the module on zero soft
  embeddings: `post_norm(embeddings + down(geglu(gate(pre_norm(0)), up(pre_norm(0)))))`. It does
  not skip it. Skipping it is a real bug, measured at 129 of 156 top labels and 0.043 mean |Δp|.
- Masks: full layers `.none`; sliding layers `.none` up to 1,023 prompt tokens. Past that, keep
  only the last 1,023 encoder positions of keys and values, as `language.py:225-246` does, and the
  mask's last `1,023 + canvas` columns, which are all true. An all-true array and `.none` give the
  same bits.
- RoPE offset for the canvas is the prompt length, the cache offset. Offset 0 is a real bug
  (105 of 156).
- Logits: `softcap(embedTokens.asLinear(norm(h)))` with the compiled
  `tanh(x.asType(.float32) / 30) * 30`.
- Slot extraction: per slot, a one-dimensional float32 row, `row - logSumExp(row)`,
  `argPartition(-lp, kth: 20)[..<20]` united with the labels, sorted by token id. It was not
  tested whether batching the slot rows gives the same bits.
- Sliding-layer storage: a ring of 1,024 slots is exact for reads (the cross-feed control). mlx-vlm
  itself keeps every prompt position after a one-shot prefill.
- Prefill must be one piece, as `MlxRuntime._prefill` does it. A chunked prefill changes reads by up
  to 0.62.
- D-015 (slot-only projection): neither mlx-vlm nor the fork does it. Projecting only the slot
  rows changes the shape of the tied-head matmul and may change the kernel. Prove it bit-identical
  in the exact tier before shipping it.

### For #27 (weight loading)

- Upstream's `loadWeights` with `BaseConfiguration.perLayerQuantization` loads this checkpoint into
  mlx-vlm's module tree with `verify: [.all]`. Nothing is missing and nothing is unexpected. There
  are 300 quantized modules: 8 bits for the embedding, attention, dense MLP and router; 4 bits for
  the experts, self-conditioning and `embed_vision.embedding_projection`.
- `sanitize` as mlx-vlm: drop `rotary_emb` and `lm_head.weight`, and keep only the `layer_scalar`s
  under `model.encoder.language_model`. The expert rename is a no-op for this checkpoint, whose
  experts are already named `experts.gate_up_proj.weight`, and it has no clipping calibration
  tensors. A text-only load drops `model.encoder.vision_tower.*` and `model.encoder.embed_vision.*`.
- Vision keys carry `.linear.` because mlx-vlm's `ClippableLinear` wraps a `Linear` named `linear`.
  The fork strips it and adds quantization path aliases. Keeping mlx-vlm's tree needs neither.
- Measured (warm page cache): 0.7 to 3.8 s wall time, 15.4 to 16.6 GiB footprint after load.

### For #28 (self-conditioning and steps 2 to 8)

- Quantized embedding, so the logits path applies:
  `softmax(previousLogits, precise: true)` in float32, then
  `quantizedMM(probs.asType(.bfloat16), weight, scales:, biases:, transpose: false, groupSize: 64, bits: 8)`,
  then `.asType(.bfloat16) * √2816`, then the module. Steps 2 and 3 are bit-exact in the
  transliteration, argmaxes included.
- Between steps: `canvas[0, slotPositions] = argMax(logits[0, slotPositions], axis: -1)`, the full
  float32 logits kept as the next conditioning, `eval(canvas, logits)`. Other canvas positions are
  never written.
- Under native kernels the written argmaxes flip where the oracle's step-1 margin is small (4 of
  6 reads equal for both the fork and the transliteration). Compare them exactly only in the exact
  tier.
- The fork's `DiffusionGemmaSoftEmbedding` kernel is off by default and applies only to a 256-token
  canvas. It is not needed.

### Where the fork departs from mlx-vlm

| Place | The fork | mlx-vlm | The port |
|---|---|---|---|
| MLX core | Layr-Labs/mlx-swift `0f4fe403`: MLX 0.32.2 plus patches to quantized kernels, SDPA, compile and the allocator, and a Gemma 4 expert qmm route used at exactly 4,096, 8,192 or 16,384 assignments (512, 1,024 or 2,048-token prefill chunks). No oracle read reaches the route. | Stock MLX 0.32.2 | Stock mlx-swift 0.32.2 |
| Conditioning logits dtype | float32 for a quantized embedding, the embedding dtype for a dense one (`selfConditioningLogitsDType`) | The same two paths (`diffusion_self_conditioning`, `_embed_canvas`) | Follow mlx-vlm; for this checkpoint float32 |
| Sliding cache | A 1,024-slot ring. A one-token encoder update sees physical ring order, to match mlx-vlm's `_update_in_place`; snapshots and denoising see temporal order. | `RotatingKVCache`: keeps every position after a one-shot prefill; in-place rotation for single tokens | A ring is exact for reads. Mirror the physical order for single-token commits when generation lands. |
| Encoder scalars | Only the 30 `layer_scalar`s are registered; the decoder owns all weights | The same, through a weak reference | Same |
| Decoder masks | `.none`, with the prefix cut to the last 1,023 | None, or a mask equivalent to the cut | Either; bit-identical |
| Expert reduction | A fused unsort-and-sum kernel, on by default | Scatter, multiply, sum | mlx-vlm's; the fused kernel was bit-identical here |
| GeGLU, softcap | GeGLU compiled only when `isCompiledDecodeSupported`; softcap not compiled | Both compiled | Compile both, as mlx-vlm |
| Vision projection | RMSNorm before the projection | The same (`MultimodalEmbedder`) | Same. The fork's comment warns that upstream Swift's autoregressive Gemma 4 wrapper differs. |
| Vision keys | `.linear.` stripped, quantization aliases, `use_clipped_linears` must be false | `.linear.` kept | mlx-vlm's tree |
| Input checks | Token range and dtype, cache owner and dtype, capacity; throws | None | Validate, as the fork does |
| Product features | Chunked and paged prefill, prefix checkpoints, persistence, continuous batching | Chunked prefill | Not for reads; a chunked prefill changes results |

## Reproducing

From the repository root, on an Apple silicon Mac with the Metal Toolchain installed and about
35 GB free for both checkpoints:

```bash
make upstream
python3.14 -m venv Tools/oracle/.venv
Tools/oracle/.venv/bin/pip install -r Tools/oracle/requirements.txt
Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py
PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/mlx_vlm_oracle.py
swift run --package-path Tools/oracle/Probe -c release Probe
swift run --package-path Tools/oracle/UpstreamProbe -c release TokenizerCheck
swift run --package-path Tools/oracle/UpstreamProbe -c release Transliteration
swift run --package-path Tools/oracle/UpstreamProbe -c release Transliteration --metallib "$PWD/Tools/oracle/.venv/lib/python3.14/site-packages/mlx/lib/mlx.metallib" --oracle-rope
```

For Part D, fetch the Gemma 4 checkpoint and load it:

```bash
Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py --repo mlx-community/gemma-4-26B-A4B-it-4bit --revision 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
swift run --package-path Tools/oracle/UpstreamProbe -c release Gemma4Load
```

The rest of the evidence comes from these runs:

- `Probe --cache-limit-gb 4` and `Transliteration --cache-limit-gb 4`, and the oracle with `--check --cache-limit-gb 4`.
- `Probe --dump DIR --dump-reads IDS` then `Tools/oracle/crossfeed.py DIR`.
- `Tools/oracle/sensitivity.py`.
- `Transliteration` with `TRANSLITERATION_BUG` set to `skip_self_conditioning`, `rope_offset_zero`, `no_window` or `upstream_router`.
- `Tools/oracle/tolerance_stats.py` and `Tools/oracle/summarize_runs.py`, which rebuild the tables.

The D-014 table, planted bugs included:

```bash
for bug in skip_self_conditioning rope_offset_zero no_window upstream_router; do TRANSLITERATION_BUG=$bug swift run --package-path Tools/oracle/UpstreamProbe -c release Transliteration; done
Tools/oracle/.venv/bin/python Tools/oracle/tolerance_stats.py Tools/oracle/results/fork_reads.json "planted bug: skip self-conditioning=Tools/oracle/results/transliteration_run_planted_bug_skip_self_conditioning.json" "planted bug: RoPE offset 0=Tools/oracle/results/transliteration_run_planted_bug_rope_offset_zero.json" "planted bug: no sliding window=Tools/oracle/results/transliteration_run_planted_bug_no_window.json" "upstream Gemma4 router=Tools/oracle/results/transliteration_run_planted_bug_upstream_router.json"
```

Their outputs are in `Tools/oracle/results/`. Without `--out`, the probe and the transliteration
name each result file after the run's options. Only a default run writes the native baselines,
`probe_run.json` and `transliteration_run.json`. Only the wheel's metallib together with the
oracle's RoPE table writes `transliteration_run_exact.json`. That file and
`transliteration_run_cache_limit_4gb.json` were written with `--summary-only`, which leaves out
per-read slot maps equal to the oracle's or to the unlimited run's. The planted-bug runs are
summarized in `tolerance_stats.json` rather than kept whole. With Xcode 27 the first build of each
scratch package takes about 4 minutes; later builds take seconds.

## Deviations from the brief

- **The oracle's script is `Tools/fixtures/mlx_vlm_oracle.py`, not `Tools/oracle/mlx_vlm_oracle.py`.** `Tests/OpenJevCoreTests/Fixtures/FixturePinTests.swift` requires every JSON file under `Fixtures/` to name a generator script under `Tools/fixtures/`. Keeping the script there leaves the main package untouched. The issue also places the generator in `Tools/fixtures/`. It runs from `Tools/oracle/.venv`, which holds MLX. The fetch script, the probes, the sensitivity, cross-feed and statistics scripts, and all timings live in `Tools/oracle/`.
- **The fork probe cannot link `OpenJevDiffusionGemma`**, for the SwiftPM conflict shown in Method. It compiles OpenJevCore through a symlink. The tokenizer runs in the second package, which does depend on the main package by path.
- **Beyond the brief:** the sensitivity experiment, the cross-feed, the Swift transliteration with its stage bisection, and the planted bugs. Without them the fork's numbers could not be read. They measure the exact tier D-014 now requires.
- **Not done:** image reads (the fork's vision path was read, not run), timings on an idle machine (R5), and a batched or slot-only projection check (D-015).
