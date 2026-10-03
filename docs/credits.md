# Models, credits and licenses

OpenJevSwift trains no model. It serves other people's models, behind an API other people
designed, and ports code and behaviour from other projects. This page credits each, with its
license and the revision this port pins. [THIRD_PARTY.md](../THIRD_PARTY.md) lists everything the
project references, at its pinned revision, including the projects whose behaviour it reproduces
and the benchmarks and client SDKs its tests use.

OpenJevSwift itself is Apache-2.0, the license of upstream OpenJev ([LICENSE](../LICENSE)). It is
an independent project, not affiliated with or endorsed by TypeSafe AI (the makers of Jev), Google
DeepMind or NVIDIA (DiffusionGemma), or the authors of the other models it serves
([NOTICE](../NOTICE)). No weights are part of this repository; each model keeps its own license
and terms, and the Gemma Terms of Use that apply to DiffusionGemma's weights are the user's to
follow (D-010).

## The models this port serves

| Model | Served as | Authors | License | Checkpoint | Pinned revision |
|---|---|---|---|---|---|
| DiffusionGemma 26B-A4B | `openjev-0.1` | Google DeepMind; the MLX conversion by mlx-community | Apache-2.0 plus the Gemma Terms of Use | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it), converted as [mlx-community/diffusiongemma-26B-A4B-it-4bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) | `a7a81407613811e8ba63af92ac0d852b809e191f` |
| Verdict 1.4 | `verdict-1.4` | Heman10x ([Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev)) | Apache-2.0 | [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) | `8af2496eb63c7fa66d7d234e1f62629380030eb4` |
| Laya 1.0 | `laya-1.0` | Nandakishor M / Convai Innovations ([laya](https://github.com/NandhaKishorM/laya)) | Apache-2.0 | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | `1a793eb568e6718f15941d08f85432581df534e3` |
| JevK5 0.2 | `jevk5-0.2` | Alibi Serikbay ([jevk5](https://github.com/allebee/jevk5)); base model Qwen3.5-4B by the Qwen team | Apache-2.0 | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5), tag `v0.2` | `ea4804e93a3db07c2250315c400f59683f54db6f` |

- **DiffusionGemma 26B-A4B** is Google DeepMind's diffusion language model: 25.2B parameters, of
  which 3.8B are active, plus a vision tower. This port loads mlx-community's 4-bit MLX conversion
  (converted with mlx-vlm 0.6.3), and two other conversions at pinned revisions:
  [8-bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-8bit) at `7b95e388` and
  [bfloat16](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-bf16) at `2cd36f95`.
  Upstream's vLLM backend serves NVIDIA's NVFP4 conversion, which this port does not use. The
  4-bit model card declares Apache-2.0 and links Google's Gemma 4 license.
- **Verdict** is a GLiClass model over ModernBERT-base (its base model is
  `knowledgator/gliclass-modern-base-v2.0`, 151M parameters), calibrated with RLCD, with the v1.4
  inference engine. This port runs upstream's Verdict prompt contract, read path and calibrator.
- **Laya** is a ModernBERT-large model (421M parameters) fine-tuned on the typed-decisions
  workflows. This port runs laya 0.3.6's `render_options`, `render_criterion`, `serialize_state`,
  `build_sequence`, `temp_bucket` and `clamp_temperature`, the calibration of `Agent.system_one`,
  and upstream's `LayaEngine.read_batch`.
- **JevK5** is Qwen3.5-4B with a LoRA distilled from Qwen3.6-27B, merged into the weights, read
  as a softmax over its answer letters' next-token logits (SemIf's readout, TheoLeeCJ/SemIf, MIT)
  under one calibration temperature, 1.532. This port runs the `jevk5` package's prompt and
  many-option readout at v0.2.2 and upstream's `JevK5Engine`, on MLX through mlx-swift-lm's
  Qwen3.5 text model. Its NOTICE, which the conversions carry, credits the Qwen team and SemIf.
  The checkpoint's `main` has held JevK5 v0.3, other weights and temperatures, since 2026-09-25;
  this port pins the author's `v0.2` tag, the model upstream names `jevk5-0.2`.

The descriptions `GET /v1/models` lists for Verdict, Laya and JevK5 are upstream's, word for word,
and credit the authors wherever the listing is shown.

## Where the weights come from

| Model | Weights | Tokenizer and calibration | Hosted by |
|---|---|---|---|
| DiffusionGemma | The mlx-community checkpoint, downloaded on first use into the Hugging Face cache | In the checkpoint | The Hugging Face Hub; this project hosts no copy |
| Verdict | `verdict-m18-fp16`, a float16 Core ML conversion of the checkpoint (306 MB), release `verdict-m18-fp16-v1` | `tokenizer.json`, `tokenizer_config.json` and `calibrator.json` from the checkpoint at its pinned revision, not re-hosted | [Algorythm-Canada/openjev-models](https://github.com/Algorythm-Canada/openjev-models) releases |
| Laya on a Mac | `laya-m18-fp16`, a float16 Core ML conversion (849 MB), release `laya-m18-fp16-v1` | The checkpoint's `tokenizer/` and `rl_agent_config.json` at its pinned revision, not re-hosted | openjev-models releases |
| Laya on an iPhone | `laya-f18-b1s128-fp16`, `laya-f18-b1s256-fp16`, `laya-f18-b1s512-fp16` and `laya-f18-b1s1024-fp16` (843 to 845 MB each), releases `laya-f18-b1s128-fp16-v1` to `laya-f18-b1s1024-fp16-v1` | As on a Mac | openjev-models releases |
| JevK5 | `jevk5-0.2-mlx-4bit` (2.37 GB of weights) and `jevk5-0.2-mlx-8bit` (4.47 GB), MLX conversions made by `Tools/jevk5/convert.py` with mlx-lm 0.32.0 | The checkpoint's `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `jevk5_config.json`, copied unchanged into each conversion | Not yet published: until the maintainer creates `Algorythm-Canada/jevk5-0.2-mlx-4bit` and `-8bit` on the Hugging Face Hub (D-051), a deployment converts the checkpoint itself and names the folder in `OPENJEV_JEVK5_MODEL` |

The JevK5 conversions keep the checkpoint's Apache-2.0 license; each carries JevK5's own `LICENSE`
and `NOTICE` and a model card that credits Alibi Serikbay and says what was changed (the weights
quantized, nothing else).

The Core ML packages are conversions of the authors' checkpoints, made with
[Tools/encoders](../Tools/encoders/README.md) and published under the checkpoints' Apache-2.0
license. The openjev-models repository carries that license and a NOTICE crediting Verdict by
Heman10x and Laya by Nandakishor M / Convai Innovations. Each release holds one package version,
its files uploaded one asset each, and the `OpenJevEncoders` library embeds every file's SHA-256 and
refuses a file that does not match (D-033).

## Models upstream serves that this port does not yet

| Model | Served as | Authors | License | Checkpoint | Upstream's pin | Issue |
|---|---|---|---|---|---|---|
| CLM 0.1 | `clm-v0.1` | Contrastive-LM ([CLM](https://github.com/Contrastive-LM/CLM)) | Apache-2.0 | [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | package 0.1.0 | #59 |

A server lists them, with upstream's descriptions, when `OPENJEV_MODEL_ROUTES` forwards them to a
server that serves them.

## Upstream projects

| Project | What this port takes from it | Pinned | License |
|---|---|---|---|
| [razorback16/openjev](https://github.com/razorback16/openjev) | The compatibility target: the wire API, the engine, the MLX backend, the encoder backends and their tests, ported throughout | `dcd2094` (0.5.0) | Apache-2.0 |
| [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) | The array framework DiffusionGemma runs on | 0.32.2 | MIT |
| [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | Model primitives: the switch layers and their quantized form, the experts' gather and scatter, weight loading with per-layer quantization, and the Gemma 4 vision configuration; the Qwen3.5 text model JevK5 runs on | `c043fb3` | MIT |
| [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | `mlx_lm.convert`, which converts JevK5 to MLX (`Tools/jevk5`); not linked | 0.32.0 | MIT |
| [allebee/jevk5](https://github.com/allebee/jevk5) | JevK5's prompt and many-option readout (`jevk5/prompt.py`), and its published v0.2 JevBench run, the parity reference | v0.2.2 (`0571ef3`) | Apache-2.0 |
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | The DiffusionGemma implementation the Swift model follows, operation for operation, and the oracle its parity tests compare with | 0.6.15 | MIT |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | The tokenizers and chat templates | 1.3.4 (researched at `af520cf`) | Apache-2.0 |

Code ported from mlx-vlm keeps its copyright notice, Copyright © 2025 Prince Canuma, in each
file's header. [Layr-Labs/mlx-swift-lm](https://github.com/Layr-Labs/mlx-swift-lm) (MIT), a fork
with its own DiffusionGemma, was a second reference, not a dependency. The random number and
`json.loads` ports follow CPython (PSF-2.0; the MT19937 reference code BSD-3-Clause), and the
request reading follows FastAPI (MIT) and Starlette (BSD-3-Clause), behaviour only.

## The Swift packages the build resolves

`Package.resolved` pins 35 packages. All are Apache-2.0 except four under MIT, and ten of the
Apache-2.0 ones add the Swift Runtime Library Exception.

| License | Packages |
|---|---|
| MIT | mlx-swift, mlx-swift-lm, EventSource, yyjson |
| Apache-2.0 with the Runtime Library Exception | swift-algorithms, swift-argument-parser, swift-async-algorithms, swift-atomics, swift-collections, swift-docc-plugin, swift-docc-symbolkit, swift-numerics, swift-syntax, swift-system |
| Apache-2.0 | async-http-client, hummingbird, swift-asn1, swift-certificates, swift-configuration, swift-crypto, swift-distributed-tracing, swift-http-structured-headers, swift-http-types, swift-huggingface, swift-jinja, swift-log, swift-metrics, swift-nio, swift-nio-extras, swift-nio-http2, swift-nio-ssl, swift-nio-transport-services, swift-service-context, swift-service-lifecycle, swift-transformers |

swift-nio-ssl embeds BoringSSL, under its OpenSSL, SSLeay and ISC terms, in the `openjev` binary.
A Linux build resolves the same list without the eight packages only Apple platforms use
([development.md](development.md)).
