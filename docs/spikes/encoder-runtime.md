# Spike #56: Core ML or MLX for Verdict and Laya, on iOS and macOS

Issue [#56](https://github.com/Algorythm-Canada/OpenJevSwift/issues/56), run on 2026-09-30.
Verdict (`verdict-1.4`, ModernBERT-base with a GLiClass head, 151M parameters) and Laya
(`laya-1.0`, ModernBERT-large with a decision head, 421M) were converted to Core ML, checked
against PyTorch on 200 questions, and measured on a MacBook Pro (M3 Max) and an iPhone 13 Pro Max
(A15, iOS 27.0). An MLX port was estimated, not built. The raw results are in
[encoder-runtime/](encoder-runtime/), the reference fixtures in
[Fixtures/encoders](../../Fixtures/encoders/README.md), and every script in
[Tools/encoders](../../Tools/encoders/README.md).

## Answer

- **Both models run on Core ML.** Both convert from PyTorch with coremltools 9.0, and the float16
  packages stay inside the variation upstream's own bfloat16 serving carries. Swift tokenizes every
  question exactly as upstream does. An MLX port would take 5 to 6 days and could not use the Neural
  Engine.
- **Verdict on the iPhone uses the Neural Engine.** With `.cpuAndNeuralEngine`, a question takes
  9.7 ms at 128 tokens, 17 ms at 256 and 46 ms at 512 (batch 1, median, starting from a cool
  phone), with the app at 106 MB. That is about 2.5 times faster than the iPhone's GPU, with a third
  of its memory.
- **Laya on the iPhone uses the GPU.** FILL_LAYA_ANSWER
- **On the Mac, both use the GPU.** Verdict takes 7.5 to 20 ms per question and Laya 15 to 82 ms.
- **The packages need iOS 18.** Core ML's CPU backend crashes on the enumerated-shape packages iOS
  17 would need, on macOS 27.0.1 and on iOS 27.0. iOS 18 multifunction packages avoid the crash, and
  iOS 18 runs on the same iPhones as iOS 17.
- **Long Laya reads should go to a server.** A 1,024-token Laya read takes 1.3 s on the iPhone, so
  an app should send such reads to an OpenJev server over the same wire API.

## Decision, per model

| | Verdict | Laya |
|---|---|---|
| Runtime | Core ML | Core ML |
| Package | `verdict-m18-fp16`: iOS 18 and macOS 15, one function per shape, float16, 306 MB | `laya-m18-fp16`: the same, 849 MB, with the two graph rewrites described under Conversion |
| iPhone compute units | `.cpuAndNeuralEngine` | `.cpuAndGPU` FILL_LAYA_UNITS_NOTE |
| Mac compute units | `.cpuAndGPU` | `.cpuAndGPU` |
| Batch | one question per call on the iPhone; up to 16 per call on the Mac GPU | one question per call |
| Avoid | `.all` (flaky loads) and the enumerated packages wherever a CPU segment runs (the crash) | the same |
| iPhones | iOS 18 devices (XS, XR and later); measured on an A15 with 6 GB | the same; 855 MB at peak on the GPU |
| Fallback | a server read over the same wire API when a request's reads exceed the app's time budget | a server read for states near 1,024 tokens, and whenever the app's budget is short |

Why not MLX: an MLX port needs 5 to 6 days and about 700 lines, and MLX cannot use the Neural
Engine, which on the iPhone ran Verdict about 2.5 times faster than the GPU with a third of the
memory. On the Mac and on the iPhone's GPU, Core ML already runs both models with no model code
to maintain. Revisit MLX only if Core ML's packaging problems (below) reach the configurations
chosen here.

### Packaging: download on first use, not the app bundle

- **Size.** 306 MB and 849 MB are too large to put in every copy of an app, and an app update
  should not be the only way to update a model.
- **First launch.** After a download, `MLModel.compileModel(at:)` takes 1.2 s (Verdict) and 2.7 s
  (Laya) on the iPhone. The first load of each function then takes 7 to 9 s for Verdict's batch-1
  functions on the Neural Engine (14 to 43 s for its batch-16 functions) and 3 to 11 s for Laya's
  on the GPU; once Core ML has cached them, a load takes about 0.2 s. An app should compile and
  load the functions it will use right after the download, in the background, not on the first
  read.
- **Where.** Publish each converted package with its SHA-256 next to the pinned checkpoint
  revisions (the organisation's Hugging Face account is one place; both checkpoints are
  Apache-2.0, so the packages carry the upstream license). Keep the compiled model in Application
  Support, excluded from backup.
- **Which functions.** An iPhone that reads one question at a time needs only the batch-1
  functions. Leaving the batch-16 functions out of the iOS package saves their compile time and
  the memory they take when loaded.

## Measurements

### iPhone 13 Pro Max (A15, 6 GB, iOS 27.0), the numbers that decide it

The settled run waited for the nominal thermal state before each configuration, and every
configuration reached the serious state before it finished. Batch 1 is the median latency of a
single question; batch 16 is the median latency of a call divided by 16.

| Model, compute units | Batch 1 ms at 128 | 256 | 512 | 1,024 | Batch 16 ms per question at 128 | at 512 | Peak footprint MB | Max probability difference | Top answers kept |
|---|---|---|---|---|---|---|---|---|---|
| Verdict, Neural Engine | 9.7 | 17.0 | 46.2 |  | 7.8 | 84.8 | 106 | 0.0073 | 197 of 200 |
| Verdict, all | 13.8 | 18.5 | 47.0 |  | 7.8 | 97.8 | 176 | 0.0073 | 197 of 200 |
| Verdict, CPU | 22.1 | 38.7 | 88.8 |  | 17.8 | 141.3 | 264 | 0.0106 | 197 of 200 |
| Verdict, GPU | 26.8 | 49.1 | 107.8 |  | 23.2 | 173.9 | 289 | 0.0018 | 199 of 200 |
| Laya, GPU | 81.7 | 151.7 | 385.2 | 1,346.4 | 146.3 | 619.6 | 855 | 0.0021 | 200 of 200 |

The CPU beats the GPU on this phone for Verdict. The Neural Engine is fastest and lightest by far,
and batching 16 questions pays only at 128 tokens (7.8 against 9.7 ms per question) and loses at
512 (85 against 46 ms). The first run of the Verdict packages, without waiting for the phone to
cool, is in [encoder-runtime/iphone](encoder-runtime/iphone/): later configurations of that run
started in the serious or critical state and ran up to twice as slow.

### MacBook Pro (M3 Max, 128 GB, macOS 27.0.1)

| Model, compute units | Batch 1 ms at 128 | 256 | 512 | 1,024 | Batch 16 ms per question at 128 | at 512 | Peak footprint MB | Max probability difference | Top answers kept |
|---|---|---|---|---|---|---|---|---|---|
| Verdict, GPU | 7.5 | 14.2 | 20.3 |  | 4.3 | 19.3 | 994 | 0.0014 | 199 of 200 |
| Verdict, Neural Engine | 4.2 | 10.8 | 30.8 |  | 5.0 | 35.0 | 108 | 0.0076 | 198 of 200 |
| Verdict, CPU | 14.2 | 26.3 | 50.8 |  | 8.9 | 49.8 | 287 | 0.0106 | 197 of 200 |
| Laya, GPU | 14.7 | 23.4 | 42.0 | 82.1 | 8.6 | 35.9 | 2,856 | 0.0039 | 200 of 200 |
| Laya, CPU | 36.9 | 69.7 | 141.2 | 354.2 | 23.8 | 143.8 | 9,948 | 0.0136 | 200 of 200 |
| Laya, Neural Engine, one-shape packages | 10.7 |  |  | 209.4 |  |  | 64 and 87 | 0.0111 and 0.0147 | 93 of 94 and 200 of 200 |

On the Mac the Neural Engine answers short prompts one at a time faster than the GPU (Verdict 4.2
against 7.5 ms at 128 tokens; Laya 10.7 against 14.7 ms, from a package with only that shape), but
the GPU is faster from 512 tokens (Laya 82 against 209 ms at 1,024), at batch 16, and closer to
PyTorch. A server batches, so the Mac uses the GPU for both models. The GPU's footprint is large
(1 GB for Verdict, 2.9 GB for Laya), which a server can afford; on the Neural Engine both models
stay near 100 MB. The complete tables, for every package and compute-unit setting on both
machines, are at the end of this report.

## Where Core ML fails

| Package | Compute units | macOS 27.0.1, M3 Max | iOS 27.0, A15 |
|---|---|---|---|
| `verdict-m18-fp16` | CPU, GPU, CPU and Neural Engine | runs | runs |
| `verdict-m18-fp16` | all | batch-16 functions fail to load (the `functionName` error) | failed to load in the first run; ran in the settled run |
| `verdict-e17-fp16` | CPU | crash in BNNS (SIGSEGV) | crash in BNNS (SIGSEGV) |
| `verdict-e17-fp16` | GPU | runs | runs |
| `verdict-e17-fp16` | CPU and Neural Engine | crash in BNNS (SIGTRAP) after a 98 s first load | runs |
| `verdict-e17-fp16` | all | runs, on the GPU | killed (SIGKILL) during batch 16, with no crash or jetsam report |
| `verdict-e17-fp32` | CPU, GPU, all | runs | runs |
| `verdict-e17-fp32` | CPU and Neural Engine | crash in BNNS (SIGTRAP) | runs, all on the CPU |
| `laya-m18-fp16` | CPU, GPU | runs | GPU runs; CPU not run |
| `laya-m18-fp16` | CPU and Neural Engine, all | fails to load (the `functionName` error) | FILL_LAYA_M18_ANE_IOS |
| `laya-e17-fp16` | CPU | crash in BNNS (SIGSEGV) | not run |
| `laya-e17-fp16` | GPU, all | runs | not run |
| `laya-e17-fp16` | CPU and Neural Engine | crash in BNNS (SIGTRAP) after a 192 s first load | first load still compiling after 40 minutes (before the rewrites) and 22 minutes (after); both stopped by hand |

The `functionName` error reads "`MLModelConfiguration`'s `.functionName` property must be `nil`
unless the model type is ML Program", for a package that is an ML program. The iPhone's failure
log is [encoder-runtime/iphone/failures.txt](encoder-runtime/iphone/failures.txt); the Mac's is
[encoder-runtime/macos/failures.txt](encoder-runtime/macos/failures.txt).

### Why the packages need iOS 18

1. **One enumerated input.** Before iOS 18, Core ML allows one input with enumerated shapes, so
   the models take a single stacked `tokens` input. That part is only awkward.
2. **Core ML's CPU backend crashes on enumerated shapes.** On macOS 27.0.1 and on iOS 27.0,
   running an enumerated-shape package with part of it on the CPU ends the process inside
   `BNNSGraphContextExecute_v2` (`libBNNS.dylib`), called from Espresso's
   `E5RT::Ops::BnnsCpuInferenceOperation::ExecuteSync()`, with `EXC_BREAKPOINT` or `EXC_BAD_ACCESS`.
   The iPhone's report for `verdict-e17-fp16` on the CPU has the same frames as the Mac's, and the
   crash happens from Swift as from Python.

A bisect over reduced models on the Mac (float16 unless noted, from Python) shows that it needs
enumerated shapes:

| Converted model | Shapes | CPU | CPU and Neural Engine | GPU |
|---|---|---|---|---|
| Verdict | enumerated | crash | crash | runs |
| Verdict, float32 | enumerated | runs | crash | runs |
| Verdict | one fixed shape, (1, 2, 128) | runs | runs | not run |
| Embeddings and masks only | enumerated | runs | crash | not run |
| Embeddings and GLiClass head, no layers | enumerated | runs | crash | not run |
| One encoder layer | enumerated | crash | crash | runs |
| That layer's attention (SDPA) | enumerated | crash | crash | runs |
| That layer's attention (eager) | enumerated | returns NaN | crash | runs |
| That layer's MLP, LayerNorm or QKV projection | enumerated | runs | crash | runs |

An iOS 18 multifunction package holds one fixed shape per function and stores the weights once,
so it avoids the crash at no cost in size. iOS 18 runs on the same iPhones as iOS 17 (XS, XR and
later, [Apple](https://support.apple.com/en-eg/guide/iphone/iphe3fa5df43/ios)), so no device is
lost. The main package keeps its iOS 17 platform; the encoder backends become available from
iOS 18 and macOS 15.

## Method

1. **Reference (part A).** `Tools/encoders/reference.py` builds 200 questions from the requests
   in Fixtures/schemas/schemas.json and reads them with both models through upstream's own code at
   `dcd2094` (`EncoderEngine.build_schema`, `VerdictEngine.read_batch`, `LayaEngine.read_batch`),
   in float32 on the CPU, 16 questions per forward pass, as upstream loads them. It records the
   prompts, token ids, marker positions, raw logits, calibrated probabilities and laya's answers,
   and the same reads with float16 and bfloat16 weights as the precision floor. Two regenerations
   gave byte-identical files.
2. **Conversion (part B).** `convert_verdict.py` and `convert_laya.py` trace wrappers around the
   models with `torch.jit.trace`, convert them with coremltools 9.0, and read all 200 questions
   again through upstream's read path with each package in place of the PyTorch model, from Python.
3. **Swift (part C).** The harness (`Tools/encoders/Harness`) tokenizes with swift-transformers
   1.3.4, builds Verdict's inputs and Laya's sequences, runs a package with Core ML, applies the
   calibration and compares with the reference. For each package and compute-unit setting it
   measures compile time, a first and a second load, Core ML's compute plan, every question alone
   (batch 1) and the corpus in calls of 16, with the app's physical footprint sampled every 5 ms.
   Questions run grouped by the shape they pad to (128, 256, 512 or 1,024 tokens); the first call
   of each shape is a warmup and is not timed. On the Mac each configuration is its own process
   (`run_macos.sh`); on the iPhone, `run_ios.sh` launches `HarnessApp.swiftpm`, because Xcode
   cannot host a package's tests on a device.
4. **MLX (part D).** Read, not built: mlx-swift-lm at `c043fb3` and mlx-swift 0.32.2.

| Machine or input | Details |
|---|---|
| Mac | MacBook Pro (Mac15,9), Apple M3 Max, 16 cores, 128 GB, macOS 27.0.1 (26A434), Xcode 27.0 (27A266a), Swift 6.4 |
| iPhone | iPhone 13 Pro Max (iPhone14,3), A15, 6 GB, iOS 27.0 (24A437), wired |
| Python | CPython 3.12.2, torch 2.13.0, transformers 5.17.0, coremltools 9.0, gliclass 0.1.20, laya 0.3.6, numpy 2.3.5 |
| Checkpoints | `heman10x/rlcd-modernbert-151m` at `8af2496`, `convaiinnovations/laya-typed-decisions` at `1a793eb` |

## Reference outputs and the precision floor (part A)

The corpus is 200 questions in 26 requests, all taken from Fixtures/schemas/schemas.json: 96
nouls, 60 choices with 2 to 24 options, 44 scores with 2 to 10 levels, 5 JSON states, 7 states
with non-ASCII text, and long states built from the fixture's own texts, so that 70 Verdict
prompts pass 512 tokens and 25 Laya states are cut at 1,024. The fixture's choices over 24 options
are cut to their first N options, which reaches every per_k entry of Verdict's calibrator and
three option counts that use its global temperature.
[Fixtures/encoders](../../Fixtures/encoders/README.md) describes every field.

Upstream itself serves both models in bfloat16 on a GPU ("the answers match fp32's to within
about 0.02 in probability", `encoders.py:219-222`). Reading the corpus with the weights cast, in
PyTorch on the CPU, gives the floor a Core ML package should be judged against:

| Model, weights | Largest logit difference | Largest probability difference | Top answers changed |
|---|---|---|---|
| Verdict, float16 | 0.029 | 0.0011 | 1 of 200 |
| Verdict, bfloat16 | 0.165 | 0.0115 | 3 of 200 |
| Laya, float16 | 0.012 | 0.0016 | 0 of 200 |
| Laya, bfloat16 | 0.144 | 0.0191 | 0 of 200 |

## Tokenization

swift-transformers 1.3.4 (`AutoTokenizer.from(modelFolder:)` on each checkpoint's tokenizer.json
and tokenizer_config.json) reproduces upstream's token ids for all 200 Verdict prompts, including
the `<<LABEL>>` and `<<SEP>>` markers, non-ASCII text, control characters and the truncation at
512 (the first 510 tokens between [CLS] and [SEP]). A Swift port of laya's `build_sequence`,
given the strings laya tokenizes, reproduces all 200 Laya sequences and their [MASK] marker
positions, with the 48-token option cap, the 256-token head budget and the cut at 1,024. The
harness's `ParityTests` check both on macOS and in the iOS Simulator, and every iPhone and Mac
result repeats the check (200 of 200 each time). On the iPhone, loading a tokenizer takes 0.2 to
0.3 s, and tokenizing a question takes 9 ms at the median and 38 ms at the 95th percentile, which
is as much as a short Verdict read, so it belongs in a backend's latency budget.

Upstream's calibration applied in Swift (Verdict's per_k temperatures, softmax and abstention
drop; Laya's temperature buckets and the [0.5, 5] clamp) reproduces the recorded probabilities
from the recorded logits within 1e-6.

## Conversion (part B)

Both models convert from PyTorch with coremltools 9.0 (`torch.jit.trace`, then `ct.convert` to
an ML program). The shipped ONNX export is not a route with this toolchain: coremltools 9.0 has no
ONNX frontend (`ct.convert` accepts only TensorFlow, PyTorch and MIL sources,
`coremltools/converters/_converters_entry.py:1033`), and handing it `model_fp16.onnx` (opset 17)
fails ("Unable to determine the type of the model", recorded in
[encoder-runtime/verdict-coreml.json](encoder-runtime/verdict-coreml.json)). The archived
standalone `onnx-coreml` converter was not tried.

### What the converted models take and return

| Model | Input `tokens`, int32 [batch, planes, sequence] | Output | Shapes |
|---|---|---|---|
| Verdict | token ids (padding 50283), attention mask | `logits` float32 [batch, 25] | batch 1 or 16, sequence 128, 256 or 512 |
| Laya | token ids, attention mask, question type (0 choice, 1 score, 2 noul) | `scores` float32 [batch, sequence], the scorer at every position | batch 1 or 16, sequence 128, 256, 512 or 1,024 |

Verdict's calibration (per_k temperatures, softmax, dropping "insufficient evidence") and Laya's
marker gather, temperature bucket, clamp and softmax stay outside the model, in Swift, so the graph
has no data-dependent shapes. Laya's action head is not converted; upstream does not use it.

### Packages

| Package | Kind | Precision | Minimum OS | Size | Converted in |
|---|---|---|---|---|---|
| `verdict-m18-fp16` | one function per shape (`b1_s128` to `b16_s512`), weights stored once | float16 | iOS 18, macOS 15 | 306 MB | 53 s |
| `verdict-e17-fp16` | one program, enumerated shapes | float16 | iOS 17, macOS 14 | 304 MB | 8 s |
| `verdict-e17-fp32` | one program, enumerated shapes | float32 | iOS 17, macOS 14 | 607 MB | 5 s |
| `laya-m18-fp16` | one function per shape (`b1_s128` to `b16_s1024`), weights stored once | float16 | iOS 18, macOS 15 | 849 MB | 159 s |
| `laya-e17-fp16` | one program, enumerated shapes | float16 | iOS 17, macOS 14 | 845 MB | 13 s |
| `laya-f18-b1s128-fp16`, `laya-f18-b1s1024-fp16` | one program for one fixed shape, (1, 3, 128) or (1, 3, 1,024) | float16 | iOS 18, macOS 15 | 843 and 845 MB | 10 s each |
| `verdict-m18-w8`, `laya-m18-w8` | as `*-m18-fp16`, with 8-bit weights | int8 weights, float16 computation | iOS 18, macOS 15 | 154 and 427 MB | 68 and 199 s |

Read from Python on the Mac, with each package in place of the PyTorch model in upstream's read
path:

Verdict:

| Package | Units, batch | Max abs logit difference | Max abs probability difference | Mean abs probability difference | Top answers kept |
|---|---|---|---|---|---|
| verdict-e17-fp16 | CPU_AND_GPU batch 1 | 0.026 | 1.39e-03 | 2.2e-04 | 199/200 |
| verdict-e17-fp16 | CPU_AND_GPU batch 16 | 0.021 | 1.13e-03 | 2.2e-04 | 199/200 |
| verdict-e17-fp16 | ALL batch 1 | 0.026 | 1.39e-03 | 2.2e-04 | 199/200 |
| verdict-e17-fp16 | ALL batch 16 | 0.021 | 1.13e-03 | 2.2e-04 | 199/200 |
| verdict-e17-fp32 | CPU_ONLY batch 1 | 2.6e-05 | 1.70e-06 | 2.7e-07 | 200/200 |
| verdict-e17-fp32 | CPU_ONLY batch 16 | 2.6e-05 | 1.70e-06 | 2.7e-07 | 200/200 |
| verdict-e17-fp32 | CPU_AND_GPU batch 1 | 2.6e-05 | 1.91e-06 | 2.7e-07 | 200/200 |
| verdict-e17-fp32 | CPU_AND_GPU batch 16 | 2.5e-05 | 2.09e-06 | 2.5e-07 | 200/200 |
| verdict-e17-fp32 | ALL batch 1 | 2.6e-05 | 1.91e-06 | 2.7e-07 | 200/200 |
| verdict-e17-fp32 | ALL batch 16 | 2.5e-05 | 2.09e-06 | 2.5e-07 | 200/200 |
| verdict-m18-fp16 | CPU_AND_GPU batch 1 | 0.026 | 1.39e-03 | 2.2e-04 | 199/200 |
| verdict-m18-fp16 | CPU_AND_GPU batch 16 | 0.021 | 1.13e-03 | 2.2e-04 | 199/200 |
| verdict-m18-fp16 | ALL batch 1 | 0.026 | 1.39e-03 | 2.2e-04 | 199/200 |
| verdict-m18-fp16 | ALL batch 16 | load failed: "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." | | | |
| verdict-m18-w8 | CPU_AND_GPU batch 1 | 0.313 | 2.61e-02 | 2.4e-03 | 199/200 |
| verdict-m18-w8 | CPU_AND_GPU batch 16 | 0.294 | 2.49e-02 | 2.4e-03 | 199/200 |

Laya, whose probabilities are compared before laya's rounding; the rounded answers can match only when a difference stays under 5e-5:

| Package | Units, batch | Max abs logit difference | Max abs probability difference | Mean abs probability difference | Top answers kept | Rounded answers identical |
|---|---|---|---|---|---|---|
| laya-e17-fp16 | CPU_AND_GPU batch 1 | 0.029 | 3.95e-03 | 2.0e-04 | 200/200 | 19/200 |
| laya-e17-fp16 | CPU_AND_GPU batch 16 | 0.029 | 3.95e-03 | 2.1e-04 | 200/200 | 14/200 |
| laya-e17-fp16 | ALL batch 1 | 0.079 | 1.27e-02 | 6.3e-04 | 200/200 | 5/200 |
| laya-e17-fp16 | ALL batch 16 | 0.079 | 1.27e-02 | 6.4e-04 | 200/200 | 2/200 |
| laya-m18-fp16 | CPU_AND_GPU batch 1 | 0.029 | 3.95e-03 | 2.0e-04 | 200/200 | 19/200 |
| laya-m18-fp16 | CPU_AND_GPU batch 16 | 0.029 | 3.95e-03 | 2.1e-04 | 200/200 | 14/200 |
| laya-m18-fp16 | ALL batch 1 | load failed: "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." | | | | |
| laya-m18-fp16 | ALL batch 16 | load failed: "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." | | | | |
| laya-m18-w8 | CPU_AND_GPU batch 1 | 0.307 | 3.96e-02 | 4.8e-03 | 194/200 | 0/200 |
| laya-m18-w8 | CPU_AND_GPU batch 16 | 0.322 | 4.13e-02 | 4.8e-03 | 194/200 | 1/200 |

`convert_verdict.py --only verdict-m18-w8` and `convert_laya.py --only laya-m18-w8` also write
packages with 8-bit weights, at half the size. That costs too much accuracy for either model:
Verdict's probabilities moved by up to 0.026 (1 of 200 top answers changed) and Laya's by up to
0.041 (6 of 200 changed), about twice upstream's own bfloat16 variation (0.0115 and 0.019), so
neither is a candidate.

### Problems and workarounds

1. **numpy.** numpy 2.4 made `int()` of a one-element array an error; coremltools 9.0 does that
   while converting shape arithmetic (`TypeError: only 0-dimensional arrays can be converted to
   Python scalars`). numpy is pinned to 2.3.5.
2. **Untested torch.** coremltools 9.0 warns that it was tested up to torch 2.7; torch 2.13, the
   version upstream's images use, converted both models without an unsupported operator.
3. **Masks.** transformers' `masking_utils` is replaced by additive float masks built with plain
   operations: -1e4 on padding keys, and in the local layers where |i - j| > 64. -1e4 is finite in
   float16, so a padded query whose whole window is padding does not produce NaN.
4. **Positions.** The rotary cos and sin tables and the local-window band are computed once for
   the longest sequence and sliced, so the program holds no position arithmetic. PyTorch runs the
   wrapped models within 1.6e-5 (Verdict) and 6.9e-5 (Laya) of the originals in float32.
5. **PyTorch's fast path.** With it on, `nn.TransformerEncoderLayer` (Laya's head) calls the fused
   `torch._transformer_encoder_layer_fwd`, which `torch.jit.trace` refuses ("Cannot insert a
   Tensor that requires grad as a constant"); the converter turns it off. laya 0.3.6 uses the
   stock layer; there is no `_DynamicMultiheadAttention` in this version.
6. **Laya and the Neural Engine.** Converted as is, Laya never ran on the Neural Engine: Core ML's
   plan put all 1,184 of its operations on the CPU, although the encoder, the head and the scorer
   were each planned on the Neural Engine when converted alone. What failed was the question
   type's embedding (a gather indexed by an input) feeding the head. The converter writes the
   embedding as a product with a one-hot row, and the head layers' norm-first forward with rank-4
   tensors (PyTorch's own path goes through a rank-5 tensor); the model then plans with 1,167
   operations on the Neural Engine and 16 on the CPU, and PyTorch runs it within 6.9e-5 of the
   original. `laya_ane_plan.py` reproduces the bisect in about five minutes:

   | Laya converted at (1, 3, 128) | Operations planned on the Neural Engine | On the CPU |
   |---|---|---|
   | encoder | 1,123 | 12 |
   | encoder and type embedding | 1,121 | 25 |
   | encoder and head | 1,160 | 12 |
   | encoder and scorer | 1,127 | 12 |
   | encoder, type embedding and head | 0 | 1,184 |
   | encoder, one-hot type embedding, head and scorer | 1,167 | 16 |

7. **Multifunction loading.** Each function loaded as its own `MLModel` keeps its own copy of the
   weights once it has run (on the Mac GPU, six Verdict functions took 1.5 GB for a 306 MB
   package), so the runners load a function when a batch first needs it and keep one (Swift) or
   two (Python). Some loads fail with the misleading `functionName` error: Verdict's batch-16
   functions under `.all`, and every Laya function wherever the Neural Engine is allowed, on the
   Mac and on the iPhone, before and after the rewrites in point 6 (a package with only the four
   batch-1 functions, tried before them, failed the same way). A package that holds one program
   for one fixed shape loads: after the rewrites, `laya-f18-b1s128-fp16` and `laya-f18-b1s1024-fp16`
   run on the Mac's Neural Engine with 1,167 of their 1,183 operations there.
8. **coremltools' Python binding** crashed the interpreter once while releasing an input after a
   multifunction prediction (`_PyObject_Free` from `-[MLFeatureValue dealloc]` in
   `-[MLE5ExecutionStream _reset]`). The Python runner keeps its inputs alive, and the CPU and
   Neural Engine settings of the multifunction packages are measured from Swift only.

## The MLX alternative (estimated, not built)

Read in mlx-swift-lm at `c043fb3`, the revision the main package pins, and mlx-swift 0.32.2.

What exists:

- `MLXEmbedders` registers `bert`, `roberta`, `xlm-roberta`, `distilbert`, `nomic_bert`, `qwen3`,
  `lfm2`, `gemma3`, `gemma3_text` and `gemma3n` (`ModelFactory.swift:30-46`). Nothing mentions
  ModernBERT.
- `NomicBert.swift` (1,039 lines) is the closest in shape: a fused `Wqkv` projection under the same
  weight name as ModernBERT (line 354), configurable Linear biases (156-163, 394-399) and RoPE on q
  and k (416-421, 457-459). But it has one RoPE base for every layer, a SwiGLU MLP with separate
  `fc11` and `fc12` and a hidden size rounded up to a multiple of 256 (143-148; ModernBERT's 1,152
  and 2,624 are not multiples), post-norm blocks without a final norm, and attention written out
  as matmul and softmax with the full score matrix (464-472).
- `Bert.swift` (818 lines) uses absolute positions and biases everywhere.
- The Gemma 3 embedder already chooses the RoPE base per layer from a sliding-window pattern
  (`Gemma3.swift:185-199`) and ends with a norm, but its sliding mask is causal.
- `MLXLMCommon`'s `createBidirectionalSlidingWindowMask` tests absolute positions (`kIdx <
  windowSize`, `BidirectionalMasks.swift:55-73`); its own documentation says a distance window must
  be built inline.
- MLXNN has what the blocks need: `LayerNorm(dimensions:eps:affine:bias:)` without a bias
  (`Normalization.swift:91-99`), `Linear(_:_:bias:)`, `RoPE(dimensions:traditional:base:scale:)`,
  exact `gelu`, and `MLXFast.scaledDotProductAttention` with an additive mask.
- A model type cannot be added to `MLXEmbedders` from outside the package: `EmbeddingModelOutput`
  (`EmbeddingModel.swift:8-11`) has no public initializer, so another module cannot conform to
  `EmbeddingModel`. A port would define its own `Module` types in OpenJevSwift and load weights
  with `MLXLMCommon.loadWeights`, as the DiffusionGemma port does.

What ModernBERT and the two heads need on top:

| Piece | What it takes | Lines |
|---|---|---|
| Configuration | `layer_types` or `global_attn_every_n_layers`, `local_attention`, two RoPE bases, `norm_eps`, sizes | 50 |
| Embeddings | token embedding, LayerNorm without bias | 15 |
| Attention | fused `Wqkv` without bias, RoPE with base 160,000 in global and 10,000 in local layers, SDPA, `Wo` without bias | 55 |
| Masks | padding mask; local layers add \|i - j\| <= 64 | 20 |
| MLP | GeGLU: `Wi` to twice the intermediate size, exact GELU of one half times the other, `Wo`; no biases | 20 |
| Layers and model | pre-norm, no `attn_norm` in layer 0, final norm | 40 |
| Weight loading | key mapping for both checkpoints (`model.encoder_model.*` in Verdict, `encoder.*` and the head in Laya) | 80 |
| GLiClass head | first 25 `<<LABEL>>` positions, two GELU projectors with bias, `[CLS]` pooling, dot product | 60 |
| Laya head | type embedding, two norm-first `TransformerEncoderLayer`s (fused `in_proj` with bias, ReLU feed-forward, key padding mask), scorer MLP | 90 |
| Tests | parity with Fixtures/encoders on 200 questions per model; shape tests on random weights for CI (D-028) | 250 |

That is about 700 lines: 3 days for the model code and loading, 1 day for the tests, and 1 to 2 days
on an iPhone for float16 weights, `Memory.cacheLimit`, footprint and latency, so 5 to 6 days in
all. The prompt builders, tokenization and calibration are shared with a Core ML backend and are
not counted. The Core ML route needed no model code: the converters in `Tools/encoders` and a
runner of about 150 lines (`Harness/Sources/EncoderHarness/CoreMLEncoder.swift`).

MLX's iOS memory behaviour, from its documentation and code, not measured here: MLX keeps the
weights in its own Metal buffers, so they count fully in the app's footprint (about 300 MB for
Verdict and 840 MB for Laya in float16). Freed buffers stay in MLX's cache up to
`Memory.cacheLimit`, which defaults to the memory limit, itself based on the GPU's
`recommendedMaxWorkingSetSize`; mlx-swift's own documentation recommends lowering it on iOS, where
jetsam limits apply (`Memory.swift:35-47` and `227-235`). MLX runs on the GPU and the CPU; it
cannot use the Neural Engine.

## Scope notes for #57 (Verdict) and #58 (Laya)

Both:

- **Runtime.** Core ML, from the multifunction float16 packages (`*-m18-fp16`), minimum iOS 18
  and macOS 15, with the compute units in the decision table. The backend loads the function a
  batch needs and keeps one or two loaded; each loaded function holds its own copy of the weights
  once it has run.
- **Packaging.** Download on first use, verify the SHA-256, compile once with
  `MLModel.compileModel(at:)`, keep the compiled model in Application Support, and load the batch-1
  functions in the background after the download. On macOS the server uses the same packages.
- **Conversion.** `Tools/encoders/convert_verdict.py` and `convert_laya.py` are the pipeline;
  re-run them when a checkpoint revision, coremltools or the minimum OS moves, and gate a new
  package on the parity they print.
- **Tokenizer.** swift-transformers 1.3.4 (`AutoTokenizer.from(modelFolder:)`) reproduces every
  recorded token id. Tokenizing costs 9 ms per question at the median on the iPhone; tokenizing a
  request's state once for all its questions would save most of it.
- **Tests.** Fixtures/encoders is the oracle. Tokenization compares exactly, calibration within
  1e-6, and model outputs with the tolerances measured here: a probability within 0.01 of float32
  PyTorch and the top answer unchanged except on the listed near ties.
- **Wire contract.** Unchanged from upstream's `EncoderEngine`: `build_schema`, the 400s for
  images, steps, samples, think and sequential, at most 16 questions per pass, and
  `usage.input_tokens` as the attention-mask sum.
- **Batch.** One question per call on the iPhone; up to 16 on the Mac GPU.

#57, Verdict:

- Port `verdict_prompt` (upstream `encoders.py:224-237`) byte for byte, and truncate as the
  harness does: the first 510 tokens between [CLS] and [SEP].
- Calibration: `per_k[k]` or the global temperature, softmax over the first k logits, drop the
  abstention, renormalise, uniform when not finite (`VerdictCalibration` in the harness).
- 24 options at most: the head has 25 logits.

#58, Laya:

- Port `build_sequence` with its budgets (the harness's `LayaSequence` matches all 200 sequences)
  and the rendering in front of it: `render_options`, `render_criterion` (compact JSON with
  `", "` and `": "` separators, `ensure_ascii=False`, `default=str`) and `serialize_state`, on the
  core's order-preserving JSON model and Python float formatting (D-007, the python-json
  fixtures). Laya replaces "[MASK]" in every text with a space.
- The model returns a score per position; read it at the markers, divide by the bucket's
  temperature (`temperature_by_options[bucket]`, else `temperature[qtype]`, both clamped to
  [0.5, 5]; `choice:11+` is 0.1006 raw and 0.5 applied), take the softmax, round to 4 decimals as
  Python's `round` does, and renormalise as upstream does. The action head is not converted.
- Float16 moves probabilities by up to 0.004 on the GPU, so laya's 4-decimal answers differ from
  PyTorch's in most questions while the top answers hold; tests compare before rounding, and the
  rounding needs its own test against the recorded answers.
- FILL_58_ANE

## Is #55 (JevK5 on an iPhone) worth trying next?

Not before Verdict and Laya ship, and then only on an 8 GB iPhone. This is an estimate scaled from
this spike's measurements, not a measurement of JevK5:

- JevK5 is Qwen3.5-4B, about ten times Laya's size, and reads each question with one forward pass
  over a prompt of hundreds to thousands of tokens. MLX can only use the GPU.
- On this iPhone's GPU, Laya read 1,024 tokens in 1.35 s, about 0.6 TFLOPS of useful work
  (roughly 0.8 GFLOP per token). A 4B model needs about 7 GFLOP per token, so at that rate about
  12 ms per prompt token: 6 s for a 500-token prompt and 18 s for 1,500, per question, before the
  cost of 4-bit dequantization.
- Its 4-bit weights are about 2.5 GB. A 6 GB iPhone would need the increased-memory-limit
  entitlement, and the phone reached the serious thermal state within minutes of reads by models
  a tenth of that size.

So JevK5 on a phone would take seconds per question and heat the device: acceptable for an
offline demo, not for interactive reads. A one-day measurement on an iPhone 15 Pro or later (8 GB,
a faster GPU) would settle it; until then an app should send JevK5 reads to a server over the
wire API.

## Problems and open items

1. **Thermal state.** Every configuration took the phone from nominal to serious within one to
   seven minutes, and the first run's later configurations started in the critical state and ran
   up to twice as slow. Sustained reads on a phone will run slower than the settled numbers.
2. **Core ML bugs worth reporting to Apple** (Feedback Assistant; not filed): the BNNS crash with
   enumerated shapes (`run_macos.sh verdict-e17-fp16`), the misleading `functionName` load error
   (`run_macos.sh laya-m18-fp16`), the silent CPU fallback of the unmodified Laya program
   (`laya_ane_plan.py`), and the SIGKILL of `verdict-e17-fp16` under `.all` on the phone with no
   crash or jetsam report.
3. **Laya on the Neural Engine.** FILL_OPEN_LAYA_ANE
4. **One device.** Only an A15 with 6 GB was measured. Older supported iPhones (A12 to A14, 3 to
   4 GB) are slower, and Laya's GPU path peaked at 855 MB.
5. **The Laya rounding.** Swift must round Laya's probabilities to 4 decimals exactly as Python's
   `round` does before renormalising; #58 needs its own test for it against the recorded answers.

## Reproducing

```bash
make upstream
/usr/local/bin/python3.12 -m venv Tools/encoders/.venv
Tools/encoders/.venv/bin/python -m pip install -r Tools/encoders/requirements.txt
Tools/encoders/.venv/bin/python Tools/encoders/reference.py
Tools/encoders/.venv/bin/python Tools/encoders/convert_verdict.py
Tools/encoders/.venv/bin/python Tools/encoders/convert_laya.py
swift test --package-path Tools/encoders/Harness
Tools/encoders/run_macos.sh
DEVELOPMENT_TEAM=<team> RESULTS_DIR=$PWD/docs/spikes/encoder-runtime/iphone-settled SETTLE=600 COOLDOWN=0 PASSES16=1 Tools/encoders/run_ios.sh <device id> verdict-m18-fp16:cpuAndNeuralEngine laya-m18-fp16:cpuAndGPU
Tools/encoders/.venv/bin/python Tools/encoders/summarize.py docs/spikes/encoder-runtime/iphone-settled
```

[Tools/encoders/README.md](../../Tools/encoders/README.md) describes each step. The reference run
takes about 37 minutes, and the converters about 5 and 15.

## All results

### iPhone, settled run

Each configuration waited for the nominal thermal state before it started (`iphone-settled/`).

Device: iPhone14,3 (D64AP), Version 27.0 (Build 24A437), 6 cores, 6 GiB

| Package | Units | Compile s | Load s (first, second) | Batch 1 median ms | Batch 1 p95 ms | Batch 16 per call median ms | Batch 16 p95 ms | Batch 16 per question ms | Peak footprint MB | Thermal |
|---|---|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 1.2 | 1.2, 0.0 | 83.5 | 90.8 | 2,004.1 | 2,381.1 | 125.3 | 264 | nominal to serious |
| verdict-m18-fp16 | cpuAndGPU | 1.2 | 1.5, 0.1 | 103.8 | 114.2 | 2,715.6 | 5,217.0 | 169.7 | 289 | nominal to serious |
| verdict-m18-fp16 | cpuAndNeuralEngine | 1.2 | 7.4, 0.2 | 42.5 | 51.6 | 1,350.1 | 1,465.9 | 84.5 | 106 | nominal to serious |
| verdict-m18-fp16 | all | 1.1 | 6.2, 0.2 | 43.2 | 47.8 | 1,514.5 | 1,709.3 | 94.7 | 176 | nominal to serious |
| laya-m18-fp16 | cpuAndGPU | 2.7 | 14.6, 0.2 | 299.3 | 1,380.9 | 9,987.5 | 24,491.9 | 624.2 | 855 | nominal to serious |

| Package | Units | Batch 1 median ms at 128, 256, 512, 1024 tokens | Batch 16 median ms at 128, 256, 512, 1024 | Function loads s | Footprint after first call MB |
|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 22.1, 38.7, 88.8, none | 284.4, 787.9, 2,261.5, none | b16_s128 1.2, b16_s256 1.2, b16_s512 1.3, b1_s128 0.0, b1_s256 1.0, b1_s512 1.5 | 35 to 85 |
| verdict-m18-fp16 | cpuAndGPU | 26.8, 49.1, 107.8, none | 371.5, 819.5, 2,782.5, none | b16_s128 1.5, b16_s256 1.3, b16_s512 3.0, b1_s128 0.1, b1_s256 1.2, b1_s512 1.7 | 119 to 249 |
| verdict-m18-fp16 | cpuAndNeuralEngine | 9.7, 17.0, 46.2, none | 125.6, 437.1, 1,356.4, none | b16_s128 14.0, b16_s256 43.1, b16_s512 38.8, b1_s128 0.2, b1_s256 7.1, b1_s512 9.1 | 43 to 102 |
| verdict-m18-fp16 | all | 13.8, 18.5, 47.0, none | 124.5, 379.2, 1,565.3, none | b16_s128 11.8, b16_s256 42.0, b16_s512 45.8, b1_s128 0.2, b1_s256 9.2, b1_s512 14.6 | 54 to 126 |
| laya-m18-fp16 | cpuAndGPU | 81.7, 151.7, 385.2, 1,346.4 | 2,340.6, 4,741.3, 9,913.5, 22,201.8 | b16_s1024 3.3, b16_s128 4.9, b16_s256 3.4, b16_s512 2.9, b1_s1024 9.0, b1_s128 8.0, b1_s256 11.4, b1_s512 10.2 | 74 to 601 |

| Package | Units | Max abs probability difference (batch 1, 16) | Mean (batch 1) | Max abs logit difference | Top answers kept (batch 1, 16) | Non-finite | Tokenization | Planned cost by device |
|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 1.06e-02, 1.04e-02 | 1.8e-03 | 0.195 | 197, 197 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-m18-fp16 | cpuAndGPU | 1.79e-03, 1.79e-03 | 2.4e-04 | 0.053 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-m18-fp16 | cpuAndNeuralEngine | 7.35e-03, 8.19e-03 | 9.9e-04 | 0.227 | 197, 198 of 200 | 0, 0 | 200/200 | ANE 67%, CPU 33% |
| verdict-m18-fp16 | all | 7.35e-03, 8.19e-03 | 9.9e-04 | 0.227 | 197, 198 of 200 | 0, 0 | 200/200 | ANE 75%, GPU 25% |
| laya-m18-fp16 | cpuAndGPU | 2.14e-03, 2.14e-03 | 1.8e-04 | 0.022 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |

Failures:

launch 1 did not start (verdict-m18-fp16:cpuAndNeuralEngine,verdict-m18-fp16:cpuAndGPU,verdict-m18-fp16:cpuOnly,laya-m18-fp16:cpuAndGPU,verdict-m18-fp16:all,laya-e17-fp16:cpuAndNeuralEngine)
    ERROR: The application failed to launch. (com.apple.dt.CoreDeviceError error 10002 (0x2712))
           ----------------------------------------
               The operation couldn?t be completed. Invalid argument (NSPOSIXErrorDomain error 22 (0x16))
laya-e17-fp16:cpuAndNeuralEngine: stopped by hand (SIGTERM) at 17:25, after 22 minutes in its first load with ANECompilerService compiling; no result; device report JetsamEvent-2026-09-30-165011.ips
    Launched application with org.openjevswift.encoderharness bundle identifier.
    Waiting for the application to terminate...
    App terminated due to signal 15.

### iPhone, first run

Twelve Verdict configurations in one launch with 30-second pauses, and the Laya GPU configuration with the package before the graph rewrites (`iphone/`). Later configurations started hot; the thermal column shows how hot.

Device: iPhone14,3 (D64AP), Version 27.0 (Build 24A437), 6 cores, 6 GiB

| Package | Units | Compile s | Load s (first, second) | Batch 1 median ms | Batch 1 p95 ms | Batch 16 per call median ms | Batch 16 p95 ms | Batch 16 per question ms | Peak footprint MB | Thermal |
|---|---|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 1.1 | 1.4, 0.0 | 85.5 | 95.7 | 2,058.8 | 6,434.4 | 128.7 | 275 | serious to serious |
| verdict-m18-fp16 | cpuAndGPU | 1.1 | 2.4, 0.1 | 106.6 | 140.6 | 2,566.6 | 2,601.0 | 160.4 | 367 | serious to serious |
| verdict-m18-fp16 | cpuAndNeuralEngine | 1.2 | 6.9, 0.2 | 42.1 | 46.4 | 1,295.9 | 1,446.5 | 81.0 | 114 | nominal to serious |
| verdict-e17-fp16 | cpuAndGPU | 0.7 | 1.4, 0.2 | 109.8 | 124.2 | 3,197.1 | 3,736.5 | 201.0 | 235 | serious to serious |
| verdict-e17-fp16 | cpuAndNeuralEngine | 0.7 | 121.4, 0.1 | 65.7 | 80.0 | 1,558.5 | 1,811.3 | 97.4 | 90 | critical to critical |
| verdict-e17-fp32 | cpuOnly | 1.3 | 8.0, 0.0 | 233.3 | 331.3 | 8,462.2 | 9,123.3 | 529.5 | 407 | critical to critical |
| verdict-e17-fp32 | cpuAndGPU | 1.9 | 3.9, 0.4 | 254.7 | 336.5 | 4,698.3 | 4,891.7 | 293.6 | 400 | critical to critical |
| verdict-e17-fp32 | cpuAndNeuralEngine | 3.0 | 8.8, 0.0 | 342.2 | 478.8 | 9,307.7 | 10,240.6 | 581.7 | 408 | critical to critical |
| verdict-e17-fp32 | all | 1.3 | 3.8, 0.4 | 126.7 | 154.1 | 5,068.9 | 6,457.3 | 319.8 | 569 | critical to critical |
| laya-m18-fp16 | cpuAndGPU | 1.8 | 16.4, 0.2 | 302.7 | 2,038.1 | 11,958.4 | 31,378.2 | 747.4 | 856 | fair to serious |

| Package | Units | Batch 1 median ms at 128, 256, 512, 1024 tokens | Batch 16 median ms at 128, 256, 512, 1024 | Function loads s | Footprint after first call MB |
|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 22.5, 40.2, 93.4, none | 308.3, 746.3, 3,738.6, none | b16_s128 1.1, b16_s256 1.6, b16_s512 1.3, b1_s128 0.0, b1_s256 1.5, b1_s512 1.4 | 51 to 76 |
| verdict-m18-fp16 | cpuAndGPU | 27.0, 57.1, 131.2, none | 441.0, 1,305.9, 2,574.7, none | b16_s128 1.2, b16_s256 1.3, b16_s512 1.2, b1_s128 0.1, b1_s256 1.3, b1_s512 1.4 | 67 to 255 |
| verdict-m18-fp16 | cpuAndNeuralEngine | 9.8, 16.9, 46.0, none | 123.2, 435.8, 1,308.6, none | b16_s128 10.8, b16_s256 41.2, b16_s512 39.7, b1_s128 0.2, b1_s256 8.1, b1_s512 9.2 | 52 to 114 |
| verdict-e17-fp16 | cpuAndGPU | 33.2, 56.2, 121.2, none | 439.4, 1,703.5, 3,328.3, none | main 0.2 | 66 to 211 |
| verdict-e17-fp16 | cpuAndNeuralEngine | 11.1, 25.0, 72.3, none | 205.2, 502.6, 1,604.2, none | main 0.1 | 48 to 75 |
| verdict-e17-fp32 | cpuOnly | 63.2, 109.7, 243.0, none | 2,720.3, 4,949.8, 8,537.6, none | main 0.0 | 35 to 58 |
| verdict-e17-fp32 | cpuAndGPU | 44.0, 159.5, 289.5, none | 1,047.6, 2,135.5, 4,719.2, none | main 1.0 | 61 to 354 |
| verdict-e17-fp32 | cpuAndNeuralEngine | 83.4, 171.7, 454.8, none | 1,596.9, 3,968.1, 9,508.2, none | main 0.0 | 33 to 80 |
| verdict-e17-fp32 | all | 35.9, 62.7, 148.1, none | 543.3, 3,584.9, 5,250.1, none | main 1.6 | 68 to 354 |
| laya-m18-fp16 | cpuAndGPU | 81.7, 147.8, 346.6, 1,433.9 | 2,457.3, 5,101.9, 11,707.3, 25,206.4 | b16_s1024 3.6, b16_s128 4.0, b16_s256 3.3, b16_s512 4.5, b1_s1024 4.1, b1_s128 6.9, b1_s256 5.8, b1_s512 5.0 | 70 to 601 |

| Package | Units | Max abs probability difference (batch 1, 16) | Mean (batch 1) | Max abs logit difference | Top answers kept (batch 1, 16) | Non-finite | Tokenization | Planned cost by device |
|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 1.06e-02, 1.04e-02 | 1.8e-03 | 0.195 | 197, 197 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-m18-fp16 | cpuAndGPU | 1.79e-03, 1.79e-03 | 2.4e-04 | 0.053 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-m18-fp16 | cpuAndNeuralEngine | 7.35e-03, 8.19e-03 | 9.9e-04 | 0.227 | 197, 198 of 200 | 0, 0 | 200/200 | ANE 67%, CPU 33% |
| verdict-e17-fp16 | cpuAndGPU | 1.79e-03, 1.79e-03 | 2.4e-04 | 0.053 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-e17-fp16 | cpuAndNeuralEngine | 7.35e-03, 8.19e-03 | 9.9e-04 | 0.227 | 197, 198 of 200 | 0, 0 | 200/200 | ANE 67%, CPU 33% |
| verdict-e17-fp32 | cpuOnly | 1.72e-06, 1.72e-06 | 2.7e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-e17-fp32 | cpuAndGPU | 1.83e-06, 1.81e-06 | 2.5e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-e17-fp32 | cpuAndNeuralEngine | 1.72e-06, 1.72e-06 | 2.7e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-e17-fp32 | all | 1.83e-06, 1.81e-06 | 2.5e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| laya-m18-fp16 | cpuAndGPU | 2.14e-03, 2.14e-03 | 1.8e-04 | 0.022 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |

Failures:

# Written by Tools/encoders/run_ios.sh. The report names on three lines were corrected by hand:
# the first version of the script picked the newest report by name, not by time.
HARNESS_ERROR verdict-m18-fp16 all: Error Domain=com.apple.CoreML Code=0 "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." UserInfo={NSLocalizedDescription=`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.}
verdict-e17-fp16:all: the process ended while it ran (SIGKILL during batch 16, after batch 1 at a 735 ms median); the device kept no crash or jetsam report for it
    Launched application with org.openjevswift.encoderharness bundle identifier.
    Waiting for the application to terminate...
    App terminated due to signal 9.
verdict-e17-fp16:cpuOnly: the process ended while it ran; device report EncoderHarnessApp-2026-09-30-143401.ips (EXC_BAD_ACCESS in BNNSGraphContextExecute_v2)
    Launched application with org.openjevswift.encoderharness bundle identifier.
    Waiting for the application to terminate...
    App terminated due to signal 11.
HARNESS_ERROR laya-m18-fp16 cpuAndNeuralEngine: Error Domain=com.apple.CoreML Code=0 "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." UserInfo={NSLocalizedDescription=`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.}
laya-e17-fp16:cpuAndNeuralEngine: stopped by hand (SIGTERM) after 40 minutes in its first load, with ANECompilerService still compiling (ANECompilerService.cpu_resource-2026-09-30-145641.ips). Earlier, JetsamEvent-2026-09-30-144546.ips shows the system killing ANECompilerService for exceeding its per-process memory limit, during the laya-m18-fp16 cpuAndGPU run
    Launched application with org.openjevswift.encoderharness bundle identifier.
    Waiting for the application to terminate...
    App terminated due to signal 15.

### Mac

One process per configuration (`macos/`). The first `verdict-e17-fp16` GPU run was disturbed (non-monotonic by length, in the fair thermal state) and was repeated on an idle machine; the file holds the repeat.

Device: arm64 (Mac15,9), Version 27.0.1 (Build 26A434), 16 cores, 128 GiB

| Package | Units | Compile s | Load s (first, second) | Batch 1 median ms | Batch 1 p95 ms | Batch 16 per call median ms | Batch 16 p95 ms | Batch 16 per question ms | Peak footprint MB | Thermal |
|---|---|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 0.6 | 1.3, 0.0 | 49.6 | 52.5 | 732.2 | 914.2 | 46.5 | 287 | nominal to nominal |
| verdict-m18-fp16 | cpuAndGPU | 0.5 | 1.4, 0.1 | 16.1 | 23.7 | 277.6 | 352.7 | 17.3 | 994 | fair to nominal |
| verdict-m18-fp16 | cpuAndNeuralEngine | 0.5 | 5.1, 0.2 | 30.7 | 32.4 | 559.0 | 578.2 | 34.9 | 108 | nominal to nominal |
| verdict-e17-fp16 | cpuAndGPU | 0.1 | 1.6, 0.2 | 16.1 | 16.8 | 193.5 | 213.5 | 12.1 | 1,054 | nominal to nominal |
| verdict-e17-fp16 | all | 0.1 | 5.6, 0.5 | 16.8 | 29.4 | 332.2 | 425.6 | 20.8 | 1,101 | fair to fair |
| verdict-e17-fp32 | cpuOnly | 0.2 | 2.3, 0.0 | 129.8 | 182.7 | 2,064.6 | 2,183.5 | 129.7 | 468 | fair to fair |
| verdict-e17-fp32 | cpuAndGPU | 0.1 | 2.2, 0.3 | 23.5 | 39.1 | 589.5 | 899.0 | 36.8 | 1,811 | fair to fair |
| verdict-e17-fp32 | all | 0.2 | 3.1, 0.4 | 23.3 | 41.6 | 625.0 | 690.2 | 39.2 | 1,759 | fair to fair |
| laya-m18-fp16 | cpuOnly | 0.9 | 3.4, 0.0 | 135.6 | 409.7 | 2,330.6 | 6,106.0 | 145.7 | 9,948 | fair to fair |
| laya-m18-fp16 | cpuAndGPU | 0.9 | 3.2, 0.1 | 41.7 | 83.7 | 577.1 | 1,459.8 | 36.1 | 2,856 | nominal to nominal |
| laya-e17-fp16 | cpuAndGPU | 0.2 | 3.2, 0.5 | 41.8 | 87.5 | 579.8 | 1,557.8 | 36.2 | 2,888 | fair to fair |
| laya-e17-fp16 | all | 0.2 | 181.8, 0.1 | 74.4 | 214.0 | 1,240.7 | 4,181.0 | 77.5 | 861 | fair to nominal |

| Package | Units | Batch 1 median ms at 128, 256, 512, 1024 tokens | Batch 16 median ms at 128, 256, 512, 1024 | Function loads s | Footprint after first call MB |
|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 14.2, 26.3, 50.8, none | 142.4, 295.2, 796.8, none | b16_s128 0.8, b16_s256 0.8, b16_s512 0.8, b1_s128 0.0, b1_s256 0.8, b1_s512 0.8 | 93 to 102 |
| verdict-m18-fp16 | cpuAndGPU | 7.5, 14.2, 20.3, none | 68.9, 137.0, 308.1, none | b16_s128 1.1, b16_s256 1.2, b16_s512 0.9, b1_s128 0.1, b1_s256 1.0, b1_s512 1.1 | 391 to 944 |
| verdict-m18-fp16 | cpuAndNeuralEngine | 4.2, 10.8, 30.8, none | 79.6, 184.1, 559.3, none | b16_s128 9.0, b16_s256 28.6, b16_s512 24.6, b1_s128 0.2, b1_s256 5.4, b1_s512 6.0 | 75 to 108 |
| verdict-e17-fp16 | cpuAndGPU | 9.2, 10.4, 16.3, none | 46.3, 92.3, 193.9, none | main 0.2 | 372 to 1,016 |
| verdict-e17-fp16 | all | 8.7, 11.1, 20.9, none | 65.2, 143.4, 348.0, none | main 0.3 | 378 to 1,004 |
| verdict-e17-fp32 | cpuOnly | 33.2, 60.3, 143.9, none | 326.4, 826.1, 2,117.4, none | main 0.0 | 93 to 99 |
| verdict-e17-fp32 | cpuAndGPU | 12.0, 15.5, 34.7, none | 106.4, 282.4, 650.7, none | main 0.3 | 482 to 1,660 |
| verdict-e17-fp32 | all | 12.4, 16.0, 33.4, none | 123.5, 279.0, 652.0, none | main 0.4 | 482 to 1,709 |
| laya-m18-fp16 | cpuOnly | 36.9, 69.7, 141.2, 354.2 | 380.6, 869.5, 2,300.4, 5,874.6 | b16_s1024 2.2, b16_s128 1.6, b16_s256 1.6, b16_s512 1.8, b1_s1024 1.6, b1_s128 0.0, b1_s256 2.4, b1_s512 2.7 | 85 to 9,176 |
| laya-m18-fp16 | cpuAndGPU | 14.7, 23.4, 42.0, 82.1 | 138.1, 278.1, 574.1, 1,397.2 | b16_s1024 1.8, b16_s128 1.8, b16_s256 1.9, b16_s512 1.9, b1_s1024 2.0, b1_s128 0.1, b1_s256 1.9, b1_s512 2.0 | 430 to 2,345 |
| laya-e17-fp16 | cpuAndGPU | 14.9, 23.4, 42.0, 83.8 | 138.3, 278.2, 578.7, 1,392.9 | main 0.4 | 458 to 2,650 |
| laya-e17-fp16 | all | 12.4, 28.1, 78.3, 207.7 | 205.3, 490.8, 1,237.3, 4,176.6 | main 185.4 | 325 to 633 |

| Package | Units | Max abs probability difference (batch 1, 16) | Mean (batch 1) | Max abs logit difference | Top answers kept (batch 1, 16) | Non-finite | Tokenization | Planned cost by device |
|---|---|---|---|---|---|---|---|---|
| verdict-m18-fp16 | cpuOnly | 1.06e-02, 1.04e-02 | 1.8e-03 | 0.195 | 197, 197 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-m18-fp16 | cpuAndGPU | 1.39e-03, 1.17e-03 | 2.2e-04 | 0.026 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-m18-fp16 | cpuAndNeuralEngine | 7.61e-03, 7.61e-03 | 9.9e-04 | 0.219 | 198, 198 of 200 | 0, 0 | 200/200 | ANE 75%, CPU 25% |
| verdict-e17-fp16 | cpuAndGPU | 1.39e-03, 1.17e-03 | 2.2e-04 | 0.026 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-e17-fp16 | all | 1.39e-03, 1.17e-03 | 2.2e-04 | 0.026 | 199, 199 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-e17-fp32 | cpuOnly | 1.72e-06, 1.72e-06 | 2.7e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | CPU 100% |
| verdict-e17-fp32 | cpuAndGPU | 1.92e-06, 2.19e-06 | 2.7e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| verdict-e17-fp32 | all | 1.92e-06, 2.19e-06 | 2.7e-07 | 2.6e-05 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| laya-m18-fp16 | cpuOnly | 1.36e-02, 1.36e-02 | 1.0e-03 | 0.107 | 200, 200 of 200 | 0, 0 | 200/200 | CPU 100% |
| laya-m18-fp16 | cpuAndGPU | 3.95e-03, 3.95e-03 | 2.0e-04 | 0.029 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| laya-e17-fp16 | cpuAndGPU | 3.95e-03, 3.95e-03 | 2.0e-04 | 0.029 | 200, 200 of 200 | 0, 0 | 200/200 | GPU 100% |
| laya-e17-fp16 | all | 1.27e-02, 1.27e-02 | 6.3e-04 | 0.079 | 200, 200 of 200 | 0, 0 | 200/200 | ANE 85%, GPU 15% |

Failures:

verdict-e17-fp16 cpuOnly: exit status 139
    verdict-e17-fp16 cpuOnly: tokenized 200 questions, 200 match the reference
    verdict-e17-fp16 cpuOnly: compiled in 0.129 s, loads 1.106 s and 0.001 s
verdict-e17-fp16 cpuAndNeuralEngine: exit status 133
    verdict-e17-fp16 cpuAndNeuralEngine: tokenized 200 questions, 200 match the reference
    verdict-e17-fp16 cpuAndNeuralEngine: compiled in 0.131 s, loads 98.042 s and 0.075 s
verdict-e17-fp32 cpuAndNeuralEngine: exit status 133
    verdict-e17-fp32 cpuAndNeuralEngine: tokenized 200 questions, 200 match the reference
    verdict-e17-fp32 cpuAndNeuralEngine: compiled in 0.121 s, loads 2.256 s and 0.036 s
verdict-m18-fp16 all: exit status 1
    verdict-m18-fp16 all: compiled in 0.507 s, loads 2.132 s and 0.208 s
    verdict-m18-fp16 all: batch 1 median 15.995 ms
    encoder-harness: verdict-m18-fp16 all: Error Domain=com.apple.CoreML Code=0 "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." UserInfo={NSLocalizedDescription=`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.}
laya-m18-fp16 all: exit status 1
    laya-m18-fp16 all: tokenized 200 questions, 200 match the reference
    encoder-harness: laya-m18-fp16 all: Error Domain=com.apple.CoreML Code=0 "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." UserInfo={NSLocalizedDescription=`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.}
laya-m18-fp16 cpuAndNeuralEngine: exit status 1
    laya-m18-fp16 cpuAndNeuralEngine: tokenized 200 questions, 200 match the reference
    encoder-harness: laya-m18-fp16 cpuAndNeuralEngine: Error Domain=com.apple.CoreML Code=0 "`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program." UserInfo={NSLocalizedDescription=`MLModelConfiguration`'s `.functionName` property must be `nil` unless the model type is ML Program.}
laya-e17-fp16 cpuOnly: exit status 139
    laya-e17-fp16 cpuOnly: tokenized 200 questions, 200 match the reference
    laya-e17-fp16 cpuOnly: compiled in 0.164 s, loads 3.165 s and 0.002 s
laya-e17-fp16 cpuAndNeuralEngine: exit status 133
    laya-e17-fp16 cpuAndNeuralEngine: tokenized 200 questions, 200 match the reference
    laya-e17-fp16 cpuAndNeuralEngine: compiled in 0.162 s, loads 192.477 s and 0.081 s
