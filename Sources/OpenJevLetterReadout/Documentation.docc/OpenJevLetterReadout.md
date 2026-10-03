# ``OpenJevLetterReadout``

JevK5, the letter-readout model upstream OpenJev serves as `jevk5-0.2`, on MLX for macOS and iOS.

## Overview

JevK5 is Alibi Serikbay's model (github.com/allebee/jevk5, Apache-2.0): Qwen3.5-4B with a LoRA
distilled from Qwen3.6-27B, merged into the weights. It answers a typed decision without
generating anything. Each question's options are lettered `A` to `P` in a fixed JSON prompt, and
the answer is a softmax over the letters' next-token logits under one calibration temperature,
SemIf's readout. Upstream reads the logits from a vLLM server; this module runs the model in the
process, through mlx-swift-lm's Qwen3.5 text model, behind ``/OpenJevCore/QuestionReadBackend``,
so ``/OpenJevCore/EncoderDecisionEngine`` answers requests with it in an app or in the server.

``JevK5Prompt`` and ``JevK5Option`` write upstream's prompt byte for byte, and
``JevK5Readout`` reads more than 16 options in several passes as the `jevk5` package does, to
the last bit on the same logits. ``JevK5Backend`` reads the questions of a request concurrently,
one pass at a time on its actor, bills every pass and refuses a prompt over 16,383 tokens with
upstream's 400.

The model is a conversion of the checkpoint to MLX that `Tools/jevk5/convert.py` writes and
``JevK5Checkpoint`` pins. ``JevK5ModelFiles`` finds it in a folder or downloads it from the
Hugging Face Hub at a pinned revision, through ``/OpenJevDiffusionGemma/ModelResolver``. By
default ``JevK5Backend/load(_:cache:token:cacheLimitGB:resolver:)`` takes
``JevK5Checkpoint/platformDefault``: the 8-bit conversion on macOS, which gives the author's
published top answer on 230 of JevBench's 231 items where the 4-bit one gives it on 209, and the
4-bit one on iOS, half the size.

```swift
let backend = try await JevK5Backend.load(
    .directory(URL(fileURLWithPath: "/models/jevk5-0.2-mlx-8bit")), cacheLimitGB: 4)
let engine = EncoderDecisionEngine(backend: backend)
let decision = try await engine.decide(request)
```

MLX needs Apple silicon, and the module exists only where OpenJevDiffusionGemma does.

## Topics

### Essentials

- ``JevK5Backend``
- ``JevK5Checkpoint``
- ``JevK5ModelFiles``

### Prompt and readout

- ``JevK5Prompt``
- ``JevK5Option``
- ``JevK5Readout``
- ``JevK5PromptLimit``
- ``JevK5Calibration``

### The model on MLX

- ``LetterReadoutModel``
- ``LetterReadoutTokenizing``
- ``Qwen35LetterReadoutModel``
- ``JevK5Tokenizer``

### Errors

- ``JevK5LoadError``
- ``JevK5ModelError``

### Version

- ``openJevLetterReadoutVersion``
