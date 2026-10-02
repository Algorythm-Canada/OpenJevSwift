# Reading with DiffusionGemma

Load the checkpoint, put the runtime behind a decision engine, and keep its memory in bounds.

## Overview

``DiffusionGemmaRuntime/load(_:configuration:cache:token:resolver:progress:)`` resolves a
checkpoint, loads its tokenizer and weights, applies the memory settings and runs a warm-up read.
The runtime it returns is the backend of a ``/OpenJevCore/DecisionEngine``:

```swift
import OpenJevCore
import OpenJevDiffusionGemma

let runtime = try await DiffusionGemmaRuntime.load(.fourBit)
let engine = try DecisionEngine(backend: runtime)
let decision = try await engine.decide(request)
```

The runtime is an actor and runs every MLX evaluation inside itself, one at a time, as upstream's
one-worker executor does. Concurrent decisions queue on it, so concurrency buys prefill reuse and
overlapping CPU work, not GPU parallelism. On an M3 Max a release build answered a three-question
request in about 0.3 s, and 1 to 16 concurrent callers shared 3.0 to 3.4 requests per second; the
repository's `docs/benchmarks.md` has the measurements.

### Where the weights come from

A ``ModelSource`` is a local directory or a Hugging Face repository at a revision:

| Source | Checkpoint |
|---|---|
| ``ModelSource/fourBit`` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407`, the checkpoint every fixture and parity test was made with |
| ``ModelSource/eightBit`` | the 8-bit conversion, pinned; no fixture covers it |
| ``ModelSource/bf16`` | the bfloat16 conversion, pinned; no fixture covers it |
| ``ModelSource/directory(_:)`` | a folder with `config.json`, the shard index, the shards and the tokenizer files, opened without network access |
| ``ModelSource/hub(repository:revision:)`` | any repository at a commit, a branch or a tag |

``ModelSource/init(setting:)`` reads the `OPENJEV_MLX_MODEL` setting's forms: a path, a
repository, or `repository@revision`. A preset's repository without a revision takes the preset's
pinned commit.

A Hub source is downloaded by ``ModelResolver`` into the cache a ``HubCacheLocation`` names, in
huggingface_hub's layout, so upstream OpenJev, mlx-vlm and this module share one copy. Each file
is resumed after an interruption, checked by SHA-256 or git blob digest, and linked from the
snapshot; when the Hub cannot be reached, a complete snapshot of a pinned commit is used offline.
``HubCacheLocation/init(environment:)`` follows `HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME`, and
``HubCacheLocation/token(environment:)`` reads `HF_TOKEN`, from an environment dictionary the
caller passes: the module never reads the process environment.

### Settings and memory

``DiffusionGemmaRuntime/Configuration`` holds upstream's settings:

- ``DiffusionGemmaRuntime/Configuration/maxPromptTokens``, 32,768: a longer prompt is refused
  with upstream's message before anything runs.
- ``DiffusionGemmaRuntime/Configuration/promptCacheEntries``, 12, and
  ``DiffusionGemmaRuntime/Configuration/promptCacheTokens``, 16,384: the prefills kept for
  reuse, bounded by both. A read of a cached prompt skips its prefill.
- ``DiffusionGemmaRuntime/Configuration/cacheLimitGB``: MLX's buffer pool limit. `nil` leaves MLX
  alone, and 0 disables the pool.
- ``DiffusionGemmaRuntime/Configuration/warmUp``: one small read at load, so the first request
  does not pay for compiling the kernels.

Loaded, the 4-bit model holds 14.35 GiB of MLX memory; in service with short prompts MLX holds
about 17.3 GiB, and cached long prompts can add about 3.6 GB more.
``DiffusionGemmaRuntime/memoryReport()`` gives MLX's active, cached and peak bytes and the
process's resident size, ``DiffusionGemmaRuntime/statistics()`` the reads, the prefill cache's
hits and the time in the model, and ``DiffusionGemmaRuntime/removeCachedPrefills()`` empties the
prefill cache.

### What the runtime does not do yet

The runtime's ``DiffusionGemmaRuntime/capabilities`` turn `think` and images off, so the engine
answers `openjev-0.1 does not support think` and `openjev-0.1 does not support images` before
any read. Images arrive with issues #46 to #48, and text generation, `think` and
`POST /v1/chat/completions` with issues #50 to #53.

## See Also

- <doc:/OpenJevCore/GettingStarted>
