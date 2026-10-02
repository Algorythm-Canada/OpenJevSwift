# ``OpenJevDiffusionGemma``

DiffusionGemma 26B-A4B on MLX: the model, its download and the runtime that reads canvases for
the decision engine.

## Overview

This module ports mlx-vlm 0.6.15's DiffusionGemma to Swift on mlx-swift and mlx-swift-lm
primitives, and upstream OpenJev's MLX runtime as one actor, ``DiffusionGemmaRuntime``, which
adopts ``/OpenJevCore/DecisionBackend``. A read is the model's read-only decision pass: one prefill
of the prompt, then one or more denoise passes over a canvas of noise and answer slots, without
generating text. ``/OpenJevCore/DecisionEngine`` builds the prompts and canvases and turns the reads
into answers.

It needs an Apple silicon Mac. The pinned 4-bit checkpoint is 16.58 GB on disk and takes about
16 GB of memory to load; a Mac with 32 GB or more is recommended. The module compiles for iOS,
which CI checks, but no iPhone holds the model.

The port matches mlx-vlm bit for bit on all 63 recorded oracle reads when it runs on the oracle's
Metal library, and stays within the bounds of decisions D-014 and D-048 on mlx-swift's own kernels,
which is how it runs in production. `steps`, `samples` and `sequential` work; images
(issues #46 to #48) and `think` with text generation (issues #50 to #53) do not yet, and the engine
refuses them with upstream's messages.

<doc:ReadingWithDiffusionGemma> loads the model and covers its settings, memory and downloads.

## Topics

### Essentials

- <doc:ReadingWithDiffusionGemma>
- ``DiffusionGemmaRuntime``
- ``ModelSource``
- ``HubCacheLocation``

### Loading

- ``ModelResolver``
- ``ModelResolverError``
- ``TokenizerFiles``
- ``TokenizerFilesError``
- ``SwiftTransformersTokenizer``
- ``LoadMetrics``
- ``DiffusionGemmaRuntimeError``

### The prefill cache

- ``PrefillCache``
- ``PrefillKey``
- ``PrefillCacheDefaults``

### The model

- ``DiffusionGemmaModel``
- ``DiffusionGemmaConfiguration``
- ``DiffusionGemmaTextConfiguration``
- ``DiffusionGemmaQuantization``
- ``DiffusionGemmaGenerationConfiguration``
- ``DiffusionGemmaConfigurationError``
- ``WeightLoadingError``

### Reads

- ``PromptCache``
- ``SlotRequest``
- ``ReadOutput``
- ``ReadInputError``
- ``TensorDigest``

### Blocks

- ``Backbone``
- ``DecoderModel``
- ``EncoderModel``
- ``EncoderLanguageModel``
- ``EncoderLayerScalar``
- ``DecoderLayer``
- ``Attention``
- ``DenseMLP``
- ``Router``
- ``Experts``
- ``SelfConditioning``
- ``LayerCache``
- ``StageObserver``
- ``makeSoftcap(_:)``

### Version

- ``openJevDiffusionGemmaVersion``
