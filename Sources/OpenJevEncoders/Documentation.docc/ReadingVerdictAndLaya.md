# Reading Verdict and Laya

Load an encoder on a Mac or an iPhone, from downloaded or local packages, and answer requests
with it.

## Overview

Each backend loads from an ``EncoderPackageStore``, which finds the model's files or downloads
them, and is answered through an ``/OpenJevCore/EncoderDecisionEngine``:

```swift
import OpenJevCore
import OpenJevEncoders

let store = try EncoderPackageStore(environment: [:])
let backend = try await LayaBackend.load(from: store)
let engine = EncoderDecisionEngine(backend: backend)
try await engine.warmUp()
let decision = try await engine.decide(request)
```

`VerdictBackend.load(from:functionCapacity:)` loads Verdict the same way.

### Packages and downloads

The weights are float16 Core ML conversions of the authors' checkpoints, published as releases of
[Algorythm-Canada/openjev-models](https://github.com/Algorythm-Canada/openjev-models); the
tokenizers and calibration files come from the checkpoints on Hugging Face at pinned revisions.
The store downloads each file on first use into `Application Support/OpenJevSwift/encoders`,
excluded from backups, and moves it into place only when its size and SHA-256 match the manifest
the library embeds (``EncoderPackageManifest/verdict``, ``EncoderPackageManifest/laya`` and
``EncoderPackageManifest/layaByLength``). Later launches download only what is missing. The
package is compiled once, and the compiled copy is kept beside it.

| Model | Package | Download |
|---|---|---|
| Verdict | `verdict-m18-fp16`, one function per shape (batch 1 and 16 by 128, 256 and 512 tokens) | 306 MB, with about 4 MB of tokenizer and calibrator |
| Laya on a Mac | `laya-m18-fp16`, one function per shape (batch 1 and 16 by 128 to 1,024 tokens) | about 850 MB |
| Laya on an iPhone | `laya-f18-b1s128-fp16` to `laya-f18-b1s1024-fp16`, one per sequence length | 843 to 845 MB each, when the app asks |

An `OPENJEV_ENCODER_MODELS` entry in the environment dictionary names a folder of packages
converted with the repository's `Tools/encoders`, which the store then uses without downloading;
the server reads it from the process environment.

### On a Mac

Both backends run on the GPU and read up to 16 questions per Core ML call, in the smallest
function that holds the batch and its longest question. A function loads when a read first needs
its shape and stays loaded, holding its own copy of the weights: Verdict keeps up to its 6
functions and Laya its 8, which costs 1.6 to 2.8 GB for Verdict and 4.7 to 8.9 GB for Laya.
`functionCapacity` caps the functions kept, at the price of loading one again when a request needs
it; the server sets it from `OPENJEV_ENCODER_FUNCTIONS`. On an M3 Max, Verdict reads one question
in 7.5 to 20.3 ms, depending on its length.

### On an iPhone

Both backends run on the Neural Engine, one question per call. Verdict loads as on a Mac and keeps
one function loaded. Laya cannot load its multifunction package for the Neural Engine, so it uses
one package per sequence length, each kept loaded once a read has needed it:
``LayaBackend/load(from:packageSet:functionCapacity:)``
fetches only the tokenizer and the configuration file, and the app downloads the packages it
wants with ``LayaBackend/prefetch(lengths:)``, at least the 128-token one before the warm-up read.
A question whose sequence is longer than every package the device holds throws
``EncoderLoadError/noPackage(length:package:held:)``, which names the package to fetch. The first
load of a Laya package took 33 to 56 s on an A15.

On an iPhone 13 Pro Max, Verdict read a question in 9.7 ms at 128 tokens, 17 ms at 256 and 46 ms at
512, and Laya in 27.9 ms at 128, 137 ms at 512 and 513 ms at 1,024. When a device cannot meet an
app's time budget, or does not hold the package a question needs, the app can send that request
to an OpenJev server, which answers the same wire API with the same models.

### What the backends refuse

Like upstream's encoder engines, the engine refuses `images`, `steps` or `samples` above 1,
`think` and `sequential` before any read. Verdict takes at most 24 options per choice and Laya
255, though Laya's 256-token budget for a question and its options makes about 20 the practical
limit: a question whose options do not fit is refused with upstream's message. Both truncate a
long state rather than refusing it, Verdict at 512 tokens and Laya at 1,024.

## See Also

- <doc:/OpenJevCore/GettingStarted>
