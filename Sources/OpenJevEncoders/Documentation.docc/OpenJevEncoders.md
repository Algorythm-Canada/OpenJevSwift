# ``OpenJevEncoders``

Verdict and Laya, the encoder models upstream OpenJev serves, on Core ML for iOS and macOS.

## Overview

Two of the models upstream puts behind Jev's API are encoders that score a question's options in
one forward pass: Verdict (`verdict-1.4`, Heman10x's ModernBERT-base and GLiClass model, 151M
parameters) and Laya (`laya-1.0`, Nandakishor M and Convai Innovations' ModernBERT-large model,
421M parameters). This module runs both on Core ML, from float16 conversions of their
checkpoints, behind ``/OpenJevCore/QuestionReadBackend``, so ``/OpenJevCore/EncoderDecisionEngine``
answers requests with them in an app or in the server.

``VerdictBackend`` and ``LayaBackend`` port upstream's prompts, sequences, truncation and
calibration, which match upstream byte for byte and, for the calibration, within 1e-6 or bit for
bit. What is left is float16 rounding inside the network: on 666 JevBench and TypeSafe items the
Swift server gave upstream's top answer on every item, with probabilities at most 0.0026 apart.

The Core ML types need macOS 15 or iOS 18, because the multifunction packages do; the loaders
build for the package's macOS 14 and iOS 17 floors and throw
``EncoderLoadError/unsupportedOperatingSystem(_:)`` on an older system. Core ML does not exist on
Linux, so neither does this module.

<doc:ReadingVerdictAndLaya> loads either model, on a Mac or on an iPhone.

## Topics

### Essentials

- <doc:ReadingVerdictAndLaya>
- ``VerdictBackend``
- ``LayaBackend``
- ``EncoderPackageStore``

### Packages and downloads

- ``EncoderPackageManifest``
- ``EncoderPackageLocations``
- ``EncoderTokenizerLocations``
- ``EncoderPackageError``
- ``LayaPackageSet``

### Core ML

- ``EncoderModelRunner``
- ``CoreMLEncoderModel``
- ``CoreMLPackagesByLength``
- ``CompiledEncoderModel``
- ``EncoderPackageSpec``
- ``EncoderComputeUnits``
- ``EncoderModelError``
- ``EncoderLoadError``

### Verdict's prompt and calibration

- ``VerdictPrompt``
- ``VerdictTokenizing``
- ``VerdictTokenizer``
- ``VerdictCalibration``

### Laya's prompt, sequence and calibration

- ``LayaPrompt``
- ``LayaSequence``
- ``LayaTokenizing``
- ``LayaTokenizer``
- ``LayaSpecialTokens``
- ``LayaCalibration``

### Version

- ``openJevEncodersVersion``
