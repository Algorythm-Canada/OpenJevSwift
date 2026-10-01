# Encoder model tools (spike #56)

Scripts and a Swift harness that answer spike #56: should Verdict and Laya, upstream's two
ModernBERT encoder models, run on Core ML or on an MLX port, on iOS and macOS? The findings are in
[docs/spikes/encoder-runtime.md](../../docs/spikes/encoder-runtime.md). The converters and
`manifest.py` also produce and describe the packages that `OpenJevEncoders` downloads. Nothing here
is built by the main package, and no weights, converted packages or tokenizer files are committed.

| Path | What it is |
|---|---|
| `requirements.txt` | The complete lock for `.venv` (CPython 3.12) |
| `common.py` | Pins (upstream commit, checkpoint revisions), paths and helpers shared by the scripts |
| `reference.py` | Writes [Fixtures/encoders](../../Fixtures/encoders/README.md): the 200-question corpus and both models' PyTorch reference outputs, read through upstream's own engine code |
| `coreml_common.py` | The ModernBERT wrapper that converts cleanly, the conversion helpers and the Core ML runner used for parity |
| `convert_verdict.py`, `convert_laya.py` | Convert each model to Core ML packages in `~/Library/Caches/OpenJevSwift/encoders` and check them against the reference from Python; reports in `docs/spikes/encoder-runtime/` |
| `laya_ane_plan.py` | Prints where Core ML plans each part of Laya under the Neural Engine setting: the bisect that found why the unmodified model ran on the CPU |
| `Harness/` | A Swift package: swift-transformers tokenization, the prompt and sequence builders, calibration, a Core ML runner and the measurement loop; a macOS command (`encoder-harness`) and XCTest parity tests for macOS and the iOS Simulator |
| `HarnessApp.swiftpm/` | An iPhone app around the harness, because Xcode cannot host a package's test bundle on a device |
| `stage_harness.sh` | Copies fixtures, tokenizers and packages into the app for a device run |
| `run_macos.sh`, `run_ios.sh` | Measure every package with every compute-unit setting, one process each, on the Mac or a connected iPhone |
| `summarize.py` | Turns the measurement results into the report's tables |
| `manifest.py` | Writes the manifest of Verdict's package that `OpenJevEncoders` embeds (every file's URL, size and SHA-256) and prints the commands that publish the package as a GitHub release (D-033). Standard library only |

## Running it

From the repository root, once:

```bash
make upstream
/usr/local/bin/python3.12 -m venv Tools/encoders/.venv
Tools/encoders/.venv/bin/python -m pip install -r Tools/encoders/requirements.txt
```

Then, in order:

```bash
Tools/encoders/.venv/bin/python Tools/encoders/reference.py
Tools/encoders/.venv/bin/python Tools/encoders/convert_verdict.py
Tools/encoders/.venv/bin/python Tools/encoders/convert_laya.py
Tools/encoders/.venv/bin/python Tools/encoders/convert_laya.py --only laya-f18-b1s128-fp16,laya-f18-b1s256-fp16,laya-f18-b1s512-fp16,laya-f18-b1s1024-fp16
swift test --package-path Tools/encoders/Harness
Tools/encoders/run_macos.sh
DEVELOPMENT_TEAM=<team id> SETTLE=300 Tools/encoders/run_ios.sh <device id> verdict-m18-fp16:cpuAndNeuralEngine laya-f18-b1s128-fp16:cpuAndNeuralEngine
```

`reference.py` downloads both checkpoints (about 1.5 GB) into the Hugging Face cache and takes
about 37 minutes on an M3 Max, most of it in the float16 and bfloat16 passes. The two converters
take about five and fifteen minutes and write about 1.2 GB and 1.7 GB of packages. The
`--only` line writes Laya's one-shape packages, one per sequence length, 845 MB each: on an iPhone
they are the only Laya packages that run on the Neural Engine. The harness tests skip when the
tokenizers or fixtures are missing; the measurement tests also need `ENCODER_HARNESS_BENCHMARK=1`
and converted packages.

## Publishing a package

`OpenJevEncoders` downloads Verdict's package on first use and checks every file against the
manifest it embeds, `Sources/OpenJevEncoders/Store/EncoderPackageManifest+Verdict.swift` (D-033).
Until the package is published, the manifest keeps downloads off (`PACKAGE_DOWNLOADS_ENABLED` in
`manifest.py`). After converting a new package, from the repository root:

```bash
python3 Tools/encoders/manifest.py
```

It rewrites the manifest from `verdict-m18-fp16.mlpackage` (in `OPENJEV_ENCODER_MODELS` or
`~/Library/Caches/OpenJevSwift/encoders`) and the checkpoint's tokenizer and calibrator (in the
Hugging Face cache), and prints the commands that create `Algorythm-Canada/openjev-models` once
(with the Apache-2.0 license as its first commit, since a release needs a commit to tag), copy the
three package files under their asset names and upload them to a release. It uploads nothing itself;
`--gh gh` prints `gh` instead of `ghp`, and `--check` fails when the committed manifest no longer
matches the package. Every conversion writes new identifiers into the package's Manifest.json, so a
package converted on another machine never matches the published digests: publish from the machine
whose package the manifest describes. A new package needs a new tag (`--tag`), and a published asset
is never replaced. Once the uploaded assets match the manifest, set
`PACKAGE_DOWNLOADS_ENABLED = True`, run the script again and commit the manifest.

## Known problems

- numpy is pinned to 2.3: numpy 2.4 made `int()` of a one-element array an error, and coremltools
  9.0 still does that during conversion.
- On macOS 27.0.1, Core ML's CPU backend traps or returns NaN on a float16 package with
  enumerated input shapes, and a package with enumerated shapes also traps under
  `cpuAndNeuralEngine`. The multifunction packages (one fixed shape per function) avoid it. The
  converters therefore check the enumerated float16 packages from Python only on the GPU.
- coremltools' Python binding crashed once when releasing an input after a multifunction
  prediction; the runner keeps its inputs alive, and the CPU and Neural Engine settings of the
  multifunction packages are measured from Swift instead.
- Laya's multifunction package does not load wherever the Neural Engine is allowed, on macOS
  27.0.1 and iOS 27.0: Core ML reports that `functionName` must be `nil`. The one-shape packages
  load and run there.
- `run_ios.sh` sees the app's output only when a launch ends, so the crash or jetsam report it
  attaches to a failure is the newest since the launch began and can belong to an earlier
  configuration; compare its time with the result files.
