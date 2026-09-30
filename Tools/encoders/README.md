# Encoder model tools (spike #56)

Scripts and a Swift harness that answer spike #56: should Verdict and Laya, upstream's two
ModernBERT encoder models, run on Core ML or on an MLX port, on iOS and macOS? The findings are in
[docs/spikes/encoder-runtime.md](../../docs/spikes/encoder-runtime.md). Nothing here is built by
the main package, and no weights, converted packages or tokenizer files are committed.

| Path | What it is |
|---|---|
| `requirements.txt` | The complete lock for `.venv` (CPython 3.12) |
| `common.py` | Pins (upstream commit, checkpoint revisions), paths and helpers shared by the scripts |
| `reference.py` | Writes [Fixtures/encoders](../../Fixtures/encoders/README.md): the 200-question corpus and both models' PyTorch reference outputs, read through upstream's own engine code |
| `coreml_common.py` | The ModernBERT wrapper that converts cleanly, the conversion helpers and the Core ML runner used for parity |
| `convert_verdict.py`, `convert_laya.py` | Convert each model to Core ML packages in `~/Library/Caches/OpenJevSwift/encoders` and check them against the reference from Python; reports in `docs/spikes/encoder-runtime/` |
| `Harness/` | A Swift package: swift-transformers tokenization, the prompt and sequence builders, calibration, a Core ML runner and the measurement loop; a macOS command (`encoder-harness`) and XCTest parity tests for macOS and the iOS Simulator |
| `HarnessApp.swiftpm/` | An iPhone app around the harness, because Xcode cannot host a package's test bundle on a device |
| `stage_harness.sh` | Copies fixtures, tokenizers and packages into the app for a device run |
| `run_macos.sh`, `run_ios.sh` | Measure every package with every compute-unit setting, one process each, on the Mac or a connected iPhone |
| `summarize.py` | Turns the measurement results into the report's tables |

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
swift test --package-path Tools/encoders/Harness
Tools/encoders/run_macos.sh
DEVELOPMENT_TEAM=<team id> Tools/encoders/run_ios.sh <device id> verdict-m18-fp16:cpuAndNeuralEngine
```

`reference.py` downloads both checkpoints (about 1.5 GB) into the Hugging Face cache and takes
about 37 minutes on an M3 Max, most of it in the float16 and bfloat16 passes. The two converters take about five and fifteen minutes and write
about 1.2 GB and 1.7 GB of packages. The harness tests skip when the tokenizers or fixtures are
missing; the measurement tests also need `ENCODER_HARNESS_BENCHMARK=1` and converted packages.

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
