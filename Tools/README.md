# Tools

Scripts that support development and testing. They are not part of the Swift package, and SwiftPM
does not build anything in this directory. The testing strategy they serve is described in
[docs/09-conformance-and-testing.md](../docs/09-conformance-and-testing.md).

| Directory | Language | What it holds |
|---|---|---|
| `fixtures/` | Python | The scripts that generate [Fixtures/](../Fixtures/README.md), run together by `make fixtures` from a virtual environment at `Tools/fixtures/.venv` (`make fixtures-venv`, packages pinned in `requirements.txt`). `upstream_tables.py` drives upstream OpenJev at the pinned commit with the real DiffusionGemma tokenizer: it imports `openjev.engine` and `openjev.api` directly and stubs the model read the way upstream's `test_api.py` does (issue #6). `wire_tables.py` records upstream's HTTP contract into `Fixtures/wire` with a stand-in tokenizer (issue #5). `python_json_tables.py` writes the CPython JSON reference tables in `Fixtures/python-json` (issue #3), `json.loads` errors included (issue #35). The script that records model parity data from mlx-vlm through upstream's `MlxRuntime.read` will join them with issue #31. |
| `fixtures/checkpoint_tables.py` | Python | Copies the checkpoint's `config.json`, `generation_config.json` and `model.safetensors.index.json` at the pinned revision into [Fixtures/model](../Fixtures/model/README.md), with their digests, for the configuration tests (issue #23) and the weight coverage test (issue #27). It downloads no weights and involves no upstream code, so its files pin `model_repo` and `model_revision` instead of the upstream commit. `make fixtures` runs it after `upstream_tables.py`. |
| `encoders/` | Python and Swift | Spike #56: `reference.py` writes [Fixtures/encoders](../Fixtures/encoders/README.md) from upstream's own Verdict and Laya read paths, `convert_*.py` convert both models to Core ML, and `Harness/` (a separate Swift package) with `HarnessApp.swiftpm` measures them on macOS and an iPhone. `function_capacity.py` and the harness's `encoder-capacity` measured what each loaded Core ML function costs a Mac and how often each number of functions kept loads one again (D-042). Nothing in it is built by the main package; [encoders/README.md](encoders/README.md) explains the order to run them. |
| `jevbench/` | Python | Issues #61 and #62: `harness.py` runs JevBench v1's 231 public items, and SemIf's 102-row TypeSafe subset, against any `/v1/systemone` server through the benchmark's own adapter, scores them with the benchmarks' own code (vendored unchanged under `vendor/`) and compares two runs item by item; `servers.py` starts the Swift or upstream server for one backend and records its versions; `report.py` renders the comparison tables of [docs/quality.md](../docs/quality.md) and, for issue #62, `calibration.py` its calibration tables (ECE, Brier score, reliability bins, `confidence` against accuracy, SemIf's temperature scaling fitted offline). Standard library only; upstream's server runs from `.venv` (`requirements-upstream.txt`). The datasets are downloaded at pinned revisions into a cache outside the repository and checked by SHA-256; `results/` holds one file per run. `smoke_test.py` runs in CI. [jevbench/README.md](jevbench/README.md) explains the order to run them. |
| `oracle/` | Python and Swift | Spike #22 (D-014): `fetch_checkpoint.py` downloads the DiffusionGemma checkpoint, `Probe/` and `UpstreamProbe/` (packages of their own) read the oracle's 27 reads through the Layr-Labs fork and a Swift transliteration of mlx-vlm, and `sensitivity.py`, `crossfeed.py`, `stage_dump.py`, `tolerance_stats.py` and `summarize_runs.py` measure them, into `results/`; [docs/spikes/backend-validation.md](../docs/spikes/backend-validation.md) explains them. `item_reads.py` with `UpstreamProbe`'s `ItemReads` reads any JevBench or TypeSafe item through upstream's engine and the port's and compares the two read by read, bit for bit (issue #62, [below](#reading-benchmark-items-against-upstream)). |
| `bench/results/` | JSON | Issue #32: one file per machine and day, `<date>-<machine>.json`, each run of `openjev-bench` (the executable target in `Sources/openjev-bench`) appended by `--json`, with the machine, macOS and AC power recorded. [docs/benchmarks.md](../docs/benchmarks.md) holds the tables from them and [docs/development.md](../docs/development.md#the-read-benchmark) the commands. |
| `sdk-compat/` | Python, JavaScript and Swift | The SDK compatibility suite (issue #39): `run.py` starts `openjev-stub-server` three times and runs TypeSafe's Python SDK (`typesafe-sdk` 0.7.2, locked in `requirements.txt` for `.venv`) and TypeScript SDK (`@typesafe-ai/sdk` 0.6.0, locked in `typescript/package-lock.json`) against them through a recording proxy: decoded answers, the listing, error classes with request ids, retries and `retry-after`, the 422 message and a routed model. `swift/`, a package of its own, runs NSStudent's JevSwiftSDK the same way. `make sdk-compat-venv` and `make sdk-compat`; [sdk-compat/README.md](sdk-compat/README.md). The CI job `SDK compatibility` runs it. |
| `docs/` | Shell and HTML | Issue #64: `build-site.sh` builds the DocC catalogs of the four library modules into one static site for GitHub Pages, every DocC warning an error, with `index.html` as its front page, and `serve-site.sh` serves it locally under `/OpenJevSwift/`. `make docs` and `make docs-preview`; the Documentation workflow (`.github/workflows/docs.yml`) runs the first and publishes the result. [docs/development.md](../docs/development.md#api-documentation) explains the catalogs and their links. |

## Reading benchmark items against upstream

`Tools/oracle/item_reads.py` takes D-014's exact tier, where the port must give mlx-vlm's reads bit
for bit, from the oracle's 27 reads to any JevBench or TypeSafe item. `bodies` rebuilds the request
bodies as `Tools/jevbench/harness.py` sends them and checks them against `body_sha256` in the
DiffusionGemma result files; `upstream` records every read of upstream's own `MlxEngine.decide` on
them, and with `--variants` replays those reads through `sensitivity.py`'s variants; the `ItemReads`
target records the port's `DecisionEngine` and model on the same bodies and replays upstream's
reads, once per configuration of kernels and RoPE table; `compare` holds each configuration to
upstream read by read and prefill layer by layer, and `--summary` writes the result without
TypeSafe's text or prompt ids. The work files go to `~/Library/Caches/OpenJevSwift/item-reads`.
Each model process loads about 16 GB, so run them one after the other. The four long-prompt items
of PR #105, from the repository root with `Tools/jevbench/.venv` made as
[docs/quality.md](../docs/quality.md#rerunning-every-table) says:

```bash
make upstream
python3 Tools/jevbench/harness.py fetch
PY=Tools/jevbench/.venv/bin/python
WHEEL_METALLIB="$PWD/Tools/jevbench/.venv/lib/python3.12/site-packages/mlx/lib/mlx.metallib"
$PY Tools/oracle/item_reads.py bodies jevbench:hard-opus-c-long_policy-04 jevbench:hard-sol-b-long_policy-06 jevbench:hard-opus-a-long_policy-19 typesafe102:e2e58201a90c11192f70edbf
$PY Tools/oracle/item_reads.py upstream --variants chunked_prefill
swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads --metallib "$WHEEL_METALLIB" --oracle-rope
swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads
swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads --metallib "$WHEEL_METALLIB"
swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads --oracle-rope
$PY Tools/oracle/item_reads.py compare --summary Tools/oracle/results/item_reads_long_flips.json
$PY Tools/oracle/item_reads.py long-slots
```

`ItemReads` without `--metallib` loads the `default.metallib` that its own build ships, which is
the Swift server's: the same mlx-swift 0.32.2 sources compiled by the same toolchain (SHA-256
`282550b0…` with Xcode 27.0, as in a release build of `openjev`). Each result file records the
library's SHA-256, and `compare` names a configuration by it. `long-slots` needs no model: it
computes D-014's long-prompt figure over the oracle fixture's few-label slots from committed files.

## Rules

- Pin every upstream input (commit, package version, tokenizer revision) in the script, and write
  the pins into each output file so that the fixtures can be checked against
  [THIRD_PARTY.md](../THIRD_PARTY.md).
- Read model weights and tokenizer files from a local cache outside the repository. Never commit
  them.
- Swift code does not go here. Anything the package builds lives under `Sources/` or `Tests/`.
  The exceptions are packages of their own that the main package never builds: a spike's
  measurement harness, as `encoders/` keeps, and the SDK suite's JevSwiftSDK driver,
  `sdk-compat/swift`.
