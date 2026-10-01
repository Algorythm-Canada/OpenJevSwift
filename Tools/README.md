# Tools

Scripts that support development and testing. They are not part of the Swift package, and SwiftPM
does not build anything in this directory. The testing strategy they serve is described in
[docs/09-conformance-and-testing.md](../docs/09-conformance-and-testing.md).

| Directory | Language | What it holds |
|---|---|---|
| `fixtures/` | Python | The scripts that generate [Fixtures/](../Fixtures/README.md), run together by `make fixtures` from a virtual environment at `Tools/fixtures/.venv` (`make fixtures-venv`, packages pinned in `requirements.txt`). `upstream_tables.py` drives upstream OpenJev at the pinned commit with the real DiffusionGemma tokenizer: it imports `openjev.engine` and `openjev.api` directly and stubs the model read the way upstream's `test_api.py` does (issue #6). `wire_tables.py` records upstream's HTTP contract into `Fixtures/wire` with a stand-in tokenizer (issue #5). `python_json_tables.py` writes the CPython JSON reference tables in `Fixtures/python-json` (issue #3), `json.loads` errors included (issue #35). The script that records model parity data from mlx-vlm through upstream's `MlxRuntime.read` will join them with issue #31. |
| `fixtures/checkpoint_tables.py` | Python | Copies the checkpoint's `config.json`, `generation_config.json` and `model.safetensors.index.json` at the pinned revision into [Fixtures/model](../Fixtures/model/README.md), with their digests, for the configuration tests (issue #23) and the weight coverage test (issue #27). It downloads no weights and involves no upstream code, so its files pin `model_repo` and `model_revision` instead of the upstream commit. `make fixtures` runs it after `upstream_tables.py`. |
| `encoders/` | Python and Swift | Spike #56: `reference.py` writes [Fixtures/encoders](../Fixtures/encoders/README.md) from upstream's own Verdict and Laya read paths, `convert_*.py` convert both models to Core ML, and `Harness/` (a separate Swift package) with `HarnessApp.swiftpm` measures them on macOS and an iPhone. Nothing in it is built by the main package; [encoders/README.md](encoders/README.md) explains the order to run them. |
| `jevbench/` | Python | Issue #61: `harness.py` runs JevBench v1's 231 public items, and SemIf's 102-row TypeSafe subset, against any `/v1/systemone` server through the benchmark's own adapter, scores them with the benchmarks' own code (vendored unchanged under `vendor/`) and compares two runs item by item; `servers.py` starts the Swift or upstream server for one backend and records its versions; `report.py` renders the tables of [docs/quality.md](../docs/quality.md). Standard library only; upstream's server runs from `.venv` (`requirements-upstream.txt`). The datasets are downloaded at pinned revisions into a cache outside the repository and checked by SHA-256; `results/` holds one file per run. `smoke_test.py` runs in CI. [jevbench/README.md](jevbench/README.md) explains the order to run them. |
| `sdk-compat/` | Python and TypeScript | Smoke tests that run the official TypeSafe SDKs (`typesafe-sdk` and `@typesafe-ai/sdk`) against a Swift server started with the stub backend. They check decoded answers, error mapping and request-id headers. Added by issue #39. |

## Rules

- Pin every upstream input (commit, package version, tokenizer revision) in the script, and write
  the pins into each output file so that the fixtures can be checked against
  [THIRD_PARTY.md](../THIRD_PARTY.md).
- Read model weights and tokenizer files from a local cache outside the repository. Never commit
  them.
- Swift code does not go here. Anything the package builds lives under `Sources/` or `Tests/`.
  The one exception is a spike's measurement harness kept as its own package, as `encoders/` does.
