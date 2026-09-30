# Tools

Scripts that support development and testing. They are not part of the Swift package, and SwiftPM
does not build anything in this directory. The testing strategy they serve is described in
[docs/09-conformance-and-testing.md](../docs/09-conformance-and-testing.md).

| Directory | Language | What it holds |
|---|---|---|
| `fixtures/` | Python | The scripts that generate [Fixtures/](../Fixtures/README.md) from upstream OpenJev at the pinned commit, with the real DiffusionGemma tokenizer. They import `openjev.engine` and `openjev.api` directly and stub the model read the way upstream's `test_api.py` does. This directory also holds the script that records model parity data from mlx-vlm on the 4-bit checkpoint through upstream's `MlxRuntime.read`. Added by issue #6. |
| `sdk-compat/` | Python and TypeScript | Smoke tests that run the official TypeSafe SDKs (`typesafe-sdk` and `@typesafe-ai/sdk`) against a Swift server started with the stub backend. They check decoded answers, error mapping and request-id headers. Added by issue #39. |

## Rules

- Pin every upstream input (commit, package version, tokenizer revision) in the script, and write
  the pins into each output file so that the fixtures can be checked against
  [THIRD_PARTY.md](../THIRD_PARTY.md).
- Read model weights and tokenizer files from a local cache outside the repository. Never commit
  them.
- Swift code does not go here. Anything the package builds lives under `Sources/` or `Tests/`.
