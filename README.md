# OpenJevSwift

[![CI](https://github.com/Algorythm-Canada/OpenJevSwift/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Algorythm-Canada/OpenJevSwift/actions/workflows/ci.yml) [![Fixtures](https://github.com/Algorythm-Canada/OpenJevSwift/actions/workflows/fixtures.yml/badge.svg?branch=main)](https://github.com/Algorythm-Canada/OpenJevSwift/actions/workflows/fixtures.yml)

A native Swift implementation of [OpenJev](https://github.com/razorback16/openjev), the open,
Jev-compatible "System One" decision server. Send it a state and typed questions (`noul`,
`choice`, `score`); it returns a probability distribution and a confidence for every answer,
read directly from a model's token probabilities. No text is generated, so an answer cannot
go off-schema.

**Status: research and planning.** There is no implementation yet, on purpose. This
repository currently holds the research findings, the proposed architecture, the recorded
decisions, the risk register and a complete set of GitHub issues that decompose the
implementation into independently workable units. See
[docs/08-implementation-plan.md](docs/08-implementation-plan.md) for where to start.

OpenJevSwift is an independent project. It is not affiliated with or endorsed by TypeSafe AI
(the makers of Jev), by Google DeepMind or NVIDIA (DiffusionGemma), or by the authors of the
other models it plans to serve. Jev, TypeSafe, Gemma and other names are the property of their
respective owners.

## What it will be

| Piece | What it does | Runs on |
|---|---|---|
| `OpenJevCore` | The backend-agnostic decision engine ported from upstream: question schema, prompt and answer templates, single-token label slots, seeded canvases, slot distributions, confidence, re-read and sequential policies, wire types. Pure Swift, no model. | macOS, iOS, Linux |
| `OpenJevDiffusionGemma` | DiffusionGemma 26B-A4B on MLX Swift: model port, quantized weight loading, a read-only decision pass, a prefill cache, and later the generation loop for `think` and chat. | Apple silicon Macs with 32 GB or more unified memory |
| `OpenJevServer` and the `openjev` CLI | A Hummingbird server speaking OpenJev's exact wire API (`POST /v1/systemone`, `GET /v1/models`, `POST /v1/chat/completions`), so TypeSafe's own SDKs work against it unchanged. | macOS |
| Additional models (later) | Verdict, Laya, JevK5 and CLM behind the same API, as upstream does. Small encoder models are the path to iOS. | macOS, iOS for the small models |

## Documentation

| Document | Contents |
|---|---|
| [docs/01-upstream-openjev.md](docs/01-upstream-openjev.md) | How the upstream Python implementation works, read from its source, not its README |
| [docs/02-jev-wire-api.md](docs/02-jev-wire-api.md) | The wire contract to be compatible with: shapes, limits, errors, headers, SDK behaviour |
| [docs/03-diffusiongemma.md](docs/03-diffusiongemma.md) | The model: architecture, checkpoint layout, tokens, what a "read" is numerically |
| [docs/04-swift-inference-landscape.md](docs/04-swift-inference-landscape.md) | Evaluation of MLX Swift, the Layr-Labs fork, swift-transformers, Core ML, llama.cpp, server frameworks |
| [docs/05-architecture.md](docs/05-architecture.md) | Proposed package layout, public API, engine and runtime design, platform support |
| [docs/06-decisions.md](docs/06-decisions.md) | Decision records with the alternatives that were rejected and why |
| [docs/07-risks-and-unknowns.md](docs/07-risks-and-unknowns.md) | Risk register with mitigations and the spike issues that resolve each unknown |
| [docs/08-implementation-plan.md](docs/08-implementation-plan.md) | Milestones, ordering, dependency graph and the issue index |
| [docs/09-conformance-and-testing.md](docs/09-conformance-and-testing.md) | Golden fixtures from upstream, parity tolerances, live tests, SDK compatibility tests, CI |
| [docs/10-other-models.md](docs/10-other-models.md) | Verdict, Laya, JevK5 and CLM: formats, calibration, Swift feasibility |
| [docs/deployment.md](docs/deployment.md) | Running `openjev serve` on a Mac: build, settings, launchd, logs, shutdown, exit statuses |

## Compatibility target

The compatibility target is upstream OpenJev at commit
[`dcd2094`](https://github.com/razorback16/openjev/commit/dcd2094) (version 0.5.0, 2026-09-29),
which in turn implements TypeSafe's published OpenAPI 0.2.0 for `/v1/systemone`. Two things
are pinned deliberately:

- The wire API, including the error contract, headers and model aliases (`jev-latest`,
  `jev-preview`, `openjev-latest`, `openjev-0.1`).
- The prompt, answer template, label and canvas construction, byte for byte, so that answers can
  be compared with the Python implementation on the same weights.

## License

Apache-2.0, the same as upstream OpenJev. Code ported from MIT-licensed projects (mlx-vlm,
mlx-swift-lm and its Layr-Labs fork) keeps its attribution; see [THIRD_PARTY.md](THIRD_PARTY.md).
Model weights are not part of this repository and keep their own licenses and terms.
