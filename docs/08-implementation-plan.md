# Implementation plan

Eight milestones, in dependency order. Each milestone has a tracking issue that lists its
issues as sub-issues. The issue index at the bottom is generated from GitHub after the issues were
created and links every issue.

## Ordering rationale

1. **Foundations first (milestone 0).** Order-preserving JSON, wire types and the fixture
   tooling are used by every later test. The package skeleton fixes the Swift tools version,
   platforms and dependency pins that the model code will build against. Two spikes here retire
   cheap unknowns early: Python-compatible canonical JSON, and whether hosted CI can run MLX.
2. **The engine core without a model (milestone 1).** Upstream's algorithm is mostly text and
   integer manipulation. Porting it first, against fixtures, means the model work later has a
   correct harness to plug into, and that most of the project can be developed and reviewed on
   any Mac or Linux box.
3. **The model (milestone 2).** Starts with three spikes (tokenizer parity, chat template parity,
   backend validation with the fork and mlx-vlm as oracles), then the port in the order the data
   flows: configuration, blocks, encoder, decoder read, weights, multi-step, runtime actor,
   download, parity tests, warmup, performance baseline.
4. **The server (milestone 3).** Only needs the core and a stub backend to be complete and
   testable; it is ordered after the model so that live tests can run as soon as it lands, but it
   can be developed in parallel with milestone 2 by a second engineer.
5. **Read extensions and images (milestone 4).** `steps`, `samples` and `sequential` are engine
   plumbing over the read pass; images need the Gemma 4 vision tower and processor.
6. **Generation (milestone 5).** The diffusion generation loop, then `think`, then chat.
7. **Other models (milestone 6).** JevK5 first (no new model code), then Verdict and Laya after
   the Core ML versus MLX spike, then a CLM decision.
8. **Quality and release (milestone 7).** JevBench harness, calibration report, benchmarks,
   DocC, release 0.1.0, upstream tracking.

## Dependency graph (milestones)

```
M0 Foundations
 └─► M1 Engine core ──► M2 DiffusionGemma reads ──► M4 Extensions and images ──► M5 Generation
                   └──► M3 Server (needs only M1 + a stub backend; live tests need M2)
                                                          M6 Other models (needs M1, M3; MLXLLM for JevK5)
                                                          M7 Quality and release (needs M2, M3)
```

## Recommended first issue

The package skeleton and dependency pins (milestone 0), because every other issue creates files
inside the targets it defines, and because resolving `mlx-swift-lm` at a pinned commit with Swift
tools 6.2 and strict concurrency on both macOS and Linux is the first thing that can fail.
Immediately after it, in parallel: the order-preserving JSON model and the fixture generation
tooling.

## Parallel tracks

Once milestone 0 is done, three tracks can proceed independently:

- Track A (any machine): milestone 1, then the server (milestone 3) against the stub backend.
- Track B (Apple silicon with 32 GB or more and the weights): milestone 2 spikes, then the port.
- Track C (any machine, later): JevK5 prompt and readout plumbing; encoder spike preparation.

## Sizing

Rough effort at a senior Swift engineer's pace, excluding review:

| Milestone | Estimate |
|---|---|
| 0 Foundations | 1.5 weeks |
| 1 Engine core | 2 weeks |
| 2 DiffusionGemma reads | 4 to 6 weeks (dominated by parity debugging) |
| 3 Server | 2 weeks |
| 4 Extensions and images | 3 weeks (images are two thirds of it) |
| 5 Generation | 3 to 4 weeks |
| 6 Other models | 3 to 5 weeks depending on the Core ML spike |
| 7 Quality and release | 2 weeks |

## Issue index

See the GitHub milestones: https://github.com/Algorythm-Canada/OpenJevSwift/milestones.
The table below is maintained by hand when issues are added or closed.

<!-- ISSUE_INDEX_START -->
(generated after issue creation)
<!-- ISSUE_INDEX_END -->
