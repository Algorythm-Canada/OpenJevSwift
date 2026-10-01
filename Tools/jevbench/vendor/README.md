# Vendored benchmark code

Unchanged copies of the files the harness scores with, so that a run, the smoke test and CI use
each benchmark's own code without a download. `harness.py` pins every file's SHA-256 (`VENDORED`),
refuses to run when one differs, and `python3 Tools/jevbench/harness.py fetch` also downloads each
file at its pinned commit and requires the same bytes. Never edit these files; to move a pin,
copy the files from the new commit and update `VENDORED`, the commit constants and
[THIRD_PARTY.md](../../../THIRD_PARTY.md) together.

| Folder | Source | Files | License |
|---|---|---|---|
| `jevbench/` | [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) at `bb05a33` (2026-09-29) | `LICENSE`; `jevbench/__init__.py`, `tasks.py` (the item format and its validation), `scoring.py` (validation of a distribution, the 1e-3 and 2e-2 sum tolerances, argmax with the smallest label on a tie), `metrics.py` (Brier score, ECE over 10 equal-width bins of top-label confidence, latency percentiles), `summarize.py` (`metric` and `summarize`, which the published results come from); `jevbench/adapters/base.py` and `typesafe.py` (the `/v1/systemone` request and the reading of its answer) | MIT, Copyright (c) 2026 Florian Standhartinger and contributors |
| `semif/` | [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) at `23cf1f3` (2026-09-24) | `LICENSE`; `build_typesafe.py` (rebuilds the 102 TypeSafe rows from the case snapshots and checks their hashes), `evaluate_external.py` (`type_safe`: equal-case modal agreement and total variation) | MIT, Copyright (c) 2026 TheoLeeCJ |

`jevbench/jevbench/adapters/__init__.py` is the one file written for this project: JevBench's own
imports every adapter it ships, most of which need packages the harness does not install, and this
one imports nothing. The adapters' copyright notices are in each folder's `LICENSE`.
