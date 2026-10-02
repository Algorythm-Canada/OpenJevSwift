# Regression fixtures

The Swift port's own answers for a fixed request set, so a refactor that changes what the port
computes fails a test (issue #31). **This is the port's output, not upstream's.** The oracle that
says whether the port agrees with upstream is [Fixtures/oracle](../oracle/README.md), compared
under D-014; this file only says whether the port still agrees with itself.

## reads.json

Written by `RegressionTests` in
[Tests/OpenJevDiffusionGemmaTests/Runtime/RegressionTests.swift](../../Tests/OpenJevDiffusionGemmaTests/Runtime/RegressionTests.swift)
on the pinned 4-bit checkpoint, under mlx-swift's own kernels (D-014's native tier).

- `generator` holds the pins: the test that wrote it, the checkpoint repository and revision, the
  mlx-swift version Package.resolved pins, the macOS version, the GPU's name and the day it was
  recorded.
- `entries` lists 29 requests:
  - the 27 reads of [Fixtures/oracle/reads.json](../oracle/README.md), each sent to
    `DiffusionGemmaRuntime.read` with the oracle's prompt ids, canvas, slots and steps;
  - `engine/quickstart`, the wire quickstart of [Fixtures/wire/cases.json](../wire/README.md),
    and `engine/readme`, upstream's README example ("Everything is down and we have a demo with
    our biggest client at noon." under the urgent, team and tone questions), each through
    `DecisionEngine` with the default configuration.
- Each entry holds every read it made (`steps`, `promptTokens`, and per slot the label
  `probabilities`, the top-k `entropy`, the `topLabel` index and its `topLabelID`). An engine
  request's reads are sorted by seed, steps and canvas, and it also records `inputTokens` and each
  answer's probabilities in request order (a noul's yes probability, a choice's or a score's
  distribution).

## The comparison

When the pins match (the same checkpoint revision, mlx-swift version, macOS version and GPU), every
probability, entropy and answer probability must equal the recorded one exactly, the top labels and
prompt tokens too: five runs of the suite on the M3 Max reproduced every value bit for bit, so the
tolerance is 0 (D-044). On another machine the kernels round differently, so the test holds the
port to D-014's aggregate bounds instead (mean label probability difference at most 0.02, top label
on at least 90% of slots) and says to record that machine's own file.

## Recording

From Xcode, run `RegressionTests` with `OPENJEV_RECORD_REGRESSION=1` in the test action's
environment (a scheme of your own, so the setting is not shared), or from a shell:

```bash
OPENJEV_RECORD_REGRESSION=1 swift test --filter RegressionTests
```

The checkpoint comes from `OPENJEV_TEST_MODEL`, else the Hugging Face cache snapshot, as for every
live test. Record again only when a change is meant to move the answers, and say why in the commit
message; the diff of this file shows what moved.
