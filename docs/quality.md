# Answer quality against upstream

How the Swift server's answers compare with upstream OpenJev's on public benchmarks, and both with
the results the benchmarks publish (issue #61). `Tools/jevbench` ran JevBench v1's 231 public items
and the 102 TypeSafe public-evaluation rows that SemIf compares with Jev against both servers, on
one Mac, with both encoder models this port serves: Verdict (`verdict-1.4`) and Laya (`laya-1.0`).
The DiffusionGemma run waits for the `mlx` backend's parity tests (issue #31); its commands are
under [DiffusionGemma](#diffusiongemma), and it is the one piece of issue #61 left. How the harness
maps items onto requests and scores them is in
[Tools/jevbench/README.md](../Tools/jevbench/README.md) and D-041.

All runs: 2026-10-01, an Apple M3 Max with 128 GB and macOS 27.0.1. The Swift server is the release
build of this branch, whose Swift package is main's at `93edceb`, serving the float16 Core ML
packages on the GPU (D-011); upstream is razorback16/openjev at `dcd2094` serving the PyTorch
checkpoints in float32 on the CPU, as it does on any machine without CUDA: the arithmetic of the
reference in [Fixtures/encoders](../Fixtures/encoders/README.md), on PyTorch's default of 12
threads where the reference pins 8, which moves only the last bits. Every request asked one
question, one request at a time.

## What the runs show

- **The Swift server gives upstream's top answer on every item.** For both models and both
  datasets, 666 items in all: no top answer differs, no item is right on one server and wrong on
  the other, accuracy is the same, and the Brier score and ECE differ by at most 0.0003, overall
  and per tier.
- **The probabilities differ by float16, and stay well inside the parity bound.** The largest
  difference is 0.0026 (Laya) and 0.0022 (Verdict) and the mean about 0.0002 to 0.0003, against
  D-034's and D-037's bound of 0.02 largest and 0.003 mean. Of the 36 items whose top two answers
  upstream puts less than 0.01 apart, every one kept its top answer.
- **Verdict reproduces the benchmark's published Verdict 1.4 row,** on both servers: the same
  outcome on 226 of 231 items, every choice and score among them. The five that differ are nouls,
  whose prompt the benchmark's own adapter writes differently. The published Laya row is another
  checkpoint and agrees on 85%.
- **Both encoders are far from Jev on TypeSafe's evaluations.** On SemIf's subset Verdict agrees
  with the reference answer 0.53 of the time and Laya 0.43 to 0.44, against Jev's published
  0.88.

## The tables

Rendered by `python3 Tools/jevbench/harness.py report` from the result files in
`Tools/jevbench/results/` and the pinned datasets. In every comparison upstream's run is the
reference. "items" counts the questions asked; nothing was skipped or refused. The latency columns
are the caller's time and the `model` part of the server's `server-timing` header, in milliseconds.

### The runs

| dataset | model | server | items | answered | skipped | refused, failed or invalid | accuracy | Brier | ECE | p50 ms | p95 ms | model p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jevbench | laya-1.0 | swift | 231 | 231 | 0 | 0 | 53.7% | 0.5090 | 0.0769 | 23.4 | 428.2 | 21.1 |
| jevbench | laya-1.0 | upstream | 231 | 231 | 0 | 0 | 53.7% | 0.5090 | 0.0769 | 170.5 | 758.0 | 167.8 |
| jevbench | verdict-1.4 | swift | 231 | 231 | 0 | 0 | 56.3% | 0.5381 | 0.0788 | 18.9 | 95.4 | 18.2 |
| jevbench | verdict-1.4 | upstream | 231 | 231 | 0 | 0 | 56.3% | 0.5382 | 0.0788 | 77.3 | 188.8 | 68.0 |
| typesafe102 | laya-1.0 | swift | 102 | 102 | 0 | 0 | 47.1% | 0.5866 | 0.0727 | 133.4 | 518.2 | 132.4 |
| typesafe102 | laya-1.0 | upstream | 102 | 102 | 0 | 0 | 47.1% | 0.5865 | 0.0727 | 677.0 | 1200.3 | 669.6 |
| typesafe102 | verdict-1.4 | swift | 102 | 102 | 0 | 0 | 52.9% | 0.5408 | 0.1015 | 62.1 | 204.2 | 61.2 |
| typesafe102 | verdict-1.4 | upstream | 102 | 102 | 0 | 0 | 52.9% | 0.5409 | 0.1014 | 152.7 | 1312.0 | 150.2 |

### Swift against upstream

| model | dataset | items | top answer agrees | identical answers | mean abs diff | largest abs diff | right only on Swift, only upstream | McNemar p |
|---|---|---|---|---|---|---|---|---|
| laya-1.0 | jevbench | 231 | 231 of 231 | 10 | 2.1e-04 | 0.0026 | 0 and 0 | 1 |
| verdict-1.4 | jevbench | 231 | 231 of 231 | 0 | 2.8e-04 | 0.0022 | 0 and 0 | 1 |
| laya-1.0 | typesafe102 | 102 | 102 of 102 | 16 | 1.6e-04 | 0.0012 | 0 and 0 | 1 |
| verdict-1.4 | typesafe102 | 102 | 102 of 102 | 0 | 2.9e-04 | 0.0020 | 0 and 0 | 1 |

| model | dataset | type | items | top answer agrees | mean abs diff | largest abs diff | largest at (label) |
|---|---|---|---|---|---|---|---|
| laya-1.0 | jevbench | choice | 139 | 139 of 139 | 1.7e-04 | 0.0019 | original-routing-03-1 (coding_agent) |
| laya-1.0 | jevbench | noul | 74 | 74 of 74 | 3.6e-04 | 0.0026 | easy-fact-11 (yes) |
| laya-1.0 | jevbench | score | 18 | 18 of 18 | 2.5e-04 | 0.0014 | original-ordinal-06-1 (2) |
| verdict-1.4 | jevbench | choice | 139 | 139 of 139 | 2.6e-04 | 0.0022 | hard-opus-b-tradeoff-12 (refund_now) |
| verdict-1.4 | jevbench | noul | 74 | 74 of 74 | 3.4e-04 | 0.0010 | original-policy-06-1 (yes) |
| verdict-1.4 | jevbench | score | 18 | 18 of 18 | 2.8e-04 | 0.0015 | hard-opus-b-multi_hop-05 (0) |
| laya-1.0 | typesafe102 | choice | 36 | 36 of 36 | 1.8e-04 | 0.0012 | f892cb9fd5fc7590f4b6525b (card_declined) |
| laya-1.0 | typesafe102 | noul | 66 | 66 of 66 | 1.4e-04 | 4.0e-04 | 71a2393072e9e2b2ac08da0c (yes) |
| verdict-1.4 | typesafe102 | choice | 36 | 36 of 36 | 3.0e-04 | 0.0020 | d5943297a4a36b3aefe79e7b (not_stated) |
| verdict-1.4 | typesafe102 | noul | 66 | 66 of 66 | 2.9e-04 | 8.5e-04 | 42a77bfee1a2fa1e4209799f (yes) |

The 5 largest deviations of each model and dataset:

| model | dataset | item | type | label | Swift | upstream | abs diff |
|---|---|---|---|---|---|---|---|
| laya-1.0 | jevbench | easy-fact-11 | noul | yes | 0.1758 | 0.1732 | 0.0026 |
| laya-1.0 | jevbench | original-routing-03-1 | choice | coding_agent | 0.5170 | 0.5151 | 0.0019 |
| laya-1.0 | jevbench | hard-sol-c-judge_hard-03 | noul | yes | 0.6803 | 0.6787 | 0.0016 |
| laya-1.0 | jevbench | original-adequacy-01-1 | noul | no | 0.5082 | 0.5096 | 0.0014 |
| laya-1.0 | jevbench | original-ordinal-06-1 | score | 2 | 0.5275 | 0.5261 | 0.0014 |
| verdict-1.4 | jevbench | hard-opus-b-tradeoff-12 | choice | refund_now | 0.4047 | 0.4025 | 0.0022 |
| verdict-1.4 | jevbench | easy-intent-02 | choice | change_address | 0.7631 | 0.7649 | 0.0018 |
| verdict-1.4 | jevbench | hard-opus-a-temporal_numeric-07 | choice | sep_26 | 0.1769 | 0.1752 | 0.0018 |
| verdict-1.4 | jevbench | hard-opus-b-multi_hop-05 | score | 0 | 0.4221 | 0.4236 | 0.0015 |
| verdict-1.4 | jevbench | hard-opus-b-multi_hop-07 | choice | credit_full_month | 0.2238 | 0.2223 | 0.0014 |
| laya-1.0 | typesafe102 | f892cb9fd5fc7590f4b6525b | choice | card_declined | 0.1152 | 0.1140 | 0.0012 |
| laya-1.0 | typesafe102 | 371be0b5bafdcd596293a496 | choice | money_back | 0.2360 | 0.2350 | 0.0010 |
| laya-1.0 | typesafe102 | 6b6f1cdebb551365aeea7b30 | choice | refund_request | 0.1343 | 0.1352 | 9.1e-04 |
| laya-1.0 | typesafe102 | 1c3247fbed2ed8ec6e7a037e | choice | more | 0.3303 | 0.3312 | 8.7e-04 |
| laya-1.0 | typesafe102 | 4f1c8783235a838e7ebad206 | choice | unsure | 0.2132 | 0.2125 | 7.0e-04 |
| verdict-1.4 | typesafe102 | d5943297a4a36b3aefe79e7b | choice | not_stated | 0.5568 | 0.5587 | 0.0020 |
| verdict-1.4 | typesafe102 | 4f1c8783235a838e7ebad206 | choice | less | 0.4464 | 0.4448 | 0.0016 |
| verdict-1.4 | typesafe102 | 0bcfbba5435af97f4c7f7ded | choice | tool_or_service | 0.3231 | 0.3247 | 0.0016 |
| verdict-1.4 | typesafe102 | 3a5f273468b2614da64aafd5 | choice | tool_or_service | 0.3231 | 0.3247 | 0.0016 |
| verdict-1.4 | typesafe102 | 59f2f955c1c757549fd8f0b5 | choice | same | 0.3468 | 0.3455 | 0.0013 |

The items whose upstream top two are less than 0.01 apart, where the parity bound of D-034 and D-037
allows a changed top answer (`harness.py compare` lists each):

| model | dataset | near ties | top answer kept | closest (top-two margin) | disagreements |
|---|---|---|---|---|---|
| laya-1.0 | jevbench | 6 | 6 of 6 | hard-opus-c-long_policy-08 (5.0e-04 upstream, 7.0e-04 Swift) | none |
| verdict-1.4 | jevbench | 6 | 6 of 6 | hard-sol-a-multi_hop-12 (4.3e-04 upstream, 8.6e-04 Swift) | none |
| laya-1.0 | typesafe102 | 11 | 11 of 11 | 692da1b417be96e6e96f0389 (2.0e-04 upstream, 0.0000 Swift) | none |
| verdict-1.4 | typesafe102 | 13 | 13 of 13 | 767af54164feae1d84407b1e (0.0019 upstream, 0.0023 Swift) | none |

### Against JevBench's published rows

| model | server | published row | public items | ours | published | same outcome | right only here | right only there | McNemar p | published sealed |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-1.0 | swift | laya | 231 | 53.7% | 58.4% | 84.8% | 12 | 23 | 0.0895 | 30.8% |
| laya-1.0 | upstream | laya | 231 | 53.7% | 58.4% | 84.8% | 12 | 23 | 0.0895 | 30.8% |
| verdict-1.4 | swift | openjev-verdict-1.4 | 231 | 56.3% | 57.6% | 97.8% | 1 | 4 | 0.375 | 27.9% |
| verdict-1.4 | upstream | openjev-verdict-1.4 | 231 | 56.3% | 57.6% | 97.8% | 1 | 4 | 0.375 | 27.9% |

| model | server | tier | items | ours | published |
|---|---|---|---|---|---|
| laya-1.0 | swift | easy | 48 | 97.9% | 95.8% |
| laya-1.0 | swift | standard | 72 | 65.3% | 69.4% |
| laya-1.0 | swift | hard | 111 | 27.0% | 35.1% |
| laya-1.0 | upstream | easy | 48 | 97.9% | 95.8% |
| laya-1.0 | upstream | standard | 72 | 65.3% | 69.4% |
| laya-1.0 | upstream | hard | 111 | 27.0% | 35.1% |
| verdict-1.4 | swift | easy | 48 | 89.6% | 87.5% |
| verdict-1.4 | swift | standard | 72 | 69.4% | 69.4% |
| verdict-1.4 | swift | hard | 111 | 33.3% | 36.9% |
| verdict-1.4 | upstream | easy | 48 | 89.6% | 87.5% |
| verdict-1.4 | upstream | standard | 72 | 69.4% | 69.4% |
| verdict-1.4 | upstream | hard | 111 | 33.3% | 36.9% |

Items whose outcome differs from the published row's (upstream's run; Swift's is the same):

| model | type | items | which |
|---|---|---|---|
| laya-1.0 | choice | 21 | 21 items |
| laya-1.0 | noul | 11 | 11 items |
| laya-1.0 | score | 3 | original-ordinal-01-1, original-ordinal-06-0, hard-opus-b-multi_hop-05 |
| verdict-1.4 | noul | 5 | easy-fact-11, hard-opus-a-long_policy-13, hard-opus-a-long_policy-19, hard-opus-a-temporal_numeric-06, hard-sol-a-trap-15 |

### The TypeSafe subset with SemIf's metrics

| model | server | rows | cases | modal agreement (equal-case) | total variation (equal-case) | accuracy (pooled) | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| laya-1.0 | swift | 102 of 102 | 20 | 0.426 | 0.500 | 47.1% | 0.5866 | 0.0727 |
| laya-1.0 | upstream | 102 of 102 | 20 | 0.436 | 0.500 | 47.1% | 0.5865 | 0.0727 |
| verdict-1.4 | swift | 102 of 102 | 20 | 0.532 | 0.466 | 52.9% | 0.5408 | 0.1015 |
| verdict-1.4 | upstream | 102 of 102 | 20 | 0.532 | 0.466 | 52.9% | 0.5409 | 0.1014 |

The answers TypeSafe's snapshots publish, over the same rows:

| model | rows | cases | modal agreement (equal-case) | total variation (equal-case) |
|---|---|---|---|---|
| typesafe (typesafe:v13_snowy_elephant) | 102 | 20 | 0.883 | 0.127 |
| opus (anthropic:claude-opus-5) | 102 | 20 | 0.912 | 0.101 |
| sol (openai:gpt-5.6-sol) | 101 | 20 | 0.906 | 0.103 |

### JevBench by tier

| model | server | tier | items | accuracy | Brier | ECE | ordinal MAE |
|---|---|---|---|---|---|---|---|
| laya-1.0 | swift | easy | 48 | 97.9% | 0.1213 | 0.2297 | n/a |
| laya-1.0 | swift | standard | 72 | 65.3% | 0.4588 | 0.1764 | 0.512 |
| laya-1.0 | swift | hard | 111 | 27.0% | 0.7091 | 0.1706 | 0.692 |
| laya-1.0 | upstream | easy | 48 | 97.9% | 0.1212 | 0.2296 | n/a |
| laya-1.0 | upstream | standard | 72 | 65.3% | 0.4590 | 0.1765 | 0.512 |
| laya-1.0 | upstream | hard | 111 | 27.0% | 0.7092 | 0.1707 | 0.692 |
| verdict-1.4 | swift | easy | 48 | 89.6% | 0.1509 | 0.1166 | n/a |
| verdict-1.4 | swift | standard | 72 | 69.4% | 0.5108 | 0.2204 | 0.598 |
| verdict-1.4 | swift | hard | 111 | 33.3% | 0.7232 | 0.1614 | 0.662 |
| verdict-1.4 | upstream | easy | 48 | 89.6% | 0.1509 | 0.1166 | n/a |
| verdict-1.4 | upstream | standard | 72 | 69.4% | 0.5107 | 0.2204 | 0.598 |
| verdict-1.4 | upstream | hard | 111 | 33.3% | 0.7234 | 0.1614 | 0.662 |

### The Swift server's Core ML function loads

Requests that had to load a Core ML function on the Swift server, from a simulation of its
two-function cache over each server's requests, and the median model time of the other requests by
input shape:

| model | requests | first loads | loaded again | load ms | other requests' median ms by shape |
|---|---|---|---|---|---|
| verdict-1.4 | 333 | 3 | 0 | 684 to 764 | 128: 14, 256: 15, 512: 48 |
| laya-1.0 | 333 | 4 | 15 | 453 to 1815 | 128: 18, 256: 29, 512: 50, 1024: 133 |

### Machines and versions

| datasets | model | server | code | runtime | machine | run on |
|---|---|---|---|---|---|---|
| jevbench, typesafe102 | laya-1.0 | swift | OpenJevSwift 0.1.0-dev at 93edceb | Core ML, float16 multifunction package, .cpuAndGPU, up to 16 questions per call | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | laya-1.0 | upstream | openjev 0.5.0 at dcd2094, Python 3.12.2 | PyTorch 2.13.0 on the cpu, float32, 12 threads | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | verdict-1.4 | swift | OpenJevSwift 0.1.0-dev at 93edceb | Core ML, float16 multifunction package, .cpuAndGPU, up to 16 questions per call | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | verdict-1.4 | upstream | openjev 0.5.0 at dcd2094, Python 3.12.2 | PyTorch 2.13.0 on the cpu, float32, 12 threads | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |


## What the numbers mean

### A disagreement between the servers

The two servers do not run the same arithmetic. Upstream reads each encoder with PyTorch in
float32 on the CPU; the Swift server reads a Core ML package whose weights and activations are
float16, on the GPU, then applies the same prompt, truncation and calibration in Swift, which the
encoder tests check byte for byte and, for the calibration, within 1e-6 or bit for bit (D-034,
D-037). What is left is float16 rounding inside the network. Spike #56 measured it on 200
reference questions: Verdict's package stays within 0.0014 of PyTorch float32 on the Mac's GPU
(D-034) and Laya's within 0.0039 before its 4-decimal rounding (D-037), and both backends are held
to the spike's bound: the largest difference at most 0.02, the mean at most 0.003, and the top
answer unchanged wherever the reference's top two are at least 0.01 apart.

These runs are a second, larger sample of the same difference: 666 items, with JevBench's long
hard-tier states and TypeSafe's long documents filling Verdict's 512 tokens and Laya's 1,024. The
largest differences (Verdict 0.0022, Laya 0.0026) are a little above the spike's corpus figure for
Verdict and below it for Laya, about a tenth of the bound's 0.02, and the means, 0.0002 to 0.0003,
a tenth of its 0.003. So a disagreement between the two servers means one of two things:

- **Below a 0.01 margin, float16 rounding.** A probability moves by about 0.001, so a top answer
  can only change where upstream's top two are about that close, and the bound allows it there. The
  near-tie table counts the 36 such items; all of them kept their top answer, the closest at a
  margin of 0.0002. A flip there in a later run is expected now and then and is not a defect.
- **Above it, a defect.** A different prompt, truncation, calibration or rounding moves a
  distribution far more than float16 does, and would break the bound; the encoder tests in
  `OpenJevEncodersTests` are where it should then be caught.

Laya's 4-decimal rounding makes 26 answers bit-identical across the two servers, and once makes a
tie. On TypeSafe row `692da1b417be96e6e96f0389` upstream answers 0.4999 for true and the Swift
server exactly 0.5. JevBench breaks a tie towards the smaller label, "no", so its accuracy, Brier
score and ECE do not move; SemIf's evaluator takes the first option, "true", against the reference
"false", which costs the Swift run one row of a five-row case, 0.010 of SemIf's equal-case agreement
(0.426 against 0.436). It is a rounding tie, not a difference between the models.

### Against the published results

- **Verdict.** JevBench's `openjev-verdict-1.4` row ran the same weights through the author's v1.4
  engine with the benchmark's `verdict_local` adapter, on a Ryzen CPU. Every one of the 139 choice
  and 18 score items has the published outcome, on both servers. The five items that differ are 5
  of the 74 nouls: the benchmark's adapter adds a noul's criteria to its proposition,
  `(true: ...; false: ...)`, which the engine writes into both labels and the text, while upstream's
  prompt, and the port's, ignores a noul's criteria ([10-other-models.md](10-other-models.md)).
  Where the prompts are the same, upstream's Verdict gives the author's engine's answers, as its
  README says, and so does the port: 56.3% against the published 57.6% (McNemar p = 0.375).
- **Laya.** The `laya` row is another checkpoint, `convaiinnovations/laya` (its repository root),
  read through the `laya` package with a 512-token budget, where upstream serves
  `laya-typed-decisions` with 1,024 tokens. The two agree on 85% of the items and the published row
  scores 58.4% against 53.7% (p = 0.09), mostly on the hard tier: 35.1% against 27.0%. It is a
  comparison of checkpoints, not of the port.
- **The board's other numbers.** JevBench publishes its Brier score and ECE over the hard tier with
  its held-out half, 220 items (Verdict 0.6941 and 0.1157, Laya 0.7661 and 0.2055), and its tier
  accuracies with the held-out and imported items, so only the public accuracy and the per-item
  outcomes compare directly. Its sealed accuracies, 27.9% (Verdict) and 30.8% (Laya), are about
  half the public ones.
- **Upstream issue #6** reports only the DiffusionGemma rows (v1.4: #21 at 36.85 with the NVFP4
  checkpoint on vLLM, 81.8% on the public items and 28.6% sealed; #54 for the thinking row). No
  benchmark result upstream publishes covers Verdict, Laya or the TypeSafe subset; SemIf publishes
  the subset's Jev value (0.883, which the harness recomputes from TypeSafe's snapshots) and its
  own Qwen3.5-4B scorer's, 0.845.

### Speed, which this is not a benchmark of

One request at a time, the Swift server reads on the GPU and upstream on the CPU, on a MacBook Pro
that was doing other work, so the latency columns say little about a loaded server and move between
reruns, while the scores do not. The Swift server's 95th percentile shows something else: on a Mac
the encoder keeps two Core ML functions loaded (`functionCapacity`, D-037 item 3), one per input
shape, and one-question requests whose lengths move among three or four shapes load a function
again. The function-load table above counts them in the recorded runs: Laya loaded a function 19
times in its 333 requests, 4 first loads of 1.7 to 1.8 s and 15 reloads of 0.45 to 1.15 s, where an
ordinary read took tens of milliseconds, and Verdict's requests used three shapes, each loaded once,
in about 0.7 s. A function's first load varied between 0.2 and 1.9 s across the runs on this Mac.
The answers do not change. A server that sees mixed lengths may want more functions loaded, at the
cost of one more copy of the weights in memory for each.

## Rerunning every table

From the repository root, on a Mac with the converted packages in
`~/Library/Caches/OpenJevSwift/encoders` (or without `--encoder-models`, which downloads the
published packages, the same bytes) and about five minutes for the four runs:

```bash
make upstream
python3 Tools/jevbench/harness.py fetch
python3 Tools/jevbench/smoke_test.py
/usr/local/bin/python3.12 -m venv Tools/jevbench/.venv
Tools/jevbench/.venv/bin/python -m pip install -r Tools/jevbench/requirements-upstream.txt
swift build -c release --product openjev
python3 Tools/jevbench/servers.py --server swift --backend verdict --encoder-models ~/Library/Caches/OpenJevSwift/encoders --force
python3 Tools/jevbench/servers.py --server swift --backend laya --encoder-models ~/Library/Caches/OpenJevSwift/encoders --force
python3 Tools/jevbench/servers.py --server upstream --backend verdict --force
python3 Tools/jevbench/servers.py --server upstream --backend laya --force
python3 Tools/jevbench/harness.py report
```

The scores are deterministic: five sets of runs on this machine, on two builds, gave the same
answers to the last digit (the earlier ones are not committed), and only the timings move. Each
comparison also prints item by item:

```bash
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/verdict-1.4-swift.json Tools/jevbench/results/verdict-1.4-upstream.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/typesafe102/laya-1.0-swift.json Tools/jevbench/results/typesafe102/laya-1.0-upstream.json
python3 Tools/jevbench/harness.py published Tools/jevbench/results/verdict-1.4-swift.json Tools/jevbench/results/laya-1.0-swift.json
python3 Tools/jevbench/harness.py summary Tools/jevbench/results/*.json
```

## DiffusionGemma

Issue #61 asks first for this comparison: the Swift server on DiffusionGemma 4-bit against
upstream's Python MLX server on the same machine and weights
(`mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407`), with the same harness. The runtime
(issue #29) reached main on 2026-10-01, so both halves of these commands run today; the recorded
comparison waits for issue #31, the backend's parity with mlx-vlm:

```bash
python3.14 -m venv Tools/oracle/.venv
Tools/oracle/.venv/bin/python -m pip install -r Tools/oracle/requirements.txt
swift build -c release --product openjev
python3 Tools/jevbench/servers.py --server swift --backend mlx
python3 Tools/jevbench/servers.py --server upstream --backend mlx --python Tools/oracle/.venv/bin/python
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/openjev-0.1-swift.json Tools/jevbench/results/openjev-0.1-upstream.json
python3 Tools/jevbench/harness.py published Tools/jevbench/results/openjev-0.1-swift.json Tools/jevbench/results/openjev-0.1-upstream.json
python3 Tools/jevbench/harness.py report
```

`Tools/oracle/.venv` is the CPython 3.14 environment of the mlx-vlm oracle (mlx-vlm 0.6.15,
upstream's pin, with upstream's own dependencies); `servers.py` points upstream at the checkpoint's
snapshot in the Hugging Face cache. Both halves were checked on 2026-10-01 on three JevBench items,
0.4 to 0.5 s each, with the same top answers and probabilities within 0.0007; that trial is not
recorded. The servers' model times do not compare: upstream's `server-timing` gives 0, because it
times only its calls to vLLM, and the Swift server's sums the reads it runs in parallel (D-038 item
7), so it can exceed the request's total. The caller's time compares. The published row this run
is compared with, `openjev-razorback16`, ran the NVFP4 checkpoint on vLLM, other weights and other
kernels, so its per-item agreement will be lower than the encoders'; upstream issue #6's 81.8% on
the public items is its accuracy. The model reads at a few requests a second on a Mac, so the 333
requests take minutes, and the 4-bit weights need about 16 GB.
