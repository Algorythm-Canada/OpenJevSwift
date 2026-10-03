# Answer quality against upstream

How the Swift server's answers compare with upstream OpenJev's on public benchmarks, and both with
the results the benchmarks publish (issue #61), and how well DiffusionGemma's probabilities are
calibrated (issue #62). `Tools/jevbench` ran JevBench v1's 231 public items and the 102 TypeSafe
public-evaluation rows that SemIf compares with Jev against both servers, on one Mac, with three of
the models this port serves: Verdict (`verdict-1.4`), Laya (`laya-1.0`) and DiffusionGemma 26B-A4B
in 4 bits (`openjev-0.1`). The fourth, JevK5 (`jevk5-0.2`, issue #55), is compared with its author's
published JevBench run instead, since upstream serves it from vLLM on an NVIDIA GPU
([JevK5](#jevk5)). How the harness maps items onto requests and scores them is in
[Tools/jevbench/README.md](../Tools/jevbench/README.md) and D-041; the calibration is under
[Calibration of DiffusionGemma's reads](#calibration-of-diffusiongemmas-reads) and in D-046.

The encoder runs: 2026-10-01, an Apple M3 Max with 128 GB and macOS 27.0.1. The Swift server is a
release build of the package as main's `c77cca9` left it, serving the float16 Core ML packages on
the GPU (D-011); upstream is razorback16/openjev at `dcd2094` serving the PyTorch checkpoints in
float32 on the CPU, as it does on any machine without CUDA: the arithmetic of the reference in
[Fixtures/encoders](../Fixtures/encoders/README.md), on PyTorch's default of 12 threads where the
reference pins 8, which moves only the last bits.

The DiffusionGemma runs: 2026-10-02, the same Mac. The Swift server is a release build of the
package as main's `2414408` left it, reading the checkpoint with MLX on the GPU (D-039); upstream is
`dcd2094` with MLX 0.32.2 and mlx-vlm 0.6.15 on the GPU, from `Tools/jevbench/.venv`. Both read
`mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407` from the Hugging Face cache, with
upstream's read policy (three more reads, averaged with the first, when a slot is uncertain) and
`OPENJEV_MLX_CACHE_LIMIT_GB=4`, which [DiffusionGemma](#diffusiongemma) explains. Every request
asked one question, one request at a time.

The JevK5 runs: 2026-10-03 UTC, the same Mac. The Swift server is a release build of this branch as
`db1d51b` left the package, reading JevK5 v0.2 with MLX on the GPU (D-052) from the pinned 8-bit
conversion, the server's default, with `OPENJEV_MLX_CACHE_LIMIT_GB=4`; the same build ran JevBench
again on the 4-bit conversion and on the unquantized bfloat16 weights. The reference is the `jevk5`
package's published v0.2 run (`results/public231/jevk5-v0.2.jsonl` in allebee/jevk5 at `0571ef3`):
transformers with the bfloat16 weights on the author's NVIDIA GPU, through JevBench's runner, which
`harness.py author-run` turns into a result file. The author published no TypeSafe run.

## What the runs show

- **On the encoders the Swift server gives upstream's top answer on every item.** For both encoder
  models and both datasets, 666 items in all: no top answer differs, no item is right on one server
  and wrong on the other, accuracy is the same, and the Brier score and ECE differ by at most
  0.0003, overall and per tier.
- **The encoders' probabilities differ by float16, and stay well inside the parity bound.** The
  largest difference is 0.0026 (Laya) and 0.0022 (Verdict) and the mean about 0.0002 to 0.0003,
  against D-034's and D-037's bound of 0.02 largest and 0.003 mean. Of the 36 items whose top two
  answers upstream puts less than 0.01 apart, every one kept its top answer.
- **Verdict reproduces the benchmark's published Verdict 1.4 row,** on both servers: the same
  outcome on 226 of 231 items, every choice and score among them. The five that differ are nouls,
  whose prompt the benchmark's own adapter writes differently. The published Laya row is another
  checkpoint and agrees on 85%.
- **Both encoders are far from Jev on TypeSafe's evaluations.** On SemIf's subset Verdict agrees
  with the reference answer 0.53 of the time and Laya 0.43 to 0.44, against Jev's published 0.88.
- **On DiffusionGemma the two servers agree on 326 of 333 top answers,** 225 of 231 on JevBench and
  101 of 102 on TypeSafe, every one of the 299 whose upstream top two are at least 0.5 apart among
  them, with a mean probability difference of 0.0135 and 0.0069. Seven items are right on one server
  only, five of them upstream's (McNemar p = 0.45). That meets every D-014 bound an answer
  carries. On JevBench's 41 prompts over 1,024 tokens the mean difference is 0.0396, past the 0.01
  D-014 set there, a bound that many-option labels diluted and that D-048 replaced, and in D-014's
  exact tier the port's reads of the four items whose top answers differ at a wide margin are
  mlx-vlm's bit for bit, so the difference is the kernels' ([A disagreement on
  DiffusionGemma](#a-disagreement-on-diffusiongemma)).
- **DiffusionGemma scores like the benchmark's published row, and like Jev on TypeSafe.** JevBench's
  `openjev-razorback16` row, the NVFP4 weights on vLLM, has 81.8% on the public items; these runs
  have 81.4% (Swift) and 82.3% (upstream), with the published outcome on 96.1% of the items. On
  SemIf's TypeSafe subset DiffusionGemma agrees with the reference answer 0.888 and 0.892 of the
  time, against Jev's published 0.883.
- **JevK5 on its 8-bit conversion gives the author's top answer on 230 of the 231 JevBench items
  and bills the same tokens on all 231,** so every prompt is the author's. The one item that
  differs, `hard-sol-a-multi_hop-10`, is a near-tie in the author's run, its top two 0.040 apart;
  the mean probability difference is 0.0049 and the largest 0.083, where upstream measured 0.055 on
  vLLM. Issue #55 asks for all 231.
- **Quantization moves JevK5's answers; the port does not.** The bfloat16 weights give the author's
  top answer on 228 items, the three others within 0.031 of a tie in the author's run, and a largest
  difference of 0.051. The 4-bit conversion gives it on 209, with differences up to 0.53 and 18 of
  the 22 changed answers on items whose author top two are at least 0.05 apart. Accuracy follows:
  85.7% for 8 bits and bfloat16 and 85.3% for 4 bits on JevBench, against the author's 86.1%, and
  on SemIf's TypeSafe subset 86.3% for 8 bits and bfloat16 and 81.4% for 4 bits (the 8-bit
  conversion's modal agreement is 0.845, against Jev's published 0.883). D-052 makes the 8-bit
  conversion the server's default for that reason.
- **DiffusionGemma's probabilities are overconfident on hard questions.** On JevBench it is right
  81% to 82% of the time at a mean top probability of 0.92, an ECE of 0.106 (Swift) and 0.097
  (upstream), nearly all of the excess in the hard tier. A temperature of about 2, fitted offline,
  lowers the ECE out of fold to about 0.06. D-046 keeps the server's probabilities upstream's and
  leaves the rescaling to the client ([Calibration of DiffusionGemma's
  reads](#calibration-of-diffusiongemmas-reads)).

## The tables

Rendered by `python3 Tools/jevbench/harness.py report` from the result files in
`Tools/jevbench/results/` and the pinned datasets. In every comparison upstream's run is the
reference, or for JevK5 its author's. "items" counts the questions asked; nothing was skipped or
refused. The latency columns are the caller's time and the `model` part of the server's
`server-timing` header, in milliseconds; the DiffusionGemma and JevK5 runs' are not reported,
because the first shared the Mac with other work and the second were not taken under a benchmark's
protocol ([Speed](#speed-which-this-is-not-a-benchmark-of)).

### The runs

| dataset | model | server | items | answered | skipped | refused, failed or invalid | accuracy | Brier | ECE | p50 ms | p95 ms | model p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jevbench | jevk5-0.2 | author | 231 | 231 | 0 | 0 | 86.1% | 0.2104 | 0.0587 | not reported | not reported | not reported |
| jevbench | jevk5-0.2 | swift | 231 | 231 | 0 | 0 | 85.7% | 0.2105 | 0.0514 | not reported | not reported | not reported |
| jevbench | laya-1.0 | swift | 231 | 231 | 0 | 0 | 53.7% | 0.5090 | 0.0769 | 30.3 | 510.2 | 29.4 |
| jevbench | laya-1.0 | upstream | 231 | 231 | 0 | 0 | 53.7% | 0.5090 | 0.0769 | 171.3 | 761.2 | 158.5 |
| jevbench | openjev-0.1 | swift | 231 | 231 | 0 | 0 | 81.4% | 0.2452 | 0.1060 | not reported | not reported | not reported |
| jevbench | openjev-0.1 | upstream | 231 | 231 | 0 | 0 | 82.3% | 0.2385 | 0.0966 | not reported | not reported | not reported |
| jevbench | verdict-1.4 | swift | 231 | 231 | 0 | 0 | 56.3% | 0.5381 | 0.0788 | 19.4 | 110.9 | 18.6 |
| jevbench | verdict-1.4 | upstream | 231 | 231 | 0 | 0 | 56.3% | 0.5382 | 0.0788 | 77.0 | 202.5 | 73.2 |
| typesafe102 | jevk5-0.2 | swift | 102 | 102 | 0 | 0 | 86.3% | 0.2094 | 0.0454 | not reported | not reported | not reported |
| typesafe102 | laya-1.0 | swift | 102 | 102 | 0 | 0 | 47.1% | 0.5866 | 0.0727 | 166.7 | 760.4 | 164.9 |
| typesafe102 | laya-1.0 | upstream | 102 | 102 | 0 | 0 | 47.1% | 0.5865 | 0.0727 | 684.2 | 892.4 | 681.2 |
| typesafe102 | openjev-0.1 | swift | 102 | 102 | 0 | 0 | 89.2% | 0.1941 | 0.0919 | not reported | not reported | not reported |
| typesafe102 | openjev-0.1 | upstream | 102 | 102 | 0 | 0 | 90.2% | 0.1892 | 0.0962 | not reported | not reported | not reported |
| typesafe102 | verdict-1.4 | swift | 102 | 102 | 0 | 0 | 52.9% | 0.5408 | 0.1015 | 52.6 | 213.3 | 51.8 |
| typesafe102 | verdict-1.4 | upstream | 102 | 102 | 0 | 0 | 52.9% | 0.5409 | 0.1014 | 160.3 | 216.5 | 158.4 |

### Swift against upstream

| model | dataset | items | top answer agrees | identical answers | mean abs diff | largest abs diff | right only on Swift, only upstream | McNemar p |
|---|---|---|---|---|---|---|---|---|
| laya-1.0 | jevbench | 231 | 231 of 231 | 10 | 2.1e-04 | 0.0026 | 0 and 0 | 1 |
| openjev-0.1 | jevbench | 231 | 225 of 231 | 0 | 0.0135 | 0.4066 | 2 and 4 | 0.688 |
| verdict-1.4 | jevbench | 231 | 231 of 231 | 0 | 2.8e-04 | 0.0022 | 0 and 0 | 1 |
| laya-1.0 | typesafe102 | 102 | 102 of 102 | 16 | 1.6e-04 | 0.0012 | 0 and 0 | 1 |
| openjev-0.1 | typesafe102 | 102 | 101 of 102 | 0 | 0.0069 | 0.1605 | 0 and 1 | 1 |
| verdict-1.4 | typesafe102 | 102 | 102 of 102 | 0 | 2.9e-04 | 0.0020 | 0 and 0 | 1 |

| model | dataset | type | items | top answer agrees | mean abs diff | largest abs diff | largest at (label) |
|---|---|---|---|---|---|---|---|
| laya-1.0 | jevbench | choice | 139 | 139 of 139 | 1.7e-04 | 0.0019 | original-routing-03-1 (coding_agent) |
| laya-1.0 | jevbench | noul | 74 | 74 of 74 | 3.6e-04 | 0.0026 | easy-fact-11 (yes) |
| laya-1.0 | jevbench | score | 18 | 18 of 18 | 2.5e-04 | 0.0014 | original-ordinal-06-1 (2) |
| openjev-0.1 | jevbench | choice | 139 | 137 of 139 | 0.0120 | 0.2960 | hard-sol-b-long_policy-06 (no_award) |
| openjev-0.1 | jevbench | noul | 74 | 71 of 74 | 0.0230 | 0.4066 | hard-opus-c-long_policy-04 (yes) |
| openjev-0.1 | jevbench | score | 18 | 17 of 18 | 0.0070 | 0.0656 | hard-opus-b-multi_hop-05 (2) |
| verdict-1.4 | jevbench | choice | 139 | 139 of 139 | 2.6e-04 | 0.0022 | hard-opus-b-tradeoff-12 (refund_now) |
| verdict-1.4 | jevbench | noul | 74 | 74 of 74 | 3.4e-04 | 0.0010 | original-policy-06-1 (yes) |
| verdict-1.4 | jevbench | score | 18 | 18 of 18 | 2.8e-04 | 0.0015 | hard-opus-b-multi_hop-05 (0) |
| laya-1.0 | typesafe102 | choice | 36 | 36 of 36 | 1.8e-04 | 0.0012 | f892cb9fd5fc7590f4b6525b (card_declined) |
| laya-1.0 | typesafe102 | noul | 66 | 66 of 66 | 1.4e-04 | 4.0e-04 | 71a2393072e9e2b2ac08da0c (yes) |
| openjev-0.1 | typesafe102 | choice | 36 | 36 of 36 | 0.0037 | 0.1068 | acae0a13f9c7e72851015c11 (speak_to_human) |
| openjev-0.1 | typesafe102 | noul | 66 | 65 of 66 | 0.0112 | 0.1605 | e2e58201a90c11192f70edbf (yes) |
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
| openjev-0.1 | jevbench | hard-opus-c-long_policy-04 | noul | yes | 0.3178 | 0.7244 | 0.4066 |
| openjev-0.1 | jevbench | hard-sol-b-long_policy-06 | choice | no_award | 0.6029 | 0.3069 | 0.2960 |
| openjev-0.1 | jevbench | hard-opus-c-long_policy-02 | choice | erase_but_retain_transaction_records | 0.6150 | 0.8705 | 0.2555 |
| openjev-0.1 | jevbench | hard-opus-b-tradeoff-08 | noul | yes | 0.7522 | 0.4993 | 0.2529 |
| openjev-0.1 | jevbench | hard-opus-b-multi_hop-03 | choice | tier2_department_head | 0.7005 | 0.4955 | 0.2050 |
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
| openjev-0.1 | typesafe102 | e2e58201a90c11192f70edbf | noul | yes | 0.4412 | 0.6016 | 0.1605 |
| openjev-0.1 | typesafe102 | 071dd61c989b844c00d9a044 | noul | yes | 0.7852 | 0.8945 | 0.1093 |
| openjev-0.1 | typesafe102 | acae0a13f9c7e72851015c11 | choice | speak_to_human | 0.9091 | 0.8023 | 0.1068 |
| openjev-0.1 | typesafe102 | 4cd12c18602cee6acf9d714d | noul | yes | 0.6662 | 0.5716 | 0.0946 |
| openjev-0.1 | typesafe102 | 36d0a4df0d8697ca942472a1 | choice | speak_to_human | 0.3844 | 0.4729 | 0.0886 |
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

DiffusionGemma's answers against D-014's aggregate bounds, which the 63 oracle reads are held to
(here over answers, each the mean of up to four reads, with upstream's as the reference; `harness.py
compare` lists every disagreement). Past 1,024 tokens D-048 bounds each read slot's largest
difference, which an answer averages, so the long prompts' figures are shown and not bounded:

| model | dataset | items | mean abs diff, every label (at most 0.02) | over prompts of more than 1,024 tokens: mean of each item's largest abs diff, and mean abs diff over every label (not bounded) | top answer agrees (at least 90%) | where upstream's top two are at least 0.5 apart (at least 97%) | within every bound |
|---|---|---|---|---|---|---|---|
| openjev-0.1 | jevbench | 231 | 0.0135 | 0.0741 and 0.0396 over 41 | 225 of 231 (97.4%) | 200 of 200 (100.0%) | yes |
| openjev-0.1 | typesafe102 | 102 | 0.0069 | 0.0102 and 0.0075 over 76 | 101 of 102 (99.0%) | 99 of 99 (100.0%) | yes |

### Swift against the model author's published run

JevK5's reference is its author's own published v0.2 run, as it is upstream's: upstream's server
reads JevK5's letters from vLLM, which needs an NVIDIA GPU (D-052). Equal input tokens mean equal
prompts.

| model | dataset | items | top answer agrees | input tokens equal | mean abs diff | median of each item's largest abs diff | largest abs diff | right only on Swift, only the author's | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| jevk5-0.2 | jevbench | 231 | 230 of 231 | 231 of 231 | 0.0049 | 0.0024 | 0.0834 | 0 and 1 | 1 |

| model | dataset | type | items | top answer agrees | mean abs diff | largest abs diff | largest at (label) |
|---|---|---|---|---|---|---|---|
| jevk5-0.2 | jevbench | choice | 139 | 138 of 139 | 0.0041 | 0.0834 | hard-sol-b-long_policy-05 (delete_profile_omar_april_and_table) |
| jevk5-0.2 | jevbench | noul | 74 | 74 of 74 | 0.0087 | 0.0609 | original-policy-03-1 (yes) |
| jevk5-0.2 | jevbench | score | 18 | 18 of 18 | 0.0038 | 0.0175 | original-ordinal-01-0 (0) |

The 5 largest deviations:

| model | dataset | item | type | label | Swift | author | abs diff | top answer agrees |
|---|---|---|---|---|---|---|---|---|
| jevk5-0.2 | jevbench | hard-sol-b-long_policy-05 | choice | delete_profile_omar_april_and_table | 0.2540 | 0.3374 | 0.0834 | yes |
| jevk5-0.2 | jevbench | hard-opus-b-tradeoff-07 | choice | p2_fix_30d | 0.3495 | 0.4140 | 0.0645 | yes |
| jevk5-0.2 | jevbench | hard-opus-a-probability-07 | choice | resolved_first_contact | 0.4967 | 0.5577 | 0.0609 | yes |
| jevk5-0.2 | jevbench | original-policy-03-1 | noul | yes | 0.4391 | 0.5000 | 0.0609 | yes |
| jevk5-0.2 | jevbench | hard-sol-a-multi_hop-12 | choice | needs_authentication | 0.4340 | 0.4945 | 0.0606 | yes |

### JevK5's conversions against the author's run

The same runs with each MLX conversion of JevK5 v0.2 that Tools/jevk5/convert.py pins, from
jevk5-conversions/: the default is the main result files' conversion, and bfloat16 is the
unquantized weights. A near-tie, the author's top two less than 0.05 apart, can turn on bfloat16
rounding alone (D-052).

| model | conversion | top answer agrees | top answer agrees where the author's top two are at least 0.05 apart | input tokens equal | mean abs diff | median of each item's largest abs diff | largest abs diff | accuracy | TypeSafe accuracy |
|---|---|---|---|---|---|---|---|---|---|
| jevk5-0.2 | jevk5-0.2-mlx-8bit, the default | 230 of 231 | 219 of 219 | 231 of 231 | 0.0049 | 0.0024 | 0.0834 | 85.7% | 86.3% |
| jevk5-0.2 | jevk5-0.2-mlx-4bit | 209 of 231 | 201 of 219 | 231 of 231 | 0.0440 | 0.0358 | 0.5299 | 85.3% | 81.4% |
| jevk5-0.2 | jevk5-0.2-mlx-bf16 | 228 of 231 | 219 of 219 | 231 of 231 | 0.0037 | 0.0017 | 0.0507 | 85.7% | 86.3% |
| jevk5-0.2 | the author's published run |  |  |  |  |  |  | 86.1% | not published |

### Against JevBench's published rows

| model | server | published row | public items | ours | published | same outcome | right only here | right only there | McNemar p | published sealed |
|---|---|---|---|---|---|---|---|---|---|---|
| laya-1.0 | swift | laya | 231 | 53.7% | 58.4% | 84.8% | 12 | 23 | 0.0895 | 30.8% |
| laya-1.0 | upstream | laya | 231 | 53.7% | 58.4% | 84.8% | 12 | 23 | 0.0895 | 30.8% |
| openjev-0.1 | swift | openjev-razorback16 | 231 | 81.4% | 81.8% | 96.1% | 4 | 5 | 1 | 28.6% |
| openjev-0.1 | upstream | openjev-razorback16 | 231 | 82.3% | 81.8% | 96.1% | 5 | 4 | 1 | 28.6% |
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
| openjev-0.1 | swift | easy | 48 | 100.0% | 100.0% |
| openjev-0.1 | swift | standard | 72 | 98.6% | 97.2% |
| openjev-0.1 | swift | hard | 111 | 62.2% | 64.0% |
| openjev-0.1 | upstream | easy | 48 | 100.0% | 100.0% |
| openjev-0.1 | upstream | standard | 72 | 98.6% | 97.2% |
| openjev-0.1 | upstream | hard | 111 | 64.0% | 64.0% |
| verdict-1.4 | swift | easy | 48 | 89.6% | 87.5% |
| verdict-1.4 | swift | standard | 72 | 69.4% | 69.4% |
| verdict-1.4 | swift | hard | 111 | 33.3% | 36.9% |
| verdict-1.4 | upstream | easy | 48 | 89.6% | 87.5% |
| verdict-1.4 | upstream | standard | 72 | 69.4% | 69.4% |
| verdict-1.4 | upstream | hard | 111 | 33.3% | 36.9% |

Items whose outcome differs from the published row's (upstream's run, and the Swift run's where it
is not the same: laya-1.0 the same, openjev-0.1 differs, verdict-1.4 the same):

| model | server | type | items | which |
|---|---|---|---|---|
| laya-1.0 | upstream | choice | 21 | 21 items |
| laya-1.0 | upstream | noul | 11 | 11 items |
| laya-1.0 | upstream | score | 3 | original-ordinal-01-1, original-ordinal-06-0, hard-opus-b-multi_hop-05 |
| openjev-0.1 | upstream | choice | 6 | original-routing-04-1, hard-opus-a-temporal_numeric-07, hard-opus-b-ambiguous-09, hard-opus-b-multi_hop-03, hard-sol-b-long_policy-06, hard-sol-c-multi_hop-12 |
| openjev-0.1 | upstream | noul | 3 | hard-opus-a-long_policy-19, hard-opus-c-long_policy-04, hard-sol-c-judge_hard-07 |
| openjev-0.1 | swift | choice | 6 | original-routing-04-1, hard-opus-a-temporal_numeric-07, hard-opus-b-ambiguous-09, hard-opus-b-multi_hop-03, hard-opus-c-temporal_numeric-08, hard-sol-c-multi_hop-12 |
| openjev-0.1 | swift | noul | 2 | hard-opus-b-tradeoff-08, hard-sol-c-judge_hard-07 |
| openjev-0.1 | swift | score | 1 | hard-opus-c-long_policy-05 |
| verdict-1.4 | upstream | noul | 5 | easy-fact-11, hard-opus-a-long_policy-13, hard-opus-a-long_policy-19, hard-opus-a-temporal_numeric-06, hard-sol-a-trap-15 |

### The TypeSafe subset with SemIf's metrics

| model | server | rows | cases | modal agreement (equal-case) | total variation (equal-case) | accuracy (pooled) | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| jevk5-0.2 | swift | 102 of 102 | 20 | 0.845 | 0.208 | 86.3% | 0.2094 | 0.0454 |
| laya-1.0 | swift | 102 of 102 | 20 | 0.426 | 0.500 | 47.1% | 0.5866 | 0.0727 |
| laya-1.0 | upstream | 102 of 102 | 20 | 0.436 | 0.500 | 47.1% | 0.5865 | 0.0727 |
| openjev-0.1 | swift | 102 of 102 | 20 | 0.888 | 0.131 | 89.2% | 0.1941 | 0.0919 |
| openjev-0.1 | upstream | 102 of 102 | 20 | 0.892 | 0.131 | 90.2% | 0.1892 | 0.0962 |
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
| jevk5-0.2 | author | easy | 48 | 100.0% | 0.0119 | 0.0385 | n/a |
| jevk5-0.2 | author | standard | 72 | 95.8% | 0.0979 | 0.1410 | 0.100 |
| jevk5-0.2 | author | hard | 111 | 73.9% | 0.3692 | 0.0662 | 0.473 |
| jevk5-0.2 | swift | easy | 48 | 100.0% | 0.0122 | 0.0378 | n/a |
| jevk5-0.2 | swift | standard | 72 | 95.8% | 0.0961 | 0.1383 | 0.098 |
| jevk5-0.2 | swift | hard | 111 | 73.0% | 0.3704 | 0.0706 | 0.466 |
| laya-1.0 | swift | easy | 48 | 97.9% | 0.1213 | 0.2297 | n/a |
| laya-1.0 | swift | standard | 72 | 65.3% | 0.4588 | 0.1764 | 0.512 |
| laya-1.0 | swift | hard | 111 | 27.0% | 0.7091 | 0.1706 | 0.692 |
| laya-1.0 | upstream | easy | 48 | 97.9% | 0.1212 | 0.2296 | n/a |
| laya-1.0 | upstream | standard | 72 | 65.3% | 0.4590 | 0.1765 | 0.512 |
| laya-1.0 | upstream | hard | 111 | 27.0% | 0.7092 | 0.1707 | 0.692 |
| openjev-0.1 | swift | easy | 48 | 100.0% | 6.3e-06 | 0.0011 | n/a |
| openjev-0.1 | swift | standard | 72 | 98.6% | 0.0141 | 0.0160 | 4.6e-03 |
| openjev-0.1 | swift | hard | 111 | 62.2% | 0.5011 | 0.2189 | 0.645 |
| openjev-0.1 | upstream | easy | 48 | 100.0% | 9.8e-06 | 0.0012 | n/a |
| openjev-0.1 | upstream | standard | 72 | 98.6% | 0.0163 | 0.0182 | 4.6e-03 |
| openjev-0.1 | upstream | hard | 111 | 64.0% | 0.4858 | 0.2001 | 0.680 |
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
| verdict-1.4 | 333 | 3 | 0 | 698 to 1213 | 128: 14, 256: 16, 512: 47 |
| laya-1.0 | 333 | 4 | 15 | 461 to 838 | 128: 20, 256: 29, 512: 62, 1024: 163 |

### Machines and versions

| datasets | model | server | code | runtime | machine | run on |
|---|---|---|---|---|---|---|
| jevbench | jevk5-0.2 | author | allebee/jevk5 0.2.2 at 0571ef3, published | the jevk5 package's own runtime (JevK5 0.2.2: transformers, the bf16 v0.2 weights, CUDA graphs) through JevBench's runner, as the author published it | not recorded by the published run | 2026-09-22 |
| jevbench, typesafe102 | jevk5-0.2 | swift | OpenJevSwift 0.1.0-dev at db1d51b | MLX on the GPU, Qwen3.5 through mlx-swift-lm (D-052), OPENJEV_JEVK5_MODEL=~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-8bit, OPENJEV_MLX_CACHE_LIMIT_GB=4 | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-03 |
| jevbench, typesafe102 | laya-1.0 | swift | OpenJevSwift 0.1.0-dev at c77cca9 | Core ML, float16 multifunction package, .cpuAndGPU, up to 16 questions per call | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | laya-1.0 | upstream | openjev 0.5.0 at dcd2094, Python 3.12.2 | PyTorch 2.13.0 on the cpu, float32, 12 threads | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | openjev-0.1 | swift | OpenJevSwift 0.1.0-dev at 2414408 | MLX on the GPU (D-039), OPENJEV_MLX_CACHE_LIMIT_GB=4 | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-02 |
| jevbench, typesafe102 | openjev-0.1 | upstream | openjev 0.5.0 at dcd2094, Python 3.12.2 | MLX 0.32.2 and mlx-vlm 0.6.15 on the GPU, the checkpoint's weights, OPENJEV_MLX_CACHE_LIMIT_GB=4 | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-02 |
| jevbench, typesafe102 | verdict-1.4 | swift | OpenJevSwift 0.1.0-dev at c77cca9 | Core ML, float16 multifunction package, .cpuAndGPU, up to 16 questions per call | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |
| jevbench, typesafe102 | verdict-1.4 | upstream | openjev 0.5.0 at dcd2094, Python 3.12.2 | PyTorch 2.13.0 on the cpu, float32, 12 threads | Apple M3 Max, 128 GB, macOS 27.0.1 (26A434) | 2026-10-01 |


## What the numbers mean

### A disagreement on the encoders

The two servers do not run the same arithmetic. Upstream reads each encoder with PyTorch in float32
on the CPU; the Swift server reads a Core ML package whose weights and activations are float16, on
the GPU, then applies the same prompt, truncation and calibration in Swift, which the encoder tests
check byte for byte and, for the calibration, within 1e-6 or bit for bit (D-034, D-037). What is
left is float16 rounding inside the network. Spike #56 measured it on 200 reference questions:
Verdict's package stays within 0.0014 of PyTorch float32 on the Mac's GPU (D-034) and Laya's within
0.0039 before its 4-decimal rounding (D-037), and both backends are held to the spike's bound: the
largest difference at most 0.02, the mean at most 0.003, and the top answer unchanged wherever the
reference's top two are at least 0.01 apart.

These runs are a second, larger sample of the same difference: 666 items, with JevBench's long
hard-tier states and TypeSafe's long documents filling Verdict's 512 tokens and Laya's 1,024. The
largest differences (Verdict 0.0022, Laya 0.0026) are a little above the spike's corpus figure for
Verdict and below it for Laya, about a tenth of the bound's 0.02, and the means, 0.0002 to 0.0003, a
tenth of its 0.003. So a disagreement between the two servers means one of two things:

- **Below a 0.01 margin, float16 rounding.** A probability moves by about 0.001, so a top answer can
  only change where upstream's top two are about that close, and the bound allows it there. The
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

### A disagreement on DiffusionGemma

The Swift server and upstream's run the same read on different kernels: mlx-swift compiles MLX's
elementwise kernels at run time where the mlx-metal wheel ships them precompiled, so transcendental
functions round differently in the last bit, and 23 of the 64 full-attention RoPE frequencies come
out one or two float32 steps away from mlx-vlm's (spike #22,
[spikes/backend-validation.md](spikes/backend-validation.md)). The read is chaotic in bfloat16: such
last-bit differences move label probabilities by up to 0.62 and change top labels even where the
oracle's margin is 0.68, so D-014 bounds the difference in aggregate only, and holds the port to
mlx-vlm's 63 oracle reads bit for bit when it runs on the wheel's metallib and the oracle's RoPE
table (D-036, D-048). The prompts are not part of it: both servers count the same prompt tokens on all 333
items.

Over these 333 answers, each the mean of up to four reads, the difference is what D-014 measured on
short prompts and grows past 1,024 tokens, the length of the model's sliding window:

- **Up to 1,024 tokens** (190 JevBench items and 26 TypeSafe rows) the mean difference is 0.0074 and
  0.0057. The top answers agree on 188 of 190 and 26 of 26, and the two that differ are close calls:
  `hard-opus-b-tradeoff-08` at an upstream margin of 0.0015 and `hard-opus-c-temporal_numeric-08` at
  0.026.
- **Over 1,024 tokens** JevBench's 41 prompts (up to 3,940 tokens) differ by 0.0396 on average, four
  times the 0.01 D-014 set over long prompts before D-048 replaced it, and TypeSafe's 76 (up to
  11,600 tokens) by 0.0075. Most of what separates them is how uncertain the answers are: JevBench's long items are the
  hard tier's long-policy and multi-hop questions, and an answer whose upstream top two are less
  than 0.5 apart moves by 0.06 to 0.09 on average at any length, while TypeSafe's long rows are
  mostly confident. At the same confidence a long prompt still moves more: on JevBench, answers with
  an upstream margin of 0.9 or more differ by 0.0049 over 1,024 tokens and by 0.0007 under it.
- **Four top answers differ at a wide margin,** all on long prompts: `hard-opus-c-long_policy-04`
  (3,264 tokens, upstream margin 0.45), `hard-sol-b-long_policy-06` (2,647 tokens, 0.27),
  `hard-opus-a-long_policy-19` (2,318 tokens, 0.25) and TypeSafe row `e2e58201a90c11192f70edbf`
  (2,925 tokens, 0.20). Upstream's answer is the expected one on all four, which four items cannot
  tell from chance (sign test p = 0.125).

No question type always differs (on JevBench noul 71 of 74, choice 137 of 139, score 17 of 18; on
TypeSafe noul 65 of 66, choice 36 of 36), and no type's probabilities lean one way: the Swift server
gives upstream's top choice a higher probability on 81 items and a lower one on 58 (sign test p =
0.06), and a higher P(yes) on 36 nouls and a lower one on 38. The four wide-margin flips settle
where the difference comes from: read under D-014's exact tier, with the mlx-metal wheel's metallib
and the oracle's RoPE table in the five full-attention layers, the port gives mlx-vlm's reads of all
four bit for bit.

- **The same reads.** Upstream's own `MlxEngine.decide` on the harness's request bodies (their
  SHA-256 is the result files' `body_sha256`) gives upstream's four answers bit for bit, and the
  port's `DecisionEngine` builds the same prompt ids, canvases, slots and steps from the same
  bodies: four reads per item, at the request seed and the three re-read seeds, each one step over
  a 16-token canvas with the slot at position 7.
- **The same numbers.** All 16 reads are identical in every token id and float32 logprob of the top
  20 and the labels, and so are the prefill caches of all 30 layers, the 25 sliding layers' last
  1,023 positions (what a read sees) and the prompt token counts. The answers equal upstream's
  result files to the last bit: 0.7244154751137403, 0.3748895786709909, 0.6016278724196832, and
  `east_ward` with confidence 0.2770513449574379.
- **The control.** On the Swift server's own kernels and RoPE table the same harness gives the Swift
  result files' four answers bit for bit, so it reads what the server read. There the first
  difference is layer 0's keys, after RoPE: their sums of squares differ from mlx-vlm's by 5e-9 to
  3e-8 relative, and the values are equal. With the wheel's metallib and the port's own table,
  layers 0 to 4 are equal and the first difference is layer 5, the first full-attention layer.

These states move with any last-bit change, mlx-vlm's own included. Against upstream's 16 reads,
over their 40 label probabilities, and in the top label of the four answers:

| read by | mean \|Δp\| | largest \|Δp\| | answers flipped |
|---|---|---|---|
| the port on the Swift server's kernels and its own RoPE table, as the server reads | 0.210 | 0.500 | 4 of 4 |
| the port on the wheel's metallib, with its own table | 0.083 | 0.247 | 1 of 4 |
| the port on the Swift server's kernels, with the oracle's table | 0.226 | 0.913 | 2 of 4 |
| mlx-vlm itself with a 64-token chunked prefill (`Tools/oracle/sensitivity.py`) | 0.119 | 0.476 | 1 of 4 |
| the port in the exact tier, the wheel's metallib and the oracle's table | 0 | 0 | 0 of 4 |

mlx-vlm's own chunked prefill, exact in real arithmetic, flips `hard-sol-b-long_policy-06` to
`no_award` at 0.628, as the Swift server did, and moves the other three answers by 0.05 to 0.14;
upstream's own four reads of `hard-opus-a-long_policy-19` put P(yes) anywhere from 0.06 to 0.63.
Either kernel difference alone flips some of these items, and the oracle's table alone would not
bring the Swift server's reads closer. `Tools/oracle/item_reads.py` runs the check on any JevBench
or TypeSafe item ([Tools/README.md](../Tools/README.md#reading-benchmark-items-against-upstream)),
and `Tools/oracle/results/item_reads_long_flips.json` keeps this run, without TypeSafe's text.

The long-prompt excess led to D-048, which widened the oracle fixture and replaced D-014's
long-prompt row. That bound was a mean over the 1,496 label probabilities of the oracle fixture's
50 slots on prompts over 1,024 tokens (four prompts of 1,572 to 2,939 tokens, where the port met it
with 0.0054, D-044), and 1,440 of them belong to the 26 slots with 10 to 255 labels, most of them
near 0.
The other 24 slots have two or three labels, the only long slots of the fixture with no more labels
than a benchmark question (two to six on JevBench, two to eight on TypeSafe). Over their 56 labels
spike #22's committed runs give 0.0428 for mlx-vlm's chunked prefill and 0.0537 for the
transliteration on native kernels, against 0.0064 and 0.0054 over all labels, and 27 of the 50
slots have an oracle top-two margin under 0.5. JevBench's 0.0396 over its 41 long prompts is of
that size, a little under both. D-048 added 36 reads of nine of JevBench's long items, PR #105's
three long-policy flips among them, each one question with two to four labels as the harness asks
it; the port reads all 63 of the fixture's reads bit for bit in the exact tier. Over the 60 long
slots with at most four labels the widened fixture's figures are 0.0628 for mlx-vlm's chunked
prefill and 0.0757 for the transliteration, and the bound is now the mean of each long slot's
largest difference, which many-option labels cannot dilute: at most 0.14, where the port gives
0.1006 and the planted no-window bug 0.1766. An answer averages up to four reads, so the comparison
above reports its long-prompt figures instead of bounding them: the mean of each item's largest
difference is 0.0741 over JevBench's 41 long items and 0.0102 over TypeSafe's 76.
`item_reads.py long-slots` recomputes the fixture's figures from the committed files.

### Against the published results

- **Verdict.** JevBench's `openjev-verdict-1.4` row ran the same weights through the author's v1.4
  engine with the benchmark's `verdict_local` adapter, on a Ryzen CPU. Every one of the 139 choice
  and 18 score items has the published outcome, on both servers. The five items that differ are 5 of
  the 74 nouls: the benchmark's adapter adds a noul's criteria to its proposition, `(true: ...;
  false: ...)`, which the engine writes into both labels and the text, while upstream's prompt, and
  the port's, ignores a noul's criteria ([10-other-models.md](10-other-models.md)). Where the
  prompts are the same, upstream's Verdict gives the author's engine's answers, as its README says,
  and so does the port: 56.3% against the published 57.6% (McNemar p = 0.375).
- **Laya.** The `laya` row is another checkpoint, `convaiinnovations/laya` (its repository root),
  read through the `laya` package with a 512-token budget, where upstream serves
  `laya-typed-decisions` with 1,024 tokens. The two agree on 85% of the items and the published row
  scores 58.4% against 53.7% (p = 0.09), mostly on the hard tier: 35.1% against 27.0%. It is a
  comparison of checkpoints, not of the port.
- **DiffusionGemma.** The `openjev-razorback16` row ran upstream on vLLM with the NVFP4 checkpoint
  on an RTX PRO 4500 Blackwell, through JevBench's `typesafe` adapter: the same requests, other
  weights and other kernels. It scores 81.8% on the public items, 189 of 231; the Swift run scores
  81.4% and upstream's 82.3%, each with the published outcome on 96.1% of the items (McNemar p = 1).
  Per tier the runs are within two items of the row: 100% easy, 98.6% standard against the published
  97.2%, and 62.2% (Swift) and 64.0% (upstream) hard against 64.0%.
- **The board's other numbers.** JevBench publishes its Brier score and ECE over the hard tier with
  its held-out half, 220 items (Verdict 0.6941 and 0.1157, Laya 0.7661 and 0.2055, DiffusionGemma
  NVFP4 0.4844 and 0.1779), and its tier accuracies with the held-out and imported items, so only
  the public accuracy and the per-item outcomes compare directly. The public hard tier alone gives
  DiffusionGemma 0.5011 and 0.2189 here on the Swift server and 0.4858 and 0.2001 on upstream's. The
  sealed accuracies, 27.9% (Verdict), 30.8% (Laya) and 28.6% (DiffusionGemma), are about half the
  public ones or less.
- **Upstream issue #6** reports only the DiffusionGemma rows (v1.4: #21 at 36.85 with the NVFP4
  checkpoint on vLLM, 81.8% on the public items and 28.6% sealed; #54 for the thinking row). No
  benchmark result upstream publishes covers Verdict, Laya or the TypeSafe subset; SemIf publishes
  the subset's Jev value (0.883, which the harness recomputes from TypeSafe's snapshots) and its own
  Qwen3.5-4B scorer's, 0.845. Neither publishes a DiffusionGemma figure for the subset; these runs
  give 0.888 (Swift) and 0.892 (upstream).

### Speed, which this is not a benchmark of

On the encoders, one request at a time, the Swift server reads on the GPU and upstream on the CPU,
on a MacBook Pro that was doing other work, so the latency columns say little about a loaded server
and move between reruns, while the scores do not. The Swift server's 95th percentile shows something
else: on a Mac the encoder kept two Core ML functions loaded when these runs were recorded
(`functionCapacity`, D-037 item 3), one per input shape, and one-question requests whose lengths
moved among three or four shapes loaded a function again. The function-load table above counts them
in the recorded runs: Laya loaded a function 19 times in its 333 requests, 4 first loads and 15
reloads, and Verdict's requests used three shapes, each loaded once. A load took about half a second
to a second, several times an ordinary read, and a function's first load varied between 0.2 and 1.9
s across the runs on this Mac. The answers do not change. Since D-042 a Mac keeps every function
loaded once a read has needed it, each holding its own copy of the weights, 805 MB for Laya and 289
MB for Verdict ([spikes/encoder-function-capacity.md](spikes/encoder-function-capacity.md)), unless
`OPENJEV_ENCODER_FUNCTIONS` caps them; a new run records the number its server kept, and the table
above simulates that number.

The DiffusionGemma runs' timings are left out of the tables (`UNTIMED` in `report.py`). Another
worktree's Xcode tests, using 15 to 33 GB, ran beside the Swift runs, and the first Swift run
swapped, so the times say nothing about either server; the result files keep them. Their model times
would not compare either: upstream's MLX engine writes `model;dur=0.0`, and the Swift server's sums
the reads it runs in parallel (D-038 item 7). The read baseline of both servers on this Mac,
measured under a protocol, is in [benchmarks.md](benchmarks.md) (D-044).

The JevK5 runs' timings are left out too. They were not taken under that protocol: a second run of
the 8-bit conversion, in a new process, gave every answer bit for bit and a median 36% slower. The
author's run was timed on another machine and runtime. [deployment.md](deployment.md#jevk5) gives
the times these runs saw, as a guide.

## Rerunning every table

From the repository root, on a Mac with the converted packages in
`~/Library/Caches/OpenJevSwift/encoders` (or without `--encoder-models`, which downloads the
published packages, the same bytes) and the 4-bit checkpoint in the Hugging Face cache (either
server downloads it on first use, 16.58 GB). The four encoder runs take about five minutes, each
DiffusionGemma run about twenty; run the two DiffusionGemma servers one after the other, since each
loads about 16 GB:

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
python3 Tools/jevbench/servers.py --server swift --backend mlx --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --force
python3 Tools/jevbench/servers.py --server upstream --backend mlx --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --force
python3 Tools/jevbench/harness.py report
python3 Tools/jevbench/harness.py calibration
```

The JevK5 runs need the three conversions, which `Tools/jevk5/convert.py` makes from the checkpoint
in its own environment ([Tools/jevk5/README.md](../Tools/jevk5/README.md); 2.4, 4.5 and 8.4 GB),
and the author's run, which `author-run` downloads and checks. Each run takes about three minutes
for JevBench and six for TypeSafe:

```bash
for bits in 4 8 16; do ~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/convert.py --bits $bits; done
python3 Tools/jevbench/harness.py author-run --force
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --force
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --jevk5-model ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-4bit --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --output-dir Tools/jevbench/results/jevk5-conversions/4bit --force
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --jevk5-model ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-bf16 --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --output-dir Tools/jevbench/results/jevk5-conversions/bf16 --force
```

The scores are deterministic: six sets of encoder runs on this machine, on three builds, gave the
same answers to the last digit, and two DiffusionGemma runs of the Swift server, in separate
processes with and without the memory cap, gave all 231 JevBench answers bit for bit (the earlier
runs are not committed); only the timings move. Each comparison also prints item by item:

```bash
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/verdict-1.4-swift.json Tools/jevbench/results/verdict-1.4-upstream.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/typesafe102/laya-1.0-swift.json Tools/jevbench/results/typesafe102/laya-1.0-upstream.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/openjev-0.1-swift.json Tools/jevbench/results/openjev-0.1-upstream.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/typesafe102/openjev-0.1-swift.json Tools/jevbench/results/typesafe102/openjev-0.1-upstream.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/jevk5-0.2-swift.json Tools/jevbench/results/jevk5-0.2-author.json
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/jevk5-conversions/4bit/jevk5-0.2-swift.json Tools/jevbench/results/jevk5-0.2-author.json
python3 Tools/jevbench/harness.py published Tools/jevbench/results/verdict-1.4-swift.json Tools/jevbench/results/openjev-0.1-swift.json Tools/jevbench/results/openjev-0.1-upstream.json
python3 Tools/jevbench/harness.py summary Tools/jevbench/results/*.json
```

## DiffusionGemma

Issue #61 asked first for this comparison: the Swift server on DiffusionGemma 4-bit against
upstream's Python MLX server on the same machine and weights, with the same harness. It waited for
the backend's parity tests (issue #31, D-044), and these runs record it.

- **Upstream's environment** is `Tools/jevbench/.venv`, whose lock carries upstream's `mlx` extra
  since issue #32 (mlx-vlm 0.6.15 on MLX 0.32.2, D-044 item 5). `servers.py` points upstream at the
  checkpoint's snapshot in the Hugging Face cache, and the Swift server resolves the same snapshot,
  so neither downloads the weights.
- **The memory cap.** Without `OPENJEV_MLX_CACHE_LIMIT_GB`, MLX keeps the GPU buffers a read frees
  for reuse, up to its memory limit, as upstream's README says under "MLX memory". The first Swift
  run had no cap. On JevBench's hard tier the server's footprint grew to 102 GB of the Mac's 128 GB,
  beside another worktree's tests holding up to 33 GB; swap reached 73 GB and the disk had 24 GB
  left, so that run was stopped in its TypeSafe rows and made again with `--setting
  OPENJEV_MLX_CACHE_LIMIT_GB=4`, which kept the server at 19 to 24 GB. The cap changes no answer:
  the capped JevBench run gave the uncapped one's 231 answers bit for bit, from the same binary in a
  new process. Upstream ran with the same cap, at up to 25 GB. A deployment that reads long prompts
  should set a cap ([deployment.md](deployment.md)). The settings line of the server that ran,
  `2414408`, did not print it, so its log cannot show whether a run had one; the result files
  record it, and since PR #106 the settings line prints `mlx_cache_limit_gb`.
- **What is recorded** is the capped runs: `openjev-0.1-swift.json` and `openjev-0.1-upstream.json`
  for JevBench and their pair under `typesafe102/`. The uncapped run's JevBench file is not
  committed, since it is the same.

[A disagreement on DiffusionGemma](#a-disagreement-on-diffusiongemma) says what the comparison
shows, and the next section what the probabilities are worth.

## JevK5

Issue #55 asks JevK5 for the `jevk5` package's published v0.2 run on JevBench's 231 public items:
the same top answers and token counts, as upstream reached on vLLM, and the probability deviations
recorded, which upstream measured at 0.055 at most. Upstream's `jevk5` backend reads the letters
from a vLLM server, which needs an NVIDIA GPU, so the reference here is the author's run itself, as
it was for upstream.

- **The prompts are the author's.** On every conversion every item bills the author's token count,
  231 of 231, and the count is the prompt's: the system text, the evidence and the options rendered
  through the pinned template, then tokenized. `Fixtures/jevk5` holds the prompts byte for byte and
  the readout bit for bit on recorded logits (D-052), so what is left is the model's logits.
- **The 8-bit conversion meets the criterion except at one near-tie.** It changes one top answer,
  `hard-sol-a-multi_hop-10`, where the author gives `approve_45_days` 0.507 and the next 0.467, and
  the Swift server `reduce_to_30_days` 0.528. Its largest difference is 0.083, on
  `hard-sol-b-long_policy-05`, an answer it keeps.
- **The bfloat16 weights isolate the port from the quantization.** They change three top answers,
  each a near-tie in the author's run: two exact ties, whose top two labels the author's run gives
  the same probability (`hard-opus-b-tradeoff-07` and `hard-sol-a-multi_hop-12`), and one at 0.030
  (`hard-opus-b-ambiguous-10`). Their largest difference, 0.051, is near what upstream saw on vLLM.
  On the same 4-bit weights the Swift model's letter logits are mlx-lm's within 0.375, 0.068 on
  average, over the fixture's 324 passes (the live tests, D-052): two implementations ordering
  bfloat16 arithmetic differently.
- **The 4-bit conversion is the one that moves answers.** It changes 22 top answers, 18 of them on
  items whose author top two are at least 0.05 apart and four at margins above 0.6, such as
  `original-routing-02-0` (the author's `coding` at 0.818, the Swift server's `coding_agent` at
  0.499). On TypeSafe's rows it loses five points of accuracy to the other two. It is half the
  8-bit conversion's size, the one an iPhone app would load, and a Mac server should not.
- **The author's 86.1%.** Upstream's README says its run and the author's both scored 86.6%, 200 of
  the 231 items. The author's records score 199: two items are exact ties in the author's run,
  `original-policy-03-1`, a noul at exactly 0.5, and `hard-sol-a-multi_hop-12`, and the author's
  runner broke both against the expected answer. A scorer that breaks either tie the other way
  counts 200. These tables score every run from its own records.
- **What is recorded.** `jevk5-0.2-swift.json` and `typesafe102/jevk5-0.2-swift.json` are the
  8-bit conversion's runs, the server's default; `jevk5-conversions/4bit/` and
  `jevk5-conversions/bf16/` hold the other two, each with its TypeSafe run; `jevk5-0.2-author.json`
  is the author's run as `harness.py author-run` reads it. All five Swift runs used one build, and
  the 8-bit run, repeated in a new process, gave all 231 answers bit for bit.

## Calibration of DiffusionGemma's reads

Issue #62. Upstream applies no calibration to DiffusionGemma: an answer's probabilities are the
model's own. Each read takes a softmax over the label tokens' log-probabilities at the answer's slot
(`slot_distribution` in upstream's `engine.py`), and the answer is the mean of the reads: the first,
and three more with other noise when any slot's top-k entropy exceeds 0.1 (`read_group`,
`OPENJEV_AUTO_THRESHOLD` and `OPENJEV_AUTO_MAX`). A choice or score answer's `confidence` is `1 -
H(p)/ln K`: how peaked the distribution is, 1 when certain and 0 when uniform, not how often such an
answer is right. A noul answer carries only `noul`, P(yes). Upstream's README calls the decisions
calibrated and, under Caveats, asks users to evaluate the model on their own tasks.

`python3 Tools/jevbench/harness.py calibration` renders the tables below from the result files the
comparison above uses, over every item with an expected label: JevBench's 231 items, whose labels
are the benchmark's, and the 102 TypeSafe rows, whose label is the top option of the reference
distribution, the mean of the reference answers TypeSafe's evaluation publishes for the question
(SemIf's `build_typesafe.py`), so that a TypeSafe "accuracy" is agreement with that reference. The
formulas, each from its source (`Tools/jevbench/calibration.py` cites them):

- **Brier**: the mean over items of `sum_k (p_k - y_k)^2` over the item's labels, JevBench's
  `summarize.metric`; for a noul, `2 (p_yes - y)^2`. It is the Brier column of the runs table.
- **ECE**: `sum_b (n_b / N) |acc_b - conf_b|` over 10 equal-width bins of the top label's
  probability, a value `c` falling in bin `min(floor(10 c), 9)`: JevBench's `metrics.ece_top_label`,
  and the same definition as SemIf's `calibrate.ece`. It is the ECE column of the runs table, and
  the reliability tables are its bins.
- **NLL**: `-(1/N) sum_i ln max(p_i(expected), 1e-12)`, SemIf's `calibrate.mean_nll`, which
  temperature scaling minimises.
- **AUROC**: the probability that a right answer's value is higher than a wrong one's, ties counting
  one half. It measures how well the top probability, or `confidence`, ranks right answers above
  wrong ones, which a temperature cannot change for two labels.
- **`confidence`**: upstream's `1 - H(p)/ln K` with `H(p) = -sum_k p_k ln p_k`, as the answer
  carries it; for a noul, computed from `(p_yes, p_no)` with the same formula.
- **Temperature scaling**, SemIf's method ([docs/CALIBRATION.md and benchmarks/calibrate.py at
  `23cf1f3`](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/23cf1f39fc9534fe81437200959b6dfc7106e45a/docs/CALIBRATION.md)):
  `q = softmax(z / T)` with one scalar T per workload, fitted by golden-section search over [0.05,
  20] to minimise the NLL; the ECE is reported out of fold, each of 5 folds scored with the T fitted
  on the other four, the folds dealt by group (a JevBench paraphrase pair stays together; a TypeSafe
  row's group is its case), with 95% bootstrap intervals from 1,000 resamples of the groups. An
  answer carries no logits, so `z = ln p` and `q_k = p_k^(1/T) / sum_j p_j^(1/T)`. Dividing by T
  keeps the order of the labels, so accuracy does not move. A T is fitted on the hard-labelled
  JevBench runs only: SemIf leaves rows with a reference distribution, as TypeSafe's are, out of its
  fits, and here JevBench's T is applied to them instead. Two departures from SemIf's script: the
  folds and resamples come from Python's `random` with SemIf's seed 217, not NumPy's, since the
  harness uses only the standard library, so the assignment of groups differs from what SemIf's
  script would draw; and beside SemIf's test (the two unpaired ECE intervals do not overlap) the
  tables give the paired interval of each change, over the same resampled groups.

### The raw reads

| dataset | server | items | accuracy | mean p(top) | Brier | ECE | NLL | AUROC p(top) | AUROC confidence | mean confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| jevbench | swift | 231 | 81.4% | 0.920 | 0.2452 | 0.1060 | 0.566 | 0.870 | 0.878 | 0.830 |
| jevbench | upstream | 231 | 82.3% | 0.919 | 0.2385 | 0.0966 | 0.563 | 0.865 | 0.867 | 0.827 |
| typesafe102 | swift | 102 | 89.2% | 0.973 | 0.1941 | 0.0919 | 0.546 | 0.775 | 0.758 | 0.909 |
| typesafe102 | upstream | 102 | 90.2% | 0.972 | 0.1892 | 0.0962 | 0.541 | 0.742 | 0.723 | 0.909 |

Reliability on jevbench: the probability of the chosen answer against how often it is right, 10
equal-width bins:

| p(top) | swift n | swift mean p | swift accuracy | upstream n | upstream mean p | upstream accuracy |
|---|---|---|---|---|---|---|
| 0.0 to 0.1 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.1 to 0.2 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.2 to 0.3 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.3 to 0.4 | 3 | 0.343 | 33.3% | 5 | 0.372 | 20.0% |
| 0.4 to 0.5 | 6 | 0.417 | 16.7% | 5 | 0.433 | 20.0% |
| 0.5 to 0.6 | 9 | 0.552 | 11.1% | 5 | 0.561 | 20.0% |
| 0.6 to 0.7 | 9 | 0.647 | 33.3% | 12 | 0.645 | 33.3% |
| 0.7 to 0.8 | 8 | 0.727 | 37.5% | 10 | 0.746 | 40.0% |
| 0.8 to 0.9 | 14 | 0.838 | 50.0% | 19 | 0.852 | 68.4% |
| 0.9 to 1.0 | 182 | 0.992 | 94.5% | 175 | 0.995 | 94.9% |

Reliability on typesafe102: the probability of the chosen answer against how often it is right, 10
equal-width bins:

| p(top) | swift n | swift mean p | swift accuracy | upstream n | upstream mean p | upstream accuracy |
|---|---|---|---|---|---|---|
| 0.0 to 0.1 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.1 to 0.2 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.2 to 0.3 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.3 to 0.4 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.4 to 0.5 | 0 | n/a | n/a | 0 | n/a | n/a |
| 0.5 to 0.6 | 2 | 0.573 | 50.0% | 2 | 0.540 | 100.0% |
| 0.6 to 0.7 | 1 | 0.666 | 100.0% | 1 | 0.602 | 100.0% |
| 0.7 to 0.8 | 1 | 0.785 | 100.0% | 0 | n/a | n/a |
| 0.8 to 0.9 | 2 | 0.834 | 50.0% | 6 | 0.864 | 50.0% |
| 0.9 to 1.0 | 96 | 0.990 | 90.6% | 93 | 0.993 | 92.5% |

By tier, with the temperature each tier alone would be fitted to:

| dataset | tier | server | items | accuracy | mean p(top) | ECE | Brier | NLL | the tier's own T |
|---|---|---|---|---|---|---|---|---|---|
| jevbench | easy | swift | 48 | 100.0% | 0.999 | 0.0011 | 6.3e-06 | 0.001 | none: every answer is right |
| jevbench | standard | swift | 72 | 98.6% | 0.990 | 0.0160 | 0.0141 | 0.023 | 0.80 |
| jevbench | hard | swift | 111 | 62.2% | 0.841 | 0.2189 | 0.5011 | 1.163 | 2.91 |
| jevbench | easy | upstream | 48 | 100.0% | 0.999 | 0.0012 | 9.8e-06 | 0.001 | none: every answer is right |
| jevbench | standard | upstream | 72 | 98.6% | 0.988 | 0.0182 | 0.0163 | 0.027 | 0.81 |
| jevbench | hard | upstream | 111 | 64.0% | 0.840 | 0.2001 | 0.4858 | 1.154 | 2.90 |

### Confidence and accuracy by question type

| dataset | type | server | items | accuracy | mean p(top) | mean confidence | ECE | Brier | AUROC confidence |
|---|---|---|---|---|---|---|---|---|---|
| jevbench | noul | swift | 74 | 79.7% | 0.945 | 0.824 | 0.1481 | 0.2950 | 0.747 |
| jevbench | choice | swift | 139 | 83.5% | 0.911 | 0.838 | 0.0763 | 0.2168 | 0.940 |
| jevbench | score | swift | 18 | 72.2% | 0.885 | 0.802 | 0.1689 | 0.2599 | 1.000 |
| jevbench | noul | upstream | 74 | 81.1% | 0.945 | 0.820 | 0.1339 | 0.2836 | 0.726 |
| jevbench | choice | upstream | 139 | 83.5% | 0.909 | 0.833 | 0.0742 | 0.2090 | 0.944 |
| jevbench | score | upstream | 18 | 77.8% | 0.895 | 0.810 | 0.1888 | 0.2809 | 0.929 |
| typesafe102 | noul | swift | 66 | 90.9% | 0.971 | 0.884 | 0.0839 | 0.1591 | 0.694 |
| typesafe102 | choice | swift | 36 | 86.1% | 0.977 | 0.954 | 0.1389 | 0.2583 | 0.884 |
| typesafe102 | noul | upstream | 66 | 92.4% | 0.972 | 0.886 | 0.0726 | 0.1521 | 0.633 |
| typesafe102 | choice | upstream | 36 | 86.1% | 0.973 | 0.951 | 0.1393 | 0.2573 | 0.852 |

By option count (upstream's runs):

| dataset | type | options | items | accuracy | mean p(top) | mean confidence | ECE | AUROC confidence |
|---|---|---|---|---|---|---|---|---|
| jevbench | noul | 2 | 74 | 81.1% | 0.945 | 0.820 | 0.1339 | 0.726 |
| jevbench | choice | 3 | 15 | 46.7% | 0.821 | 0.592 | 0.3876 | 0.821 |
| jevbench | choice | 4 | 53 | 88.7% | 0.915 | 0.837 | 0.0786 | 0.968 |
| jevbench | choice | 5 | 55 | 87.3% | 0.942 | 0.907 | 0.0694 | 0.964 |
| jevbench | choice | 6 | 16 | 87.5% | 0.854 | 0.790 | 0.0742 | 1.000 |
| jevbench | score | 4 | 17 | 82.4% | 0.913 | 0.834 | 0.1658 | 0.929 |
| jevbench | score | 5 | 1 | 0.0% | 0.579 | 0.405 | 0.5787 | n/a |
| typesafe102 | noul | 2 | 66 | 92.4% | 0.972 | 0.886 | 0.0726 | 0.633 |
| typesafe102 | choice | 2 | 2 | 100.0% | 0.994 | 0.946 | 0.0063 | n/a |
| typesafe102 | choice | 3 | 3 | 100.0% | 0.995 | 0.972 | 0.0052 | n/a |
| typesafe102 | choice | 4 | 8 | 100.0% | 0.997 | 0.988 | 0.0025 | n/a |
| typesafe102 | choice | 5 | 13 | 69.2% | 0.932 | 0.898 | 0.3156 | 0.806 |
| typesafe102 | choice | 6 | 6 | 100.0% | 0.997 | 0.990 | 0.0027 | n/a |
| typesafe102 | choice | 8 | 4 | 75.0% | 0.994 | 0.983 | 0.2440 | 1.000 |

Accuracy by the answer's `confidence` (upstream's runs; a noul's is computed):

| dataset | confidence | noul n | noul accuracy | choice n | choice accuracy | score n | score accuracy |
|---|---|---|---|---|---|---|---|
| jevbench | 0.0 to 0.2 | 8 | 25.0% | 8 | 12.5% | 1 | 100.0% |
| jevbench | 0.2 to 0.4 | 3 | 100.0% | 7 | 42.9% | 1 | 0.0% |
| jevbench | 0.4 to 0.6 | 3 | 66.7% | 13 | 30.8% | 3 | 0.0% |
| jevbench | 0.6 to 0.8 | 3 | 66.7% | 11 | 90.9% | 0 |  |
| jevbench | 0.8 to 1.0 | 57 | 89.5% | 100 | 98.0% | 13 | 100.0% |
| typesafe102 | 0.0 to 0.2 | 2 | 100.0% | 0 |  | 0 |  |
| typesafe102 | 0.2 to 0.4 | 0 |  | 0 |  | 0 |  |
| typesafe102 | 0.4 to 0.6 | 4 | 75.0% | 2 | 50.0% | 0 |  |
| typesafe102 | 0.6 to 0.8 | 9 | 88.9% | 1 | 0.0% | 0 |  |
| typesafe102 | 0.8 to 1.0 | 51 | 94.1% | 33 | 90.9% | 0 |  |

### Temperature scaling fitted offline

| dataset | server | items | groups | fitted T | fold Ts | ECE at T = 1 (95%) | ECE out of fold (95%) | intervals separate |
|---|---|---|---|---|---|---|---|---|
| jevbench | swift | 231 | 195 | 2.01 | 1.85 to 2.18 | 0.1060 (0.071 to 0.153) | 0.0627 (0.042 to 0.110) | no |
| jevbench | upstream | 231 | 195 | 2.00 | 1.86 to 2.16 | 0.0966 (0.062 to 0.139) | 0.0581 (0.042 to 0.106) | no |

Out of fold against T = 1 on the same items, with the paired 95% interval of each change over the
same resampled groups (below 0 is better):

| dataset | server | ECE change (95%) | Brier | Brier change (95%) | NLL | NLL change (95%) | mean p(top) |
|---|---|---|---|---|---|---|---|
| jevbench | swift | -0.063 to -0.005 | 0.2452 to 0.2266 | -0.035 to -0.003 | 0.566 to 0.446 | -0.226 to -0.036 | 0.920 to 0.842 |
| jevbench | upstream | -0.057 to 0.010 | 0.2385 to 0.2249 | -0.029 to 0.002 | 0.563 to 0.446 | -0.224 to -0.031 | 0.919 to 0.841 |

The T fitted on JevBench applied to the TypeSafe rows, scored against the reference's top option:

| dataset | server | JevBench's T | rows | ECE, T = 1 to JevBench's T | Brier | NLL |
|---|---|---|---|---|---|---|
| typesafe102 | swift | 2.01 | 102 | 0.0919 to 0.0367 | 0.1941 to 0.1743 | 0.546 to 0.350 |
| typesafe102 | upstream | 2.00 | 102 | 0.0962 to 0.0347 | 0.1892 to 0.1708 | 0.541 to 0.349 |

One T for every type against a T per type, out of fold on the same folds:

| dataset | server | type | items | the type's own T | ECE at T = 1 | ECE own T, out of fold | ECE one T, out of fold | one minus own, 95% |
|---|---|---|---|---|---|---|---|---|
| jevbench | swift | noul | 74 | 3.65 | 0.1481 | 0.0903 | 0.1221 | -0.061 to 0.076 |
| jevbench | swift | choice | 139 | 1.63 | 0.0763 | 0.0451 | 0.0553 | -0.029 to 0.040 |
| jevbench | swift | score | 18 | 1.40 | 0.1689 | 0.1635 | 0.1867 | -0.018 to 0.053 |
| jevbench | upstream | noul | 74 | 3.66 | 0.1339 | 0.0603 | 0.0994 | -0.053 to 0.064 |
| jevbench | upstream | choice | 139 | 1.60 | 0.0742 | 0.0416 | 0.0759 | -0.017 to 0.060 |
| jevbench | upstream | score | 18 | 1.48 | 0.1888 | 0.1879 | 0.2086 | -0.014 to 0.045 |

### What the calibration shows

The two servers' figures are within a few items of each other throughout, so what follows is a
property of DiffusionGemma's reads, not of the port.

- **The reads are overconfident on hard questions and calibrated on easy ones.** On JevBench the
  answers are right 81% to 82% of the time at a mean top probability of 0.92: an ECE of 0.106
  (Swift) and 0.097 (upstream), above the two calibrated encoders' 0.079 (Verdict) and 0.077 (Laya)
  although the accuracy is far higher. The 175 to 182 answers at 0.9 or more are right 94.5% to
  94.9% of the time at a mean of 0.99; the 49 to 56 below 0.9, at a mean of 0.65 to 0.68, are right
  33% to 43% of the time, and the bins from 0.3 to 0.8 hold answers that are right 11% to 40% of the
  time. The easy and standard tiers are calibrated (ECE 0.001 to 0.018, 98.6% to 100% right at
  0.99), and the hard tier is right 62% to 64% of the time at 0.84 (ECE 0.20 to 0.22). On TypeSafe
  the reads agree with the reference 89% to 90% of the time at 0.97 (ECE 0.092 to 0.096), seven to
  eight points too sure as well.
- **By question type and option count.** JevBench's choices are the best calibrated type (ECE 0.074
  to 0.076 at 83.5% right), its nouls (0.13 to 0.15 at 80% to 81%) and its 18 scores (0.17 to 0.19
  at 72% to 78%) less so. The 15 three-option choices, all hard-tier probability, ambiguity and
  temporal questions, are right 46.7% of the time at 0.82 (ECE 0.39), where four to six options are
  right 87% to 89% of the time. The option count is not the cause, the questions are: on TypeSafe
  the five-option choices are the overconfident ones (69% at 0.93).
- **`confidence` ranks answers, and is not a probability.** Its AUROC is 0.87 to 0.88 on JevBench
  and 0.72 to 0.76 on TypeSafe, about that of the top probability. At a confidence of 0.8 or more,
  89.5% of JevBench's nouls, 97% to 98% of its choices and 13 of its 13 scores are right; below 0.6,
  about 30% of its choices are. The values themselves do not read as probabilities: a noul at 0.75
  against 0.25 has a confidence of 0.19, and a choice of five at 0.92 with the rest spread evenly
  one of 0.76. A deployment that gates on `confidence` should choose the threshold on its own
  labelled items.
- **A temperature of about 2 helps, modestly.** The T fitted on JevBench is 2.01 (Swift) and 2.00
  (upstream), 1.85 to 2.18 over the folds. Out of fold it lowers the ECE from 0.106 to 0.063 and
  from 0.097 to 0.058, the Brier score by 0.014 to 0.019 and the NLL from 0.56 to 0.45, and applied
  to the TypeSafe rows it lowers their ECE from 0.092 to 0.037 and from 0.096 to 0.035. The NLL's
  paired interval excludes zero on both servers' runs; the ECE's and the Brier score's do on the
  Swift run's and just include it on upstream's; and SemIf's own test, the two ECE intervals not
  overlapping, fails on both. So a scalar T is a real but small gain on these sets, at their sizes.
- **The temperature belongs to the workload.** The tiers alone would be fitted to 0.8 (standard,
  slightly underconfident) and 2.9 (hard), and every easy answer is right; the types to about 3.7
  (noul), 1.6 (choice) and 1.4 to 1.5 (score), which 74, 139 and 18 items cannot separate from one T
  (every paired interval of the per-type control includes zero). One T fitted on a benchmark would
  be too weak for a harder workload and would flatten an easier one's answers, which are right and
  should stay sharp.

D-046 records the decision these numbers support: the server keeps upstream's probabilities, with no
temperature-scaling hook, and a deployment that wants calibrated probabilities fits its own T
offline and rescales the answers it receives. With `T` from `harness.py calibration` over its own
labelled items (run with `harness.py run --items`), a client computes `q_k = p_k^(1/T) / sum_j
p_j^(1/T)` from a choice's or score's `probabilities`, or from `noul` and `1 - noul`, and from `q`
the `confidence` (`1 - H(q)/ln K`) and a score's expected level (`sum_i i q_i`). It keeps the
argmax, so the answer does not change.
