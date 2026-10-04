# Benchmarks: DiffusionGemma reads

The performance and memory baseline of issue #32, measured with `openjev-bench`
(`Sources/openjev-bench`, [development.md](development.md#the-read-benchmark)) against the Swift
port and against upstream's Python MLX backend on the same Mac and weights. The raw runs are in
[Tools/bench/results](../Tools/bench/results); the choices behind the method are D-044 in
[06-decisions.md](06-decisions.md).

## The machines

| Machine | Memory | macOS | Measured |
|---|---|---|---|
| Apple M3 Max (Mac15,9), 12 performance and 4 efficiency CPU cores, a MacBook Pro | 128 GB | 27.0.1 (26A434) | 2026-10-01, every table below |
| A Mac with 32 GB or 48 GB | 32 or 48 GB | | not measured: none was available. R4 in [07-risks-and-unknowns.md](07-risks-and-unknowns.md) carries the same note |

Every figure on this page is from the M3 Max, on AC power, with the 4-bit checkpoint
(`mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407`), mlx-swift 0.32.2, the release build
Xcode makes (a user scheme with a Release run configuration), and upstream at `dcd2094` on
mlx-vlm 0.6.15 and MLX 0.32.2 from `Tools/jevbench/.venv`. The package has since moved to
mlx-swift 0.32.3 and mlx-swift-lm 3.32.3 (D-053), and nothing here was measured again: 0.32.3
changes one logging call, the kernels are unchanged (the vendored MLX core is the same, and so is
the SHA-256 of the compiled `default.metallib`), and the reads came out bit for bit the same.

## How it was measured

- **Requests.** `reads` times 1, 3 and 12 questions (upstream's README urgent question; its three
  README questions; those three and nine nouls), each with `"samples": 1` and a state no earlier
  request sent ("Ticket {tag}-{n}: Everything is down..."), so every request prefills: 5 untimed
  requests, then 50 timed ones, one at a time. `concurrency` sends 48 three-question requests per
  level from 1, 4 or 16 concurrent callers. p50 and p95 are NumPy's linear percentiles.
- **Swift and upstream the same way.** Over HTTP, `openjev-bench --url` sends the same bodies to
  any `/v1/systemone` server and records the HTTP time and `server-timing`. `Tools/jevbench/servers.py
  --command` starts each server as the JevBench runs do (127.0.0.1, warm-up on) and runs the bench
  against it. Upstream's MLX engine writes `model;dur=0.0` in `server-timing`, so only its HTTP
  time is comparable. In process, `Tools/bench/upstream_stages.py` times upstream's own
  `MlxRuntime` prefill and read the way `openjev-bench prefill` and `profile` time the port's.
- **Hygiene.** Each run started only when `pgrep -fl 'swift-build|xcodebuild|openjev|python'` and
  a few more patterns (Xcode's test runners, Docker, another worktree's servers) showed nothing but
  the run itself, on AC power, after two minutes idle and with the thermal state
  (`ProcessInfo.thermalState`) at nominal; every run records the thermal state and the power
  source at its start and end. A watcher discarded any run during which another build, test or server appeared, or the Mac
  went on battery, and ran it again.
- **Why the protocol.** The first runs of the day were taken right after one another and
  disagreed by up to 80%: the same three-question read took 302 ms at nominal and 380 to 539 ms
  once heavy runs (10,000-token prefills) had brought the M3 Max laptop to the fair thermal state,
  and Swift's and upstream's servers swapped places depending on which ran first. Those runs, and
  three that overlapped another session's model server, a Docker build or the Mac's switch to
  battery, are not reported.

## Read latency

`openjev-bench reads`, the engine in process (`DecisionEngine` over `DiffusionGemmaRuntime`).
Thermal state nominal at start and end. M3 Max, 128 GB, macOS 27.0.1, AC power.

| Questions | p50 ms | p95 ms | mean ms | model time p50 ms | input tokens |
|---|---|---|---|---|---|
| 1 | 209.8 | 245.1 | 214.7 | 208.6 | 112 |
| 3 | 301.6 | 327.7 | 304.6 | 299.5 | 180 |
| 12 | 517.5 | 550.6 | 519.7 | 515.5 | 354 |

Upstream's README reports 0.2 to 0.4 s for a three-question read on an M3 Ultra and 0.39 s on an
M4 Max. The port's 0.30 s on an M3 Max is within that range. The engine adds 1 to 2 ms to the
model time.

## Swift against upstream over HTTP

`openjev-bench reads --url` against each server in turn, Swift first, each started by
`servers.py`. M3 Max, 128 GB, macOS 27.0.1, AC power, nominal thermal state at each start.

| Server, round | 1 question p50 / p95 ms | 3 questions p50 / p95 ms | 12 questions p50 / p95 ms | thermal state at end |
|---|---|---|---|---|
| Swift `openjev serve`, 1 | 208.9 / 242.8 | 289.6 / 318.8 | 482.1 / 552.4 | fair |
| upstream `python -m openjev`, 1 | 210.0 / 248.4 | 302.1 / 1,216.8 | 495.6 / 708.4 | nominal |
| Swift `openjev serve`, 2 | 210.6 / 238.6 | 311.8 / 391.1 | 492.8 / 1,969.2 | fair |

At the median the two servers are within 4% of each other on every request size, and the order
changes between rounds: for three questions Swift's first round was 4% faster than upstream's and
its second 3% slower, so the difference is within the run-to-run spread. Upstream's p95 for three
questions (1.2 s) and Swift's round-2 p95 for twelve (2.0 s) come from a few slow requests among
the 50 on either server, not from a slower median. The HTTP exchange adds about 2 ms to the model
time on the Swift server (`server-timing` `model` p50 286.9 ms against 289.6 ms over HTTP for three
questions).

## Concurrency

`openjev-bench concurrency`, the engine in process, 48 three-question requests per level, each
with its own state. Nominal thermal state at start and end. M3 Max, 128 GB, macOS 27.0.1, AC
power.

| Concurrent callers | p50 ms | p95 ms | requests/s |
|---|---|---|---|
| 1 | 291.6 | 325.3 | 3.39 |
| 4 | 1,283.5 | 1,337.5 | 3.13 |
| 16 | 5,189.4 | 5,386.4 | 3.02 |

The runtime runs one read at a time on the GPU (R14), so concurrency buys no throughput: callers
queue, and latency grows with the queue. Upstream's README reports about 4 requests/s for 16
concurrent requests on its machines, which also read one at a time. The engine's CPU work for one
request overlaps the GPU work for another, so throughput does not fall below the serial rate
either; it drifts from 3.4 to 3.0 requests/s.

## Memory

`openjev-bench memory`: `memoryReport()` after loading (warm-up included), after 20 reads and after
200 unique three-question prompts, without a cache limit and with `cacheLimitGB` 4. Each in its own
process. M3 Max, 128 GB, macOS 27.0.1, AC power.

| `cacheLimitGB` | When | MLX active GiB | MLX cache GiB | MLX peak GiB | resident GiB | cached prefills |
|---|---|---|---|---|---|---|
| unset | after load | 14.35 | 1.32 | 15.41 | 15.72 | 0 |
| unset | after 20 reads | 14.83 | 2.19 | 15.41 | 15.72 | 12 |
| unset | after 200 unique prompts | 14.83 | 2.48 | 15.41 | 15.73 | 12 |
| 4 | after load | 14.35 | 1.32 | 15.41 | 15.72 | 0 |
| 4 | after 20 reads | 14.83 | 2.19 | 15.41 | 15.72 | 12 |
| 4 | after 200 unique prompts | 14.83 | 2.48 | 15.41 | 15.72 | 12 |

Short prompts fill the prefill cache's 12 entries by the 12th request and then hold MLX's live
arrays at 14.83 GiB; the buffer pool grows to 2.48 GiB and stops there, below the 4 GiB limit, so
the limit changes nothing for this workload. The resident size (15.7 GiB) does not count all of
MLX's buffers: active plus pool is 17.3 GiB here, which is the figure a machine needs room for, and
more for long prompts (R4: a cached prefill costs about
0.22 MB per prompt token, up to the 16,384-token budget).

## Prefill

`openjev-bench prefill`: per size, a one-question read whose state was never sent (cold) and the
same read again from the prefill cache (cached); the prefill is the difference of the medians, 5
pairs after one warm-up pair. Upstream's row is `Tools/bench/upstream_stages.py` on its own
runtime (`diffusion_prefill_cache` evaluated alone), 5 runs after 2 warm-up runs. M3 Max, 128 GB,
macOS 27.0.1, AC power, nominal thermal state at each start (fair at the end of the Swift run; the
Python script does not record it).

| State | Swift prompt tokens | Swift prefill ms | Swift tokens/s | upstream prompt tokens | upstream prefill ms | upstream tokens/s |
|---|---|---|---|---|---|---|
| quickstart-sized | 180 | 206.2 | 873 | 182 | 201.1 | 905 |
| about 1,000 tokens | 1,091 | 838.4 | 1,301 | 1,002 | 782.0 | 1,281 |
| about 10,000 tokens | 10,091 | 13,330.8 | 757 | 10,002 | 14,118.2 | 708 |

The 180-token row is the profile's unstaged prefill below. The two implementations prefill at the
same rate within 5%. The rate falls from about 1,300 to about 750 tokens/s between 1,000 and
10,000 tokens: at 10,000 tokens 42% of the prefill time is above the 1,000-token rate, the
quadratic share of attention (follow-up issue below). A 10,000-token state costs about 13 s before
its first read; later reads of it come from the prefill cache in 70 ms.

## Where the time goes

`openjev-bench profile`: a three-question read of the 180-token README prompt (canvas 32), 20 runs
after 3 warm-up runs, each stage evaluated where the model's stage observer sees it. Evaluating at
each of the about 185 stage boundaries costs one GPU round trip each, 0.48 ms in the decoder pass
and 0.58 ms in the prefill on average (the staged total minus the unstaged one, over the number of
boundaries); the "corrected" column removes that cost, and the corrected stages add up to the
unstaged total within 2%. The staged read is bit-identical to the unstaged one. M3 Max, 128 GB,
macOS 27.0.1, AC power, nominal thermal state at start and end.

| Phase | Stage | staged ms | corrected ms | share of the unstaged pass |
|---|---|---|---|---|
| prefill | attention (projections, RoPE, SDPA, cache writes) | 74.4 | 57.2 | 27.7% |
| prefill | dense MLP | 38.7 | 21.4 | 10.4% |
| prefill | router | 30.5 | under 1 | within the method's error |
| prefill | experts (gathered quantized matmuls) | 143.3 | 126.0 | 61.1% |
| prefill | norms, residuals, layer scalars | 23.5 | 6.2 | 3.0% |
| prefill | **staged total / unstaged, as the runtime runs it** | 311.0 | 206.2 | |
| decoder pass | embedding and self-conditioning | 0.6 | 0.1 | 0.1% |
| decoder pass | attention (over the prompt cache) | 30.0 | 15.6 | 17.9% |
| decoder pass | dense MLP | 19.9 | 5.5 | 6.3% |
| decoder pass | router | 28.1 | under 1 | within the method's error |
| decoder pass | experts (gathered quantized matmuls) | 68.3 | 53.9 | 61.9% |
| decoder pass | norms, residuals, layer scalars | 19.7 | 5.3 | 6.1% |
| decoder pass | output projection (tied 262,144-row embedding) | 5.0 | 4.5 | 5.2% |
| decoder pass | softcap | 1.1 | 0.7 | 0.7% |
| decoder pass | slot log-softmax, top 20 and host copies | 2.9 | 2.4 | 2.7% |
| decoder pass | **staged total / unstaged, as the runtime runs it** | 175.9 | 87.2 | |

Upstream's own runtime on the same prompt size takes 201.1 ms to prefill and 82.3 ms to read
(`upstream_stages.py`), against the port's 206.2 and 87.2 ms. A cold three-question read is
therefore about two thirds prefill and one third decoder pass, and in both the mixture-of-experts
matmuls take three fifths of the GPU time.

The kernels are not the difference people expected (R5). With the mlx-metal wheel's
`mlx.metallib`, the library upstream runs (`openjev-bench profile --metallib`), the same profile
took 185.3 ms to prefill and 78.8 ms to read, against 185.2 and 74.7 ms with the package's own
library in the run before it: equal within run-to-run spread.

Instruments could not record on this machine: `xctrace record` 27.0 (27A266a) stops with
`Assertion failed: (_coreForNextRun != core)` in `XRAugmentationManager` for every template (Time
Profiler, CPU Profiler, Metal System Trace) and every target, `/bin/sleep` included, inside and
outside the sandbox, and writes an empty trace. The stage profile above is the GPU-side answer. A
host-side call tree (macOS's `sample` on a running `openjev-bench reads`) was queued but not taken:
the Mac went on battery for the rest of the session, and the protocol times only on AC power.

## Follow-up issues

Each optimisation with more than 10% headroom by the figures above has an issue in milestone 7
(label `area/diffusiongemma`):

- [#100](https://github.com/Algorythm-Canada/OpenJevSwift/issues/100) **The expert matmuls** take
  61% of the prefill and 62% of the decoder pass: measure the sort threshold (64 assignments,
  mlx-vlm's), the gathered-matmul path and compiling the expert block.
- [#101](https://github.com/Algorythm-Canada/OpenJevSwift/issues/101) **Long-prompt prefill**: the
  rate falls from 1,301 to 757 tokens/s from 1,000 to 10,000 tokens, and the sliding layers build a
  dense `[L, L]` boolean band mask, 100 million entries at 10,000 tokens (from the code; a profile
  at 10,000 tokens, `openjev-bench profile --state-tokens 10000`, was not taken).
- [#102](https://github.com/Algorythm-Canada/OpenJevSwift/issues/102) **Throughput under
  concurrency** stays at 3.0 to 3.4 requests/s from 1 to 16 callers; the decoder passes of queued
  requests (87 ms of a 300 ms read) could run as one batch.

Below 10%, so not filed: the slot-only projection (D-015; the output projection is 5.2% of a
decoder pass and about 1.5% of a cold read, and its result is not bit-identical), compiling the
softcap (0.7%), and the host round trips of the slot extraction (2.7% of a decoder pass, six
`asArray` copies for three slots) and of the separate prefill evaluation (one round trip, about
0.5 ms).

## Not measured

- A 32 GB or 48 GB Mac (none was available).
- Swift's and upstream's servers under concurrency over HTTP, a third `reads` round of each, the
  10,000-token stage profile and the `sample` call tree: queued under the protocol, but the Mac ran
  on battery from 22:20 to the end of the session. The in-process concurrency table above is the
  port's; upstream's README figure (about 4 requests/s at 16 concurrent, on an M3 Ultra or M4 Max)
  is the only upstream row.
- Instruments traces, for the reason above.
