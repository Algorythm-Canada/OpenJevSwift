# How many Core ML functions a Mac's encoder keeps loaded

Run on 2026-10-01 on the MacBook Pro of spike #56 (M3 Max, 128 GB, macOS 27.0.1), after issue #61's
JevBench runs found Laya loading Core ML functions again in the middle of a run. On a Mac, Verdict's
and Laya's backends run a multifunction package with one function per input shape (batch 1 and 16 by
128 to 512 tokens for Verdict, to 1,024 for Laya), and `CoreMLEncoderModel` kept the two used most
recently. This report counts the reloads, measures what a loaded function costs, serves both models
end to end with two functions and with all of them, and compares the options; the decision is D-042
in [06-decisions.md](../06-decisions.md). The raw results are in
[encoder-function-capacity/](encoder-function-capacity/), and the tools are
`Tools/encoders/function_capacity.py` and the harness's `encoder-capacity` command
([Tools/encoders](../../Tools/encoders/README.md)). MB and GB are 2^20 and 2^30 bytes throughout,
vmmap's units.

## Answer

- **The reloads are the cache's.** Replaying the run's 333 requests through a two-entry cache, each
  one-question request through the batch-1 function of its length after the warm-up's `b16_s128`,
  predicts 19 loads, 4 first loads and 15 reloads: in a rerun they were exactly the 19 requests
  whose model time was over 300 ms, and in issue #61's recorded run, taken while the Mac was busy,
  19 of its 31 (the other 12 loaded nothing).
- **The benchmark's order hides most of them.** JevBench sends its short tiers first: 120 requests
  at 128 tokens, then a mix of 256 to 1,024. In random orders of the same requests, two functions
  reload 36% of Laya's requests and 13% of Verdict's. Once one-question requests and batches mix,
  any capacity below the package's function count reloads: at 4, 30 to 37% of Laya's requests.
- **A reload costs Laya 0.44 to 1.85 s and Verdict 0.22 to 0.69 s,** where a Laya read took medians
  of 20 to 163 ms in issue #61's recorded run. `MLModel` initialises in 0.05 to 0.25 s of it; the
  first prediction takes the rest.
- **A loaded function holds 805 MB for Laya and 289 MB for Verdict:** its own copy of the weights,
  mapped from files of Core ML's, which the process's physical footprint does not count. All eight
  Laya functions hold 8.86 GB, all six of Verdict's 2.76 GB.
- **Served end to end** in a shuffled order, two functions made 102 of Laya's 333 requests wait for
  a reload (median 497 ms, against 50 ms with the function kept), and those waits were the whole of
  the 44.8 s the run took longer; keeping every function held 4.6 GB against 2.0. The answers were
  identical, for both models and both orders.
- **Decision (D-042).** On macOS every function stays loaded once a read has needed it;
  `OPENJEV_ENCODER_FUNCTIONS` caps them for a small Mac; the warm-up stays upstream's.

## The reloads in issue #61's runs

`function_capacity.py replay` over the recorded result files (Laya: `b1_s128` 120 requests,
`b1_s256` 24, `b1_s512` 54, `b1_s1024` 135; Verdict: 120, 22 and 191 of its three lengths), in the
order the server answered them, JevBench then TypeSafe:

| Functions kept | Laya first loads | Laya reloads | Verdict first loads | Verdict reloads |
|---|---|---|---|---|
| 1 | 4 | 46 | 3 | 13 |
| 2, the earlier default | 4 | 15 | 3 | 0 |
| 3 | 4 | 0 | 3 | 0 |
| 4 and more | 4 | 0 | 3 | 0 |

In a rerun of the release build with `Tools/jevbench/servers.py --server swift --backend laya` (its
result files went to a scratch folder and are not committed), the 19 requests the replay names as
loads were exactly the 19 whose model time was over 300 ms: first loads of 1.60 to 1.71 s and
reloads of 0.45 to 0.61 s, where the other reads took medians of 18 ms at 128 tokens, 28 at 256, 50
at 512 and 132 at 1,024. In the recorded run the first loads took 0.49 to 0.84 s and the reloads
0.46 to 0.78 s, and the other reads' medians were 20, 29, 62 and 163 ms. Three functions reload
nothing here only because no 128-token request comes after the first 120: from then on, three shapes
are in use.

## Other orders, and batches

The same 333 requests in 2,000 random orders, reloads per 100 requests (mean, 5th to 95th
percentile):

| Functions kept | Laya | Verdict |
|---|---|---|
| 1 | 66.6 (62.5 to 70.6) | 53.1 (48.9 to 57.1) |
| 2 | 36.4 (32.7 to 40.2) | 13.3 (11.1 to 15.3) |
| 3 | 13.1 (10.5 to 15.9) | 0 |
| 4 and more | 0 | 0 |

A request of 2 to 16 questions runs through a batch-16 function. Streams of 1,000 requests with the
recorded lengths, a share of them one question and the rest batches, reloads per 100 requests over
300 streams, by the number of functions kept:

| Laya: one-question share | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| all | 36.6 | 13.2 | 0 | 0 | 0 | 0 | 0 |
| three quarters | 60.0 | 43.8 | 29.9 | 18.7 | 9.9 | 3.5 | 0 |
| half | 67.1 | 51.8 | 37.2 | 24.1 | 13.5 | 5.2 | 0 |
| a quarter | 60.2 | 43.4 | 29.8 | 18.6 | 9.9 | 3.6 | 0 |
| none | 36.7 | 13.1 | 0 | 0 | 0 | 0 | 0 |

| Verdict: one-question share | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| all | 13.6 | 0 | 0 | 0 | 0 |
| three quarters | 45.4 | 25.6 | 11.8 | 3.4 | 0 |
| half | 54.1 | 33.3 | 14.7 | 4.6 | 0 |
| a quarter | 45.3 | 25.7 | 11.7 | 3.3 | 0 |
| none | 13.5 | 0 | 0 | 0 | 0 |

A least-recently-used cache smaller than the set of shapes in use evicts the shape the next request
needs, so no middle capacity holds for every mix: four functions are enough for one-question traffic
or for batches alone, and reload a third of the requests when the two mix.

## What a loaded function costs

`encoder-capacity` loads every function of a package in turn and keeps them all, runs each twice at
its full shape (batch 1 or 16 by its length), and records the memory after each; then it releases
them and loads the first again. On the GPU each loaded function maps its own copy of the weights
from Core ML's files `payload-*.bin` in the temporary folder: 13 regions of 64 MB per Laya function,
805 MB resident, and 5 per Verdict function, 289 MB, about the size of the package's
`weights/weight.bin` (806.7 and 289.8 MB). They are file-backed, so the physical footprint
(`phys_footprint` from `task_info`) does not count them; the resident memory below is the footprint
plus the copies. `vmmap` lists the same files as `mapped file` regions with no dirty pages (seen in
separate runs, not recorded here). The rows with the footprint at its peak add the largest
footprint, sampled every 5 ms, to the copies held at the end, so they bound the largest resident
memory from above.

| Laya functions loaded | Footprint MB | Weight copies MB | Resident GB |
|---|---|---|---|
| `b1_s128` | 247 | 804 | 1.03 |
| and `b1_s256` | 311 | 1,609 | 1.88 |
| and `b1_s512` | 406 | 2,414 | 2.75 |
| and `b1_s1024`: every batch-1 function | 547 | 3,221 | 3.68 |
| and `b16_s128`: what one-question traffic loads | 739 | 4,025 | 4.65 |
| and `b16_s256` | 892 | 4,830 | 5.59 |
| and `b16_s512` | 1,472 | 5,634 | 6.94 |
| and `b16_s1024`: all eight | 2,629 | 6,441 | 8.86 |
| all eight, with the footprint at its peak during a batch of 16 at 1,024 tokens | 3,523 | 6,441 | 9.73 |
| two, `b1_s512` and `b1_s1024` | 493 | 1,612 | 2.05 |
| two, `b16_s512` and `b16_s1024` | 2,393 | 1,613 | 3.91 |
| those two, with the footprint at its peak | 2,844 | 1,613 | 4.35 |
| four, the batch-16 functions | 2,630 | 3,221 | 5.71 |
| those four, with the footprint at its peak | 3,067 | 3,221 | 6.14 |

| Verdict functions loaded | Footprint MB | Weight copies MB | Resident GB |
|---|---|---|---|
| `b1_s128` | 229 | 289 | 0.51 |
| and `b1_s256` | 245 | 578 | 0.80 |
| and `b1_s512`: every batch-1 function | 308 | 868 | 1.15 |
| and `b16_s128`: what one-question traffic loads | 450 | 1,157 | 1.57 |
| and `b16_s256` | 679 | 1,447 | 2.08 |
| and `b16_s512`: all six | 1,095 | 1,736 | 2.76 |
| all six, with the footprint at its peak | 1,235 | 1,736 | 2.90 |
| two, `b1_s256` and `b1_s512` | 293 | 579 | 0.85 |
| two, `b16_s256` and `b16_s512` | 938 | 579 | 1.48 |
| four, `b1_s512` and the batch-16 functions | 1,192 | 1,158 | 2.29 |

Spike #56's peaks on the GPU, 994 MB for Verdict and 2,856 MB for Laya, were the footprint alone,
with one function loaded, so they left out its copy of the weights. Releasing every function
returned the copies at once; the footprint fell more slowly (1,696 MB for Laya half a second after
the release).

Loading takes two steps, both paid again by a reload. For Laya, `MLModel(contentsOf:configuration:)`
took 0.14 to 0.25 s and the first prediction 0.29 to 0.39 s more for a batch-1 function (0.95 s for
the first function of the process) and 0.44 to 1.66 s for a batch-16 one; for Verdict, 0.05 to 0.12
s (once 0.51 s) and 0.17 to 0.58 s. Loading `b1_s128` again after the release took 0.52 s for Laya
and 0.22 s for Verdict; in one run, loading Verdict's `b16_s256` again took 1.22 s.

Core ML also keeps each function it has compiled for the GPU, with its own copy of the weights
(`resources.bin` in the function's bundle, 803 to 806 MB for Laya and 289 to 290 MB for Verdict), in
`~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache`: 6.3 GB once all eight Laya functions
have run and 1.7 GB for Verdict's six, whatever the capacity
([e5rt-cache.txt](encoder-function-capacity/e5rt-cache.txt)). A function's first load in a process
varied more than its reloads: 0.49 to 0.84 s in issue #61's recorded run, about what a reload takes,
but 1.60 to 1.71 s in the rerun above and 1.6 to 2.3 s in the first served session below; the cause
was not isolated.

## Served end to end

`function_capacity.py serve` started the release build's `openjev serve` with
`OPENJEV_ENCODER_FUNCTIONS=2` and with the variable unset, and sent it JevBench's 231 items and the
102 TypeSafe rows through `Tools/jevbench`'s harness (issue #61), in the datasets' order and
shuffled (seed 61), one request at a time; vmmap read the server's memory after the warm-up and
after the reads. `function_capacity.py report` compares the runs:

| Model, functions, order | Reloads (replay) | Median model ms | p95 ms | Model time s | Resident after the reads GB |
|---|---|---|---|---|---|
| Laya, 2, datasets' order | 15 | 51.5 | 513.3 | 39.4 | 1.99 |
| Laya, all, datasets' order | 0 | 50.9 | 236.4 | 29.1 | 4.44 |
| Laya, 2, shuffled | 102 | 127.6 | 568.2 | 72.3 | 2.03 |
| Laya, all, shuffled | 0 | 50.0 | 206.0 | 27.5 | 4.60 |
| Verdict, 2, datasets' order | 0 | 25.1 | 140.3 | 15.2 | 0.85 |
| Verdict, all, datasets' order | 0 | 24.6 | 148.7 | 14.0 | 1.54 |
| Verdict, 2, shuffled | 39 | 30.7 | 250.4 | 23.8 | 0.88 |
| Verdict, all, shuffled | 0 | 25.4 | 157.7 | 15.3 | 1.52 |

Each pair answered the same requests in the same order, so a request can be compared with itself.
The requests the replay names as reloads took, with two functions and then with every function kept:
Laya in the datasets' order, 15 at a median of 545 ms (465 to 780) against 69 ms; Laya shuffled, 102
at 497 ms (439 to 633) against 50 ms; Verdict shuffled, 39 at 241 ms (219 to 515) against 15 ms. All
333 answers were identical in every pair. Shuffled, the reloads account for all of Laya's longer
model time: 45.2 s of the 44.8 s difference, the other requests running 0.6 s faster. In the
datasets' order the 10.3 s difference is 7.5 s of reloads and 4.4 s of slower first loads (the first
session's, 1.6 to 2.3 s), less 1.6 s by which the other requests ran faster. The summaries are in
[encoder-function-capacity/serve/](encoder-function-capacity/serve/); the per-item result files,
about 0.5 MB per run, are not committed, and `serve` writes them again.

## The options

| Laya | Reloads, one-question requests in random orders | Reloads, half one-question and half batches | Resident with one-question traffic | Resident at most | Notes |
|---|---|---|---|---|---|
| Two functions, the earlier default | 36% | 67% | 2.05 GB | 3.91 GB, 4.35 with the peak footprint | Each reload 0.44 to 1.85 s |
| Four | 0 | 37% | 3.68 GB | 5.71 GB, 6.14 with the peak footprint | Holds one-question traffic or batches, not the two mixed |
| Every function (D-042) | 0 | 0 | 4.65 GB | 8.86 GB, 9.73 with the peak footprint | The memory follows the shapes in use |
| `OPENJEV_ENCODER_FUNCTIONS` | as set | as set | as set | as set | For a small Mac: 2 restores the earlier behaviour |
| A default scaled to the Mac's memory | as set | as set | as set | as set | Behaviour that changes with the machine; the variable already covers a small Mac |
| Warming the batch-1 functions too | spares each length's first load, once per process | | 4.65 GB from the start | | With two functions, released again by the next reads |

| Verdict | Reloads, one-question requests in random orders | Reloads, half one-question and half batches | Resident with one-question traffic | Resident at most | Notes |
|---|---|---|---|---|---|
| Two functions, the earlier default | 13% | 54% | 0.85 GB | 1.48 GB, 1.51 with the peak footprint | Each reload 0.22 to 0.69 s |
| Four | 0 | 15% | 1.57 GB | 2.29 GB | |
| Every function (D-042) | 0 | 0 | 1.57 GB | 2.76 GB, 2.90 with the peak footprint | |

Running a request through a loaded function of a longer length or a larger batch, instead of loading
its own, was not tried: the padding changes the float16 numbers, so an answer would depend on what
happened to be loaded.

## Method and limits

- The Mac was in use while these ran: the load average was 27 to 60, with other builds running, so
  single latencies move between runs. The counts, the memory and the answers do not, and the
  end-to-end comparison pairs each request with itself.
- `replay` assumes one question per request and the warm-up's function first, as the runs had; the
  mixed streams draw lengths from the recorded runs and assume that a batch's longest question has
  the same distribution, an assumption, not a measurement.
- Some figures come from output that is not committed: the rerun of issue #61's benchmark, the
  served sessions' per-item files and vmmap's listings. The commands below produce them again.
- Only the M3 Max with 128 GB was measured. On a Mac with less memory the system may drop the
  file-backed weight pages under pressure and read them back when a function runs; that, and the
  advice of 2 functions for an 8 GB Mac, come from these figures, not from a run on such a Mac. The
  iPhone, which keeps one function, was not measured either.

## Rerunning

From the repository root, with the converted packages in `~/Library/Caches/OpenJevSwift/encoders`
and the release build:

```bash
swift build -c release --product openjev
swift build --package-path Tools/encoders/Harness -c release --product encoder-capacity
"$(swift build --package-path Tools/encoders/Harness -c release --show-bin-path)/encoder-capacity" --package laya-m18-fp16 --output docs/spikes/encoder-function-capacity/laya-m18-fp16-cpuAndGPU.json
"$(swift build --package-path Tools/encoders/Harness -c release --show-bin-path)/encoder-capacity" --package verdict-m18-fp16 --output docs/spikes/encoder-function-capacity/verdict-m18-fp16-cpuAndGPU.json
du -sh ~/Library/Caches/encoder-capacity/com.apple.e5rt.e5bundlecache/*/*
python3 Tools/encoders/function_capacity.py replay laya-1.0 Tools/jevbench/results/laya-1.0-swift.json Tools/jevbench/results/typesafe102/laya-1.0-swift.json
python3 Tools/jevbench/servers.py --server swift --backend laya --encoder-models ~/Library/Caches/OpenJevSwift/encoders --output-dir /tmp/jevbench-reload
python3 Tools/encoders/function_capacity.py serve --backend laya --functions 2 --order shuffled --binary "$(swift build -c release --show-bin-path)/openjev" --encoder-models ~/Library/Caches/OpenJevSwift/encoders --out /tmp/function-capacity
python3 Tools/encoders/function_capacity.py serve --backend laya --functions all --order shuffled --binary "$(swift build -c release --show-bin-path)/openjev" --encoder-models ~/Library/Caches/OpenJevSwift/encoders --out /tmp/function-capacity
python3 Tools/encoders/function_capacity.py report /tmp/function-capacity --backend laya
```

`encoder-capacity --functions b16_s512,b16_s1024` loads only those. Run each `encoder-capacity`
twice the first time: the first run of a new binary compiles every function into Core ML's cache.
The report's `--threshold` is the model time above which a request counts as a load in its table,
300 ms for Laya and 150 for Verdict.
