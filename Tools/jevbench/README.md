# JevBench harness (issues #61 and #62)

Runs JevBench v1's public items, and the TypeSafe subset SemIf compares with Jev, against any
`/v1/systemone` server, scores the answers with each benchmark's own code, compares two runs of
one model item by item and measures how well a model's probabilities are calibrated.
[docs/quality.md](../../docs/quality.md) holds the tables it produced for the Swift and upstream
servers and what they mean; the decisions behind its choices are D-041 and, for the calibration,
D-046 in [docs/06-decisions.md](../../docs/06-decisions.md).

| Path | What it is |
|---|---|
| `harness.py` | The runner and scorer: `fetch`, `run`, `summary`, `compare`, `published`, `author-run`, `report`, `calibration`. Standard library only |
| `servers.py` | Starts the Swift or upstream server for one backend on a free port, records its versions, runs `harness.py` against it and stops it with SIGTERM |
| `report.py` | Renders the comparison tables of docs/quality.md from `results/` |
| `calibration.py` | Renders its calibration tables (issue #62): JevBench's Brier score and ECE, reliability bins, `confidence` against accuracy by type and option count, and SemIf's temperature scaling fitted offline |
| `smoke_test.py` | The harness's own test: a fake server in the process, no model and no download. CI runs it |
| `testdata/items.jsonl` | Seven items written for the smoke test, in JevBench's format |
| `vendor/` | JevBench's and SemIf's scoring code, unchanged and pinned by SHA-256 ([vendor/README.md](vendor/README.md)) |
| `requirements.txt` | Empty: the harness needs only Python's standard library |
| `requirements-upstream.txt` | The complete lock of `.venv`, the environment upstream's server runs in |
| `results/` | One file per run: `{model}-{server}.json` for JevBench, `typesafe102/{model}-{server}.json` for the TypeSafe rows; `jevk5-0.2-author.json` is JevK5's own published run, converted by `author-run`, and `jevk5-conversions/{4bit,bf16}/` JevK5's runs on its other conversions |

## The datasets

Both are downloaded into a cache outside the repository (`~/Library/Caches/OpenJevSwift/jevbench`,
or `JEVBENCH_CACHE`, or `--cache`) and checked by size and SHA-256 before use. Nothing of either
is committed.

- **`jevbench`**: the 231 public items of
  [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) at `bb05a33`:
  `datasets/public/easy.jsonl` (48), `original.jsonl` (72, the benchmark's "standard" tier) and
  `hard.jsonl` (111). Their hashes are the ones the benchmark's own `datasets/manifest.json`
  records. With them come `results/v1.2/jevbench-v1.2-per-task.json`, the
  published outcome of every public item per system, and
  `results/v1.4.2.2/jevbench-v1.4.2.2-results.json`, the newest board. MIT.
- **`typesafe102`**: the 102 rows of [TypeSafe's public evaluations](https://evals.typesafe.ai/)
  that [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) compares with Jev (its README, "TypeSafe
  subset agreement"). SemIf's selection manifest at `23cf1f3` names the rows and pins each case
  snapshot's parsed payload; the harness downloads the four snapshots
  (`{workflow}-cases.js`, 1.4 MB together) from evals.typesafe.ai and rebuilds the rows with SemIf's
  own `build_typesafe.py`, which checks those hashes. The rows it writes are pinned by SHA-256 too,
  so a cached copy that is not the pinned one is built again rather than used. TypeSafe's snapshots
  carry no license grant, so a result file keeps only ids, digests and the server's answers: no
  TypeSafe text, reference answer or published distribution.

## How an item becomes a request

Through JevBench's own `typesafe` adapter, the one the benchmark uses for TypeSafe's API and every
rebuild of it: one request per item, `{"state": <the item's state>, "model": <--model>,
"questions": {"decision": {"type", "instructions", "criteria"}}}`, with `criteria` left out when
the item has none and every object's key order kept. The state goes as the item has it, a string or
an object. A noul answer is read as `{"yes": noul, "no": 1 - noul}`; a choice or score answer's
`probabilities` are read as they are, and a choice whose `choice` is not one of the item's labels is
a failed answer. A TypeSafe row sends TypeSafe's own question and document from the snapshot, as
TypeSafe asked Jev, not SemIf's prompt rendering of them. Requests go one at a time, with the
benchmark's 120-second timeout.

An item whose question the model cannot take is **skipped**, never sent, and counted: more than 24
options for `verdict-1.4` (its head has 25 logits, one kept for "insufficient evidence"), more than
255 options or 10 levels for any model. Skipped items lower the coverage and stay out of the
accuracy, as unattempted items do in the benchmark. No item of either dataset is skipped today:
JevBench's widest choice has 6 options and the TypeSafe rows' 8. A 4xx other than 401, 403 or 429
is a **refusal**: it is recorded with the server's detail and counts as a wrong answer, as the
benchmark counts a failed decision. Three failures in a row, or a 401, 403 or 429, stop the run and
leave the rest unattempted, as the benchmark's runner does, except that a refusal does not count
towards the three (the runner exempts only a 422; D-041).

## JevK5's reference: the author's published run

Upstream's `jevk5` server reads its letters from a vLLM server, which needs an NVIDIA GPU, so this
Mac cannot run it. The reference for the Swift `jevk5` backend is the one upstream used: the
`jevk5` package's own published v0.2 run of the 231 public items
([allebee/jevk5](https://github.com/allebee/jevk5) at `85238d7`, the v0.2.0 commit that added
`results/public231/jevk5-v0.2.jsonl`, Apache-2.0). Its `bench/SUBMISSION.md` says how it was made:
the package's in-process adapter, `jevk5_direct`, with transformers and the bf16 weights of
`alibiserikbay/JevK5` at `3c67329` (the same files as the `v0.2` tag the conversions use), through
JevBench's runner at `0caa1d0` on one H100. Its prompts were built in that process, not sent to a
server, so its token counts compare with this harness's only as far as the prompts are the same,
and equal counts on an item show that they are.
`fetch` downloads it into the cache and checks its size and SHA-256 (`AUTHOR_RUNS`), and
`author-run` turns it into a result file whose server is `author`: each item takes the
distribution the author's server returned and its `usage`, scored again with JevBench's
`score_task` as this harness scores its own runs, the published record's own outcome kept beside
it. `compare` then reads it as it reads any run, and also reports the items whose billed input
tokens differ (equal counts mean equal prompts) and the median of each item's largest difference.

## Scoring

- **JevBench** (`vendor/jevbench`): `scoring.score_task` validates each distribution over the
  item's exact labels (a sum within 1e-3, or rescaled inside the 2e-2 rounding band; anything else
  is invalid and wrong) and takes the argmax, the smallest label on a tie. `summarize.metric` and
  `summarize.summarize` then give the accuracy, the Brier score (the multi-class sum
  `sum_k (p_k - y_k)^2`, so a binary question counts both outcomes), the ECE over 10 equal-width
  bins of top-label confidence, the ordinal MAE of score items, paraphrase consistency and the
  latency percentiles, overall, per family and per tier, exactly as the benchmark computes its
  published numbers.
- **TypeSafe rows** (`vendor/semif`): SemIf's `evaluate_external.type_safe`, the equal-case modal
  agreement with the reference (each of the 20 cases weighs the same; the first option on a tie)
  and the equal-case total variation from the reference distribution, for the run and for the Jev,
  Opus and Sol answers the snapshots publish. JevBench's accuracy, Brier score and ECE are reported
  beside them, with the reference's top option as the expected label.
- **`compare A B`** takes two runs of one model and dataset, B as the reference: the top answers'
  agreement, the mean and largest absolute probability difference per question type over every
  label of every item both answered (spike #56's measure), identical answers, correctness flips with
  an exact McNemar test, every disagreement with the reference's top-two margin, the items where
  that margin is under 0.01 (where the parity bound of D-034 and D-037 allows a changed top answer),
  the largest deviations and the items only one run answered.
- **`published FILE`** compares a JevBench run with the benchmark's published row for the same
  model: the outcome of every public item, both public accuracies, per tier, and the board's
  aggregates. The rows were not produced the way this harness runs upstream's server;
  `PUBLISHED_ROWS` in `harness.py` says how each was.
- **`calibration [--model NAME] [--results DIR]`** (`calibration.py`) takes every run of one
  model (`openjev-0.1` by default) over its items with an expected label: JevBench's Brier score
  and ECE, the NLL of the expected label, the reliability bins (the top label's probability
  against observed accuracy, 10 equal-width bins with counts), how the answer's `confidence`
  (upstream's 1 - H(p)/ln K, computed for a noul, whose answer has none) relates to accuracy per
  question type and option count, and SemIf's per-workload temperature scaling (its
  `docs/CALIBRATION.md` at `23cf1f3`): one T fitted by NLL on each hard-labelled run, the ECE
  out of fold under group-disjoint 5-fold cross-validation with 95% bootstrap intervals over the
  groups, the paired intervals of the changes, a T per question type against one T, and
  JevBench's T applied to the TypeSafe rows. The module's docstring gives every formula and its
  source. A deployment's own labelled items, run with `run --items`, are fitted the same way:
  `calibration` reads every folder under `results/`, `items/` included, where such a run is
  written by default.

## Running it

From the repository root. The harness itself needs any `python3` (3.10 or later):

```bash
make upstream
python3 Tools/jevbench/harness.py fetch
python3 Tools/jevbench/smoke_test.py
```

Upstream's server needs its own environment, about 1 GB, and the two checkpoints at their pinned
revisions in the Hugging Face cache (`Tools/encoders/reference.py` downloads them, or
`servers.py` does on first use):

```bash
/usr/local/bin/python3.12 -m venv Tools/jevbench/.venv
Tools/jevbench/.venv/bin/python -m pip install -r Tools/jevbench/requirements-upstream.txt
```

The Swift server is the release build (`swift build -c release --product openjev`). Then the four
runs, each on both datasets, about five minutes in all on an M3 Max, more than half of it
upstream's Laya on the CPU:

```bash
python3 Tools/jevbench/servers.py --server swift --backend verdict --encoder-models ~/Library/Caches/OpenJevSwift/encoders
python3 Tools/jevbench/servers.py --server swift --backend laya --encoder-models ~/Library/Caches/OpenJevSwift/encoders
python3 Tools/jevbench/servers.py --server upstream --backend verdict
python3 Tools/jevbench/servers.py --server upstream --backend laya
python3 Tools/jevbench/harness.py report
```

`--encoder-models` names the folder the converters write to; without it the Swift server
downloads the published packages (D-033), which have the same bytes
(`Tools/encoders/manifest.py --check`, which `servers.py` runs and records, and which stops the run
before the server starts when a package differs). Add `--force` to replace earlier result files.

JevK5, `--backend jevk5`, reads a conversion `Tools/jevk5/convert.py` writes (`--jevk5-model`, by
default the 8-bit one, `~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-8bit`, the server's
default), which `servers.py` checks against the pinned digests of the 4-bit, 8-bit and bfloat16
conversions (`convert.py --check`) before the server starts, and records. Only the Swift server
runs it; its reference is the author's run. The main result files are the 8-bit conversion's, and
`results/jevk5-conversions/` holds the other two, which the report's table of conversions reads:

```bash
python3 Tools/jevbench/harness.py author-run
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --setting OPENJEV_MLX_CACHE_LIMIT_GB=4
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --jevk5-model ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-4bit --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --output-dir Tools/jevbench/results/jevk5-conversions/4bit
python3 Tools/jevbench/servers.py --server swift --backend jevk5 --jevk5-model ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-bf16 --setting OPENJEV_MLX_CACHE_LIMIT_GB=4 --output-dir Tools/jevbench/results/jevk5-conversions/bf16
python3 Tools/jevbench/harness.py compare Tools/jevbench/results/jevk5-0.2-swift.json Tools/jevbench/results/jevk5-0.2-author.json
```

The DiffusionGemma runs, `--backend mlx`, read `mlx-community/diffusiongemma-26B-A4B-it-4bit` at
`a7a81407` from the Hugging Face cache (both servers load about 16 GB; run them one after the
other), upstream's from the same `.venv`, whose lock includes upstream's `mlx` extra:

```bash
python3 Tools/jevbench/servers.py --server swift --backend mlx --setting OPENJEV_MLX_CACHE_LIMIT_GB=4
python3 Tools/jevbench/servers.py --server upstream --backend mlx --setting OPENJEV_MLX_CACHE_LIMIT_GB=4
python3 Tools/jevbench/harness.py report
python3 Tools/jevbench/harness.py calibration
```

`--setting OPENJEV_NAME=VALUE` (repeatable) gives the server one setting beyond its defaults, which
the result file records with the others. `servers.py` refuses the settings it chooses itself, and
any that can hold a credential the file would then record: the API key, the origin secret, the
model routes, upstream's vLLM URL and any name with KEY, SECRET, TOKEN, PASSWORD or CREDENTIAL in
it (the harness sends no key, so such a server would refuse its requests anyway). The
MLX runs cap MLX's buffer pool at 4 GB: without a cap the pool grows to the peak working set, as
upstream's README says ("MLX memory"), and on JevBench's long hard-tier states the Swift server's
footprint reached 102 GB of a 128 GB Mac. The cap changes no answer
([docs/quality.md](../../docs/quality.md#diffusiongemma)). Any other server:
`python3 Tools/jevbench/harness.py run --base-url URL --model NAME --server LABEL
[--dataset typesafe102] [--api-key-env VARIABLE]`, then `summary`, `compare`, `published` and
`calibration` on the files it writes. `--ids a,b` runs only those items.

## Result files

`schema` `openjevswift-jevbench-result/1`: the dataset and its pins; the model; the server, with its
`/v1/models` listing and the versions `servers.py` records (the last commit that changed
OpenJevSwift's package, the binary's digest, the Core ML package and its check, and the number of
Core ML functions the server keeps loaded as its settings line gives it, `function_capacity`, 2 for
a binary from before D-042; upstream's commit, Python and package versions, device, dtype, thread
count and checkpoint revision); the client (this harness's digest, the vendored files' digests); the
hardware; the published row; the summary; and one line per item. An item holds its id, tier, family,
type and labels (and, for JevBench, its paraphrase group and expected label), its status, the
request (the model and question as sent, for JevBench; ids only, for TypeSafe; the state's and the
body's SHA-256 always), the answer as the server sent it, the distribution JevBench scores, whether
it is valid and (for JevBench, since a TypeSafe row's would give its reference answer away) correct,
the published outcome where there is one, and the timing: the caller's time, the HTTP time and the
server's `server-timing` header. The JevBench files are about 400 KB and the TypeSafe ones about 110
KB; a state is kept as its digest because the hard tier's states alone are 480 KB.
