# Model oracle fixtures

What upstream's own MLX read returns on the pinned checkpoint, for the model parity tests (layer 2
in [docs/09-conformance-and-testing.md](../../docs/09-conformance-and-testing.md)). Recorded by
spike #22; the method and the findings are in
[docs/spikes/backend-validation.md](../../docs/spikes/backend-validation.md).

## reads.json

[Tools/fixtures/mlx_vlm_oracle.py](../../Tools/fixtures/mlx_vlm_oracle.py) runs
`openjev.mlx_backend.MlxRuntime(model_path).read(prompt, canvas, slots, max_tokens, steps)` from
upstream at `dcd2094`, which runs mlx-vlm 0.6.15 on MLX 0.32.2 with
`mlx-community/diffusiongemma-26B-A4B-it-4bit` at revision
`a7a81407613811e8ba63af92ac0d852b809e191f`. Its inputs come from upstream's `Engine`. Those of the
fixture requests are checked against the committed fixtures before any read runs: 23 of their 27
canvases are in `groups-and-canvases/groups_and_canvases.json`, 13 reads use prompt ids that are in
`chat-prompts/prompts.json`, and 8 reads match a read recorded in `policies/policies.json`
(`fixture_checks`). Those of the nine JevBench items (D-048) are the request bodies
`Tools/jevbench/harness.py` sends, built by JevBench's own `typesafe` adapter and checked against
the body digest both servers' DiffusionGemma result files recorded
(`Tools/jevbench/results/openjev-0.1-*.json`).

- `generator` records the pins: upstream commit, checkpoint repository and revision (it is also
  the tokenizer), Python and package versions, the GPU the file was made on, the SHA-256 of
  the `mlx.metallib` that the mlx-metal wheel ships, and the JevBench repository and commit.
- `settings` is upstream's `TOPK` (20), vocabulary size, canvas 64, step 16 and `mlx_max_prompt`.
- `rope` is the frequency table mlx-vlm's `ProportionalRoPE` passes to `mx.fast.rope` in the five
  full-attention layers, as float32 bit patterns (256 entries, the last 192 infinite) and as the 64
  finite values.
- `jevbench` maps each JevBench item's id, which is also its request name, to the split file it
  comes from, its family and question type, the SHA-256 of its request body, and `options`, the
  names of its slot's labels in upstream's order (a noul's `yes` first), as its probabilities read.
- `prompts` maps a key (`request/gGROUP`) to `{system, user, ids, tokens, in_chat_prompts_fixture,
  cache}`. `ids` is `Engine.chat_prompt_ids(system, user)`. `cache` digests mlx-vlm's prefill cache
  of layer 0 (sliding) and layer 29 (full), in temporal order: shape, dtype, SHA-256 of the raw
  bfloat16 bytes in C order, and float64 sums. For the sliding layer, `decoder_view` digests the last
  1,023 positions, which is all the decoder reads and all a 1,024-slot ring keeps.
- `reads` lists 63 reads. Each has the request name, group, format, question ids, prompt key,
  canvas seed and index, width, `canvas`, `slots` (`{pos, label_ids}`) and `steps`, then what the
  read returned: `prompt_tokens`, `logprobs` (per slot, `[token id, logprob]` pairs exactly as
  `MlxRuntime.read` returns them: the top 20 plus every label, float32 log-softmax, in token id
  order), `distributions` (per slot, `openjev.engine.slot_distribution`: `{probs, entropy}`) and
  `written` (for `steps` above 1, the argmaxes the read wrote into the slot positions between
  passes, as the decoder saw them).

The reads cover widths 16 (`single_noul`), 32 (`quickstart`, `non_ascii`, `many_choices`,
`widest_schema`, `long_state`), 48 (`nouls_8`, `six_questions`, `indexed_12_mixed`, the second group
of `nouls_24`) and 64 (`lines_10_mixed`, the first group of `nouls_24`); 1, 3 and 12 questions;
the 255-option choice (`widest_schema`); four prompts over 1,024 tokens (1,572, 2,076, 2,442 and
the 2,939-token `long_state`, the quickstart's questions over a long support thread), so the
decoder's sliding layers read only the last 1,023 encoder positions; and `steps` 2 and 3 for
`quickstart`, `lines_10_mixed` and `long_state`.

The last 36 reads are nine JevBench v1 hard-tier items (fstandhartinger/jevbench at `bb05a33`,
public, MIT), each one question with 2 to 4 labels over a state of 2,318 to 3,643 prompt tokens,
the shape of JevBench's and TypeSafe's long questions, of which the reads above hold few (D-048).
Each item has the four reads upstream's default policy makes of it: canvas index k is read k, at
the request seed plus 7919k (`read_group`), every first read being above the re-read threshold.
The mean of an item's four reads is the answer upstream's server gave in those result files. The
first three are the long-prompt flips of PR #105: `hard-opus-c-long_policy-04`,
`hard-sol-b-long_policy-06` and `hard-opus-a-long_policy-19`. Both servers answered the other six
alike, at upstream top-two margins from 0.005 to 0.99. Their states use dashes, which the file
writes as JSON escapes (`\u2013`, `\u2014`) because this repository keeps them out of its files;
they read back as the same text.

## Regenerating

From the repository root, on an Apple silicon Mac with about 25 GB free:

```bash
make upstream
python3 Tools/jevbench/harness.py fetch
python3.14 -m venv Tools/oracle/.venv
Tools/oracle/.venv/bin/pip install -r Tools/oracle/requirements.txt
Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py
PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/mlx_vlm_oracle.py --cache-limit-gb 4
```

The harness's fetch downloads JevBench's pinned items into a cache outside the repository; the
checkpoint fetch downloads 16.6 GB into the Hugging Face cache. `--cache-limit-gb 4` caps MLX's
buffer pool, as `OPENJEV_MLX_CACHE_LIMIT_GB` does, which changes no read (spike #22) and kept the
run at 24 GB. The oracle runs every read twice, the
second time from an emptied prefill cache and in reverse order, and writes nothing unless the two
passes agree bit for bit. Running it again on the same machine gives no diff. Another GPU family
may round some kernels differently; the `device` and `gpu_architecture` pins say where the file
was made. `--check` compares a run with the committed file instead of writing it: every
top-level key (the pins, settings, fixture checks, RoPE table, prompts and reads). It fails
unless both passes agree and nothing differs.

Timings and memory are machine state, not fixture data, so they go to
`Tools/oracle/results/oracle_run.json`.

## Using it from Swift

A port that follows mlx-vlm's operations reproduces every read bit for bit when it loads the
wheel's Metal library (`MLX.GPU.metallib` set to `mlx/lib/mlx.metallib` from the pinned mlx-metal
wheel, SHA-256 in the generator) and uses the `rope` table. With mlx-swift's own kernels the reads
agree only within the tolerances of decision D-014. See the spike report for both.
