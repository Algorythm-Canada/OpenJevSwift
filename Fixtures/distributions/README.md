# Distribution fixtures

`slot_distribution`, `confidence`, `to_answer` and the averaging in `Engine.read_group`, on
synthetic log-probability maps. `Tools/fixtures/upstream_tables.py` writes `distributions.json`
from a fixed seed; see [../README.md](../README.md).

## distributions.json

| Key | Contents |
|---|---|
| `auto_threshold`, `auto_max` | Upstream's defaults, 0.1 and 4: a first read whose largest slot entropy exceeds 0.1 is followed by three more |
| `slot_distribution` | `{name, top, label_ids, result: {probs, entropy, exceeds_auto_threshold}}`, or `error` |
| `confidence` | `{name, p, result}`, or `error` |
| `to_answer` | `{name, question, internal, p, answer, body_text}` |
| `read_group` | `{name, request, options, seed, label_ids, reads, means, billed, answers, answers_text}` |

- `top` is the slot's map as `[token id, log-probability]` pairs in the map's order. The order
  matters: the entropy is summed over the map's values in order. `vllm` maps hold the top 20 in
  rank order, as vLLM returns them. `mlx` maps hold the top 20 and every label, ordered by token
  id, as `MlxRuntime.read` builds them. A label missing from the map gets the smallest value
  minus 5.
- The named slot cases are Jev's example labels, labels missing from the map, no label in the
  map, a single entry, a label at probability one, equal labels, vLLM's `-9999` floor, values at
  the edge of `exp` underflow, entropies on either side of the threshold, and an empty map, which
  upstream refuses with a `ValueError`.
- `confidence` includes Jev's documented example `[0.84, 0.159, 0.001]`, which gives
  0.5942682179296132 (upstream's test accepts 0.596 within 0.01). Uniform distributions give
  0.0 or a rounding residue such as 2.220446049250313e-16; over 5 and 13 options the value
  falls just below zero and the clip returns 0.0, as it does for a slot with no label in its
  top 20. `probability_above_one` shows the upper clip, which the engine never reaches. The
  one-option and no-option cases are never reached either, because such questions are forced
  (`ZeroDivisionError`, `ValueError`).
- `to_answer` rows give the answer and its bytes as FastAPI renders them. A tie goes to the
  first option.
- `read_group` runs upstream's own `read_group` with `one_read` stubbed to return
  `slot_distribution` of synthetic maps. `reads` lists each read's seed, maps and
  distributions. `means` are the averaged label probabilities, summed over the reads in order and
  divided by their count. Samples are billed per read; the automatic re-reads (seed + 7919·k) are
  not.

Values are Python doubles written with `repr`, so they round-trip exactly. Where the Swift code
runs the same operations in the same order it should match exactly; docs/09 allows 1e-12.
