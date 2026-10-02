# Fixtures

Golden test data for the Swift tests. Every fixture file here except `regression/`, which holds
the Swift port's own answers, records what upstream OpenJev produces at the pinned commit (`dcd2094`, see [THIRD_PARTY.md](../THIRD_PARTY.md)), with the
pinned tokenizer revision where a tokenizer is involved. Tests load these files and compare byte
for byte where the computation is exact, or within a tolerance for model outputs. The full
strategy is in [docs/09-conformance-and-testing.md](../docs/09-conformance-and-testing.md).

## Regenerating

Fixture files are never edited by hand: a fixture that needs to change is regenerated. From the
repository root:

```bash
make upstream
make fixtures-venv
make fixtures
```

`make upstream` checks out upstream at the pinned commit under `Upstream/openjev`.
`make fixtures-venv` creates `Tools/fixtures/.venv` from CPython 3.14 with the packages pinned in
`Tools/fixtures/requirements.txt`. `make fixtures` runs the four scripts in
[Tools/fixtures](../Tools/README.md). The first run downloads the tokenizer files of
`mlx-community/diffusiongemma-26B-A4B-it-4bit` at revision
`a7a81407613811e8ba63af92ac0d852b809e191f` (about 32 MB) into the Hugging Face cache, outside the
repository, with the checkpoint's `config.json`, `generation_config.json` and
`model.safetensors.index.json`. No model weights are downloaded. Running `make fixtures` twice
gives no diff, and a fresh virtual environment built from the requirements file reproduces every
file byte for byte.

## Pins

Every file starts with a `generator` object. It names the script that wrote the file, the
Python version and the versions of the packages whose behaviour the file depends on. Files that
come from upstream's code also record `upstream` and `upstream_commit`. Every file written by
`upstream_tables.py`, which loads the real tokenizer, also records `tokenizer_repo`,
`tokenizer_revision` and the script's `version`.
The files in `model/` come from the checkpoint alone, not from upstream's code: they record
`model_repo`, `model_revision` and `version` instead of the upstream and tokenizer pins.
`Tests/OpenJevCoreTests/Fixtures/FixturePinTests.swift` checks every file, so a fixture
regenerated after a pin moved fails loudly. When a pin moves, update the Makefile,
THIRD_PARTY.md, the scripts and that test together, then regenerate.

## Contents

| Path | Contents | Script | Used by |
|---|---|---|---|
| [tokenizer/](tokenizer/README.md) | Special token ids; token ids, pieces and decodes for a corpus; every text upstream's engine tokenized | `upstream_tables.py` | #12, #20 |
| [chat-prompts/](chat-prompts/README.md) | `[system, user]` messages and their prompt text and ids, thinking off and on | `upstream_tables.py` | #21, #17 |
| `labels.json` | The 255 single-token choice labels, in order, below | `upstream_tables.py` | #12 |
| [schemas/](schemas/README.md) | Requests and the internal schema or the `SchemaError` (message and `loc`) they produce | `upstream_tables.py` | #10 |
| [system-texts/](system-texts/README.md) | Requests and the system text of each group, chunked and unchunked | `upstream_tables.py` | #11 |
| [templates/](templates/README.md) | Answer texts, template ids, slot positions and label ids per group, plus the error cases | `upstream_tables.py` | #11, #13 |
| [groups-and-canvases/](groups-and-canvases/README.md) | Groups, canvas widths and seeded canvases at canvas 64, 32 and 40 | `upstream_tables.py` | #14 |
| `seeds.json` | Request bodies, the bytes upstream hashes, their seeds, and Python's MT19937 draws, below | `upstream_tables.py` | #15 |
| [distributions/](distributions/README.md) | Synthetic log-probability maps and the probabilities, entropies, confidences, answers and averages they give | `upstream_tables.py` | #16, #17 |
| [policies/](policies/README.md) | Requests with `samples`, `steps`, `think`, `sequential` and images, and every read the engine makes for them | `upstream_tables.py` | #17 |
| [errors/](errors/README.md) | Error responses that `wire/cases.json` does not hold, and an index of every error row in the wire contract | `upstream_tables.py` | #35, #38 |
| [encoders/](encoders/README.md) | PyTorch reference reads of Verdict and Laya through upstream's own code (float32, with float16 and bfloat16 passes), and the question corpus they cover | `Tools/encoders/reference.py` | #56, #57, #58 |
| [model/](model/README.md) | The checkpoint's `config.json` and `generation_config.json` verbatim, and its safetensors weight map | `checkpoint_tables.py` | #23, #27 |
| [regression/](regression/README.md) | The Swift port's own answers for the 27 oracle reads, the wire quickstart and upstream's README example; not upstream's output | `RegressionTests` (`OPENJEV_RECORD_REGRESSION=1`) | #31 |
| [oracle/](oracle/README.md) | mlx-vlm 0.6.15 reads of the 4-bit checkpoint through upstream's `MlxRuntime.read`: slot logprobs, distributions, written argmaxes, prompt ids, prefill cache digests and the full-attention RoPE table | `mlx_vlm_oracle.py` | #22, #31 |
| [wire/](wire/README.md) | HTTP exchanges, answer bodies, request renderings and `/v1/models` listings, recorded with a stand-in tokenizer | `wire_tables.py` | #5, #35 |
| [python-json/](python-json/README.md) | CPython `json.dumps` and float `repr` tables, and how `json.loads` ends on valid and malformed documents. These record the Python version, not the upstream commit. | `python_json_tables.py` | #3, #35 |

`/v1/models` bodies for every backend are in `wire/models.json`, not in a `models.json` at this
level. Model parity data from mlx-vlm (layer 2) is in `oracle/`; issue #31 widens it.

### Replaying tokenization without the tokenizer

`OpenJevCore` builds on Linux, where the tokenizer (swift-transformers) is not available. Two
files let its tests replay upstream's tokenizations instead:
`tokenizer/engine_encodings.json` holds every text upstream's `Engine.enc` tokenized while these
fixtures were made, and `chat-prompts/prompts.json` holds every prompt `Engine.chat_prompt_ids`
rendered. A replay tokenizer that looks texts up in them can drive the schema, template,
grouping, canvas and policy tests against the recorded results.

### labels.json

`Engine(Settings(), tokenizer).choice_labels`: the 255 choice labels that stay one token after
the prefix `"q1: "`.

| Field | Contents |
|---|---|
| `prefix`, `base_ids` | `"q1: "` and the ids of `"q1: A"`, which every candidate is compared with |
| `candidate_count`, `candidates_examined` | 728 candidates (`A` to `Z`, `a` to `z`, `AA` to `ZZ`); discovery stops after the 261st, when it has 255 labels |
| `labels`, `label_ids` | The 255 labels in order (`A`, `B`, `C`, and so on up to `IA`) and the single token id of each after the prefix |
| `rejected` | The candidates skipped before the 255th label, with their ids and the reason. `BQ`, `BZ`, `FQ`, `FZ`, `GZ` and `HZ` are two tokens after the prefix. |
| `noul_labels`, `score_labels` | `yes`, `no` and `0` to `9` |

### seeds.json

What `api.py:262-263` computes. Every body was sent through `create_app`'s own validation in
FastAPI's `TestClient`; a spy on `Engine.decide` took the seed the route computed and what it
computed it from, and the script rebuilt the key from those and checked that it gives the same
seed.

| Key | Contents |
|---|---|
| `cases` | `{name, body_text, key_text, sha256, seed, randrange_262144, same_seed_as?}`. `body_text` is the body as sent. `key_text` is `json.dumps([state, questions] + ([image URLs] if images else []), sort_keys=True)` with `questions` as pydantic's `model_dump()` of each question, so every optional field is present. `sha256` is the digest of its UTF-8 bytes (ASCII, because `ensure_ascii` is on), and `seed` its first four bytes as a big-endian integer. `randrange_262144` is the first 64 values of `random.Random(seed).randrange(262144)`. `same_seed_as` names the other cases with the same seed. |
| `upstream_only` | The same for bodies upstream accepts only because Python's `json.loads` does: `NaN`, `Infinity`, `1e400` and a lone surrogate escape. OpenJevSwift's parser refuses them (decision D-016). |
| `mt19937_seeds` | 10,000 rows `[seed, getrandbits(32), randrange(262144)]` (the names are in `mt19937_columns`), each value the first draw of a fresh `random.Random(seed)`. The seeds are edge values, 32-bit values, values within 2**25 of 2**32 and a few up to 63 bits. Group seeds (`seed + 104729·k`) and sample seeds (`seed + 7919·k`) pass 2**32, where Python seeds the generator with a two-word key. |
| `mt19937_streams` | 1,300 `getrandbits(32)` values from seed 0 (past two state regenerations), 700 from seed 2**32 + 104729, and 1,000 `randrange(262144)` values from seed 20260929. |

What the key holds, from the cases: `sort_keys` sorts every object, including `questions` and a
choice's options, so reordering questions or options keeps the seed while the answers' labels
change. The model name and the extension fields are not in the key. `images: []` and
`images: null` add nothing, and a data URL and the equivalent `{content_type, base64}` object
give the same key. A noul's `criteria: {}` and `criteria: null` give different keys. Keys are
sorted by code point, not by UTF-16 unit. `randrange(262144)` takes the top 19 bits of a 32-bit
output and draws again while the value is 262144 or more, so one call can consume more than one
output; the script checks that reading against CPython for all 10,000 seeds.

## What never belongs here

- Model weights, tokenizer files or generated caches.
- Images larger than a few kilobytes.
- Expected values written by hand.
