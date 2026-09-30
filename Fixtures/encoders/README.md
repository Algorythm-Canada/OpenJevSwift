# Encoder model fixtures

PyTorch reference outputs of the two ModernBERT encoder models upstream serves, Verdict
(`verdict-1.4`) and Laya (`laya-1.0`), recorded for spike #56 and for the encoder backends of
issues #57 and #58. `Tools/encoders/reference.py` writes all three files by driving upstream's own
code at the pinned commit (`EncoderEngine.build_schema`, `VerdictEngine.read_batch`,
`LayaEngine.read_batch`) with both models on the CPU in float32, as upstream loads them, reading
at most 16 questions per forward pass. Small wrappers record the tensors those functions pass to
the models and get back; nothing in the read path is reimplemented.

## Regenerating

From the repository root:

```bash
make upstream
/usr/local/bin/python3.12 -m venv Tools/encoders/.venv
Tools/encoders/.venv/bin/python -m pip install -r Tools/encoders/requirements.txt
Tools/encoders/.venv/bin/python Tools/encoders/reference.py
```

The first run downloads both checkpoints into the Hugging Face cache (about 1.5 GB). A run takes
about 37 minutes on an M3 Max; almost all of it is the float16 and bfloat16 passes, which PyTorch
runs slowly on a CPU (Laya's float16 pass alone takes 17 minutes, against 55 seconds in float32). Two runs on that machine gave byte-identical files. The script pins PyTorch to 8
threads because the CPU kernels split reductions by thread; another machine or thread count can
change the last bits of a float, which is why tests compare model outputs with tolerances (D-014).

## Pins

Every file starts with a `generator` object: the script and its version, `upstream_commit`
(`dcd2094`), the checkpoints and their revisions, the Python and package versions, the CPU and
the thread count. `Tests/OpenJevCoreTests/Fixtures/FixturePinTests.swift` checks the checkpoint
revisions.

| Input | Pin |
|---|---|
| Verdict | `heman10x/rlcd-modernbert-151m` at `8af2496eb63c7fa66d7d234e1f62629380030eb4` |
| Laya | `convaiinnovations/laya-typed-decisions` at `1a793eb568e6718f15941d08f85432581df534e3` |
| Upstream | razorback16/openjev at `dcd2094` |
| Python and packages | CPython 3.12.2, torch 2.13.0, transformers 5.17.0, tokenizers 0.23.2, gliclass 0.1.20, laya 0.3.6, numpy 2.3.5 (the full lock is `Tools/encoders/requirements.txt`) |

## The corpus (`corpus.json`)

200 questions in 26 requests, each a state and its questions as the API hands them to the engine
(`type`, `instructions`, `criteria`). Every question comes from a request in
[schemas/schemas.json](../schemas/README.md):

- The fixture's 21 requests that have read questions, with their own states (91 questions).
- `six_questions` under `state_object`'s state, and upstream's warmup request
  (`encoders.WARMUP_QUESTIONS`, state `"warmup"`).
- 38 distinct fixture questions under a 16-message and a 30-message transcript of the fixture's
  text states (about 415 and 790 tokens), and `nouls_24` under an 80-message conversation as a
  JSON state (about 2,080 tokens).

Verdict refuses a choice with more than 24 options, so the fixture's 40-, 55-, 100- and
255-option choices are cut to their first N options (`truncated_choices`). The corpus has 96
nouls, 60 choices (2 to 24 options: every per_k entry of Verdict's calibrator, and 7, 12 and 20
options, which fall back to its global temperature) and 44 scores (2, 3, 4 and 10 levels); 5 JSON
states; 7 states with non-ASCII text; 70 Verdict prompts over 512 tokens and 25 Laya sequences
whose state is cut at 1,024 tokens.

## verdict.json

| Key | Contents |
|---|---|
| `calibrator` | `temperature` and `per_k` from the checkpoint's calibrator.json |
| `precision_floor` | The same reads with the weights cast to float16 and to bfloat16, against float32: the largest logit and probability differences and the questions whose top answer changed |
| `reads` | One row per question, in corpus order |

Each row: `request`, `key`, `type`; `options` (the caller's) and `k` (plus the trailing
"insufficient evidence" label); `prompt`, upstream's `verdict_prompt` text; `tokens_untruncated`
and `truncated`; `input_ids`, the tokenizer's output with truncation at 512 and the batch padding
removed; `batch` and `padded_length` of the forward pass it was read in; `temperature` and
`temperature_source` (`per_k` or `global`); `logits`, the model's first k logits in float32;
`probabilities`, upstream's calibrated distribution over the caller's options; `float16` and
`bfloat16`, the same two lists from the reduced-precision reads. A request's last row also holds
`request_input_tokens`, upstream's `usage.input_tokens` for it.

The precision floor on this corpus, against float32:

| Weights | Largest logit difference | Largest probability difference | Top answers changed |
|---|---|---|---|
| float16 | 0.029 | 0.0011 | 1 of 200 |
| bfloat16 | 0.165 | 0.0115 | 3 of 200 |

## laya.json

| Key | Contents |
|---|---|
| `special_tokens`, `max_len`, `head_max_len` | The ids build_sequence uses, 1,024 and 256 |
| `calibration` | The checkpoint's `temperature` and `temperature_by_options`, raw and clamped to [0.5, 5] as laya applies them (`choice:11+` is 0.1006 raw, 0.5 clamped) |
| `precision_floor` | The same reads with the weights cast to float16 and to bfloat16 (the action head kept in float32, as upstream keeps it on a GPU), against float32 |
| `state_texts` | Each request's state as build_sequence tokenizes it |
| `reads` | One row per question, in corpus order |

Each row: `request`, `key`, `type`, `options`; `laya_question`, what upstream passes to laya;
`texts`, the head and option strings build_sequence tokenizes; `ids` and `markers`, its token ids
and [MASK] marker positions, the tokenization oracle for issue #58; `qtype`; `state_tokens`,
`state_tokens_read` and `truncated`; `bucket` and `temperature`; `logits`, the scorer at the
markers in float32; `probabilities_unrounded`, the softmax before laya rounds to 4 decimals;
`answer`, laya's own answer; `probabilities`, the distribution upstream publishes; `float16` and
`bfloat16`, the logits and both probability lists from the reduced-precision reads. The script
checks that rounding `probabilities_unrounded` gives laya's answer exactly.

The precision floor on this corpus, against float32. Upstream serves Laya in bfloat16 on a GPU,
so the bfloat16 row is the variation upstream's own answers carry:

| Weights | Largest logit difference | Largest probability difference, before rounding | Top answers changed |
|---|---|---|---|
| float16 | 0.012 | 0.0016 | 0 of 200 |
| bfloat16 | 0.144 | 0.0191 | 0 of 200 |

## Used by

- `Tools/encoders/Harness`: tokenization parity (swift-transformers reproduces every Verdict
  `input_ids` and every Laya `ids` and `markers`), calibration parity, and the Core ML
  measurements of [docs/spikes/encoder-runtime.md](../../docs/spikes/encoder-runtime.md).
- `Tools/encoders/convert_verdict.py` and `convert_laya.py`: the parity of each converted package.
- Issues #57 (Verdict backend) and #58 (Laya backend).
