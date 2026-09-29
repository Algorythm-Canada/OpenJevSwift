# How upstream OpenJev works

Source read: [razorback16/openjev](https://github.com/razorback16/openjev) at commit `dcd2094`
(version 0.5.0, pushed 2026-09-29). Everything below comes from the code and tests, with the
README used only for measurements the author reports.

## 1. Identity and landscape

"OpenJev" is a crowded name. Within two weeks of TypeSafe's Jev launch, dozens of repositories
appeared under it. Two matter:

| Project | What it is | Status |
|---|---|---|
| [razorback16/openjev](https://github.com/razorback16/openjev) | A Jev-compatible HTTP server. Same wire API as Jev, so TypeSafe's SDKs work unchanged. Reads answers from DiffusionGemma 26B-A4B through vLLM (NVIDIA) or MLX (Apple silicon), and serves four other people's small models behind the same API. | Still named OpenJev. 531 stars, 42 forks, 54 commits, Apache-2.0, active (pushed today). Described by third parties as "the most complete OpenJev implementation". **This is the upstream for OpenJevSwift.** |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | A research scorer: reads option logits directly from autoregressive models (Qwen3.5-4B), frames decisions as NLI, ships benchmarks and a browser demo. No Jev wire API. | Renamed from OpenJev to SemIf between 2026-09-16 and 2026-09-23. 4,572 stars. Related work, not upstream. Its letter readout is what JevK5 uses. |

Others (ekzhang/openjev-sglang, SiliconLabAI/OpenJev, Heman10x's Verdict, and so on) are either
different architectures or thin API shells. The comparison article
"[Comparing 6 Open-Source Jev Clones](https://lilting.ch/en/articles/jev-clones-architecture-comparison)"
is a useful map of the space.

## 2. What the server does

`POST /v1/systemone` takes `{state, model, questions}` and returns `{model, answers, usage}`.
Each question is one of three types:

- `noul` (yes/no): optional `criteria: {true, false}` descriptions. Answer `{type, noul: P(yes)}`.
- `choice`: `criteria: {name: description}`. Answer `{type, choice, probabilities, confidence}`.
- `score`: `criteria: [level0, level1, ...]`, 1 to 10 levels. Answer
  `{type, score: Σ i·pᵢ, legend, probabilities, confidence}`.

`confidence` is `1 − H(p)/ln K`, clipped to [0, 1]. `usage.input_tokens` counts prompt tokens
(image tokens included); `usage.output_tokens` is 0 unless `think` generated a thought.

Models served (each in its own process; one origin fans out through `OPENJEV_MODEL_ROUTES`):

| Wire name | Model | Read mechanism | Runtime |
|---|---|---|---|
| `openjev-0.1`, alias `openjev-latest` (also accepts `jev-latest`, `jev-preview`) | DiffusionGemma 26B-A4B (NVFP4 on vLLM, 4-bit on MLX) | Seeded diffusion canvas, one read-only decoder pass | vLLM or in-process MLX |
| `diffusiongemma-26b` | Same weights | Text generation, `POST /v1/chat/completions` | same |
| `laya-1.0` | Laya typed-decisions, ModernBERT-large 421M | Bidirectional encoder + classification head, one sequence per question | PyTorch |
| `verdict-1.4` | Verdict, ModernBERT-base + GLiClass 151M | Same, 25 logits (24 options + abstention) | PyTorch |
| `clm-v0.1` | CLM heads over frozen Qwen3-8B | Cosine between projected state and option embeddings | vLLM pooling + PyTorch heads |
| `jevk5-0.2` | JevK5, Qwen3.5-4B with merged LoRA | Softmax over answer-letter logits (SemIf readout) | vLLM |

## 3. Repository layout

```
openjev/
  __main__.py    8   uvicorn entry point (OPENJEV_HOST, OPENJEV_PORT, OPENJEV_LOG_LEVEL)
  api.py       340   FastAPI app: validation, auth, routes, error contract, headers
  engine.py    451   The read algorithm (vLLM HTTP backend); shared helpers for all backends
  mlx_backend.py 303 In-process DiffusionGemma on Apple silicon through mlx-vlm 0.6.15
  encoders.py  445   Laya, Verdict, CLM, JevK5 backends
  chat.py      431   OpenAI-compatible /v1/chat/completions (vLLM proxy or MLX in-process)
  config.py    166   Settings from the environment; model names and aliases
  warmup.py    116   Warms vLLM shapes before the API opens
tests/
  test_api.py          527   Offline: real tokenizer, stubbed read. Pins the contract.
  test_mlx_backend.py  743   Offline: stub runtime. Pins MLX-side behaviour and the prompt cache.
  test_mlx_model.py    322   Opt-in: the real 4-bit model (OPENJEV_MLX_TEST_MODEL).
  test_encoders.py     396   Offline: Verdict prompt contract, calibration, CLM, JevK5, routes.
  test_live.py         144   Opt-in: end to end against a running server (OPENJEV_LIVE_URL).
docker/                      Dockerfiles per image, entrypoint, vLLM vision patch
```

Dependencies: `fastapi`, `uvicorn`, `httpx`, `transformers>=4.56`, `tokenizers`, `jinja2`.
Optional: `typesafe-sdk>=0.7` (tests), `mlx-vlm==0.6.15` (darwin arm64), `laya==0.3.6`,
`gliclass==0.1.20`.

## 4. Request lifecycle (`api.py`)

1. Middleware assigns `request_id = "req_" + 32 hex`, checks auth for `/v1/*` paths
   (`OPENJEV_ORIGIN_SECRET` as `X-Origin-Secret`, `OPENJEV_API_KEY` as `Authorization: Bearer`,
   constant-time comparison on bytes), and caps the body at `OPENJEV_MAX_BODY_BYTES` (64 MiB):
   a declared `Content-Length` over the cap is a 413 before any byte is read; an undeclared
   body is counted as it streams.
2. Pydantic validates the body. `state` is `str | dict | list`; `questions` is a non-empty dict
   of a discriminated union on `type`; `criteria` values are `str | dict | list | None`
   ("Described"). Extension fields: `images` (list of data URLs or `{content_type, base64}`),
   `steps` 1..8, `samples` 1..32, `think` 0..4096, `sequential` bool.
3. Model check: a routed model is forwarded whole to its container. An unknown model is
   `400 {"detail": {"error_type": "api_usage_error", "message": "Unknown model: X"}}`.
4. More than `OPENJEV_MAX_QUESTIONS` (256) questions is `400 {"detail": "at most 256 questions per request"}`.
5. Seed: `sha256(json.dumps([state, questions, images?], sort_keys=True))[:4]` as a big-endian
   integer. Same request, same noise, same answer.
6. `engine.decide(questions, state, seed, images, options)`.
7. Errors map: `SchemaError` → 400 with a plain-string `detail` (and the `loc` logged);
   `Upstream` (vLLM refused) → 400 "the model rejected this request: ..."; `Overloaded` → 529
   `overloaded_error` with `retry-after: 1`; transport failure → 503 `api_error` with
   `retry-after: 2`.
8. Every response carries `server-timing: model;dur=A, server;dur=B, total;dur=C` (model time is
   summed over a request's reads and can exceed total when reads run in parallel), plus
   `x-typesafe-request-id` and `x-request-id`.

Validation failures are `422 {"detail": [{"type", "loc", "msg", "input", ...}]}` in FastAPI's
shape, with the echoed `input` trimmed (depth 4, 20 items, 500 chars) because a 1,000-level
nested body once blew the stack while the error was being encoded. An unknown question `type`
is special-cased to `400 api_usage_error "Invalid request."`, which is what Jev returns.
Rejected bodies are never logged, only the `loc` and reason.

## 5. The read algorithm (`engine.py`)

Constants: `VOCAB = 262144`, `TURN_CLOSE = 106` (`<end_of_turn>`), `PAD = 0`, `TOPK = 20`,
`MAX_LABEL_IDS = 512` (vLLM's per-request cap on exact logprob ids, raised from 128 in the
image), `MAX_CHOICES = 255`, `SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"` (the empty
thought block the chat template leaves for the model).

### 5.1 Single-token labels

At startup the engine finds up to 255 choice labels that stay a single token after the prefix
`"q1: "`: candidates are `A`..`Z`, `a`..`z`, then `AA`..`ZZ`; a candidate is kept when
`enc("q1: " + c)` has the same length as `enc("q1: A")`, the same prefix tokens, and a last
token not yet used. The test pins that exactly 255 distinct labels exist and start `A, B, C`.
Noul labels are `yes`/`no`; score labels are `"0"`..`"9"`.

### 5.2 Schema (`build_schema`)

Questions are numbered `q1..qN` in request order; the caller's ids never reach the model. A
choice with one option, or a score with one level, is answered without a read ("forced":
probability 1.0, confidence 1.0). Limits raise `SchemaError`: "Choice question must have at
least one choice: {id}", "Too many choices. Must have at most 255 choices.", "Too many score
levels. Must have at most 10 levels." Descriptions and instructions pass through `text_of`:
strings are stripped, objects and arrays become `json.dumps(value, ensure_ascii=False)`.
The answer format is `lines` for up to 10 questions and `indexed` beyond.

### 5.3 Prompt text (`system_text`, `answer_text`)

System prompt:

```
Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; reply with exactly one label per question.

Question q1: <instructions or "Answer about the state.">
  yes: <true description>        (noul; "  yes" alone when undescribed)
  no: <false description>
Question q2: ...
  A: <name> (<description>)      (choice; "  A: <name>" when undescribed)
Question q3: ...
  0: <level description>         (score)

Reply with one line per question, in this order, formatted as "id: label".
```

The `indexed` format's instruction is "Reply on one line with each question's id immediately
followed by its label, separated by single spaces." When questions are split across reads, the
sentence " A reply may cover only some of the questions; answer every line that is present." is
appended. The user turn is the state (a string, or `json.dumps(state, ensure_ascii=False)`),
preceded by image parts when present.

Answer text is the model turn's expected reply with every question at label index 0:
`"q1: yes\nq2: A\nq3: 0"` (lines) or `"q1yes q2A q30"` (indexed).

### 5.4 Template slots (`resolve_template`)

The answer template is `head + enc(lead + answer_text)`, where `head` is the scaffold tokens for
a plain read (or nothing when the prompt already closes a thought) and `lead` is the join text
when earlier answers are already in the prompt (sequential mode). For each question and each
alternative label, the text is re-tokenized; exactly one token may differ, at the same position
for all of a question's labels, else `SchemaError("question 'id': labels do not share one
template slot")`. The slot records its position and one token id per label. Results are cached
(4,096 entries). The template plus one closing token must fit the canvas (64 by default) or
the request is refused.

### 5.5 Grouping and canvas

`groups()` splits the questions, in order, into the fewest groups whose template rows
(`len(scaffold) + len(enc(answer_text)) + 1`) fit the canvas. Groups are read in parallel with
seeds `seed + 104729·k`. The canvas width is `len(template) + 1` rounded up to a multiple of
`OPENJEV_CANVAS_STEP` (16), capped at the canvas (64). The canvas is `template + [106] + [0]...`
with each slot position replaced by `random.Random(seed).randrange(262144)` (Python's Mersenne
Twister). The noise token at a slot is what the decoder "denoises"; its distribution is the answer.

### 5.6 One read

On vLLM: a chat completion with the system and user messages, `max_tokens = len(template) + 1`,
`logprobs` with `top_logprobs = 20`, `logprob_token_ids = sorted union of all slot label ids`
(exact logprobs for every label at every position; long option lists rarely rank in the top-k),
`chat_template_kwargs = {"enable_thinking": false}`, and `vllm_xargs = {diffusion_seed_canvas,
diffusion_canvas_length, diffusion_max_steps, diffusion_read_only: true}` (plus
`diffusion_pinned` for every non-slot position when `steps > 1`). The returned per-position
logprobs at each slot position feed `slot_distribution`.

On MLX: the same prompt ids are prefilled through the encoder (cached), the canvas runs through
one decoder pass, and the slot rows are converted to log-softmax in float32. The MLX backend
returns exact logprobs for the top 20 tokens plus every label id.

`slot_distribution(top, label_ids)`: label logprobs (missing ones floored at `min(top) − 5`),
softmax over the labels only, and the entropy of the returned top-k set (used by the re-read
policy). Read-only logprobs are at temperature 1, so no rescaling is applied.

### 5.7 Policies (`read_group`, `decide`)

- **Automatic re-reads.** If any slot's top-k entropy exceeds `OPENJEV_AUTO_THRESHOLD` (0.1),
  three more reads run with seeds `seed + 7919·k` (k = 1..3) and the four label distributions
  are averaged. Re-reads are the server's policy and are not billed. `OPENJEV_AUTO_MAX` (4) is
  the total.
- **`samples: N`** replaces the automatic policy: N reads with seeds `seed + 7919·k`, all billed.
  `samples: 1` gives exactly one read.
- **`steps: S`** runs S denoise passes per read, argmax written back to the slot positions
  between passes, the rest of the canvas pinned; same prompt tokens, more GPU time.
- **`think: T`** first generates up to T tokens after `<|channel>thought\n` with thinking on,
  stops at `<channel|>`, then reads with the thought as prefix. Billing: the input is counted for
  the thought pass and again for the read; the thought's tokens are `usage.output_tokens`.
- **`sequential: true`** with more than one group: the system prompt lists all questions; each
  group's argmax labels are written into the prompt before the next group is read, so later
  answers condition on earlier ones. One read per group, in series.
- `images` cannot be combined with `think` or `sequential` (400 naming the field at fault).
- Capacity: `waiting >= OPENJEV_MAX_QUEUE` (512) raises `Overloaded`; `OPENJEV_MAX_INFLIGHT`
  (64) bounds concurrent reads through a semaphore.

Answers are assembled in request order, forced answers included. `to_answer` produces Jev's
exact shapes; a noul answer has exactly the keys `type` and `noul`.

## 6. The MLX backend (`mlx_backend.py`)

- One `ThreadPoolExecutor(max_workers=1)` owns the model from loading on. All MLX work runs on it.
- `mlx_vlm.load(model_path)` returns the model and processor; the tokenizer comes from the same
  directory (`OPENJEV_MLX_MODEL`, default `mlx-community/diffusiongemma-26B-A4B-it-4bit`).
- Prefill cache: an `OrderedDict` keyed by the prompt token tuple (or, for images, by the system
  text, state text and image digests). Two budgets: `OPENJEV_MLX_PROMPT_CACHE` entries (12) and
  16,384 tokens. Entries are evicted oldest first; no entry is exempt. Measured: about 30 to
  50 MB per cached prefill on the 4-bit weights, plateauing at 27 GB after 360 prompts, 36 GB in a
  real audit run. The entry budget is the one that binds.
- `read(prompt, canvas, slots, max_tokens, steps)`: prefill (or hit) →
  `diffusion_decoder_masks` → for each step `diffusion_decoder_logits(ids, cache,
  self_conditioning, masks)`; between steps write `argmax(logits[0, slot_positions])` into the
  slots and compute self-conditioning from the logits; return, for each slot, `{token id:
  logprob}` for the top 20 tokens and every label, in float32 log-softmax.
- `generate(...)` uses mlx-vlm's `stream_diffusion_generate` at temperature 0 with extra stop
  ids and `skip_special_token_ids`; the checkpoint's own sampling policy is used rather than a
  reimplementation.
- `OPENJEV_MLX_CACHE_LIMIT_GB` caps MLX's buffer pool (`mx.set_cache_limit`). Unset leaves MLX's
  default (the pool grows to the peak working set); `0` disables the pool (worst churn). A 4 GB
  cap turned a 36 GB process into 23.5 GB with byte-identical answers.
- `OPENJEV_MLX_MAX_PROMPT` (32,768) is the longest prompt in tokens before a 400.
- Reads run one at a time; the README calls this backend "for local use, not for serving":
  0.2 to 0.4 s for a 3-question request on an M3 Ultra, 0.39 s on an M4 Max, about 4 req/s at
  16 concurrent requests.

## 7. Encoder and letter-readout backends (`encoders.py`)

All four share `EncoderEngine`: one model thread, `build_schema` with the same forced answers
and limits, `decide` refusing `images`, `steps > 1`, `samples > 1`, `think` and `sequential`
with a 400, a queue bound, and the same `to_answer`. Questions are read in batches of
`OPENJEV_ENCODER_BATCH` (16). Details per model are in [10-other-models.md](10-other-models.md).

## 8. Text generation (`chat.py`)

OpenAI-compatible. Passthrough fields: `messages, max_tokens, stop, top_p, top_k, stream,
stream_options, tools, tool_choice, logprobs, top_logprobs, chat_template_kwargs`. Everything
else (temperature, seed, penalties, response_format, n, reasoning) is dropped because vLLM
refuses them for diffusion models. `max_tokens` must be a positive integer (default 1024, cap
`OPENJEV_GEN_MAX_TOKENS` 8192). JSON mode becomes an instruction appended to the system message
plus extraction of the first JSON object from the reply. Model names: `diffusiongemma-26b` and
`diffusiongemma`; anything else is 404 `model_not_found`. Capacity: `OPENJEV_GEN_MAX_INFLIGHT`
(8) and `OPENJEV_GEN_MAX_QUEUE` (32), else 529.

The MLX generator seeds the prompt with the empty thought scaffold, skips the thought-channel
marker tokens at the detokenizer (the model opens a channel of its own accord on some replies),
honours only single-token `stop` strings, streams over an `asyncio.Queue(64)`, cancels at the
next denoised block when the client disconnects or falls hopelessly behind, and reports usage.

## 9. Settings (`config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_BACKEND` | `vllm` | `mlx`, `laya`, `verdict`, `clm`, `jevk5` |
| `OPENJEV_MODEL_ROUTES` | unset | `name=url,...` other OpenJev servers; requests pass through unchanged |
| `OPENJEV_FORWARD_TIMEOUT` | `300` | seconds before a forwarded request is a 503 |
| `OPENJEV_UPSTREAM`, `OPENJEV_UPSTREAM_MODEL` | `http://127.0.0.1:8000`, `dgemma` | vLLM server |
| `OPENJEV_TOKENIZER` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | tokenizer for the vLLM path |
| `OPENJEV_MLX_MODEL` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | MLX weights and tokenizer |
| `OPENJEV_MLX_MAX_PROMPT` | `32768` | longest request in tokens |
| `OPENJEV_MLX_CACHE_LIMIT_GB` | unset | MLX buffer pool cap |
| `OPENJEV_MLX_PROMPT_CACHE` | `12` | cached prefills, in entries |
| `OPENJEV_CANVAS`, `OPENJEV_CANVAS_STEP` | `64`, `16` | canvas length and width rounding |
| `OPENJEV_MAX_INFLIGHT`, `OPENJEV_MAX_QUEUE` | `64`, `512` | reads in flight; waiting decisions before 529 |
| `OPENJEV_MAX_QUESTIONS` | `256` | questions per request before 400 |
| `OPENJEV_MAX_BODY_BYTES` | `67108864` | request body cap before 413 |
| `OPENJEV_API_KEY`, `OPENJEV_ORIGIN_SECRET` | unset | auth |
| `OPENJEV_AUTO_THRESHOLD`, `OPENJEV_AUTO_MAX` | `0.1`, `4` | re-read policy |
| `OPENJEV_MAX_IMAGES`, `OPENJEV_MAX_IMAGE_BYTES` | `8`, `5242880` | image limits |
| `OPENJEV_GEN_MAX_INFLIGHT`, `OPENJEV_GEN_MAX_QUEUE`, `OPENJEV_GEN_MAX_TOKENS` | `8`, `32`, `8192` | text generation |
| `OPENJEV_WARMUP` | `1` | warm up before serving |
| `OPENJEV_LAYA_MODEL`, `OPENJEV_VERDICT_MODEL`, `OPENJEV_DEVICE`, `OPENJEV_ENCODER_BATCH` | see upstream | encoder models |
| `OPENJEV_CLM_*`, `OPENJEV_JEVK5_*` | see upstream | CLM and JevK5 |

Settings are validated at startup: a zero canvas step or a zero semaphore used to turn into
500s or hangs.

Model metadata (`GET /v1/models` returns `{"models": [{name, description, release_date}]}`):
`openjev-latest` and `openjev-0.1` (2026-09-18), `diffusiongemma-26b` (2026-09-18),
`laya-1.0` and `verdict-1.4` (2026-09-22), `clm-v0.1` (2026-09-24), `jevk5-0.2` (2026-09-25).
Responses name the model version (`openjev-0.1`), not the alias the request used.

## 10. What the tests pin

`test_api.py` is the contract's executable form. It pins, among others: 255 single-token labels
starting `A, B, C`; the widest schema (255 + 10 + 2 label ids) fits one read; the quickstart
request decodes with the official `typesafe-sdk`; the confidence formula matches Jev's documented
example (`[0.84, 0.159, 0.001] → 0.596`); every read option defaults to Jev's behaviour; steps
and samples billing; think continues after the thought with output tokens billed; sequential
prefills earlier answers; single-option choices and single-level scores are answered without a
read; deeply nested bodies are rejected, not crashed; invalid requests are logged without their
body; body cap; questions cap; image validation before decoding; auth with non-ASCII
credentials. `test_mlx_backend.py` pins the prompt cache's two budgets and eviction rules, the
chat stream's cancellation and "never drop a chunk" guarantees, and that the thought channel
never reaches a chat client. `test_encoders.py` pins Verdict's prompt contract and calibration,
CLM's option order and left truncation, JevK5's letters, temperature and multi-pass reads.

## 11. Upstream caveats and open items

- The Docker image pins vLLM `1b3b88ec` with two source edits: `MAX_LOGPROB_TOKEN_IDS` 128 → 512
  and `docker/patches/vision_prefix_lm.py`, which gives image tokens the bidirectional attention
  the checkpoint asks for (`use_bidirectional_attention: "vision"`). Upstream vLLM did that for
  Gemma 4 but not yet for DiffusionGemma. The MLX backend does not need the patch: mlx-vlm's
  encoder applies the vision overlay itself.
- Open issues at the time of reading: #5 (throughput table lacks the state size), #6 (JevBench
  v1.4 results). Open PR #10 adds a "ForJev" backend for typed decisions on an existing
  Qwen/vLLM server. The model list has grown by two in a week; the compatibility target must be
  a pinned commit, tracked deliberately.
- Answer quality is DiffusionGemma's quality in this mode. Upstream applies no calibration to
  the diffusion reads; the encoder models carry their authors' fitted temperatures.
