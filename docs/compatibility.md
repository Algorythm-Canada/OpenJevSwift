# Compatibility with upstream OpenJev

OpenJevSwift is compatible with upstream OpenJev,
[razorback16/openjev](https://github.com/razorback16/openjev) at `dcd2094` (0.5.0), which implements
TypeSafe's published contract for `/v1/systemone`. This page says what is identical, what agrees
within a measured tolerance, and what differs and why, as of 2026-10-06. Each difference names the
decision in [06-decisions.md](06-decisions.md) that records it;
[09-conformance-and-testing.md](09-conformance-and-testing.md) describes how each claim is tested.
The last section is the matrix of what runs where.

## Identical to upstream

Apart from the differences the third table lists, everything up to the model's probabilities is
upstream's, byte for byte. The golden fixtures are written by upstream's own code at the pinned
commit ([Fixtures/](../Fixtures/README.md)), and the tests compare with them exactly.

| Area | What is identical | Checked against |
|---|---|---|
| Question schema | The choice labels, the forced answers of one-option choices and one-level scores, the limits and their messages | `Fixtures/schemas`, `labels.json` |
| Prompts | The system texts, the state texts, the answer templates and formats, and the chat prompt ids with thinking on and off | `Fixtures/system-texts`, `Fixtures/templates`, `Fixtures/chat-prompts` |
| Tokens | The ids of every text upstream's engine tokenizes: 917 corpus rows and 2,634 engine encodings, with the DiffusionGemma tokenizer (D-008) | `Fixtures/tokenizer` |
| Canvas and shapes | The grouping of questions into reads, the canvas widths, the slot positions and label ids, the request seeds and the Mersenne Twister noise | `Fixtures/groups-and-canvases`, `Fixtures/templates`, `seeds.json` |
| Read policies | The reads `samples`, `steps`, `sequential` and the automatic re-reads make, with their seeds and prefixes, and the billed tokens | `Fixtures/policies` |
| Answers from probabilities | Each slot's distribution and entropy from the same log-probabilities, the confidence, the choice, the score, the averaging | `Fixtures/distributions` |
| Encoder inputs | Verdict's prompts, Laya's heads, options, sequences and truncation, byte for byte; their calibration within 1e-6, Laya's bit for bit | `Fixtures/encoders` |
| JevK5 prompts and readout | The `jevk5` package's option texts and prompts byte for byte and transformers' token ids for them, on 324 passes; its letter softmax and `spread` bit for bit on the same logits; the billed tokens of all 231 JevBench items against the author's published run | `Fixtures/jevk5`, `Tools/jevbench/results` |
| Requests | Which bodies are accepted, pydantic's lax coercion of the extension fields, every 422 with its `loc`, `msg`, `input` and `ctx`, and CPython's `json_invalid` message and position for a body that is not JSON (728 recorded documents) | `Fixtures/wire/cases.json`, `Fixtures/python-json` |
| Errors | The status, body and `retry-after` of the errors both answer: 400, 401, 403, 413, 422, 503 and 529, and the 404 of an unknown path | `Fixtures/wire`, `Fixtures/errors`; the 404 against FastAPI's body in the server's application tests |
| Headers | `x-typesafe-request-id` and `x-request-id`, the same `req_` id with 32 hex digits, on every response; `server-timing` with its `model`, `server` and `total` spans | `Fixtures/wire`, the live suite |
| Response bytes | Key order, the exact key sets of each answer, Python's float formatting and FastAPI's compact separators | `Fixtures/wire/answers.json` |
| The listing | The `/v1/models` body of each backend, the served version a response names, and the accepted names, the SDK aliases `jev-latest` and `jev-preview` included | `Fixtures/wire/models.json` |
| Model routes | Which requests are forwarded, the bytes and the three headers sent, what comes back, the routed listing, and the 503 names of a failed exchange | the server's route tests (D-040) |
| Settings | Upstream's `OPENJEV_*` variables: their names, defaults and startup checks and the checks' messages, apart from the settings rows of the third table; `OPENJEV_ENCODER_MODELS` and `OPENJEV_ENCODER_FUNCTIONS` are this port's own | the settings tests, upstream's `test_settings_are_checked_at_startup` |
| Clients | TypeSafe's Python SDK 0.7.2 and TypeScript SDK 0.6.0, unchanged, decode answers, retry the 529, and raise the authentication error and the 422 as against upstream | `Tools/sdk-compat`, in CI (D-040) |
| Chat completions | `POST /v1/chat/completions` as upstream's MLX generator answers it: `Generator.normalize` on 56 bodies (the fields kept, `max_tokens` and its errors, thinking, forced streamed usage, JSON mode's instruction), `extract_json` on 44 replies, the prompt text and ids `MlxGenerator.prompt_ids` gives for 51 conversations, and the route's statuses, error bodies, whole replies and event streams, byte for byte, on 61 of 69 recorded exchanges over upstream's stub runtimes; the other 8 are the chat rows of the third table | `Fixtures/chat-completions` (D-058) |

Upstream's own end-to-end file, `tests/test_live.py`, passes against the Swift server on Verdict
and Laya, and the Swift port of it passes unchanged against both servers on all three backends
(D-043) and against the Swift server's `jevk5` backend, whose upstream counterpart needs vLLM.

## Within tolerance

The model's probabilities, and the confidences, scores and argmaxes computed from them, agree
within measured bounds rather than bit for bit, because this port runs other kernels on other
hardware. Every bound is met. Past 1,024 tokens the benchmark answers' figures are reported, not
bounded: D-048 bounds the reads there, and an answer averages up to four of them.

| Backend | Figure | Bound | Measured |
|---|---|---|---|
| `mlx`, mlx-swift's own kernels | Mean label probability difference, all 1,883 labels of the 63 oracle reads | 0.02 (D-014) | 0.0133 |
| | Mean of each slot's largest label probability difference, the 86 slots of prompts past 1,024 tokens | 0.14 (D-048) | 0.1006 |
| | Mean label probability difference, prompts past 1,024 tokens | reported, not bounded (D-048) | 0.0113 |
| | Mean entropy difference, all 192 slots | 0.2 | 0.104 |
| | The same, prompts past 1,024 tokens | 0.27 (D-048) | 0.152 |
| | Top label agreement, all slots | 90% | 91.7%, 176 of 192 |
| | Top label agreement where mlx-vlm's top two are at least 0.5 apart | 97% | 99.3%, 139 of 140 |
| | Largest label probability difference | reported, not bounded | 0.500, on a read of JevBench's `hard-opus-c-long_policy-04` |
| `mlx`, mlx-vlm's Metal library and RoPE table | Reads identical to mlx-vlm's | every read | 63 of 63, bit for bit (D-036, D-048) |
| `mlx`, the Swift server against upstream's on JevBench and TypeSafe | Mean probability difference | 0.02 (D-014) | 0.0135 over JevBench's 231 items, 0.0069 over TypeSafe's 102 |
| | Prompts past 1,024 tokens: mean of each item's largest difference, and mean over every label | reported, not bounded (D-048) | 0.0102 and 0.0075 over TypeSafe's 76; 0.0741 and 0.0396 over JevBench's 41 |
| | Top answer agreement, all items | 90% | 97.9%, 326 of 333 |
| | Top answer agreement where upstream's top two are at least 0.5 apart | 97% | 100%, 299 of 299 |
| `verdict`, Core ML float16 against upstream's PyTorch float32 | Largest probability difference | 0.02 (D-034) | 0.0022 over 333 JevBench and TypeSafe items; 0.0014 over spike #56's 200 questions |
| | Mean probability difference | 0.003 | 2.8e-4 on JevBench and 2.9e-4 on TypeSafe |
| | Top answer, wherever upstream's top two are at least 0.01 apart | unchanged | unchanged on all 333 items |
| `laya`, Core ML float16 against upstream's PyTorch float32 | Largest probability difference | 0.02, before laya's 4-decimal rounding (D-037) | 0.0026 over 333 items, between the rounded answers the servers sent; 0.0039 over spike #56's 200 questions, before rounding |
| | Mean probability difference | 0.003, before rounding | 2.1e-4 on JevBench and 1.6e-4 on TypeSafe, between the rounded answers |
| | Top answer, wherever upstream's top two are at least 0.01 apart | unchanged | unchanged on all 333 items |
| `jevk5`, the 8-bit conversion against the author's published run (transformers, bfloat16, CUDA) | Top answer agreement on JevBench's 231 items | all 231 (issue #55) | 230; the other a near-tie in the author's run, its top two 0.040 apart |
| | Top answer where the author's top two are at least 0.05 apart | unchanged | unchanged on all 219 |
| | Billed input tokens | equal | equal on all 231 |
| | Largest and mean probability difference | reported, not bounded; upstream measured 0.055 largest on vLLM | 0.083 and 0.0049 |
| `jevk5`, the unquantized bfloat16 weights against the same run | Top answer agreement; largest and mean probability difference | reported | 228 of 231, each miss within 0.031 of a tie; 0.051 and 0.0037 |
| `jevk5`, the 4-bit conversion against the same run | Top answer agreement; largest and mean probability difference | reported | 209 of 231; 0.53 and 0.044 |
| `jevk5`, the Swift model against mlx-lm's on the same 4-bit conversion | Largest and mean letter-logit difference over the fixture's 324 passes | 1.0 and 0.1 | 0.375 and 0.068; 51 passes identical |
| | Top letter, wherever mlx-lm's top two logits are more than 0.5 apart | unchanged | unchanged; 321 of all 324 passes keep it |
| | Largest answer probability difference over the fixture's 204 questions; top answer where mlx-lm's top two are at least 0.1 apart | 0.1; unchanged | 0.062; four answers changed, each where mlx-lm's top two are less than 0.1 apart |

The first `mlx` figures are D-014's bounds, with D-048's long-prompt rows, over `Fixtures/oracle`,
measured on 2026-10-02 on an M3 Max with the pinned 4-bit checkpoint
([09-conformance-and-testing.md](09-conformance-and-testing.md), layer 2). The encoder figures are
from [quality.md](quality.md): on JevBench's 231 public items and SemIf's 102 TypeSafe rows, the
Swift server gave upstream's top answer on every item for both models, including the 36 whose
upstream top two were less than 0.01 apart; accuracies are the same and Brier scores and ECEs differ
by at most 0.0003. The server-against-server `mlx` rows are quality.md's too, from the same 333
items on both servers with MLX's buffer pool capped at 4 GB: they meet every bound an answer
carries, and JevBench's 41 prompts past 1,024 tokens, mostly the hard tier's uncertain answers,
differ more than TypeSafe's. Its section
[A disagreement on DiffusionGemma](quality.md#a-disagreement-on-diffusiongemma) shows the difference
is the kernels': in D-014's exact tier the port reads the four long-prompt items whose top answers
differ at a wide margin as mlx-vlm does, bit for bit.

The `jevk5` rows are quality.md's, from the 2026-10-03 UTC runs on the same Mac with MLX's buffer
pool capped at 4 GB, and D-052's live tests. Upstream reached all 231 top answers on vLLM; on MLX
no conversion does, and the misses at 8 bits and in bfloat16 are near-ties in the author's run,
where the order of bfloat16 arithmetic decides. The 4-bit conversion changes clear answers too,
which is why the server loads the 8-bit one ([quality.md](quality.md#jevk5)).

## Different, and why

| Area | Upstream | OpenJevSwift | Decision |
|---|---|---|---|
| JSON values RFC 8259 refuses | `json.loads` accepts `NaN`, `Infinity`, floats that overflow and lone surrogate escapes, then validates the value | The 422 `json_invalid` that CPython gives at that place ("Expecting value", "Invalid \uXXXX escape") | D-016, D-031 |
| Deep nesting | Parsed up to Python's stack, then the 400 "There was an error parsing the body" | The parser stops at 1,024 levels with that 400 | D-031 |
| UTF-16 and UTF-32 bodies | Decoded | Read as UTF-8 only | D-031 |
| `server-timing` on 401, 403 and 413 | Absent: the middleware answers before routing | Present on every response | D-030, D-031 |
| Unusual `/v1/` paths | `//v1/models` is a 404; `/%761/models` is decoded and authenticated | `//v1/models` needs the key and is served; `/%761/models` is a 404 | D-031 |
| Header values that are not UTF-8 | Compared as Latin-1 bytes | Arrive as U+FFFD, so such a key never matches | D-031 |
| An engine error other than httpx's | Starlette's plain-text 500 | The 503 `inference backend unavailable: <type name>`, with `retry-after: 2` | D-031 |
| Logging | The invalid-request 400 logged with status 422; backend failures not logged; uvicorn's access log with the client address and query string | The 400 logged as 400; backend failures logged with the 503 message, type name only; one line per request with method, path, status, milliseconds and request id, never a query string | D-031, D-038 |
| Error messages' `repr` | CPython's Unicode tables decide which characters are escaped | This platform's tables, which can differ for newly assigned characters | D-018 |
| Settings errors | A bare `ValueError` naming only the text; any integer accepted | The message names the variable; an integer beyond `Int` is refused | D-030 |
| `OPENJEV_LOG_LEVEL` | uvicorn's level names | The same and swift-log's `notice` | D-030 |
| Backends | `vllm` by default, and `mlx`, `laya`, `verdict`, `clm`, `jevk5` | `mlx` by default, `laya`, `verdict` and `jevk5`; `vllm` and `clm` are unknown names (issue #59) | D-030, D-038, D-052 |
| DiffusionGemma checkpoint | The newest revision of `OPENJEV_MLX_MODEL`'s repository | The default repository loads the pinned revision `a7a81407`; `repo@revision` picks another | D-039 |
| `think` on `mlx` | Supported | `openjev-0.1 does not support think` until issue #52; with images, that refusal comes before upstream's `think needs a text state` | D-039, D-054 |
| An image that cannot be read on `mlx` | Pillow's or the processor's exception, answered as a bare 500 | The 400 `image could not be read: {reason}` at `["body", "images", i]` | D-054 |
| Image formats on `mlx` | Whatever Pillow identifies by its bytes, whatever the declared type | JPEG, PNG, WebP or GIF by their bytes, whatever the declared type; a TIFF or BMP under another label is a 400 | D-054 |
| Truncated JPEGs and images 3 pixels high on `mlx` | A truncated JPEG is a 500, though one missing only its EOI is read when libjpeg does not look past its end; an image 3 pixels high is read as channels first and answered | A truncated JPEG is a 400 where Pillow raises and read where Pillow reads it; an image 3 pixels high is a 400 | D-051, D-054, D-055 |
| A cached image prefill on `mlx` | The images are decoded again for every read | Decoded once per prefill; a cached one reuses its count, so the answers and the billing are the same | D-054 |
| `POST /v1/chat/completions` | Served by the `mlx` backend | The route is ported and answers once the `mlx` backend's model generates text (issue #51, then the wiring that follows #53); until then a 404, though `/v1/models` lists `diffusiongemma-26b` as upstream's does | D-012, D-043, D-058 |
| Chat requests upstream crashes on | A message that is not an object or whose `role` is not a string, a `chat_template_kwargs`, `response_format`, `json_schema` or streaming `stream_options` that is true but not a dict, a `stop` of another type, or a template error: a bare 500, or `dict()`'s message for a string message in JSON mode | The 400 `invalid_request_error` naming the field; messages nested past 64 levels are refused too | D-058 |
| A chat generation that fails | A bare 500, or a stream that breaks off | The 503 `inference backend unavailable: <type name>` with `retry-after: 2` before the answer starts, logged; a stream that broke off is logged | D-058 |
| A chat client that goes away | A whole reply runs to its end; a stream notices within 0.1 s and stops at the next block | Both stop at the next block, at once; a whole reply's request logs 499 | D-058 |
| A chat stream's reader 64 pieces behind | The reply ends, its end marker displacing the oldest queued piece | The reply ends after every queued piece | D-058 |
| Chat bodies `json.loads` reads and RFC 8259 refuses | Served | The 400 "The request body is not valid JSON." | D-016, D-058 |
| `null` written by the chat template | `None`, in a tool call's arguments or a tool's missing result | Nothing; an integer past `Int` is written as a float | D-058 |
| A chat object whose keys differ only in Unicode normalization | Both keys kept, and the template writes both | The 400 `invalid_request_error`: the template engine keys an object by Swift's `String`, which would keep one | D-058 |
| When a chat request counts against the capacity bound | From when its prompt has rendered, with nothing else running on the event loop meanwhile | From the capacity check, before its prompt renders, since prompts render concurrently here: a request refused afterwards (a 400 for its prompt, say) holds a place while it renders | D-058 |
| JSON mode's reply | A lone surrogate escape kept, then a 500 encoding the answer; a value nested past CPython's stack, a 500 | The surrogate is U+FFFD; a value nested past 1,024 levels leaves the reply unchanged | D-058 |
| `server-timing` `model` on `mlx` | `0.0`: the MLX engine does not time its reads | The time spent in reads, summed over reads that ran at once, so it can exceed `total` | D-038, D-044 |
| Encoder arithmetic | PyTorch, on CUDA when present, else in float32 on the CPU | Core ML packages in float16 on the GPU or the Neural Engine, within the bounds above | D-011, D-034, D-037 |
| Encoder weights | The checkpoint that `OPENJEV_VERDICT_MODEL` or `OPENJEV_LAYA_MODEL` names, on the device `OPENJEV_DEVICE` names | Converted packages from the `Algorythm-Canada/openjev-models` releases at pinned digests, or `OPENJEV_ENCODER_MODELS`' folder; those three variables are read but have no effect | D-033, D-047 |
| Encoder padding | Rows padded to the longest row | Rows padded to the Core ML function's length; the billed tokens are the same | D-034, D-037 |
| A model's malformed output | Raised from the read or broadcast silently | `EncoderModelError`, answered as the 503 | D-034, D-037 |
| Encoder memory | One PyTorch model | Each Core ML function a read needs stays loaded with its own copy of the weights, up to 6 for Verdict and 8 for Laya, unless `OPENJEV_ENCODER_FUNCTIONS` caps them | D-042 |
| A client that goes away | Its request runs to the end | Its decision or forwarded exchange is cancelled, and the log shows 499 | D-038, D-040 |
| Shutdown and exit statuses | uvicorn's | Requests in flight get `--shutdown-timeout` (30 s); exit statuses 0 to 4 for scripts and launchd | D-038 |
| Forwarded requests | httpx adds `accept`, `accept-encoding: gzip, deflate`, `connection` and `user-agent`, honours `HTTP_PROXY`, pools connections, and answers an unparsable route URL with a 500 | AsyncHTTPClient adds `host`, `content-length` and `accept-encoding: deflate, gzip`, ignores proxy variables, opens one connection per request, and answers an unparsable URL with the 503 `InvalidURL` | D-040 |
| Checkpoint downloads | huggingface_hub, with file locks | The port's own downloader in the same cache layout, without locks: two processes must not download the same file at once | D-039 |
| JevK5's runtime | A vLLM server on an NVIDIA GPU, `OPENJEV_MODEL`'s weights in bfloat16, up to `OPENJEV_JEVK5_WORKERS` reads in flight | The model in the process on MLX, an 8-bit conversion that `OPENJEV_JEVK5_MODEL` names (by default the published repository at its pinned commit, or a folder), the questions of a request read concurrently and their passes one at a time; `OPENJEV_JEVK5_WORKERS` is not read | D-052 |
| JevK5's checkpoint | `alibiserikbay/JevK5` at its newest revision, which has held v0.3 since 2026-09-25, served under the name `jevk5-0.2` | The author's `v0.2` tag, `ea4804e`, with v0.2's temperature of 1.532 | D-052 |
| JevK5's 400 | vLLM's own refusal of a prompt over its context, passed on | The same texts, from vLLM's source at upstream's pinned commit, not from a run of it | D-052 |

## What runs where

Each cell is "yes", "no" with the issue that adds it, or "n/a" where the combination does not
apply. The cells follow from each backend's `BackendCapabilities` (DiffusionGemma's runtime honours
`steps`, `samples` and `sequential`; the encoder engine checks against `.readsOnly`, as upstream's
encoder engines do), from the `openjev` tool's `BackendRegistry`, and from the platform matrix in
[05-architecture.md](05-architecture.md).

| Platform | Backend | Reads | `steps` | `samples` | `sequential` | Images | `think` | Chat |
|---|---|---|---|---|---|---|---|---|
| macOS 14 or later, Apple silicon | `mlx` | yes | yes | yes | yes | yes | no, issue #52 | the route yes; the model no, issue #51 |
| macOS 15 or later | `verdict` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| macOS 15 or later | `laya` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| macOS 14 or later, Apple silicon | `jevk5` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| macOS | routed models | yes | yes | yes | yes | yes | yes | n/a |
| iOS 18 or later | `mlx` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS 18 or later | `verdict` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS 18 or later | `laya` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS 17 or later | `jevk5` | builds, not yet run on an iPhone | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS | routed models | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `mlx` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `verdict` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `laya` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `jevk5` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | routed models | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

- **The encoders' n/a** is upstream's own contract: its encoder engines refuse `steps` and
  `samples` above 1, `think`, `sequential` and images with `"{model} does not support {field}"`,
  and its encoder containers have no chat route. This port answers the same.
- **`steps`, `samples` and `sequential` on `mlx`** run through the engine and the runtime and are
  verified end to end on the real checkpoint: upstream's read cases (D-044) and issues #43, #44 and
  #45 (D-045).
- **Routed models** are requests a server forwards to the OpenJev server `OPENJEV_MODEL_ROUTES`
  names, unchanged, so the options are whatever that server honours. Only `/v1/systemone` is
  forwarded, as upstream forwards it.
- **iOS** runs the encoders in an app, through `EncoderDecisionEngine`, from iOS 18; there is no
  server on iOS, and no iPhone holds DiffusionGemma, although the module compiles for it. Laya on
  an iPhone reads through one package per sequence length, which the app fetches.
- **Linux** builds `OpenJevCore`, the server and the `openjev` tool for the tests and the SDK
  suite's stub server, but no backend: MLX and Core ML are Apple's, so `serve` and `decide` exit
  with status 3 whatever the backend (`models` still prints a listing, which needs no model), and
  a server cannot start to forward routes.
- **`jevk5`** runs wherever MLX does: the server on an Apple silicon Mac, and in an app through
  `EncoderDecisionEngine`, which compiles for iOS but has not run on an iPhone yet. It is a reads-only
  model like the encoders, so its other cells are upstream's n/a.
- **CLM**, upstream's other model, is not served on any platform yet (issue #59).
