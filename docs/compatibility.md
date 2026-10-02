# Compatibility with upstream OpenJev

OpenJevSwift is compatible with upstream OpenJev, [razorback16/openjev](https://github.com/razorback16/openjev)
at `dcd2094` (0.5.0), which implements TypeSafe's published contract for `/v1/systemone`. This
page says what is identical, what agrees within a measured tolerance, and what differs and why,
as of 2026-10-02. Each difference names the decision in [06-decisions.md](06-decisions.md) that
records it; [09-conformance-and-testing.md](09-conformance-and-testing.md) describes how each claim
is tested. The last section is the matrix of what runs where.

## Identical to upstream

Everything up to the model's probabilities is upstream's, byte for byte. The golden fixtures are
written by upstream's own code at the pinned commit ([Fixtures/](../Fixtures/README.md)), and the
tests compare with them exactly.

| Area | What is identical | Checked against |
|---|---|---|
| Question schema | The choice labels, the forced answers of one-option choices and one-level scores, the limits and their messages | `Fixtures/schemas`, `labels.json` |
| Prompts | The system texts, the state texts, the answer templates and formats, and the chat prompt ids with thinking on and off | `Fixtures/system-texts`, `Fixtures/templates`, `Fixtures/chat-prompts` |
| Tokens | The ids of every text upstream's engine tokenizes: 917 corpus rows and 2,634 engine encodings, with the DiffusionGemma tokenizer (D-008) | `Fixtures/tokenizer` |
| Canvas and shapes | The grouping of questions into reads, the canvas widths, the slot positions and label ids, the request seeds and the Mersenne Twister noise | `Fixtures/groups-and-canvases`, `Fixtures/templates`, `seeds.json` |
| Read policies | The reads `samples`, `steps`, `sequential` and the automatic re-reads make, with their seeds and prefixes, and the billed tokens | `Fixtures/policies` |
| Answers from probabilities | Each slot's distribution and entropy from the same log-probabilities, the confidence, the choice, the score, the averaging | `Fixtures/distributions` |
| Encoder inputs | Verdict's prompts, Laya's heads, options, sequences and truncation, byte for byte; their calibration within 1e-6, Laya's bit for bit | `Fixtures/encoders` |
| Requests | Which bodies are accepted, pydantic's lax coercion of the extension fields, every 422 with its `loc`, `msg`, `input` and `ctx`, and CPython's `json_invalid` message and position for a body that is not JSON (728 recorded documents) | `Fixtures/wire/cases.json`, `Fixtures/python-json` |
| Errors | The status, body and `retry-after` of every error: 400, 401, 403, 404, 413, 422, 503 and 529 | `Fixtures/wire`, `Fixtures/errors` |
| Headers | `x-typesafe-request-id` and `x-request-id`, the same `req_` id with 32 hex digits, on every response; `server-timing` with its `model`, `server` and `total` spans | `Fixtures/wire`, the live suite |
| Response bytes | Key order, the exact key sets of each answer, Python's float formatting and FastAPI's compact separators | `Fixtures/wire/answers.json` |
| The listing | The `/v1/models` body of each backend, the served version a response names, and the accepted names, the SDK aliases `jev-latest` and `jev-preview` included | `Fixtures/wire/models.json` |
| Model routes | Which requests are forwarded, the bytes and the three headers sent, what comes back, the routed listing, and the 503 names of a failed exchange | the server's route tests (D-040) |
| Settings | The names, defaults and startup checks of every `OPENJEV_*` variable this port reads, and their messages | the settings tests, upstream's `test_settings_are_checked_at_startup` |
| Clients | TypeSafe's Python SDK 0.7.2 and TypeScript SDK 0.6.0, unchanged, decode answers, retry the 529, and raise the authentication error and the 422 as against upstream | `Tools/sdk-compat`, in CI (D-040) |

Upstream's own end-to-end file, `tests/test_live.py`, passes against the Swift server on Verdict
and Laya, and the Swift port of it passes unchanged against both servers on all three backends
(D-043).

## Within tolerance

The model's probabilities, and the confidences, scores and argmaxes computed from them, agree
within measured bounds rather than bit for bit, because this port runs other kernels on other
hardware.

| Backend | Figure | Bound | Measured |
|---|---|---|---|
| `mlx`, mlx-swift's own kernels | Mean label probability difference, all 1,763 labels of the 27 oracle reads | 0.02 (D-014) | 0.0084 |
| | The same, prompts past 1,024 tokens | 0.01 | 0.0054 |
| | Mean entropy difference, all 156 slots | 0.2 | 0.086 |
| | The same, prompts past 1,024 tokens | 0.2 | 0.131 |
| | Top label agreement, all slots | 90% | 96.2%, 150 of 156 |
| | Top label agreement where mlx-vlm's top two are at least 0.5 apart | 97% | 100%, 120 of 120 |
| | Largest label probability difference | reported, not bounded | 0.374, on the quickstart's `is_urgent` slot |
| `mlx`, mlx-vlm's Metal library and RoPE table | Reads identical to mlx-vlm's | every read | 27 of 27, bit for bit (D-036) |
| `verdict`, Core ML float16 against upstream's PyTorch float32 | Largest probability difference | 0.02 (D-034) | 0.0022 over 333 JevBench and TypeSafe items; 0.0014 over spike #56's 200 questions |
| | Mean probability difference | 0.003 | 2.8e-4 and 2.9e-4 |
| | Top answer, wherever upstream's top two are at least 0.01 apart | unchanged | unchanged on all 333 items |
| `laya`, Core ML float16 against upstream's PyTorch float32 | Largest probability difference, before laya's rounding | 0.02 (D-037) | 0.0026 over 333 items; 0.0039 over spike #56's 200 questions |
| | Mean probability difference | 0.003 | 1.6e-4 and 2.1e-4 |
| | Top answer, wherever upstream's top two are at least 0.01 apart | unchanged | unchanged on all 333 items |

The `mlx` figures are D-014's bounds over `Fixtures/oracle`, measured on 2026-10-01 on an M3 Max
with the pinned 4-bit checkpoint ([09-conformance-and-testing.md](09-conformance-and-testing.md),
layer 2). The encoder figures are from [quality.md](quality.md): on JevBench's 231 public items and
SemIf's 102 TypeSafe rows, the Swift server gave upstream's top answer on every item for both
models, including the 36 whose upstream top two were less than 0.01 apart; accuracies are the same
and Brier scores and ECEs differ by at most 0.0003. The DiffusionGemma JevBench comparison is not
recorded there yet.

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
| Backends | `vllm` by default, and `mlx`, `laya`, `verdict`, `clm`, `jevk5` | `mlx` by default, `laya` and `verdict`; `vllm`, `clm` and `jevk5` are unknown names (issues #59 and #55) | D-030, D-038 |
| DiffusionGemma checkpoint | The newest revision of `OPENJEV_MLX_MODEL`'s repository | The default repository loads the pinned revision `a7a81407`; `repo@revision` picks another | D-039 |
| `think` and `images` on `mlx` | Supported | `openjev-0.1 does not support think` and `does not support images` until issues #52 and #48 | D-039 |
| `POST /v1/chat/completions` | Served by the `mlx` backend | A 404 until issue #53, though `/v1/models` still lists `diffusiongemma-26b` as upstream's does | D-012, D-043 |
| `server-timing` `model` on `mlx` | `0.0`: the MLX engine does not time its reads | The time spent in reads, summed over reads that ran at once, so it can exceed `total` | D-038, D-044 |
| Encoder arithmetic | PyTorch, on CUDA when present, else in float32 on the CPU | Core ML packages in float16 on the GPU or the Neural Engine, within the bounds above | D-011, D-034, D-037 |
| Encoder weights | The checkpoint that `OPENJEV_VERDICT_MODEL` or `OPENJEV_LAYA_MODEL` names, on the device `OPENJEV_DEVICE` names | Converted packages from the `Algorythm-Canada/openjev-models` releases at pinned digests, or `OPENJEV_ENCODER_MODELS`' folder; those three variables are read but have no effect | D-033, D-046 |
| Encoder padding | Rows padded to the longest row | Rows padded to the Core ML function's length; the billed tokens are the same | D-034, D-037 |
| A model's malformed output | Raised from the read or broadcast silently | `EncoderModelError`, answered as the 503 | D-034, D-037 |
| Encoder memory | One PyTorch model | Each Core ML function a read needs stays loaded with its own copy of the weights, up to 6 for Verdict and 8 for Laya, unless `OPENJEV_ENCODER_FUNCTIONS` caps them | D-042 |
| A client that goes away | Its request runs to the end | Its decision or forwarded exchange is cancelled, and the log shows 499 | D-038, D-040 |
| Shutdown and exit statuses | uvicorn's | Requests in flight get `--shutdown-timeout` (30 s); exit statuses 0 to 4 for scripts and launchd | D-038 |
| Forwarded requests | httpx adds `accept`, `accept-encoding: gzip, deflate`, `connection` and `user-agent`, honours `HTTP_PROXY`, pools connections, and answers an unparsable route URL with a 500 | AsyncHTTPClient adds `host`, `content-length` and `accept-encoding: deflate, gzip`, ignores proxy variables, opens one connection per request, and answers an unparsable URL with the 503 `InvalidURL` | D-040 |
| Checkpoint downloads | huggingface_hub, with file locks | The port's own downloader in the same cache layout, without locks: two processes must not download the same file at once | D-039 |

## What runs where

Each cell is "yes", "no" with the issue that adds it, or "n/a" where the combination does not
apply. The cells follow from each backend's `BackendCapabilities` (DiffusionGemma's runtime honours
`steps`, `samples` and `sequential`; the encoder engine checks against `.readsOnly`, as upstream's
encoder engines do), from the `openjev` tool's `BackendRegistry`, and from the platform matrix in
[05-architecture.md](05-architecture.md).

| Platform | Backend | Reads | `steps` | `samples` | `sequential` | Images | `think` | Chat |
|---|---|---|---|---|---|---|---|---|
| macOS 14 or later, Apple silicon | `mlx` | yes | yes | yes | yes | no, issue #48 | no, issue #52 | no, issue #53 |
| macOS 15 or later | `verdict` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| macOS 15 or later | `laya` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| macOS | routed models | yes | yes | yes | yes | yes | yes | n/a |
| iOS 18 or later | `mlx` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS 18 or later | `verdict` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS 18 or later | `laya` | yes | n/a | n/a | n/a | n/a | n/a | n/a |
| iOS | routed models | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `mlx` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `verdict` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Linux | `laya` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
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
  suite's stub server, but no backend: MLX and Core ML are Apple's, so every backend exits with
  status 3, and a server cannot start to forward routes.
- **JevK5 and CLM**, upstream's other two models, are not served on any platform yet (issues #55
  and #59).
