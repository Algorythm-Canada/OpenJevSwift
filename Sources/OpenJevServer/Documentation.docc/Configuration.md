# Configuration reference

Every `OPENJEV_*` variable the server and the `openjev` tool read, with its default and what it
does.

## Overview

The settings are upstream OpenJev's variables, with upstream's names, defaults and startup checks,
so deployment files written for upstream keep working (decision D-013). Two are this port's own. The
`openjev` tool reads them from the process environment through
``ServerSettings/init(environment:)``, after writing its flags over their variables, and passes
`OPENJEV_ENCODER_MODELS` to the encoder store, `EncoderPackageStore(environment:)`; the libraries
never read the process environment themselves. In the table, unset means the variable may be left
out or set to the empty string with the same effect.

| Variable | Default | Backends | Meaning |
|---|---|---|---|
| `OPENJEV_BACKEND` | `mlx` | all | The backend to load: `mlx`, `verdict`, `laya` or `jevk5`. Upstream's default, `vllm`, does not exist in this port, nor does `clm` yet (issue #59); any other name is refused. |
| `OPENJEV_HOST` | `127.0.0.1` | all | The address to bind. `0.0.0.0` serves the network. |
| `OPENJEV_PORT` | `8080` | all | The port to bind. `0` picks a free one, which the `serving on` line names. |
| `OPENJEV_LOG_LEVEL` | `info` | all | `trace`, `debug`, `info`, `notice`, `warning`, `error` or `critical`, case-sensitive. `notice` is this port's addition to uvicorn's names. |
| `OPENJEV_WARMUP` | `1` | all | `0` skips the warm-up read before the server binds; any other value runs it. |
| `OPENJEV_API_KEY` | unset | all | Requires `Authorization: Bearer <key>` on `/v1/` routes. |
| `OPENJEV_ORIGIN_SECRET` | unset | all | Requires `X-Origin-Secret` on `/v1/` routes, for a server that only a proxy should reach. |
| `OPENJEV_MAX_QUEUE` | `512` | all | Decisions inside the server before a 529. `0` refuses every request, as upstream's does. |
| `OPENJEV_MAX_QUESTIONS` | `256` | all | Questions per request before a 400. |
| `OPENJEV_MAX_BODY_BYTES` | `67108864` | all | The request body limit before a 413, 64 MiB. |
| `OPENJEV_MODEL_ROUTES` | unset | all | `name=url,name=url`: models other OpenJev servers serve. A request for one this server does not serve is forwarded to `{url}/v1/systemone`, and `GET /v1/models` lists them. |
| `OPENJEV_FORWARD_TIMEOUT` | `300` | all | Seconds a forwarded request waits for each read and write of the other server before a 503. |
| `OPENJEV_MLX_MODEL` | `mlx-community/diffusiongemma-26B-A4B-it-4bit` | mlx | The checkpoint: a directory, or a Hugging Face repository with an optional `@revision`. The default repository loads its pinned revision, `a7a81407`. |
| `OPENJEV_MLX_MAX_PROMPT` | `32768` | mlx | The longest prompt in tokens; a longer one is a 400. |
| `OPENJEV_MLX_PROMPT_CACHE` | `12` | mlx | The prefills kept for reuse, which together hold at most 16,384 prompt tokens. `0` turns the cache off. |
| `OPENJEV_MLX_CACHE_LIMIT_GB` | unset | mlx, jevk5 | MLX's buffer pool limit in GB. Unset leaves MLX alone; `0` disables the pool. |
| `OPENJEV_CANVAS` | `64` | mlx | The longest canvas in tokens. Questions are read in groups whose answer template fits it. |
| `OPENJEV_CANVAS_STEP` | `16` | mlx | A canvas's width is its template's length plus one, rounded up to a multiple of this and capped at `OPENJEV_CANVAS`. |
| `OPENJEV_MAX_INFLIGHT` | `64` | mlx | Reads in flight at once. The encoders make one call at a time whatever it says, as upstream's do. |
| `OPENJEV_AUTO_THRESHOLD` | `0.1` | mlx | The top-k entropy above which a group is read again, when the request does not set `samples`. |
| `OPENJEV_AUTO_MAX` | `4` | mlx | The most reads of one group under those automatic re-reads. `1` turns them off. |
| `OPENJEV_MAX_IMAGES` | `8` | mlx | Images per request. No backend reads images yet (issue #48), so a request with images is refused first and this has no effect. |
| `OPENJEV_MAX_IMAGE_BYTES` | `5242880` | mlx | Bytes per decoded image, 5 MiB. No effect until images land (issue #48). |
| `OPENJEV_ENCODER_BATCH` | `16` | encoders | Questions per backend call. On a Mac each Core ML call reads at most 16 questions; JevK5 reads a call's questions concurrently, one pass at a time. |
| `OPENJEV_ENCODER_FUNCTIONS` | unset | encoders | This port's: the most Core ML functions an encoder keeps loaded, at least 1. Unset keeps every function a read has needed, up to Verdict's 6 and Laya's 8 (D-042). |
| `OPENJEV_ENCODER_MODELS` | unset | encoders | This port's: a folder of converted Core ML packages, used instead of downloading them (D-033). |
| `OPENJEV_VERDICT_MODEL` | `heman10x/rlcd-modernbert-151m` | verdict | Read as upstream reads it, with no effect: the backend loads the package and the checkpoint revision its manifest pins (D-033). |
| `OPENJEV_LAYA_MODEL` | `convaiinnovations/laya-typed-decisions` | laya | Read as upstream reads it, with no effect, for the same reason. |
| `OPENJEV_JEVK5_MODEL` | `Algorythm-Canada/jevk5-0.2-mlx-8bit` | jevk5 | This port's: the JevK5 checkpoint converted to MLX, a directory (`Tools/jevk5/convert.py` writes one) or a Hugging Face repository with an optional `@revision`. A conversion's repository loads its pinned commit; the default, the published 8-bit conversion, is downloaded into the Hugging Face cache on first start (D-052). Upstream reads its vLLM server's `OPENJEV_MODEL` instead. |
| `OPENJEV_DEVICE` | unset | encoders | Upstream's PyTorch device. Read, with no effect: Core ML picks the compute units for the platform (D-011). |
| `OPENJEV_GEN_MAX_INFLIGHT` | `8` | none yet | Generations in flight at once. Read and checked; text generation arrives with issue #53. |
| `OPENJEV_GEN_MAX_QUEUE` | `32` | none yet | Generations waiting before a 529. No effect until issue #53. |
| `OPENJEV_GEN_MAX_TOKENS` | `8192` | none yet | The longest generation in tokens. No effect until issue #53. |

`serve` writes `--backend`, `--host`, `--port`, `--log-level` and `--no-warmup` over
`OPENJEV_BACKEND`, `OPENJEV_HOST`, `OPENJEV_PORT`, `OPENJEV_LOG_LEVEL` and `OPENJEV_WARMUP`
before the settings are read, and `decide` and `models` take `--backend`.

### How values are read and checked

A number parses as Python's `int` and `float` parse it: surrounding whitespace, a sign and single
underscores between digits are accepted, and `float` also takes `inf` and `nan`. An empty value is
refused for every number except `OPENJEV_MLX_CACHE_LIMIT_GB`, `OPENJEV_MLX_PROMPT_CACHE` and
`OPENJEV_ENCODER_FUNCTIONS`, where it means the default. A string setting keeps an empty value.

The checks are upstream's: `OPENJEV_CANVAS`, `OPENJEV_CANVAS_STEP`, `OPENJEV_MAX_INFLIGHT`,
`OPENJEV_MAX_QUESTIONS`, `OPENJEV_MAX_BODY_BYTES`, `OPENJEV_MAX_IMAGE_BYTES`,
`OPENJEV_GEN_MAX_INFLIGHT`, `OPENJEV_GEN_MAX_TOKENS`, `OPENJEV_MLX_MAX_PROMPT`,
`OPENJEV_ENCODER_BATCH` and `OPENJEV_FORWARD_TIMEOUT` must be at least 1;
`OPENJEV_MAX_QUEUE`, `OPENJEV_GEN_MAX_QUEUE`, `OPENJEV_MAX_IMAGES`,
`OPENJEV_MLX_CACHE_LIMIT_GB` and `OPENJEV_MLX_PROMPT_CACHE` must not be negative; and
`OPENJEV_ENCODER_FUNCTIONS` must be at least 1. A route must be `name=url` with both parts. A
refused value stops `openjev` at startup with status 2 and upstream's message, which names the
variable:

```text
openjev: OPENJEV_PORT='eighty' is not a int
openjev: canvas must be at least 1, got 0 (OPENJEV_CANVAS)
```

Where upstream raises a bare `ValueError` that names only the text, this port names the variable
too (D-030).

### Other variables

| Variable | Read by | Meaning |
|---|---|---|
| `HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME` | `mlx`, `jevk5`; `verdict` and `laya` with `OPENJEV_ENCODER_MODELS` | Where the Hugging Face cache is, as `huggingface_hub` finds it: `HF_HUB_CACHE`, else `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`. DiffusionGemma's checkpoint and JevK5's conversion are kept there, and a local encoder folder's tokenizer is looked for there. |
| `HF_TOKEN` | `mlx`, `jevk5` | A Hugging Face token for a gated or private repository; an empty value counts as unset. |

Upstream also reads variables for the backends this port does not have, which it ignores:
`OPENJEV_UPSTREAM`, `OPENJEV_UPSTREAM_MODEL` and `OPENJEV_TOKENIZER` (the vLLM server), the
`OPENJEV_CLM_*` settings, and `OPENJEV_MODEL` and `OPENJEV_JEVK5_WORKERS` (JevK5). Its container
scripts' vLLM variables, such as `OPENJEV_GPU_UTIL`, have no counterpart either. The test suites'
own variables, `OPENJEV_TEST_MODEL` and `OPENJEV_LIVE_URL` among them, are in the repository's
`docs/development.md`.

## See Also

- ``ServerSettings``
- <doc:RunningTheServer>
