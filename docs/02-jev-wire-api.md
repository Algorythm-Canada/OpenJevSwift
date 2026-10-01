# The wire contract

OpenJevSwift must satisfy two clients: TypeSafe's official SDKs, which expect Jev's contract, and
anything written against upstream OpenJev, which adds optional request fields. Sources: TypeSafe's
[API reference](https://docs.typesafe.ai/api.md) and [models page](https://docs.typesafe.ai/models),
the official Python SDK ([typesafe-sdk 0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python)),
and upstream `api.py`, `config.py` and `tests/test_api.py`.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/systemone` | Decisions. `{state, model, questions}` → `{model, answers, usage}`. |
| `GET /v1/models` | `{"models": [{name, description, release_date}]}`. |
| `GET /health` | `{"status": "ok"}`. Unauthenticated. |
| `POST /v1/chat/completions` | OpenAI-style text generation with `diffusiongemma-26b` (OpenJev only). |

Authentication: `Authorization: Bearer <key>`. Jev requires it; OpenJev requires it only when
`OPENJEV_API_KEY` is set, and separately supports `X-Origin-Secret` for a front proxy.

## Request

```json
{
  "model": "jev-latest",
  "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this",
                   "criteria": {"billing": "Payment or subscription issues",
                                "technical": "Bugs or integration problems",
                                "sales": "Pricing or account questions"}},
    "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                    "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
  }
}
```

| Field | Type | Rules |
|---|---|---|
| `model` | string | Required. Jev: `jev-1.13.0`, aliases `jev-latest`, `jev-preview`. OpenJev: `openjev-0.1`, `openjev-latest`, the SDK aliases, and one encoder model name per process. A pinned Jev version such as `jev-1.13.0` gets `400 Unknown model` from OpenJev. |
| `state` | string, object or array | Required. Non-string states are serialised with `json.dumps(state, ensure_ascii=False)` before reaching the model. **Object key order is preserved.** |
| `questions` | object, at least one entry | Required. **Entry order is preserved**: it defines `q1..qN`, the answer order and the parallel read grouping. |
| `questions.*.type` | `"noul"`, `"choice"`, `"score"` | Unknown type → `400 api_usage_error "Invalid request."` |
| `questions.*.instructions` | string, object, array or null | Optional. Objects and arrays are rendered as JSON text. |
| noul `criteria` | `{true?: Described, false?: Described}` or null | Optional descriptions of each outcome. |
| choice `criteria` | `{name: Described}` | Required. 1 to 255 entries. **Order is preserved**; it defines the label letters and the `probabilities` order. One entry is answered without a read. |
| score `criteria` | `[Described, ...]` | Required. 1 to 10 levels (Jev's docs say 2 to 10; upstream answers a single level as score 0.0 at probability 1). |
| `images` (OpenJev) | up to 8 data URLs (`data:image/...;base64,`) or `{content_type, base64}` | JPEG, PNG, WebP, GIF; 5 MB each after decoding; refused before decoding when the base64 length already exceeds the limit. Not with `think` or `sequential`. |
| `steps` (OpenJev) | 1..8, default 1 | Denoise passes per read. |
| `samples` (OpenJev) | 1..32 | Fixed number of noise draws; replaces the automatic re-reads. |
| `think` (OpenJev) | 0..4096 | Thought token budget before the read. |
| `sequential` (OpenJev) | bool | Read chunks in order, conditioning on earlier answers. |

pydantic validates these four fields in lax mode, so upstream accepts more than the types say
(recorded in `Fixtures/wire/cases.json`). An integer field also takes `true` and `false` (1 and
0), an integral float (`2.0`, but not `1e20`, which is `int_parsing_size`), and a decimal string
with surrounding Unicode whitespace, a sign, single underscores between digits and an all-zero
fraction (`" 3 "`, `"+3"`, `"1_0"`, `"3.00"`; not `"3."`, `"1e3"` or `"0x10"`). `sequential` also
takes `0` and `1` as integers or floats and the strings `0`, `1`, `f`, `t`, `n`, `y`, `no`, `yes`,
`off`, `on`, `false`, `true` in any ASCII case, without trimming. A choice option's description
may be `null`; a score level may not. `null` for any optional field means unset. Unknown top-level
fields are ignored.

Jev's limits (documented): 64k tokens total context per request; 32k tokens for `state` plus the
longest single question; text only; English is the primary language. OpenJev's limits: canvas
64 tokens per read (questions are chunked, about 12 per read), 256 questions per request,
32,768 prompt tokens on MLX (400 beyond), 64 MiB body.

TypeSafe's SDKs never send the extension fields. Left unset, a request behaves exactly as Jev's
contract describes. The Python SDK's `extra_body` parameter can carry them.

## Response

```json
{
  "model": "openjev-0.1",
  "answers": {
    "department": {"type": "choice", "choice": "technical",
                   "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                   "confidence": 0.62},
    "frustration": {"type": "score", "score": 1.15,
                    "legend": {"0": "Calm, just stating facts", "1": "Frustrated but civil", "2": "Very angry, strong language"},
                    "probabilities": {"0": 0.15, "1": 0.55, "2": 0.30}, "confidence": 0.12},
    "is_urgent": {"type": "noul", "noul": 0.7}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0}
}
```

Exact key sets matter: a noul answer is `{type, noul}` only; choice is `{type, choice,
probabilities, confidence}`; score is `{type, score, legend, probabilities, confidence}` with
string keys `"0".."9"`. `probabilities` sums to 1 over the caller's options (encoder backends
drop abstention mass and renormalise). `answers` preserves the request's question order. `model`
is the served version, not the requested alias. `usage.output_tokens` is 0 except after `think`. A one-option choice or one-level score is
answered without a read (probability and confidence `1.0`, score `0.0`), so a request made only
of those gets 200 even with no backend.

The Python SDK validates answers in strict mode, drops answer types it does not know with a
warning, coerces score `legend` and `probabilities` keys to integers, exposes `.nouls`,
`.choices`, `.scores` views, and reads `usage.input_tokens` / `usage.output_tokens` as optional.
Python's `json.dumps` writes integral floats as `1.0`; Swift's default `JSONEncoder` writes `1`.
The SDK accepts both, but the fixtures compare bytes, so the Swift serialiser will write `1.0`.

## Headers

| Header | Direction | Value |
|---|---|---|
| `Authorization` | request | `Bearer <key>` |
| `x-typesafe-request-id`, `x-request-id` | response | `req_` + 32 hex characters, on every response including errors |
| `server-timing` | response | `model;dur=41.2, server;dur=2.8, total;dur=44.0` (milliseconds; upstream extension). Upstream leaves it off the 401, 403 and 413 its middleware answers before routing; OpenJevSwift sets it on every response (D-031). |
| `retry-after` | response | `1` on 529, `2` on 503 |
| `retry-after-ms` | response | honoured by the SDK when present (Jev's gateway may send it) |
| `X-TypeSafe-Retry-Count` (name per SDK constants) | request | attempt number on retries |

## Errors

| Status | Body | When |
|---|---|---|
| 400 | `{"detail": "<reason>"}` | A question the model cannot ask: no options, too many options, too many levels, labels without a shared slot, template larger than the canvas, prompt over the token limit, too many questions, unsupported option for the backend, images with think or sequential. |
| 400 | `{"detail": {"error_type": "api_usage_error", "message": "Unknown model: X"}}` | Unknown model. |
| 400 | `{"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}` | Unknown question type. |
| 400 | `{"detail": "the model rejected this request: ..."}` | The inference backend refused (a 4xx from vLLM). |
| 401 | `{"detail": {"error_type": "authentication_error", "message": "Cannot authenticate with the server. Please check your API key and try again."}}` | Wrong key. |
| 403 | `{"detail": {"error_type": "authentication_error", "message": "Must supply an API key! Check your request and try again."}}` | Missing key. |
| 403 | `{"detail": {"error_type": "permission_error", "message": "Direct access to this origin is not allowed."}}` | Wrong or missing origin secret. |
| 413 | `{"detail": {"error_type": "api_usage_error", "message": "request body is larger than N bytes"}}` | Body cap. |
| 422 | `{"detail": [{"type", "loc", "msg", "input", "ctx"?}]}` | Shape validation (missing `state`, empty `questions`, wrong field type, `samples: 33`, `steps: 9`). Every error is listed, in model field order. `url` never appears with the recorded FastAPI (0.142). |
| 422 | `{"detail": [{"type": "json_invalid", "loc": ["body", <character offset>], "msg": "JSON decode error", "input": {}, "ctx": {"error": "<Python json message>"}}]}` | A body that is not valid JSON. |
| 400 | `{"detail": "There was an error parsing the body"}` | A body that is not UTF-8. |
| 429 | Jev only | Rate limit (250,000 tokens/s, 1,200 requests/min per account). OpenJev has no rate limiter. |
| 503 | `{"detail": {"error_type": "api_error", "message": "inference backend unavailable: ..."}}` | Backend unreachable; a routed model's server down or silent (see Model routes). |
| 529 | `{"detail": {"error_type": "overloaded_error", "message": "... at capacity. Retry shortly."}}` | Queue full. |

Details of the 422 body, from the recordings:

- `loc` starts with `body`. A missing or `null` body is one `missing` error at `["body"]`; a body
  that is not an object is `model_attributes_type` at `["body"]`. A body without a JSON content
  type (`application/json`, any `+json` type) is not parsed and fails the same way, with the
  Python `bytes` repr as `input`.
- A union adds its member label to `loc`: `str`, `dict[str,any]` and `list[any]` for `state`,
  `instructions`, descriptions and score levels (one error per member), `str` and `ImageObject`
  for an image. A question adds its tag, as in `["body", "questions", "q", "choice", "criteria"]`.
  Score levels add their integer index.
- `ctx` appears only for `too_short` (`{"field_type", "min_length", "actual_length"}`), the bounds
  (`{"ge": 1}`, `{"le": 8}`), `union_tag_not_found` (`{"discriminator": "'type'"}`) and
  `json_invalid`.
- `input` is the offending value trimmed by upstream's `trim` (depth 4, 20 items, 500
  characters). A `missing` error's input is the object the field is missing from.
- A `union_tag_invalid` anywhere (a question `type` that is not `noul`, `choice` or `score`,
  including a number or `null`) replaces the whole body with the 400 `"Invalid request."`.

The Python SDK retries 408, 429 and every 5xx (and 529) by default: 2 retries, exponential
backoff from 0.5 s to 5 s with 25% jitter, `Retry-After`/`retry-after-ms` honoured, a 30 s total
budget, 10 s per-operation timeout. Its error message extraction reads `error.message`,
`message`, `detail` (string or object with `message`), or joins a 422 list as `loc: msg`.
Base URL from `TYPESAFE_BASE_URL`, key from `TYPESAFE_API_KEY`, model from
`TYPESAFE_DEFAULT_MODEL` (default `jev-latest`, base `https://api.typesafe.ai`).

## Model routes (OpenJev)

`OPENJEV_MODEL_ROUTES` (`name=url,name=url`) lets one origin serve several models, each from its
own OpenJev server. A `POST /v1/systemone` whose `model` has a route and is not served here is
passed, once its body has passed validation and before the model name is checked, to
`{url}/v1/systemone` as the bytes the client sent, with only its `authorization`,
`x-origin-secret` and `content-type` headers. The routed server's status and body come back
unchanged, with only its `content-type` and `retry-after`; the forwarding server adds its own
request ids and `server-timing`, whose `model` counts the whole exchange. A routed server that
cannot be reached, or sends nothing for `OPENJEV_FORWARD_TIMEOUT` seconds (300), is the 503
`inference backend unavailable: <httpx error>` with `retry-after: 2`, for example `ConnectError` or
`ReadTimeout`. A name the server serves itself, the SDK aliases included, is answered there even
when routed.

`GET /v1/models` lists the server's own models, then each routed name it does not serve: with
upstream's description and release date for an encoder model it knows (`laya-1.0`,
`verdict-1.4`, `clm-v0.1`, `jevk5-0.2`) and an empty description and release date for any other.
The routed servers are not asked, so the listing holds while one is down. Decision D-039 records
where OpenJevSwift differs.

## Differences between OpenJev and Jev (kept as they are)

- Model names are OpenJev's own; `jev-latest` and `jev-preview` are accepted as aliases.
- Questions beyond one canvas are read in chunks of about 12, in parallel.
- Extension fields exist. Without them, behaviour matches the Jev contract.
- No rate limiting; 529 signals capacity instead.
- The server-timing header.

## What OpenJevSwift must therefore guarantee

1. Order-preserving JSON parsing and serialisation for `state`, `questions` and `criteria`.
2. Byte-identical error bodies for every row of the table above, including FastAPI's 422 list
   shape with trimmed `input`.
3. The same headers on every response.
4. Response bodies rendered as FastAPI's `JSONResponse` renders them: compact separators `,` and
   `:`, `ensure_ascii=False`, Python float formatting, non-finite numbers refused; and
   Python-compatible `json.dumps` rendering of objects and arrays that reach the model
   (`ensure_ascii=False`, default separators).
5. The same model names, aliases, descriptions and release dates in `/v1/models`.
