# Wire fixtures

What upstream OpenJev (`razorback16/openjev` at `dcd2094`) sends and accepts on the wire, recorded
from upstream's own FastAPI app. `Tools/fixtures/wire_tables.py` builds the app with
`openjev.api.create_app(Settings(...), tokenizer=FakeTokenizer())` inside FastAPI's `TestClient`,
as upstream's tests do, and records the exchanges byte for byte. No model and no network are
involved: the engine points at `http://127.0.0.1:9`, where nothing listens, so every request that
passes validation ends in upstream's 503, and that 503 is recorded too.

Every file starts with a `generator` object that names the script, the upstream commit and the
Python, FastAPI, Starlette, pydantic, pydantic-core and httpx versions that wrote it. The 422
message texts, `ctx` contents and union member labels in `loc` paths come from pydantic, so a
different pydantic version can change them; regenerate and compare when it moves.

## Regenerating

From the repository root, with the pinned upstream checkout in place (`make upstream`):

```bash
python3 -m venv Tools/fixtures/.venv
Tools/fixtures/.venv/bin/python -m pip install "fastapi>=0.115" httpx
Tools/fixtures/.venv/bin/python Tools/fixtures/wire_tables.py
```

Running the script twice with the same versions gives identical files. Never edit these files
by hand. `make fixtures` regenerates these files together with every other fixture, from a
virtual environment with pinned versions (see [../README.md](../README.md)).

## Files

| File | Contents |
|---|---|
| `cases.json` | HTTP exchanges. Each case is `{name, settings, request, response, request_id_header_format, server_timing_present}`. `settings` lists the `Settings` fields that differ from upstream's defaults. `request` is `{method, path, headers, body_text}`; a body that is not UTF-8 is `body_base64` instead, a body too large to store has a `body_expand` placeholder (`{placeholder, repeat, count}`: replace the placeholder with `repeat` written `count` times), and a body sent without `Content-Length` has `streamed_without_content_length`. Header values were sent as UTF-8 bytes. `response` is `{status, body_text, headers}` with `content-type` and, when present, `retry-after`. Every response carried `x-request-id` and `x-typesafe-request-id`, equal and in the recorded format; `server_timing_present` says whether `server-timing` was set. |
| `answers.json` | `answers`: for each case the wire question, the probability vector and the bytes of `openjev.engine.to_answer` for them, rendered as FastAPI renders a response. `response`: a full `SystemOneResponse` body with a forced single-option choice and a forced single-level score among read answers, in request order. |
| `requests.json` | Ten request bodies, each as `json.dumps(body)` (`default`) and as FastAPI's compact rendering (`compact`). The script checks that pydantic's `model_dump(exclude_unset=True)` of each body renders to the same compact bytes, so `compact` is also what a decode and re-encode must give. |
| `models.json` | `GET /v1/models` bodies for every backend (`vllm`, `mlx` and the encoder backends `laya`, `verdict`, `clm`, `jevk5`), each with the model version a response names and the names a request may use, plus one listing with `model_routes` set. |

## What the cases cover

- `GET /health` and `GET /v1/models`, with and without authentication.
- The README quickstart and its variants, which end in the 503 with `retry-after: 2`.
- Shape validation (422): missing and mistyped fields at every level, empty `questions`, empty
  score criteria, the extension field bounds, and pydantic's lax coercions of strings, floats and
  Booleans for `steps`, `samples`, `think` and `sequential`.
- The unknown question type (400 `api_usage_error "Invalid request."`), alone and together with
  other errors.
- Semantic 400s with a plain `detail`: empty choice criteria, 256 options, 11 levels, 257
  questions, the image checks and images with `think` or `sequential`.
- Unknown and pinned Jev model names (400 `api_usage_error "Unknown model: ..."`).
- Bodies that are not a JSON object or not JSON at all, including missing and non-JSON
  content types.
- The body cap (413) with and without `Content-Length`, and the auth failures (401, 403).
