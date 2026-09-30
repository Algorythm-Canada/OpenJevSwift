# Error fixtures

HTTP error responses from upstream's FastAPI app that `wire/cases.json` does not already hold,
and an index of every error row in the wire contract. `Tools/fixtures/upstream_tables.py` writes
`cases.json`; see [../README.md](../README.md).

## cases.json

| Key | Contents |
|---|---|
| `no_backend` | `http://127.0.0.1:9`, where nothing listens: a request that passes validation and reaches a real read ends in upstream's 503 |
| `coverage` | Every row of the error table in `docs/02-jev-wire-api.md`: `{status, body, when, cases}`, each case naming a file (`wire/cases.json` or `errors/cases.json`) and a case in it. A row with no case has a `note` saying why. |
| `cases` | The recordings, in the shape of `wire/cases.json`: `{name, settings, stub, request, response, request_id_header_format, server_timing_present, note?}` |

`stub` says what stood in for a part of upstream that needs a model or a server, or is `null`:

- **Engine errors, no stub.** The canvas too small for the template, an image over the size
  limit only once decoded, a zero image limit, a full queue (529, even for a request of forced
  questions only), and images with both `think` and `sequential`.
- **The vLLM backend**, answered by `httpx.MockTransport` as upstream's chat tests do. A 4xx
  becomes 400 "the model rejected this request: ...", taking `error.message`, then `message`,
  then the raw text, cut at 500 characters. A 5xx and transport errors become 503 with the
  exception's class name.
- **Model routes**: a routed server that is down or times out (503), and one that answers an
  error, which upstream passes through keeping only `content-type` and `retry-after`.
- **The MLX backend** with a stub runtime, as upstream's `tests/test_mlx_backend.py` does: the
  prompt limit, with exact token counts for a read and for a thought.
- **The encoder backends** with nothing loaded, as upstream's `tests/test_encoders.py` does:
  "... does not support ..." for each option (the first unsupported option in upstream's order
  wins), Verdict's 24-option limit, and a full queue.

Two responses are upstream failures, marked with a `note`: a backend 4xx whose JSON `error` is a
string, and a backend 200 that is not JSON. The route raises, and Starlette answers a bare 500
`Internal Server Error` without the request-id headers (`request_id_header_format` is `null`).

Not recorded: "labels do not share one template slot" cannot be reached through the API with
the pinned tokenizer (see `templates/`), and OpenJev has no rate limiter, so there is no 429.
