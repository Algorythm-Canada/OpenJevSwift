# Requests and answers

The request a client sends, the decision an engine returns, and the JSON both take on the wire.

## Overview

``SystemOneRequest`` is the body of `POST /v1/systemone`. ``Decision`` is what an engine returns
for it, and ``SystemOneResponse`` is the body the server writes from a decision. The three keep
every order the contract depends on: the questions' order, which defines their labels and the
answers' order, and each choice's option order, which defines the order of its probabilities.
Foundation's `JSONDecoder` and `JSONSerialization` lose object order, so the wire types read and
write ``JSONValue`` instead of using `Codable`.

## The request

| Field | Swift | Rules |
|---|---|---|
| `model` | `String` | Required. The server answers a name the loaded engine serves (``ServedModels``): `openjev-0.1` or `openjev-latest` for DiffusionGemma, the encoder's own name (`verdict-1.4`, `laya-1.0`), and the SDK aliases `jev-latest` and `jev-preview` everywhere. The engines do not check it. |
| `state` | ``JSONValue`` | Required. A string, an object or an array. A string reaches the model as sent; anything else as `json.dumps(state, ensure_ascii=False)` with its key order kept (``StateText``). |
| `questions` | ``OrderedMap`` of ``Question`` | Required, at least one. The order defines `q1` to `qN`, the answer order and how the questions are grouped into reads. |
| `images` | `[ImageInput]` | Optional, an OpenJev extension: up to 8 data URLs or `{content_type, base64}` objects of JPEG, PNG, WebP or GIF, read ahead of the state. The DiffusionGemma backend reads them; the encoder backends refuse them. |
| `steps` | `Int` | Optional, 1 to 8: denoise passes per read. |
| `samples` | `Int` | Optional, 1 to 32: a fixed number of noise draws, all billed, in place of the automatic re-reads. |
| `think` | `Int` | Optional, 0 to 4096: a thought budget before the read. No backend generates yet (issue #52). |
| `sequential` | `Bool` | Optional: read the groups in order, each conditioned on the earlier answers. |

The encoders honour none of the five extensions, as upstream's encoder engines do not: they refuse
`images`, `steps` or `samples` above 1, a `think` other than 0 and `sequential: true`, with messages
such as `verdict-1.4 does not support steps`. Unknown top-level fields are ignored. The extension
fields are coerced as pydantic's lax mode coerces them, so `"3"` is an integer and `"yes"` is true,
as upstream accepts them.

### Questions

A ``Question`` is one of three kinds:

- A noul, ``Question/noul(instructions:criteria:)``, is a yes or no question. Its optional
  ``NoulCriteria`` describe the two outcomes, under the keys `true` and `false`.
- A choice, ``Question/choice(instructions:criteria:)``, picks one of its named options. The
  criteria are an object from option name to description, a ``JSONValue`` or `null`, and their
  order is the order of the answer's probabilities. A choice needs 1 to 255 options (24 for
  Verdict).
- A score, ``Question/score(instructions:criteria:)``, rates on ordered levels. The criteria are an
  array of 1 to 10 level descriptions, none of them `null`.

Instructions and descriptions may be strings, objects or arrays (``Described``). A choice with one
option or a score with one level is answered without a read, at probability and confidence 1.0, as
upstream answers it.

### Decoding a request

``JSONParser`` parses text or bytes into a ``JSONValue`` whose objects keep their order. It follows
RFC 8259, which is stricter than Python's `json.loads`: it refuses `NaN`, `Infinity`, numbers that
overflow a `Double` and lone surrogate escapes, where upstream would accept them.
``SystemOneRequest/init(json:)`` then checks the shape with ``RequestValidator``, which reports
every problem as upstream's FastAPI app does: a ``WireError`` that carries the 422 body, its `loc`
paths and its messages, or the 400 `Invalid request.` for an unknown question type.

A request built in Swift with ``SystemOneRequest/init(model:state:questions:images:steps:samples:think:sequential:)``
skips that check. The engine still applies the schema's limits, on options, levels and empty
choices, but the ranges of `steps`, `samples` and `think` are the validator's alone, so code that
builds a request keeps to them.

## The decision

A ``Decision`` holds:

- ``Decision/answers``: one ``Answer`` per question, in the request's order, forced answers
  included.
- ``Decision/inputTokens``: the prompt tokens of every billed read, which the response reports as
  `usage.input_tokens`.
- ``Decision/outputTokens``: the thought tokens generated, `usage.output_tokens`, 0 without
  `think`.
- ``Decision/modelTime``: the time spent inside backend calls, summed over calls that may have run
  at once, so it can exceed the request's wall time.

### Answers

| Kind | Swift | JSON keys |
|---|---|---|
| noul | ``Answer/noul(_:)``, the probability of yes | `type`, `noul` |
| choice | ``Answer/choice(choice:probabilities:confidence:)`` | `type`, `choice`, `probabilities`, `confidence` |
| score | ``Answer/score(score:legend:probabilities:confidence:)`` | `type`, `score`, `legend`, `probabilities`, `confidence` |

A choice's `choice` is the option with the largest probability, the first one on a tie. A score's
`score` is the expected level, the sum of each level index times its probability, and its
`legend` repeats the levels as sent. `legend` and a score's `probabilities` are objects keyed `"0"`
to `"N-1"`. The confidence is ``Confidence/compute(_:)``: 1 minus the distribution's entropy over
the logarithm of its size, clamped to 0 through 1, so 1 is certain and 0 is uniform.

## The wire format

``WireEncoder`` writes a wire value as FastAPI's `JSONResponse` writes it: compact separators,
non-ASCII text as UTF-8, every probability as a Python float (`1.0` stays `1.0`), and an error,
``JSONWriteError``, for an infinite or NaN number. The response names the served model version,
not the alias the request used. This is Verdict's answer to the request of <doc:GettingStarted>:

```json
{
  "model": "verdict-1.4",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {
        "billing": 0.2679097056388855,
        "technical": 0.5201022028923035,
        "sales": 0.21198803186416626
      },
      "confidence": 0.0699968384405778
    },
    "frustration": {
      "type": "score",
      "score": 1.001843899488449,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "probabilities": {
        "0": 0.18937484920024872,
        "1": 0.6194064617156982,
        "2": 0.19121871888637543
      },
      "confidence": 0.15515523959287614
    },
    "is_urgent": {
      "type": "noul",
      "noul": 0.7060711979866028
    }
  },
  "usage": {
    "input_tokens": 214,
    "output_tokens": 0
  }
}
```

The server writes those bytes on one line, without the indentation, with `server-timing`,
`x-request-id` and `x-typesafe-request-id` headers. The figures are what `verdict-1.4` answered
on 2026-10-02.

``SystemOneResponse/init(json:)``, ``Answer/init(json:)`` and ``ModelsResponse`` decode the other
direction and require the exact key sets, for clients and tests.

### Errors

| Error | Status | Body |
|---|---|---|
| ``WireError`` from ``RequestValidator`` | 422, or 400 for an unknown question type | FastAPI's `{"detail": [...]}` list, or `{"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}` |
| A model name the server does not serve | 400 | `{"detail": {"error_type": "api_usage_error", "message": "Unknown model: ..."}}` |
| ``SchemaError`` | 400 | `{"detail": "<message>"}`, upstream's message |
| ``BackendRefusal`` | 400 | `{"detail": "the model rejected this request: <reason>"}` |
| ``OverloadedError`` | 529, `retry-after: 1` | `{"detail": {"error_type": "overloaded_error", "message": "... at capacity. Retry shortly."}}` |
| Any other error of the engine or its backend | 503, `retry-after: 2` | `{"detail": {"error_type": "api_error", "message": "inference backend unavailable: <type name>"}}` |

The server adds `x-typesafe-request-id`, `x-request-id` and `server-timing` to every response,
errors included.

## Seeds

``DecisionEngine`` derives a request's seed as upstream does (``SeedDerivation``): the SHA-256 of
the canonical JSON of the state and the questions, the image data URLs too when there are images,
whose first four bytes seed Python's Mersenne Twister for the canvas noise. The same request
therefore reads the same canvases, on this port and upstream alike.
``DecisionEngine/decide(_:seed:)`` takes a seed to replay a recorded request. The encoder engines'
reads are deterministic and use no seed.
