# Schema fixtures

`Engine.build_schema` at the pinned upstream commit. `Tools/fixtures/upstream_tables.py` writes
`schemas.json`; see [../README.md](../README.md).

## schemas.json

`cases` holds one row per request: `{name, api_reachable, request, questions, schema}` or, when
upstream refuses it, `{..., error: {message, loc}}`.

- `request` is the request body. `questions` is what the route hands the engine
  (`api.py:256`): pydantic's `model_dump()` of each validated question, so every optional field is
  present and a noul's criteria object has both keys.
- `schema` is `build_schema(questions)`: `{questions, forced, format}`. Each internal question is
  `{key, id, type, instructions, choices, labels, legend}`, with `choices` as
  `[name, description]` pairs.
- `error` is the `SchemaError`'s message and `loc`.
- `api_reachable` is false for the two unknown-type cases, which call the engine directly:
  pydantic refuses an unknown type before the engine sees it, and `wire/cases.json` records that
  400.

The requests cover the quickstart, forced single-option choices and single-level scores alone and
among read questions, the lines and indexed formats at 10 and 11 questions, 12 mixed questions,
the widest schema (255 options, 10 levels and a noul), object and array instructions and
descriptions, non-ASCII text, whitespace to strip, blank descriptions, noul criteria variants,
choice names with quotes and newlines, and every `SchemaError` in request order.

What the rows show:

- Instructions and descriptions go through `text_of`: strings are stripped, objects and arrays
  become `json.dumps(value, ensure_ascii=False)` with the default separators, and `null` becomes
  an empty string.
- Questions are numbered `q1` to `qN` after forced questions are taken out.
- The format is `lines` for up to 10 read questions and `indexed` beyond.
- The first failing question in request order raises.

The 256-question request is left out here; `groups-and-canvases/` has it.
