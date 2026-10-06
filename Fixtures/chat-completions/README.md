# Chat completion fixtures

What upstream's `openjev/chat.py` does with `POST /v1/chat/completions` on its MLX backend, recorded
by `Tools/fixtures/chat_tables.py` from upstream's own functions and route with the pinned
tokenizer and no weights (issue #53); see [../README.md](../README.md). Each row has a `note`
saying what it shows.

## prompts.json

`MlxGenerator.prompt_ids`: OpenAI messages through the chat template with the generation prompt
and `enable_thinking`, then the empty thought scaffold.

`scaffold` is `enc("<|channel>thought\n<channel|>")`, `[100, 45518, 107, 101]`. `cases` holds one
row per request: `{name, note, messages, thinking, text, ids}`. Each request went through
`Generator.normalize` first, as the route sends it, so `messages` are the normalized ones (JSON
mode's instruction included) and `thinking` is `bool(chat_template_kwargs.enable_thinking)`.
`text` is `apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
enable_thinking=thinking)` and `ids` is `prompt_ids`, which the script checks is `text` encoded
without special tokens followed by the scaffold.

The rows cover user, system, developer and assistant turns, multi-turn conversations with and
without a system turn, consecutive assistant messages, a thought channel inside an assistant turn,
an unknown role, content as text parts in each role, image and `image_url` parts, non-ASCII text,
special token text, inner whitespace, the template's `trim` (ASCII whitespace, U+001C to U+001F,
the other characters `str.isspace()` accepts, and U+200B, which stays), empty, missing and null
content, extra message keys, OpenAI tool calls with string and object arguments and their
responses, argument keys and a legacy `tool_responses` object that jinja2's `dictsort` orders by
`str.lower()` and code point (a sharp s, a decomposed e acute, capitals, a dotted capital I, a
final sigma, CJK), `reasoning_content`, a twelve-round conversation, JSON mode with and without a
system turn, and content shapes a client may send that are not OpenAI's: a dict, parts that are not
dicts, a number, and text parts whose text is null, a number, a Boolean, a list or a dict, which
jinja2 writes as Python's `str()` does.

## normalize.json

`Generator.normalize` on request bodies: `{name, note, settings, body}` and then either
`{upstream, json_mode}`, the request it builds (with the `model` it adds for vLLM), or `error`,
`{type, message}` of what it raised. A `ValueError` is upstream's 400 with that message; the
`crash_` rows and `json_mode_number_message` raise something else, which upstream's route answers
with a bare 500.

## extract_json.json

`extract_json` on model replies: `{name, text, result}`. The texts cover prose and code fences
around a value, fences alone, Unicode whitespace, invalid values before a valid one, duplicate
keys, escapes, numbers in every form, `NaN`, `Infinity` and a float past the largest double,
literals, trailing commas, single quotes, a 5,000-digit integer, which `int()` refuses, and 200
levels of nesting.

## routes.json

HTTP exchanges with `create_app(Settings(backend="mlx", ...))` in FastAPI's `TestClient`, the
runtime replaced as upstream's `tests/test_mlx_backend.py` replaces it. `completion_id` and
`created` are the id and time every reply carries: the script fixes upstream's random id and
clock while it records. `markers` is `engine.thought_open + engine.thought_close`, which chat
passes as `skip_special`, and `scaffold` the scaffold's ids. `runtimes` describes the three stub
runtimes, `stub`, `replay` and `one_token`, which are upstream's `StubRuntime`, `ReplayRuntime` and
`OneTokenRuntime`.

`cases` holds `{name, note, settings, runtime, running, request, response, request_id_headers,
server_timing_present, prompts, encodings, generations}`:

- `settings` are the `Settings` fields that differ from upstream's defaults, and `running`, when
  not null, the generator's `running` count set before the request, as
  `test_chat_capacity_is_refused` sets it.
- `request` is `{method, path, headers, body_text}`, or `body_base64` for a body sent as bytes
  (the one that starts with a byte order mark).
- `response` is `{status, headers, body_text}` with `content-type` and `retry-after`. A streamed
  reply's `body_text` is the whole event stream.
- `prompts` lists every `prompt_ids` call the request made, `{messages, thinking, ids}`, and
  `encodings` every text `Engine.enc` tokenized for it (the `stop` strings), `{text, ids}`, so a
  test can replay them without the tokenizer.
- `generations` lists what the runtime was asked: `{prompt, max_tokens, stop_ids,
  skip_special}`.

Upstream failures are recorded as they happen and marked in their `note`: a message without a
role or that is not an object, a role that is not a string, a `response_format` string, a `stop`
number and a `chat_template_kwargs` list make the route raise, and Starlette answers a bare 500
without the request id headers. `completion_nan_in_body` is a body only Python's `json.loads`
reads (decision D-016).

The stub runtimes generate far faster than a model. Under CPython 3.14, a generation that
finishes before `run_in_executor` registers its callback completes the awaited future at once,
ahead of the chunks it emitted with `call_soon_threadsafe`, and upstream's stream then queues its
end marker before them: a fresh app's first one-token stream lost its only chunk in about half of
30 tries. CPython 3.12, which upstream's container runs, always completes it through
`call_soon_threadsafe`, after the chunks. The script's stub runtimes start each call only once its
callback is registered, which restores that order, so a recording is the same on every run.
