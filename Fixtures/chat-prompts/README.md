# Chat prompt fixtures

The prompts upstream builds for a read: `[system, user]` messages rendered by the pinned
tokenizer's chat template, with thinking off (every read) and on (the pass before a thought).
`Tools/fixtures/upstream_tables.py` writes `prompts.json`; see [../README.md](../README.md).

## prompts.json

`cases` holds one row per message pair:
`{name, source, messages, thinking_off: {text, ids}, thinking_on: {text, ids}}`.

- `messages` is `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`.
- `text` is `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
  enable_thinking=...)`.
- `ids` is `Engine.chat_prompt_ids(system, user, thinking)`, upstream's own call with
  `tokenize=True`. Newer transformers return a dict there, and upstream takes its `input_ids`.
  The script checks that `ids` equals `encode(text, add_special_tokens=False)`.
- `source` is `curated` for the pairs chosen for this file, or the fixture whose engine run
  rendered the prompt (`policies`, `errors`). Every prompt those runs rendered is here, so a
  replay tokenizer can answer `chat_prompt_ids` for the policy and error tests.

The curated pairs cover the quickstart system text chunked and unchunked, the 24-question and
255-option system texts, non-ASCII text, object, list and float states rendered as upstream
renders them, empty and whitespace-only states, surrounding whitespace, escapes and control
characters, special token text inside a state, emoji and a 400-word state.

The fifteen `trim_` rows (issue #124) hold the template's `trim`, which is jinja2's and so
Python's `str.strip()`, under the system text `Answer.`: the state `Look at the photo.` alone,
with each of U+001C to U+001F after it and before it, with U+000B, U+0085, U+00A0 and U+200B
after it, a state of the 29 characters `str.isspace()` accepts, and an empty state. Upstream's
engine hands the state over unstripped, so only the template trims it.

What the rows show: the template writes `<bos>`, trims the system and user contents, and ends
with `<|turn>model\n`. Thinking on adds `<|think|>\n` at the top of the system turn. The empty
thought block is not part of the prompt: the engine puts it at the head of the canvas. The trim
removes every character `str.isspace()` accepts, U+001C to U+001F included, and nothing else:
each `trim_` state but U+200B's gives the plain state's 21 ids, U+200B stays (`.` and U+200B
are one token, 38834), and the whitespace-only state gives the empty state's 16.
