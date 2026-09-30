# Template fixtures

Answer texts and `Engine.resolve_template` results for every group of every request that has a
question to read, and the template errors. `Tools/fixtures/upstream_tables.py` writes both files;
see [../README.md](../README.md).

## templates.json

`cases` holds one row per request: `{name, request, format, group_count, groups}`. Each group is
`{group, questions, labels, answer_text, alternatives, variants}`.

- `answer_text` is `answer_text(group, [0, 0, ...], format)`, every question at its first label:
  `"q1: A\nq2: 0\nq3: yes"` in the lines format, `"q1yes q2A q30"` in the indexed format.
- `alternatives` lists `{question, label, text}`: the answer text with one question moved to
  another label, in the order `resolve_template` tokenizes them to find each slot.
- `variants` lists `{head, lead, template, slots}` for the three ways upstream resolves a
  template. `head: "scaffold"` starts the canvas with the empty thought block
  `[100, 45518, 107, 101]` and is the plain read. `head: "none"` with an empty `lead` is a read
  after a thought. `head: "none"` with `lead` set to the join (`"\n"` or `" "`) is a sequential
  read after the first. `template` is `head + enc(lead + answer_text)`, and each slot is
  `{pos, label_ids}` with one token id per label.
- The 256-question request keeps its first two groups and its last; `group_count` says how many
  it has.

`slot_check` records a sweep over every label of a noul, a 10-level score and a 255-option
choice, at every question number from 1 to 256, between two neighbouring questions, in both
formats, as a plain read and as a sequential read. Every label kept one slot (`failures` is
empty), so "labels do not share one template slot" cannot be reached through the API with the
pinned tokenizer.

## errors.json

`cases` holds engine-level error cases and the boundaries next to them. Each case gives its
input as `request` and `questions` (the group's ids), or as `internal_questions` built by hand,
with `settings`, `format`, `head` and `lead`. Its result is `{template, slots}` or
`{error: {message, loc}}`. `label_ids_over_read_limit` is the exception: its `template` and
`slots` are the input handed to `call` (`Engine.one_read`), and `error` is the result.

- `canvas_4_*`, `canvas_8_*`, `canvas_9_*`: one noul at canvas 4, 8 and 9 with both heads. The
  check is `len(template) + 1 > canvas`: "answer template is 8 tokens; the canvas holds 7".
- `canvas_12_quickstart_group_*`: at canvas 12 `groups()` gives each quickstart question a group
  of its own, and each fits.
- `two_token_label_*`: questions built by hand with a label that is two tokens after `"q1: "`
  (`BQ`, `HZ`, `FZ`, `GZ`). In the lines format they fail with "question 'k': labels do not
  share one template slot". The key is written with Python's `repr`, so a key containing an
  apostrophe is quoted with double quotes. In the indexed format `BQ`, `HZ` and `FZ` are single
  tokens after `q1` and the template resolves.
- `label_ids_over_read_limit`: `Engine.one_read` asked for 513 label ids refuses before sending
  anything: "the questions of one read need 513 label tokens; a read allows 512. Ask them in
  separate requests." `groups()` makes this unreachable, since a whole schema needs at most
  255 + 10 + 2 ids.

`errors/cases.json` records the canvas error as an HTTP response.
