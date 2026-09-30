# System text fixtures

`Engine.system_text` for every group of every request that has a question to read, at the
default canvas (64). `Tools/fixtures/upstream_tables.py` writes `system_texts.json`; see
[../README.md](../README.md).

## system_texts.json

`cases` holds one row per request: `{name, request, format, chunked, groups, all_questions?}`.

- `groups` lists the groups `Engine.groups` makes, each `{questions, unchunked, chunked}`: the
  question ids and `system_text(group, format)` without and with `chunked=True`.
- `chunked` is true when the request has more than one group. Parallel reads then use each
  group's chunked text, which ends with the sentence about replies that cover only some
  questions.
- `all_questions` appears when there is more than one group. Its `unchunked` text lists every
  question and is the system text of sequential reads. `chunked` is included for completeness.

The requests are the ones in `schemas/` that have a question to read.
