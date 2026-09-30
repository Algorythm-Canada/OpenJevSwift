# Policy fixtures

Every read upstream's engine makes for a request, and the response it sends. Recorded through
`create_app(Settings(...), tokenizer=<the pinned tokenizer>)` in FastAPI's `TestClient`, with
`Engine.one_read` and `Engine.think` replaced exactly as upstream's `tests/test_api.py` replaces
them. Pass-through spies on `Engine.read_group` and `Engine.decide` note their arguments and call
upstream's own methods. `Tools/fixtures/upstream_tables.py` writes both files; see
[../README.md](../README.md).

The stub read gives every slot 0.7 on its first label and splits 0.3 over the others; a noul
flips to `[0.3, 0.7]` on odd seeds, so averaging over samples shows. Every read bills 123 input
tokens. The stub thought is ids 7, 8 and 9 and bills 100 input tokens. `fake_read` in each file
says the same.

## policies.json

`cases` holds one row per request: `{name, settings, request, seed, groups, thinks, reads, response}`.

- `settings` lists the `Settings` fields that differ from upstream's defaults (only
  `sequential_lines_canvas_32` sets any). `seed` is the request seed `decide` received.
- `groups` lists every `read_group` call in call order: `{questions, keys, seed, lead, prefix,
  sys_text, options}`.
- `thinks` lists every `think` call: `{group, sys_text, state_text, budget}`. `group` is `null`
  for the one thought a sequential request takes before its first group.
- `reads` lists every `one_read` call in call order:
  - `group` is the index into `groups`, and `lead` is that group's lead.
  - `seed`, `steps`, `prefix`, `sys_text` and `content` are the call's arguments, with `template`
    and `slots` (`{pos, label_ids}`).
  - `label_ids` is the sorted union of the slots' label ids, the exact ids a vLLM read asks for.
  - `canvas_width`, `canvas` and `pinned` are what `_xargs` sends: `pinned` lists the non-slot
    positions and is `null` for one step.
  - `mlx_prompt` says what `MlxEngine.one_read` would prefill: `prefix`, `chat_prompt_ids`
    (the ids for `sys_text` and `content` with thinking off, in `chat-prompts/prompts.json`) or
    `image_prompt` (the processor expands images, so there are no ids to record).
- `response` is `{status, content_type, body_text}`.

The requests: the quickstart plain, with its defaults explicit and with them `null`,
`steps: 4` with `samples: 4`, `samples: 1`, `steps: 8`, `think: 256`, 24 nouls sequential (two
groups), sequential with one group, with `think` and with `samples`, sequential in the lines format
at canvas 32, `think` with two parallel groups, 30 nouls in parallel groups, samples with two
groups, images ahead of the state with and without samples, forced questions among read ones,
forced questions only (no read), the widest schema, an object state and 12 mixed questions.

Within a group, reads are in sample order (`seed + 7919·k`). Parallel groups (`seed + 104729·k`)
interleave as asyncio schedules them, so compare reads per group rather than across groups.

## auto_rereads.json

The same recording with one change to the stub: every slot reports entropy 0.5 instead of 0.05,
above the 0.1 threshold, so the automatic re-reads run. A group then gets four reads at
`seed + 7919·k` for k from 0 to 3, billed once. `samples` replaces the re-reads, and
`auto_max: 1` turns them off.
