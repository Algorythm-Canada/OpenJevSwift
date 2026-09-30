# Group and canvas fixtures

`Engine.groups`, `Engine.canvas_width` and `Engine.build_canvas` for a set of requests at three
canvas sizes. `Tools/fixtures/upstream_tables.py` writes `groups_and_canvases.json`; see
[../README.md](../README.md).

## groups_and_canvases.json

`cases` holds one row per request and canvas size: `{name, settings, request, format, seed, groups}`.

- `settings` is `{}` (canvas 64, step 16, upstream's defaults), `{"canvas": 32}` or
  `{"canvas": 40}`. At 40 the width is capped below the rounded 48.
- `seed` is the request seed, as `seeds.json` derives it.
- Each group is `{questions, rows, template, slots, width, canvases}`. `rows` is
  `len(scaffold) + len(enc(answer_text)) + 1`, the number `groups()` compares with the canvas.
  `template` and `slots` are `resolve_template(group, format)` for a plain read, and `width` is
  `canvas_width(template)`: `len(template) + 1` rounded up to the step, capped at the canvas.
- `canvases` lists `{seed, noise, canvas}`. `canvas` is `build_canvas(template, slots, seed)`:
  the template, the turn close 106, pads (0) up to the width, and each slot position replaced by
  the next `random.Random(seed).randrange(262144)`. `noise` is those values in slot order.
- Each group has canvases for its group seed (`seed + 104729·k`), its second sample seed
  (`+ 7919`), and the fixed seeds 0, 2**32 - 1 and 2**32 + 104729. A request with more than
  eight groups has the group seed only.

Widths 16, 32, 48 and 64 all occur at the default canvas. The requests include one noul (16),
the quickstart (32), eight nouls (48), 10 mixed questions (64), 24 nouls (64 then 48), 30 and 256 nouls (many groups,
three-digit question ids), the widest schema and 12 mixed questions.
