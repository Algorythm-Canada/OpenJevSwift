# Vision fixtures

What upstream's own image path gives DiffusionGemma on the pinned checkpoint: the images'
`pixel_values`, the expanded prompts and `mm_token_type_ids` (issue #46), and upstream's reads of
the hot dog photo (for #47). Layer 2 in
[docs/09-conformance-and-testing.md](../../docs/09-conformance-and-testing.md); the method and the
findings are in [docs/spikes/vision-preprocessing.md](../../docs/spikes/vision-preprocessing.md).

## The images

Seven synthetic images, drawn by the generator and committed, each under 5 KB:

| File | Format | Size | What it exercises |
|---|---|---|---|
| `baseline.jpg` | JPEG, baseline, 4:2:0, a restart marker every 6 MCUs | 203 by 141 | libjpeg-turbo's decode and h2v2 fancy upsampling, sides that are not whole MCUs |
| `progressive.jpg` | JPEG, progressive, 4:2:2 | 157 by 99 | progressive scans and h2v1 upsampling |
| `gradients.png` | PNG, RGB | 1,040 by 624 | a slight shrink: ramps and hard-edged blocks |
| `gray.png` | PNG, grey | 1,600 by 1,000 | a 1.6 times shrink, where Pillow widens the bicubic kernel; grey to RGB |
| `small.png` | PNG, RGB noise | 32 by 20 | a 31 times enlargement |
| `frames.gif` | GIF, two frames over one palette | 96 by 64 | the first frame only (the second differs) |
| `pattern.webp` | WebP, lossless | 160 by 240 | WebP and a portrait image |

Upstream's `tests/data/hotdog.jpg` (12,860 bytes, a baseline 4:2:0 JPEG, 384 by 188) is read from
the pinned `Upstream/openjev` checkout and not committed; the fixture records its size and SHA-256.

## preprocessing.json

[Tools/fixtures/vision_oracle.py](../../Tools/fixtures/vision_oracle.py) calls upstream's
`openjev.mlx_backend.MlxRuntime._inputs` at `dcd2094` on an `ImagePrompt` for each prompt, with a
stand-in runtime that holds the processor `mlx_vlm.load` builds for the checkpoint (mlx-vlm 0.6.15,
Pillow 12.3.0) and no model.

- `generator` records the pins, as in [Fixtures/oracle](../oracle/README.md), with Pillow's version.
- `processor` is what the processor was built with: `max_soft_tokens` 280, `patch_size` 16,
  `pooling_kernel_size` 3, `rescale_factor`, `do_normalize` false, `resample` 3 (bicubic), the
  `size` it ignores, and the image tokens and their ids.
- `budget_rule` runs the processor's resize on blank images of 27 sizes: the size it resizes each
  to and the soft tokens that gives, or the error it raises (images 1 pixel high).
- `text_prompt` is `Engine.chat_prompt_ids` for the prompts' system and state texts, for
  comparison: an image prompt's system turn has one more token, a space.
- `images` maps a name to the file, content type, byte count and SHA-256, PIL's format, mode and
  frame count, the decoded RGB size and the SHA-256 of its bytes, the resized size, the soft token
  count, and `pixel_values`: shape `(1, 3, H, W)`, dtype, the SHA-256 of the float32 bytes in C
  order, per-channel `mean`, `std`, `min` and `max`, and `samples`, 4,096 `[flat position, value]`
  pairs at positions drawn from a generator seeded by the image's name.
- `prompts` has one prompt per image and `hotdog+small`, two images of different sizes, each with
  the system text upstream's `Engine` builds for the hot dog request and the state
  `Look at the photo.`: `ids`, `tokens`, `mm_token_type_ids`, `image_runs` (the `[start, end)`
  runs of soft image tokens), `soft_tokens` per image, and `pixel_values`, stacked with its shape
  or, for images of different sizes, a list of shapes.

## reads.json

The same script runs `MlxRuntime(model_path).read(prompt, canvas, slots, max_tokens, steps)` with
the image prompt for two requests built by upstream's API and `Engine` code (`image_parts`, the
seed of `api.py:256-263` with the image URLs, `Engine.groups`, `system_text`, `resolve_template`
and `build_canvas`): `hotdog`, the request of upstream's `tests/test_live.py` `test_image` (state
`Look at the photo.`, nouls `hotdog` and `cat`), and `readme_hotdog`, the README's three questions
(`tests/test_live.py`'s `QUESTIONS`) with the hot dog. Each is read at canvas index 0 (the request
seed) and 1 (seed + 7919), one step.

- `settings` is upstream's `TOPK`, vocabulary size, canvas, step, `mlx_max_prompt` and re-read
  threshold.
- `requests` holds each request's questions, state and image names; `prompts` its system text and
  expanded ids (355 and 411 tokens).
- `reads` lists the four reads: request, group, format, question ids, seed, canvas index and seed,
  width, `canvas`, `slots`, `steps`, then `prompt_tokens`, `logprobs` (per slot, `[token id,
  logprob]` pairs as `MlxRuntime.read` returns them) and `distributions`
  (`openjev.engine.slot_distribution`).

## Regenerating

From the repository root, on an Apple silicon Mac with about 25 GB free:

```bash
make upstream
python3.14 -m venv Tools/oracle/.venv
Tools/oracle/.venv/bin/pip install -r Tools/oracle/requirements.txt
Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py
PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/vision_oracle.py --cache-limit-gb 4
```

`--only preprocessing` skips the reads, so it needs only the checkpoint's small files. The script
runs the preprocessing and the reads twice, the second time in reverse order (and for the reads
from an emptied prefill cache), and writes nothing unless the two passes agree bit for bit. It
redraws the images every run, and they come out byte for byte the same. `--check` compares a run
with the committed files instead of writing them. The full tensors, the decoded RGB as bytes and
`pixel_values` as float32 (about 60 MB), go to `Tools/oracle/results/vision/`, which git ignores,
and the run's timings to `Tools/oracle/results/vision_run.json`.

## Using it from Swift

`Tests/OpenJevDiffusionGemmaTests/Vision/VisionPreprocessingTests.swift` compares the port with
`preprocessing.json`: the synthetic images everywhere, the hot dog when the upstream checkout is
present, every value when `Tools/oracle/results/vision/` is, and the prompts when the tokenizer
files are. `reads.json` waits for the image runtime (#47).
