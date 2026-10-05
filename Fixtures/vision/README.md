# Vision fixtures

What upstream's own image path gives DiffusionGemma on the pinned checkpoint: the images'
`pixel_values`, the expanded prompts and `mm_token_type_ids` (issue #46), and upstream's reads of
the hot dog photo, which the image runtime (#47) is held to. Layer 2 in
[docs/09-conformance-and-testing.md](../../docs/09-conformance-and-testing.md); the method and the
findings are in [docs/spikes/vision-preprocessing.md](../../docs/spikes/vision-preprocessing.md).

## The images

Eleven synthetic images, drawn by the generator and committed, each under 5 KB:

| File | Format | Size | What it exercises |
|---|---|---|---|
| `baseline.jpg` | JPEG, baseline, 4:2:0, a restart marker every 6 MCUs | 203 by 141 | libjpeg-turbo's decode and h2v2 fancy upsampling, sides that are not whole MCUs |
| `progressive.jpg` | JPEG, progressive, 4:2:2 | 157 by 99 | progressive scans and h2v1 upsampling |
| `gradients.png` | PNG, RGB | 1,040 by 624 | a slight shrink: ramps and hard-edged blocks |
| `gray.png` | PNG, grey | 1,600 by 1,000 | a 1.6 times shrink, where Pillow widens the bicubic kernel; grey to RGB |
| `small.png` | PNG, RGB noise | 32 by 20 | a 31 times enlargement |
| `frames.gif` | GIF, two frames over one palette | 96 by 64 | the first frame only (the second differs) |
| `pattern.webp` | WebP, lossless | 160 by 240 | WebP and a portrait image |
| `transparent.gif` | GIF, one frame, transparent index 3 (white) | 80 by 56 | a transparent pixel keeps its palette colour, where ImageIO hands (0, 0, 0, 0) over |
| `offset.gif` | GIF, a 72 by 40 frame at (30, 20), no transparency | 120 by 80 | the screen around the first frame is palette index 0 (orange) |
| `local.gif` | GIF, a local colour table unlike the global one | 64 by 64 | the frame's own table is the one used |
| `interlaced.gif` | GIF, interlaced | 72 by 60 | the four interlace passes land on their rows |

Upstream's `tests/data/hotdog.jpg` (12,860 bytes, a baseline 4:2:0 JPEG, 384 by 188) is read from
the pinned `Upstream/openjev` checkout and not committed; the fixture records its size and SHA-256.

## preprocessing.json

[Tools/fixtures/vision_oracle.py](../../Tools/fixtures/vision_oracle.py) calls upstream's
`openjev.mlx_backend.MlxRuntime._inputs` at `dcd2094` on an `ImagePrompt` for each prompt, with a
stand-in runtime that holds the processor `mlx_vlm.load` builds for the checkpoint (mlx-vlm 0.6.15,
Pillow 12.3.0) and no model.

- `generator` records the pins, as in [Fixtures/oracle](../oracle/README.md), with Pillow's version.
  Its `version` is this file's own, 2 since `state_prompts` (issue #124); `reads.json` keeps its
  own, 1, because only a run with the model rewrites it.
- `processor` is what the processor was built with: `max_soft_tokens` 280, `patch_size` 16,
  `pooling_kernel_size` 3, `rescale_factor`, `do_normalize` false, `resample` 3 (bicubic), the
  `size` it ignores, and the image tokens and their ids.
- `budget_rule` runs the processor's resize on blank images of 34 sizes: the size it resizes each
  to and the soft tokens that gives, or the error it raises (images 1 pixel high). The last three
  rows show the fewest soft tokens the rule gives: 700 by 10 gets 280, 701 by 10 gets 140, and
  1,190 by 17 gets 139.
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
- `state_prompts` has the fifteen states of issue #124, the `trim_` rows of
  [chat-prompts/prompts.json](../chat-prompts/README.md), each as the text of a prompt with
  `gradients.png` and the same system text: `ids`, `tokens`, `mm_token_type_ids`, `image_runs` and
  `soft_tokens`. mlx-vlm strips the user's text with Python's `str.strip()` before the template
  trims it again, so the states come out as in the text prompts: U+001C to U+001F go at either
  end, U+200B stays (354 tokens, 38834 last before `<turn|>`), and the whitespace-only state
  gives the empty state's 349 tokens.
- `gif_cases` maps a name to a small GIF for the rest of Pillow's GIF reader: its byte count,
  SHA-256 and the bytes in base64, and what upstream's `ImagePrompt.pil` (the decode in
  `MlxRuntime._inputs`) makes of it, `decoded` (the size and the SHA-256 of the RGB bytes) or
  `error`, the exception it raises. They cover a canvas grown to hold the frame, the transparent
  index around an offset frame, indices past the colour table, tables that are the grey ramp
  (global, local with a global table, local alone) and none at all, extensions and stray bytes
  ahead of the image, an extension whose first sub-block is the terminator, a short interlaced
  frame and LZW code size 12, and, raised on, an early end code, cut data, code size 13, a frame 0
  pixels wide, a broken code, a short graphic control extension, a screen past the decompression
  bomb limit and a file with no image.

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
`pixel_values` as float32 (about 90 MB), go to `Tools/oracle/results/vision/`, which git ignores,
and the run's timings to `Tools/oracle/results/vision_run.json`. The committed record is a full
run's, the reads included; pass `--run-out` with another path when redoing only the
preprocessing, so the reads' record stays.

## Using it from Swift

`Tests/OpenJevDiffusionGemmaTests/Vision/VisionPreprocessingTests.swift` compares the port with
`preprocessing.json`: the synthetic images and the GIF cases everywhere, the hot dog when the
upstream checkout is present, every value when `Tools/oracle/results/vision/` is, and the prompts
when the tokenizer files are. `Tests/OpenJevDiffusionGemmaTests/Model/ImageReadOracleTests.swift`
reads `reads.json`'s four reads through the image runtime: bit for bit in D-014's exact tier,
within its bounds natively (D-054). `Tools/oracle/stage_dump.py --image hotdog` writes mlx-vlm's
own stages of the hot dog prefill to `Tools/oracle/results/vision/hotdog.stages.safetensors` for
`ImageStageTests`.
