# Spike #46: Gemma 4 image preprocessing parity

Question. Can the port turn a request's images into the prompt ids, `mm_token_type_ids` and
`pixel_values` that upstream's MLX backend gives DiffusionGemma, within 1e-3, and can it reuse
`MLXVLM`'s Gemma 4 processor to do it (R9, R18)?

Answer. Yes, bit for bit, but not with `MLXVLM`'s processor. On upstream's hot dog photo and seven
synthetic images (JPEG baseline and progressive, RGB and grey PNG, a two-frame GIF, a lossless
WebP), every decoded byte and every `pixel_values` value equals mlx-vlm's, and every expanded
prompt and its `mm_token_type_ids` equal the processor's. Two things stood in the way, and the port
reproduces both in Swift: Pillow's 8-bit bicubic resize, which Core Image's bicubic misses by up to
0.69, and libjpeg-turbo's JPEG decode, which ImageIO's misses by up to 57 levels. The decision is
D-051 in [../06-decisions.md](../06-decisions.md).

Versions the result holds for: upstream at `dcd2094`, mlx-vlm 0.6.15, Pillow 12.3.0 (with its
bundled libjpeg-turbo 3.1.4.1 and libwebp 1.6.0), NumPy 2.5.3, transformers 5.17.0, the
`mlx-community/diffusiongemma-26B-A4B-it-4bit` checkpoint at `a7a81407`, mlx-swift-lm `c043fb3`,
macOS 27.0.1 on an M3 Max. Run on 2026-10-02.

## What upstream does with an image

`MlxEngine.one_read` (`openjev/mlx_backend.py:290-301`) turns the image parts ahead of the state into
an `ImagePrompt`, and `MlxRuntime._inputs` (lines 95 to 114) builds the model's inputs from it:

1. `ImagePrompt.pil` base64-decodes the text after the data URL's comma and opens it with
   `PIL.Image.open(...).convert("RGB")`. The declared content type plays no part: PIL sniffs the
   bytes, and opens a GIF at its first frame.
2. `processor.apply_chat_template` renders `[system, user]`, the user content
   `[{"type": "image"}] * n` then the state text, with the generation prompt and thinking off.
3. `prepare_inputs(processor, images, prompts)` calls `DiffusionGemma4Processor`, which is mlx-vlm's
   `Gemma4Processor` (`models/gemma4/processing_gemma4.py`) with audio refused. It resizes and
   rescales each image, replaces the n-th `<|image|>` of the text with `<|image>`, that image's
   soft tokens of `<|image|>` and `<image|>`, tokenizes without adding special tokens, and marks
   `mm_token_type_ids`.

## The budget rule, from the code

`processor_config.json` lists `size` 224 by 224, `image_seq_length` and `max_soft_tokens` 280.
`Gemma4ImageProcessor.preprocess` never reads `size`. Its rule
(`aspect_ratio_preserving_resize`):

- The budget is `max_soft_tokens` (280) soft tokens, so 280 * 3² = 2,520 patches of 16 by 16
  pixels, 645,120 pixels.
- The image is scaled by `sqrt(645120 / (height * width))`, up or down, and each side is floored to
  a multiple of 48 (`pooling_kernel_size * patch_size`). A side that floors to 0 becomes 48, and
  the other side `floor(long / short) * 48`, at most 13,440.
- Its soft tokens are `(height / 16) * (width / 16) / 9`: at most 280, fewer for most aspect
  ratios. The hot dog (384 by 188) becomes 1,104 by 528 and 253 soft tokens. Over the table in
  `Fixtures/vision/preprocessing.json` (27 sizes), the sizes upstream processes correctly get 236
  to 280. A square becomes 768 by 768 (256), 4:3 912 by 672 (266), 16:9 1,056 by 576 (264).

The 70, 140, 560 and 1,120 that docs/03 listed are the values `Gemma4VideoProcessor` accepts for its
own `max_soft_tokens` (`_SUPPORTED_SOFT_TOKENS`, default 70 per frame). The image processor
validates nothing, upstream never passes another budget, and the checkpoint sets 280, so an image
read's budget is always 280 and its token count follows from the image's size.

Two sizes upstream cannot process. Transformers' `infer_channel_dimension_format` reads an
`(height, width, 3)` array whose height is 1 or 3 as channels first. An image 1 pixel high then
raises a `TypeError` in Pillow. An image 3 pixels high is resized as if it were 3 pixels wide:
100 by 3 becomes 96 by 4,608. The port refuses both with a `VisionError` (D-051). It also refuses,
as Pillow's `Image.open` does, an image of more than 178,956,970 pixels (`2 * MAX_IMAGE_PIXELS`),
reading the size from the header before it decodes anything.

The resize is `PIL.Image.resize((w, h), BICUBIC)` on the 8-bit image, then
`image.astype(np.float32) * rescale_factor`, which NumPy 2 computes in float32 with the factor cast
to float32. `do_normalize` is false (mean 0, std 1). Channels go first.

## The prompt

mlx-vlm's `prompt_utils.apply_chat_template` turns every message's content into a list of text
parts, the system message's included, and Gemma 4's template writes each system text part as
`item['text'] | trim + ' '`. So an image prompt's system turn ends `... <turn|>` with one more
token (236743, a space) than the same text prompt's. Up to the system turn's `<turn|>`, the hot
dog prompt equals `Engine.chat_prompt_ids` for the same texts but for that token. The port renders
the same message shape through swift-jinja and so gets the same space.

The expansion replaces every `<|image|>` of the rendered text, so a system or state text that
spells `<|image|>` out takes an image's place; with more placeholders than images mlx-vlm's
`re.sub` runs out of replacements and raises, and the port throws. `mm_token_type_ids` is 1 at
the soft image tokens only (not at `<|image>` or `<image|>`), 2 at `<|video|>` and 3 at
`<|audio|>`, 0 elsewhere.

`pixel_values` is one `(n, 3, H, W)` array when every image has the same size and otherwise a list
of `(3, H, W)` arrays, as `preprocess` stacks only equal shapes. The two-image prompt of the
fixture (the hot dog and the small PNG, 528 by 1,104 and 624 by 1,008) is a list, 630 tokens.
`MLXVLM`'s processor instead zero-pads every image onto the largest canvas.

## How it was measured

[Tools/fixtures/vision_oracle.py](../../Tools/fixtures/vision_oracle.py) runs upstream's own
`MlxRuntime._inputs` on a stand-in that has the processor but no model, loaded the way
`mlx_vlm.load` loads it, and writes [Fixtures/vision](../../Fixtures/vision/README.md). It runs
everything twice in opposite orders and writes only when both passes agree; `--check` reran the
whole oracle, reads included, and found no difference.

The Swift tests (`Tests/OpenJevDiffusionGemmaTests/Vision/VisionPreprocessingTests.swift`) compare
the shape, per-channel mean, std, min and max and the 4,096 sampled values of every image within
1e-3, and record whether the float32 bytes hash to the oracle's digest. The committed images run
in CI. With the pinned upstream checkout present, the hot dog is checked too, and with
`Tools/oracle/results/vision/` present every value of every tensor is compared. With the tokenizer
files, every fixture prompt's ids, `mm_token_type_ids`, soft tokens and pixel shapes are compared.

## Results

| Image | Decoded | Resized to | Soft tokens | Prompt tokens | Port: largest difference | Core Image: largest difference |
|---|---|---|---|---|---|---|
| `hotdog.jpg` (upstream, baseline 4:2:0) | 384 by 188 | 1,104 by 528 | 253 | 355 | 0, bit for bit | 0.517 |
| `baseline.jpg` (4:2:0, restart markers) | 203 by 141 | 960 by 624 | 260 | 362 | 0, bit for bit | 0.613 |
| `progressive.jpg` (progressive 4:2:2) | 157 by 99 | 1,008 by 624 | 273 | 375 | 0, bit for bit | 0.654 |
| `gradients.png` (RGB) | 1,040 by 624 | 1,008 by 576 | 252 | 354 | 0, bit for bit | 0.179 |
| `gray.png` (grey) | 1,600 by 1,000 | 1,008 by 624 | 273 | 375 | 0, bit for bit | 0.174 |
| `small.png` (RGB noise) | 32 by 20 | 1,008 by 624 | 273 | 375 | 0, bit for bit | 0.624 |
| `frames.gif` (two frames) | 96 by 64 | 960 by 624 | 260 | 362 | 0, bit for bit | 0.690 |
| `pattern.webp` (lossless) | 160 by 240 | 624 by 960 | 260 | 362 | 0, bit for bit | 0.589 |

The differences are over every value of the full tensors (1.7 to 1.9 million per image). Core
Image's is `MLXVLM.Gemma4Processor.preprocess(image:processing:)` with the checkpoint's settings:
it gets the same sizes, but differs at 307,440 (the GIF, mostly flat colour) to 1,886,973 values,
and its sampled values miss by 0.070 (`gradients.png`) to 0.466 (`small.png`). It resizes in float
in Core Image's sRGB tone-curve space, with no 8-bit result after each pass; which part of Core
Image's filter accounts for which share of the difference was not taken apart.

Decoding. ImageIO's PNG, GIF and lossless WebP samples equal PIL's exactly, read without colour
management (PIL ignores profiles, copies a grey level into three channels, looks palette indices up
and drops alpha without compositing). Its JPEG decode does not: on the hot dog it is up to 30
levels from Pillow's at 74,176 of 216,576 samples, on `baseline.jpg` up to 57 levels at 25,739 of
85,869, on `progressive.jpg` up to 18 at 18,665 of 46,629. That alone put the hot dog 0.110 outside
the bound. The port therefore decodes JPEG with a Swift port of libjpeg-turbo's default path
(Huffman baseline and progressive decoding, the accurate integer IDCT, fancy upsampling, the
fixed-point YCbCr conversion), which gives Pillow's bytes on every fixture JPEG. Ten more JPEGs
made for this spike with Pillow and Homebrew's `cjpeg` decoded bit for bit too: 4:4:4, 4:2:0 at
quality 100, 4:4:0, 4:1:1, grey, RGB without colour conversion, two progressive 4:2:0 with a
restart every MCU row, and images of 2 by 3 and 5 by 2 pixels. They are not committed. An
eleventh, arithmetic-coded, is refused by the port and falls back to ImageIO.

The GIF test decodes frame 1 as well: it differs from frame 0, and processing it misses the
fixture's bounds, so the bounds would catch a decoder that took the wrong frame.

Planted bugs. Swapping the bicubic filter for Pillow's bilinear breaks the bounds on all seven
committed images. Keeping bicubic but not widening its support when shrinking breaks them on the two
images that shrink: by 0.031 on `gradients.png` (a 1.03 shrink) and 0.239 on `gray.png` (1.59).

Prompts. All nine fixture prompts (each image alone, and the hot dog with the small PNG) expand to
the processor's ids, `mm_token_type_ids`, soft token counts and pixel shapes.

## Scope notes

- The port's decode reads 8-bit RGB, grey and indexed images, with or without alpha, as PIL does.
  Premultiplied images are read only when every alpha is 255 (ImageIO hands opaque GIFs and WebPs
  over premultiplied). Other layouts (16-bit PNG, translucent premultiplied images, CMYK) are drawn
  into an 8-bit sRGB context, which colour-manages; none of the fixtures takes that path, so it is
  unmeasured, as are lossy WebP and CMYK JPEG.
- A request's image bytes are untrusted, so the JPEG port checks every segment as libjpeg does
  (lengths, a second frame, DC symbols above 15, progressive scan parameters, fractional sampling
  ratios, more than 10 blocks per MCU) and throws rather than reads past its input; such a JPEG
  goes to ImageIO. A frame of more than 178,956,970 pixels, or a JPEG of more than 100 scans, is
  refused outright. `JPEGRobustnessTests` feeds it each of those cases and 6,799 truncations and
  corruptions of the fixture JPEGs: 3,703 decode, 3,096 are refused, none traps. A truncated JPEG
  has no EOI, so the port leaves it to ImageIO, which may decode what is there; Pillow's `load`
  raises "image file is truncated" instead (read from Pillow's `ImageFile.load`, not run), so
  upstream would answer 500 where the port may read a partial image.
- No timing was measured: two other jobs shared the machine.
- The Layr-Labs fork's `DiffusionGemmaProcessor.swift`, `DiffusionGemmaImagePixels.swift` and
  `DiffusionGemmaBicubicRGB.swift` were not used; nothing here derives from them.

## What #47 needs next

- The runtime still refuses images (`unsupported("images")`). #47 builds `ImageReadInputs` from the
  `ImagePart`s on the runtime's side, checks the expanded length against the prompt limit after the
  expansion (upstream checks it in `_prefill`), keys the prefill cache by the text and the images'
  data URL digests (`ImagePrompt.key`), and prefills in one piece.
- The vision tower and multimodal embedder, the placeholder replacement in the embeddings, and the
  bidirectional overlay within each image block from `mm_token_type_ids`. Whether to use `MLXVLM`'s
  Gemma 4 vision tower or copy it is R18's remaining question. How the tower takes a list of
  differently sized `pixel_values` was not checked here.
- [Fixtures/vision/reads.json](../../Fixtures/vision/README.md) holds upstream's reads for the hot
  dog request of `test_live.py` and the README's questions with the hot dog (355 and 411 prompt
  tokens; `hotdog` 0.995 and 0.999, `cat` 0.0003 and 0.0001), recorded two passes bit for bit.
- How a `VisionError` reaches the wire. Upstream's `/v1/systemone` catches only `SchemaError`,
  `Upstream`, `Overloaded` and `httpx.HTTPError` (`openjev/api.py:265-272`), so a PIL or processor
  exception is uncaught and Starlette answers 500. That is read from the code; no such request was
  sent.
