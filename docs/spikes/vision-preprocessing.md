# Spike #46: Gemma 4 image preprocessing parity

Question. Can the port turn a request's images into the prompt ids, `mm_token_type_ids` and
`pixel_values` that upstream's MLX backend gives DiffusionGemma, within 1e-3, and can it reuse
`MLXVLM`'s Gemma 4 processor to do it (R9, R18)?

Answer. Yes, bit for bit, but not with `MLXVLM`'s processor. On upstream's hot dog photo and eleven
synthetic images (JPEG baseline and progressive, RGB and grey PNG, five GIFs, a lossless WebP),
every decoded byte and every `pixel_values` value equals mlx-vlm's, and every expanded prompt and
its `mm_token_type_ids` equal the processor's. Three things stood in the way, and the port
reproduces each in Swift: Pillow's 8-bit bicubic resize, which Core Image's bicubic misses by up to
0.69, libjpeg-turbo's JPEG decode, which ImageIO's misses by up to 57 levels, and Pillow's reading
of a GIF's first frame, where ImageIO leaves the transparent index and the screen around the frame
black. The spike missed the third; PR #121's review found it ([GIFs](#gifs)). The decision is
D-051 in [../06-decisions.md](../06-decisions.md).

Versions the result holds for: upstream at `dcd2094`, mlx-vlm 0.6.15, Pillow 12.3.0 (with its
bundled libjpeg-turbo 3.1.4.1 and libwebp 1.6.0), NumPy 2.5.3, transformers 5.17.0, the
`mlx-community/diffusiongemma-26B-A4B-it-4bit` checkpoint at `a7a81407`, mlx-swift-lm `c043fb3`,
macOS 27.0.1 on an M3 Max. Run on 2026-10-02; the GIF port and the review's measurements on
2026-10-03.

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
  ratios. The hot dog (384 by 188) becomes 1,104 by 528 and 253 soft tokens. A square becomes 768
  by 768 (256), 4:3 912 by 672 (266), 16:9 1,056 by 576 (264).
- Extreme aspect ratios get fewer. The scaled sides are *a* and *b* units of 48 pixels with *a*
  times *b* = 280, and when neither floors to 0 the count is floor(*a*) times floor(*b*), so a
  side just under two units keeps one. 3 by 100 gets 192, and 701 by 10 gets 140 (6,720 by 48)
  where 700 by 10 gets 280 (6,720 by 96). At 1,190 by 17 both sides are whole numbers of units in
  real arithmetic, float rounding puts each just under, and the count is 139, the fewest the rule
  gives: every size up to 6,000 by 6,000 was run through mlx-vlm's arithmetic, and 4.9% of them
  get fewer than 236. Over the table in `Fixtures/vision/preprocessing.json` (34 sizes), the sizes
  upstream processes correctly get 139 to 280.

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
whole oracle, reads included, and found no difference. After PR #121's review the preprocessing
was regenerated with the new GIFs and rerun with `--only preprocessing --check`, again with no
difference; the reads were not rerun, since nothing they read changed.

The Swift tests (`Tests/OpenJevDiffusionGemmaTests/Vision/VisionPreprocessingTests.swift`) compare
the shape, per-channel mean, std, min and max and the 4,096 sampled values of every image within
1e-3, and record whether the float32 bytes hash to the oracle's digest. The committed images run
in CI. With the pinned upstream checkout present, the hot dog is checked too, and with
`Tools/oracle/results/vision/` present every value of every tensor is compared. With the tokenizer
files, every fixture prompt's ids, `mm_token_type_ids`, soft tokens and pixel shapes are compared.
The 22 small GIF cases are decoded in CI and must give PIL's bytes, or be refused where PIL
raised.

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
| `transparent.gif` (transparent index white) | 80 by 56 | 960 by 672 | 280 | 382 | 0, bit for bit | 1.205 |
| `offset.gif` (frame at 30, 20 on a 120 by 80 screen) | 120 by 80 | 960 by 624 | 260 | 362 | 0, bit for bit | 1.024 |
| `local.gif` (local colour table) | 64 by 64 | 768 by 768 | 256 | 358 | 0, bit for bit | 0.650 |
| `interlaced.gif` (interlaced) | 72 by 60 | 864 by 720 | 270 | 372 | 0, bit for bit | 0.693 |

The differences are over every value of the full tensors (1.7 to 1.9 million per image). Core
Image's is `MLXVLM.Gemma4Processor.preprocess(image:processing:)` with the checkpoint's settings:
it gets the same sizes, but differs at 307,440 (`frames.gif`, mostly flat colour) to 1,935,355
values, and its sampled values miss by 0.070 (`gradients.png`) to 1.124 (`transparent.gif`, whose
transparent pixels Core Image's decode leaves black, as ImageIO's does). It resizes in float
in Core Image's sRGB tone-curve space, with no 8-bit result after each pass; which part of Core
Image's filter accounts for which share of the difference was not taken apart.

Decoding. ImageIO's samples of the fixture's PNGs and WebP equal PIL's exactly, read without
colour management (PIL ignores profiles, copies a grey level into three channels, looks palette
indices up and drops alpha without compositing). Its GIF decode does not, which the spike missed
because `frames.gif` has no transparent index and covers its screen: see [GIFs](#gifs). Its JPEG
decode does not either: on the hot dog it is up to 30 levels from Pillow's at 74,176 of 216,576
samples, on `baseline.jpg` up to 57 levels at 25,739 of 85,869, on `progressive.jpg` up to 18 at
18,665 of 46,629. That alone put the hot dog 0.110 outside the bound. The port therefore decodes
JPEG with a Swift port of libjpeg-turbo's default path (Huffman baseline and progressive decoding,
the accurate integer IDCT, fancy upsampling, the fixed-point YCbCr conversion), which gives
Pillow's bytes on every fixture JPEG. Ten more JPEGs made for this spike with Pillow and Homebrew's
`cjpeg` decoded bit for bit too: 4:4:4, 4:2:0 at quality 100, 4:4:0, 4:1:1, grey, RGB without
colour conversion, two progressive 4:2:0 with a restart every MCU row, and images of 2 by 3 and 5
by 2 pixels. They are not committed. An eleventh, arithmetic-coded, is one the port does not
cover, so it goes to ImageIO (0.110 off, below).
After PR #121's review the port also follows libjpeg-turbo on damaged and unusual data, and
computes the IDCT as the Arm Neon code Pillow runs on Apple silicon does
([below](#the-jpeg-decoder-after-the-review), D-055).

The GIF test decodes frame 1 as well: it differs from frame 0, and processing it misses the
fixture's bounds, so the bounds would catch a decoder that took the wrong frame.

Planted bugs. Swapping the bicubic filter for Pillow's bilinear breaks the bounds on all eleven
committed images. Keeping bicubic but not widening its support when shrinking breaks them on the two
images that shrink: by 0.031 on `gradients.png` (a 1.03 shrink) and 0.239 on `gray.png` (1.59).

Prompts. All 13 fixture prompts (each image alone, and the hot dog with the small PNG) expand to
the processor's ids, `mm_token_type_ids`, soft token counts and pixel shapes.

## GIFs

What PR #121 got wrong. ImageIO hands a GIF's first frame over as 32-bit RGBA with `alphaInfo`
`.last` (not premultiplied) and (0, 0, 0, 0) for the pixels of the transparent index, and for the
logical screen around a first frame that does not cover it. Pillow 12.3.0 opens the frame in mode
P, with the canvas filled with the transparent index (or index 0 when there is none) before the
frame is decoded into it, and `convert("RGB")` looks every index up in the palette, the
transparent one included. The port copied ImageIO's colour samples, so it read black where
upstream reads the palette colour. Measured on macOS 27.0.1 against upstream's own
`MlxRuntime._inputs`, the largest difference in `pixel_values` before the fix: Zscaler's
`loader.gif` (a white transparent colour) 1.000, at 1,336,158 of 1,769,472 values; Microsoft
Word's `Lines/Autumn Leaves.gif` 0.871; 11 of the 16 GIFs in Claude.app's resources 1.000; an
ffmpeg GIF made from an RGBA PNG (a green transparent colour) 1.000; Pillow's own RGBA-to-GIF save
0.529; and a 100 by 60 first frame at (20, 15) on a 150 by 100 screen without transparency
0.714. GIFs whose transparent colour is black, or with no transparent pixel in a frame that covers
the screen, matched.

The port. `PillowGIFDecoder` translates Pillow's frame-0 path: `GifImagePlugin.py`'s screen and
block parsing, quirks included (an extension whose first sub-block is the terminator makes it read
the next block's bytes as sub-blocks, and stray bytes between blocks are skipped), the canvas grown
to hold a frame that reaches past the screen, `GifDecode.c`'s suspendable LZW decoder driven by
`ImageFile.load`'s 65,536-byte reads, and `convert`'s palette lookup. Three of its consequences
show only in a translation. A colour table that is the grey ramp is dropped, so the indices are
grey levels, except that a frame whose local table is the ramp takes the global table (the frame
stays mode L and `Image.load` then puts the global palette on it). Indices past the table's end
are black, since Pillow 12.3.0's palette entries start as 0. And an LZW end code before the frame
is full does not stop the decoder while the file has bytes `ImageFile.load` has not read yet, so
whether such a GIF decodes or raises "image file is truncated" depends on where the reads fall.
The port refuses every GIF on which Pillow raises: an LZW code size above 12, broken codes, data
cut short, a frame 0 pixels wide, a canvas past the decompression bomb limit, headers cut short.

The evidence, from the port's sources built on their own against Pillow 12.3.0:

- The 629 GIF files on the development Mac (apps, frameworks, Homebrew): 625 decode to Pillow's
  bytes, and 4, Perl data files named `.gif`, are refused by both. PR #121's code differed on 348
  of them, among them `earth.gif` in Tk's demos, which ImageIO refuses and Pillow decodes.
- Through upstream's `_inputs`, 33 GIFs (the ones above, all 16 of Claude.app's and the review's
  others) now give `pixel_values` equal to upstream's, bit for bit.
- 123,000 generated GIFs, valid and broken (random screens, frames, tables, transparency,
  interlacing, extensions, LZW code sizes 0 to 13, early end codes, cuts and corrupted bytes):
  68,133 decode to Pillow's bytes and 54,867 are refused by both. None differs.
- 13 GIFs of up to 241 KB with end codes and cuts around the 65,536-byte reads behave as Pillow
  does.
- In the fixture, four GIFs (`transparent.gif`, `offset.gif`, `local.gif`, `interlaced.gif`) and
  22 small GIF cases hold the port to Pillow in CI. ImageIO alone misses `transparent.gif` by up to
  255 levels at 5,529 of 13,440 samples and `offset.gif` by up to 230 at 20,160 of 28,800, and
  decodes `local.gif`, `interlaced.gif` and `frames.gif` as Pillow does.

A translation rather than a recolouring of the pixels ImageIO leaves transparent, from the GIF's
own tables: a recolouring would rest on ImageIO's canvas size, palette choice and handling of
broken data, which the paragraphs above show differ from Pillow's, and only the translation could
be checked bit for bit (D-051).

## Scope notes

- The port reads the 8-bit RGB, grey and indexed images ImageIO decodes, with or without alpha,
  as PIL does. Premultiplied images are read only when every alpha is 255; on macOS 27.0.1 none of
  the images measured arrives premultiplied (opaque GIFs and WebPs come as `noneSkipLast`,
  translucent PNGs and WebPs as `last`). Other layouts (16-bit PNG, translucent premultiplied
  images, CMYK) are drawn into an 8-bit sRGB context, which colour-manages and composites over
  black; none of the fixtures takes that path, and the review's measurements of it are below.
- A request's image bytes are untrusted, so the JPEG port refuses, as a `VisionError`, the JPEGs on
  which upstream's Pillow or its libjpeg-turbo raises, and hands to ImageIO only the
  arithmetic-coded, lossless and 4-component JPEGs it does not cover, which Pillow decodes. Decoding
  work stays in proportion to the input, and a JPEG whose scans would visit more than 838,860,600
  blocks one at a time is refused (D-055, which lists the exceptions). `JPEGParityTests` holds it to
  Pillow on 185 regression cases (`Fixtures/vision/jpeg_cases.json`), and `JPEGRobustnessTests`
  feeds it truncations and corruptions of the fixture JPEGs, each of which must decode or be refused
  within the bound on work: in CI a subset of 866 that keeps every kind of mutation on every
  segment, and all 7,650 with `OPENJEV_TEST_JPEG_MUTATIONS=1` (docs/development.md). A truncated
  JPEG is refused where Pillow's `load` raises "image file is truncated", and decoded where Pillow
  decodes it, as when only the EOI is missing after a single scan.
- No timing was measured: two other jobs shared the machine.
- The Layr-Labs fork's `DiffusionGemmaProcessor.swift`, `DiffusionGemmaImagePixels.swift` and
  `DiffusionGemmaBicubicRGB.swift` were not used; nothing here derives from them.

## What PR #121's review measured

PR #121 merged before its review finished. The review measured these on macOS 27.0.1 against
upstream's own `MlxRuntime._inputs`, as the largest absolute difference in `pixel_values`, and
every figure below was rerun for this report on the same machine with the port's sources:

- Bit for bit: lossy WebP (quality 80 and 95, with alpha, an animated one's first frame, and one
  with an ICC profile and EXIF); 8-bit translucent PNGs (RGBA with random alpha, grey with alpha,
  palette with tRNS, colour-key tRNS), which ImageIO hands over not premultiplied; EXIF
  orientation, which neither side applies; ICC profiles and gAMA, which both sides ignore on the
  direct path; the resize rule, against mlx-vlm's own `aspect_ratio_preserving_resize`, on 6.77
  million sizes (6.56 million in the rerun: every size up to 2,560 by 2,560 and 5,678 others); and
  the `Resample.c` port against Pillow on 292 random resizes.
- Still different: 16-bit PNGs in RGB, opaque RGBA and grey with alpha by 0.0118 (one level);
  16-bit grey by 1.000 (Pillow opens it as `I;16` and clips everything above 255 to white); a
  16-bit Display P3 PNG by 0.471; 16-bit translucent RGBA by 1.000 (the drawn path composites over
  black); CMYK and Adobe CMYK JPEGs, which go to the ImageIO fallback, by 0.525 to 0.529; an
  arithmetic-coded JPEG by 0.110.
- Accepted by the port where upstream raises, for #47 to decide on the wire rather than here:
  HEIC bytes labelled `image/jpeg` (Pillow's `UnidentifiedImageError`, which upstream answers with
  a 500), which are no JPEG, so ImageIO decodes them. The review's other two kinds, JPEGs missing
  only their EOI on which Pillow raises "image file is truncated" and JPEGs cut short (the same
  error), went to ImageIO too; the JPEG port now refuses them (D-055, the fixture's `eof_` cases)
  and still decodes `baseline.jpg` without its EOI, as Pillow does.

## The JPEG decoder after the review

The review's fuzzing ran the JPEG port and Pillow 12.3.0 over 211,936 generated JPEGs. The port gave
Pillow's bytes for 33,286 of them; it gave other bytes for 15,799, decoded 2,584 on which Pillow
raises, and handed 160,252 to ImageIO, 12,300 of them ones Pillow decodes. D-055 records what
changed and why; in short, the port now follows libjpeg-turbo 3.1.4.1 and Pillow where the review
found it did not (scan data that runs out, libjpeg-turbo's and Pillow's refusals, the standard
Huffman tables, codes longer than 16 bits, restart resynchronisation, the limit of 10 blocks per MCU
per scan), adds block smoothing, and holds less memory at the pixel limit. Three more things decide
Pillow's bytes, and are ported too: the inverse DCT is libjpeg-turbo's Arm Neon code, whose 16-bit
arithmetic differs from jidctint.c's on corrupt data; libjpeg-turbo's fast Huffman path leaves the
coefficients it wrote when it meets a marker or FF FF and the slow path decodes the MCU again over
them; and Pillow's 65,536-byte reads decide which MCUs take the fast path, and where reading stops
after a single scan's last row.

Method. Each corpus went through Pillow (`Image.open(...).convert("RGB")` in `Tools/oracle/.venv`,
run afresh) and through a standalone optimized build of the decoder's three source files, and the
outcomes were compared per file: the same RGB bytes, or a refusal where Pillow raises. Ablations
showed that the files made for the fast path and the 65,536-byte reads tell those rules apart:
without the fast path's stale coefficients 802 of those 3,973 files differ from Pillow, and without
the 65,536-byte reads 219 do. The libjpeg-turbo sources were read for every rule, and its Homebrew
build (3.1.4.1, linked into small C programs) settled questions about coefficients and
`last_good_iMCU_row` where Pillow shows only pixels. Now every file of the review's corpora and of
7,847 more made for this change decodes to Pillow's bytes or is refused where Pillow raises, except
41 arithmetic-coded or lossless files, which go to ImageIO as D-051 decided (2 of them lossless
files Pillow refuses, D-055 item 7); an AddressSanitizer build gives the same outcomes with no
report.

The review's recorded Pillow results for its batches `st1` to `st8` (JPEGs with random tables)
did not reproduce: 2,275 files decode to the same size with other bytes when Pillow is run again.
Every one of them equals the review's own run with libjpeg-turbo's SIMD code disabled, so those
records came from the C inverse DCT, not the Neon one Pillow runs by default; the counts here use
fresh runs, which repeat bit for bit.

## What #47 needs next

Done by #47 and #48 (D-054): the runtime builds `ImageReadInputs`, keys the prefill by
`ImagePrompt.key`, checks the cap after the expansion and prefills in one piece through a port of
mlx-vlm's tower (MLXVLM's is private), the four hot dog reads are bit for bit in the exact tier, and
a `VisionError` is a 400 at `["body", "images", i]`. The list below is as the spike left it.

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
