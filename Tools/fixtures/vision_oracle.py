"""Record what upstream's own image path gives DiffusionGemma, as the vision oracle.

Issue #46 (and the reads of #47). Upstream's `openjev.mlx_backend.MlxRuntime._inputs` turns an
`ImagePrompt` (system text, state text, image data URLs) into the model's inputs: PIL decodes each
data URL and converts it to RGB, mlx-vlm's `DiffusionGemma4Processor` renders the chat template
with `[{"type": "image"}] * n` then the text, and `prepare_inputs` resizes and rescales the images
and expands each image placeholder. This script runs that code, unchanged, on fixed images and
writes two fixtures:

- Fixtures/vision/preprocessing.json: per image, the decoded RGB size and digest, the size the
  processor resized to, its soft token count, and `pixel_values` (shape, dtype, the SHA-256 of
  its float32 bytes, per-channel mean, std, min and max, and 4,096 sampled values); per prompt,
  the expanded ids, `mm_token_type_ids` and the soft tokens per image, also for the states of
  issue #124 as the text of a prompt with one image (`state_prompts`); the processor's resize
  rule on a table of image sizes; and `gif_cases`, small GIFs (their bytes in base64) for the
  rest of Pillow's GIF reader, with what upstream's `ImagePrompt.pil` decodes each to (the size
  and the SHA-256 of the RGB bytes) or the exception it raises. No weights are needed: the
  processor is loaded the way `mlx_vlm.load` loads it, without the model.
- Fixtures/vision/reads.json: `MlxRuntime.read` for the hot dog request of upstream's
  tests/test_live.py (`Look at the photo.`, nouls `hotdog` and `cat`) and for the README's
  questions with the hot dog, built by upstream's API and `Engine` code: `prompt_tokens`, the
  logprobs exactly as the read returns them and `slot_distribution`. Every read runs twice, the
  second time from an emptied prefill cache and in reverse order, and nothing is written unless
  the two passes agree bit for bit, as in Tools/fixtures/mlx_vlm_oracle.py.

The images are upstream's tests/data/hotdog.jpg, read from the pinned Upstream/openjev checkout
and not committed, and eleven synthetic images this script draws and writes to
Fixtures/vision/: a baseline 4:2:0 JPEG with restart markers and a progressive 4:2:2 JPEG, a
non-square RGB PNG just over the token budget and a grayscale PNG well over it (so the resize
shrinks them), a small noise PNG (so the resize enlarges it), a two-frame GIF whose frames
differ, a lossless WebP, and four GIFs where ImageIO's first frame is not Pillow's: a transparent
index whose colour is not black, a first frame offset on a larger logical screen, a local colour
table that differs from the global one, and an interlaced frame. The script draws them
deterministically, so a second run writes the same bytes.

The full tensors (`pixel_values` as float32, and the decoded RGB as bytes, both C order) go to
Tools/oracle/results/vision/, outside the fixtures, for the Swift test that compares every value
when they are present. Timings and memory go to Tools/oracle/results/vision_run.json.

With `--check` nothing is written to Fixtures: the images are drawn and compared with the
committed files, and the run is compared with the committed JSON, every top-level key. With
`--only preprocessing` the model is not loaded and only preprocessing.json is written or checked.

Usage, from the repository root (Tools/oracle/requirements.txt pins the environment):

    make upstream
    python3.14 -m venv Tools/oracle/.venv
    Tools/oracle/.venv/bin/pip install -r Tools/oracle/requirements.txt
    Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py
    PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/vision_oracle.py --cache-limit-gb 4

The script lives in Tools/fixtures because Tests/OpenJevCoreTests/Fixtures/FixturePinTests.swift
requires every fixture's generator to be a script there. It runs from Tools/oracle/.venv because
the reads need MLX and the weights.
"""
import argparse
import base64
import gc
import hashlib
import io
import json
import os
import platform
import re
import struct
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
FIXTURES = ROOT / "Fixtures"
VISION = FIXTURES / "vision"
PREPROCESSING_OUT = VISION / "preprocessing.json"
READS_OUT = VISION / "reads.json"
TENSORS_OUT = ROOT / "Tools" / "oracle" / "results" / "vision"
RUN_OUT = ROOT / "Tools" / "oracle" / "results" / "vision_run.json"
SCRIPT = "Tools/fixtures/vision_oracle.py"
GENERATOR_VERSION = 1
UPSTREAM_COMMIT = "dcd2094"
MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
HOTDOG = UPSTREAM / "tests" / "data" / "hotdog.jpg"
HOTDOG_BYTES = 12860
SAMPLES = 4096

# Settings read the environment; clear it so the defaults are upstream's own.
for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(UPSTREAM))

import huggingface_hub  # noqa: E402
import jinja2  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import PIL  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402
from importlib.metadata import version as package_version  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import mlx_vlm.utils as mlx_vlm_utils  # noqa: E402
from openjev.api import SystemOneRequest, image_parts  # noqa: E402
from openjev.config import Settings  # noqa: E402
from openjev.engine import TOPK, VOCAB, Engine, slot_distribution  # noqa: E402
from openjev.mlx_backend import ImagePrompt, MlxRuntime  # noqa: E402


# Pins ----------------------------------------------------------------------------------------

def upstream_head():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_info():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return {"device_name": info.get("device_name"), "architecture": info.get("architecture"),
            "memory_size": info.get("memory_size")}


def generator():
    """What the fixtures depend on. Pillow decodes and resizes; the device matters only to the
    reads, whose Metal kernels may round differently on another GPU family."""
    metallib = Path(mx.__file__).parent / "lib" / "mlx.metallib"
    return {
        "script": SCRIPT,
        "version": GENERATOR_VERSION,
        "upstream": "razorback16/openjev",
        "upstream_commit": UPSTREAM_COMMIT,
        "tokenizer_repo": MODEL_REPO,
        "tokenizer_revision": MODEL_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "python": sys.version.split()[0],
        "mlx": package_version("mlx"),
        "mlx_metal": package_version("mlx-metal"),
        "mlx_vlm": package_version("mlx-vlm"),
        "pillow": PIL.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "jinja2": jinja2.__version__,
        "numpy": np.__version__,
        "device": device_info()["device_name"],
        "gpu_architecture": device_info()["architecture"],
        "metallib_sha256": hashlib.sha256(metallib.read_bytes()).hexdigest(),
    }


# Synthetic images ----------------------------------------------------------------------------

def png_bytes(rgb):
    out = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


def gradients_png():
    """1,040 by 624 RGB, just over the 280-token budget, so the resize shrinks it slightly.
    Red rises left to right, green top to bottom, and blue has two hard-edged blocks, and a
    black checkerboard. (A one-pixel diagonal would double the file.)"""
    h, w = 624, 1040
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (x * 255) // (w - 1)
    rgb[..., 1] = (y * 255) // (h - 1)
    rgb[90:260, 130:390, 2] = 255
    rgb[350:560, 600:950, 2] = 128
    checker = ((x // 8 + y // 8) % 2 == 0) & (y >= 430) & (y < 500) & (x >= 90) & (x < 230)
    rgb[checker] = 0
    return png_bytes(rgb)


def gray_png():
    """1,600 by 1,000 grayscale, which the resize shrinks about 1.6 times, so Pillow widens the
    bicubic kernel to cover 1.6 source pixels per tap: a ramp left to right, a white and a black
    block, and a 6-pixel checkerboard. PIL converts it from mode L to RGB by copying the level
    into the three channels."""
    h, w = 1000, 1600
    y, x = np.mgrid[0:h, 0:w]
    level = ((x * 255) // (w - 1)).astype(np.uint8)
    level[200:520, 300:700] = 255
    level[600:900, 1000:1450] = 0
    board = (y >= 700) & (y < 820) & (x >= 150) & (x < 450)
    level[board & ((x // 6 + y // 6) % 2 == 0)] = 255
    level[board & ((x // 6 + y // 6) % 2 == 1)] = 0
    out = io.BytesIO()
    Image.fromarray(level, "L").save(out, format="PNG", optimize=True)
    return out.getvalue()


def small_png():
    """32 by 20 of seeded noise, so the resize enlarges it about 31 times in each direction."""
    rng = np.random.default_rng(46)
    return png_bytes(rng.integers(0, 256, (20, 32, 3), dtype=np.uint8))


def frames_gif():
    """96 by 64, two frames over one eight-colour palette: vertical stripes, then horizontal
    ones. The processor reads only the first."""
    palette = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255,
               255, 255, 255] + [0] * (256 - 8) * 3
    h, w = 64, 96
    y, x = np.mgrid[0:h, 0:w]
    frames = []
    for index in ((x // 12) % 8, (y // 8 + 3) % 8):
        frame = Image.fromarray(index.astype(np.uint8), "P")
        frame.putpalette(palette)
        frames.append(frame)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0,
                   optimize=False, disposal=1)
    return out.getvalue()


def gif_frame_data(indices, interlace=False):
    """The image data Pillow writes for a frame of palette indices: the LZW code size byte (8,
    which Pillow always writes), the data sub-blocks and their terminator. The GIFs below are
    put together from these around their own headers, so each shows one thing."""
    frame = Image.fromarray(np.asarray(indices, np.uint8), "P")
    frame.putpalette([0] * 768)
    out = io.BytesIO()
    frame.save(out, format="GIF", optimize=False, interlace=interlace)
    data = out.getvalue()
    flags = data[10]
    pos = 13 + ((3 << ((flags & 7) + 1)) if flags & 0x80 else 0)
    while data[pos] == 0x21:
        pos += 2
        while data[pos]:
            pos += data[pos] + 1
        pos += 1
    if data[pos] != 0x2C or bool(data[pos + 9] & 0x40) != interlace:
        raise SystemExit("Pillow wrote an unexpected GIF frame")
    start = pos + 10 + ((3 << ((data[pos + 9] & 7) + 1)) if data[pos + 9] & 0x80 else 0)
    end = start + 1
    while data[end]:
        end += data[end] + 1
    return data[start:end + 1]


def gif_table(colors):
    """A colour table: the colours, padded with black to a power of two of at least 2 entries,
    and the 3-bit size field that says how many."""
    bits = max(1, (len(colors) - 1).bit_length())
    padded = list(colors) + [(0, 0, 0)] * ((1 << bits) - len(colors))
    return bytes(v for color in padded for v in color), bits - 1


def gif_bytes(screen, frame, data, global_colors=None, local_colors=None, transparency=None,
              interlace=False, before=b""):
    """A one-frame GIF89a: the logical screen `(width, height)`, the frame `(x, y, width,
    height)` and its image data, the global and local colour tables, a graphic control extension
    when `transparency` is an index, and `before`, raw blocks ahead of the image."""
    flags, table = 0, b""
    if global_colors is not None:
        table, size = gif_table(global_colors)
        flags = 0x80 | 0x70 | size
    out = b"GIF89a" + struct.pack("<HHBBB", screen[0], screen[1], flags, 0, 0) + table + before
    if transparency is not None:
        out += b"!\xf9\x04\x01\x00\x00" + bytes([transparency]) + b"\x00"
    flags, table = (0x40 if interlace else 0), b""
    if local_colors is not None:
        table, size = gif_table(local_colors)
        flags |= 0x80 | size
    return out + b"," + struct.pack("<HHHHB", *frame, flags) + table + data + b";"


# Eight colours, none of them black, for the GIFs.
GIF_COLORS = [(230, 120, 40), (30, 160, 60), (40, 70, 220), (255, 255, 255), (250, 210, 30),
              (150, 40, 170), (20, 200, 200), (120, 120, 120)]


def gif_pattern(h, w, colors=8):
    """Stripes, a checkerboard and rings of palette indices, so every resize has edges."""
    y, x = np.mgrid[0:h, 0:w]
    index = ((x + y) // 5) % colors
    board = (x // 4 + y // 4) % 2 == 0
    index[board & (y < h // 3)] = (index[board & (y < h // 3)] + 3) % colors
    rings = (((x - w // 2) ** 2 + (y - h // 2) ** 2) // 40) % 3 == 0
    index[rings & (y >= 2 * h // 3)] = 6 % colors
    return index


def transparent_gif():
    """80 by 56, one frame whose transparent index (3) is white and fills the corners and two
    bands. Pillow keeps a transparent pixel's palette colour; ImageIO hands it over as
    (0, 0, 0, 0), which the port read as black before it decoded GIFs itself."""
    h, w = 56, 80
    y, x = np.mgrid[0:h, 0:w]
    index = gif_pattern(h, w)
    index[index == 3] = 2
    index[((x - w // 2) ** 2) / (w // 2) ** 2 + ((y - h // 2) ** 2) / (h // 2) ** 2 > 1] = 3
    index[(y // 6) % 4 == 1] = 3
    return gif_bytes((w, h), (0, 0, w, h), gif_frame_data(index), GIF_COLORS, transparency=3)


def offset_gif():
    """A 72 by 40 first frame at (30, 20) on a 120 by 80 logical screen, with no transparency.
    Pillow fills the screen around the frame with palette index 0 (orange); ImageIO leaves it
    (0, 0, 0, 0)."""
    index = gif_pattern(40, 72)
    return gif_bytes((120, 80), (30, 20, 72, 40), gif_frame_data(index), GIF_COLORS)


def local_gif():
    """64 by 64 with a global colour table and a different local one, which the frame uses."""
    local = [(255 - r, 255 - g, 255 - b) for r, g, b in GIF_COLORS]
    global_colors = [(v, v, 0) for v in range(0, 256, 32)]
    index = gif_pattern(64, 64)
    return gif_bytes((64, 64), (0, 0, 64, 64), gif_frame_data(index), global_colors, local)


def interlaced_gif():
    """72 by 60, interlaced: the rows come in four passes, every eighth from row 0, every eighth
    from row 4, every fourth from row 2, then every second from row 1."""
    y, x = np.mgrid[0:60, 0:72]
    index = (y % 8 + x // 9) % 8
    return gif_bytes((72, 60), (0, 0, 72, 60), gif_frame_data(index, interlace=True), GIF_COLORS,
                     interlace=True)


def interlace_order(h):
    """The rows of an interlaced frame in the order its data holds them."""
    return [*range(0, h, 8), *range(4, h, 8), *range(2, h, 4), *range(1, h, 2)]


def lzw_codes(codes):
    """LZW data written by hand: `(code, width)` pairs packed least significant bit first, in
    sub-blocks of at most 255 bytes, then the terminator."""
    data, acc, n = bytearray(), 0, 0
    for code, width in codes:
        acc |= code << n
        n += width
        while n >= 8:
            data.append(acc & 255)
            acc >>= 8
            n -= 8
    if n:
        data.append(acc)
    return b"".join(bytes([len(data[i:i + 255])]) + data[i:i + 255]
                    for i in range(0, len(data), 255)) + b"\0"


def gif_cases():
    """Small GIFs for the rest of Pillow's GIF reader, decoded only: name -> bytes. Each frame is
    6 by 4 or smaller, so the records stay small."""
    four = [(200, 30, 40), (30, 160, 60), (40, 70, 220), (250, 250, 250)]
    ramp = [(i, i, i) for i in range(8)]
    y, x = np.mgrid[0:4, 0:6]
    base = (x + 2 * y) % 4
    data = gif_frame_data(base)
    wide = base.copy()
    wide[0] = [4, 9, 17, 100, 200, 255]
    wide_data = gif_frame_data(wide)
    frame = (0, 0, 6, 4)
    whole = gif_bytes((6, 4), frame, data, four)
    return {
        # The canvas grows to hold a frame that reaches past the screen, filled with index 0.
        "grown_canvas": gif_bytes((4, 3), (2, 1, 6, 4), data, four),
        # Around an offset frame, the transparent index's colour.
        "transparent_around": gif_bytes((10, 7), (2, 2, 6, 4), data, four, transparency=3),
        # Indices past the table's end are black.
        "index_past_table": gif_bytes((6, 4), frame, wide_data, four),
        # A global table that is the grey ramp is dropped: indices are grey levels, past its
        # end too.
        "ramp_global": gif_bytes((6, 4), frame, wide_data, ramp),
        # A local ramp with a global table: the indices are looked up in the global table.
        "ramp_local": gif_bytes((6, 4), frame, wide_data, four, ramp),
        # A local ramp and no global table: grey levels.
        "ramp_local_alone": gif_bytes((6, 4), frame, wide_data, None, ramp),
        # No colour table at all: grey levels.
        "no_table": gif_bytes((6, 4), frame, data),
        # A local table and a transparent index, around an offset frame.
        "local_transparent": gif_bytes((9, 6), (3, 1, 6, 4), data, four,
                                       [color[::-1] for color in four], transparency=1),
        # A transparent index past the table's end: black around the frame.
        "transparent_past_table": gif_bytes((9, 6), (3, 1, 6, 4), data, four, transparency=200),
        # Comment, looping, plain text and unknown extensions ahead of the image.
        "extensions": gif_bytes((6, 4), frame, data, four, before=(
            b"!\xfe\x05hello\x03abc\x00" + b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
            + b"!\x01\x0c" + bytes(12) + b"\x00" + b"!\x77\x02xy\x00")),
        # Stray bytes between blocks are skipped.
        "stray_bytes": gif_bytes((6, 4), frame, data, four, before=b"\x00\x01\x99"),
        # An extension whose first sub-block is the terminator: Pillow reads the next block's
        # bytes as more sub-blocks.
        "empty_extension": gif_bytes((6, 4), frame, data, four, before=b"!\x01\x00"),
        # Interlaced and 3 rows high, under the 8 of the first pass.
        "interlaced_short": gif_bytes((6, 3), (0, 0, 6, 3), gif_frame_data(base[interlace_order(3)]),
                                      four, interlace=True),
        # LZW code size 12: 13-bit codes, and an index is a code modulo 256 (300, 65, 1000).
        "code_size_12": gif_bytes((3, 1), (0, 0, 3, 1), b"\x0c" + lzw_codes(
            [(4096, 13), (300, 13), (65, 13), (1000, 13)])),
        # Refused: an end code before the frame is full, after which Pillow reads on to the end
        # of the file.
        "early_end": gif_bytes((6, 4), frame, gif_frame_data(base[:2]), four),
        # Refused: the image data cut short.
        "truncated": whole[:-4],
        # Refused: an LZW code size above 12.
        "code_size_13": gif_bytes((6, 4), frame, b"\x0d" + data[1:], four),
        # Refused: a frame 0 pixels wide.
        "zero_width": gif_bytes((6, 4), (0, 0, 0, 4), data, four),
        # Refused: a first code above the clear code.
        "broken_code": gif_bytes((2, 1), (0, 0, 2, 1), b"\x02" + lzw_codes([(4, 3), (7, 3)]), four),
        # Refused: a graphic control extension too short for its duration.
        "short_control": gif_bytes((6, 4), frame, data, four, before=b"!\xf9\x02\x00\x00\x00"),
        # Refused: a logical screen past the decompression bomb limit.
        "screen_bomb": gif_bytes((65535, 65535), frame, data, four),
        # Refused: no image before the trailer.
        "no_image": b"GIF89a" + struct.pack("<HHBBB", 6, 4, 0, 0, 0) + b";",
    }


def gif_case_records(sys_text):
    """What upstream's ImagePrompt.pil, the decode in MlxRuntime._inputs, makes of each case:
    the decoded size and the SHA-256 of its RGB bytes, or the exception it raises."""
    out = {}
    for name, data in gif_cases().items():
        record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "base64": base64.b64encode(data).decode()}
        try:
            pil = ImagePrompt(sys_text, STATE, [data_url(data, "image/gif")]).pil()[0]
            rgb = np.ascontiguousarray(np.array(pil), dtype=np.uint8)
            record["decoded"] = {"width": pil.size[0], "height": pil.size[1],
                                 "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        except Exception as error:  # noqa: BLE001, upstream answers any of them with a 500
            # Without the BytesIO's address, which changes from run to run.
            record["error"] = f"{type(error).__name__}: {re.sub(r' at 0x[0-9a-f]+', '', str(error))}"
        out[name] = record
    return out


def pattern_webp():
    """160 by 240 (portrait), lossless: a diagonal ramp, a vertical ramp and concentric rings."""
    h, w = 240, 160
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = ((x + y) * 255) // (w + h - 2)
    rgb[..., 1] = 255 - (y * 255) // (h - 1)
    rgb[..., 2] = (((x - 80) ** 2 + (y - 120) ** 2) // 300 % 2) * 255
    out = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(out, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return out.getvalue()


def jpeg_scene(h, w):
    """Three ramps, a red block and a blue disc: smooth areas and hard colour edges, so the
    chroma upsampling shows."""
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (x * 255) // (w - 1)
    rgb[..., 1] = (y * 255) // (h - 1)
    rgb[..., 2] = ((x + 2 * y) * 255) // (w + 2 * h - 3)
    rgb[h // 4:h // 2, w // 5:w // 2] = [230, 40, 30]
    rgb[(x - 3 * w // 4) ** 2 + (y - h // 2) ** 2 < (h // 5) ** 2] = [20, 30, 200]
    return rgb


def baseline_jpeg():
    """203 by 141, baseline with 4:2:0 chroma and a restart marker every 6 MCUs. Neither side
    is a multiple of the 16-pixel MCU, so the decoder's edge handling shows."""
    out = io.BytesIO()
    Image.fromarray(jpeg_scene(141, 203)).save(out, format="JPEG", quality=80, subsampling=2,
                                               restart_marker_blocks=6, optimize=False)
    return out.getvalue()


def progressive_jpeg():
    """157 by 99, progressive with 4:2:2 chroma: spectral selection and successive
    approximation scans, and horizontal-only chroma upsampling."""
    out = io.BytesIO()
    Image.fromarray(jpeg_scene(99, 157)).save(out, format="JPEG", quality=85, subsampling=1,
                                              progressive=True, optimize=False)
    return out.getvalue()


# name -> (file, content type, how the bytes are made). The hot dog is read, the others drawn.
IMAGES = {
    "hotdog": ("hotdog.jpg", "image/jpeg", None),
    "gradients": ("gradients.png", "image/png", gradients_png),
    "baseline": ("baseline.jpg", "image/jpeg", baseline_jpeg),
    "progressive": ("progressive.jpg", "image/jpeg", progressive_jpeg),
    "gray": ("gray.png", "image/png", gray_png),
    "small": ("small.png", "image/png", small_png),
    "frames": ("frames.gif", "image/gif", frames_gif),
    "pattern": ("pattern.webp", "image/webp", pattern_webp),
    "transparent": ("transparent.gif", "image/gif", transparent_gif),
    "offset": ("offset.gif", "image/gif", offset_gif),
    "local": ("local.gif", "image/gif", local_gif),
    "interlaced": ("interlaced.gif", "image/gif", interlaced_gif),
}
# Every image alone, and one prompt with two images of different sizes.
PROMPTS = {name: [name] for name in IMAGES} | {"hotdog+small": ["hotdog", "small"]}
# The sizes the budget table runs the processor's resize rule on, (width, height): the
# images', square sides around the budget's edge (803 by 803 is the largest square under it),
# photo and screen sizes, aspect ratios past 280:1, where the rule's floor gives zero, and the
# fewest soft tokens the rule gives: 701 by 10 floors its short side to one 48-pixel unit (140
# soft tokens, where 700 by 10 gets 280), and at 1,190 by 17 float rounding puts both sides
# just under whole units (139).
BUDGET_SIZES = [
    (384, 188), (203, 141), (157, 99), (1040, 624), (1600, 1000), (32, 20), (96, 64), (160, 240), (1, 1), (16, 16), (48, 48), (224, 224),
    (640, 480), (803, 803), (804, 804), (1024, 768), (1920, 1080), (4032, 3024), (3000, 10), (10, 3000),
    (700, 1), (1, 700), (13440, 48), (100000, 1), (100, 3), (100, 2), (3, 100),
    (80, 56), (120, 80), (64, 64), (72, 60), (700, 10), (701, 10), (1190, 17),
]


def image_bytes():
    """name -> (bytes, content type). The hot dog must be upstream's 12,860-byte file."""
    out = {}
    for name, (file, content_type, draw) in IMAGES.items():
        if draw is None:
            data = HOTDOG.read_bytes()
            if len(data) != HOTDOG_BYTES:
                raise SystemExit(f"{HOTDOG} is {len(data)} bytes, expected {HOTDOG_BYTES}")
        else:
            data = draw()
            if len(data) > 5000:
                raise SystemExit(f"{file} is {len(data)} bytes; the committed images stay under 5 KB")
        out[name] = (data, content_type)
    return out


def data_url(data, content_type):
    return f"data:{content_type};base64,{base64.b64encode(data).decode()}"


# The processor ---------------------------------------------------------------------------------

def load_processor(model_path):
    """The processor `mlx_vlm.load(model_path, trust_remote_code=False)` builds (as upstream's
    MlxRuntime._load calls it), without loading the model: resolving the model class registers
    DiffusionGemma4Processor with AutoProcessor, the image processor hook returns None for this
    model, and load_processor gets the config's eos ids."""
    path = Path(model_path)
    config = mlx_vlm_utils.load_config(path, trust_remote_code=False)
    mlx_vlm_utils.get_model_and_args(config)
    image_processor = mlx_vlm_utils.load_image_processor(path, trust_remote_code=False)
    processor = mlx_vlm_utils.load_processor(path, True, eos_token_ids=config.get("eos_token_id"),
                                             trust_remote_code=False)
    if image_processor is not None:
        processor.image_processor = image_processor
    return processor


def upstream_inputs(processor, prompt):
    """MlxRuntime._inputs, called on a stand-in that has the processor and mx but no model."""
    runtime = types.SimpleNamespace(processor=processor, mx=mx)
    return MlxRuntime._inputs(runtime, prompt)


def channel_stats(pixels):
    """Per channel over every image and position, in float64."""
    out = []
    for c in range(pixels.shape[1]):
        v = pixels[:, c].astype(np.float64)
        out.append({"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
                    "max": float(v.max())})
    return out


def samples(name, pixels):
    """SAMPLES distinct flat positions of the C-order tensor, from a generator seeded by the
    image's name, in increasing order, with their float32 values."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    flat = pixels.reshape(-1)
    positions = np.sort(np.random.default_rng(seed).choice(flat.size, size=min(SAMPLES, flat.size),
                                                           replace=False))
    return [[int(p), float(flat[p])] for p in positions]


def tensor_record(name, pixels):
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    return {"shape": list(pixels.shape), "dtype": "float32",
            "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "channels": channel_stats(pixels), "samples": samples(name, pixels)}


def budget_rule(processor):
    """The resize rule of Gemma4ImageProcessor.preprocess on blank RGB images of each size, given
    as PIL images as upstream gives them: the size it resizes to and the soft tokens that size
    gives, or the error it raises. Transformers' infer_channel_dimension_format reads an array
    whose first dimension is 1 or 3 as channels first, so images 1 or 3 pixels high fail."""
    rows = []
    for w, h in BUDGET_SIZES:
        row = {"width": w, "height": h}
        try:
            data, tokens = processor.image_processor.preprocess([Image.new("RGB", (w, h))])
            pixels = data["pixel_values"]
            shape = pixels[0].shape if isinstance(pixels, list) else pixels.shape[1:]
            row.update({"target_width": int(shape[-1]), "target_height": int(shape[-2]),
                        "soft_tokens": int(tokens[0])})
        except (ValueError, TypeError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def runs(ids):
    """The [start, end) runs of positions where a 0/1 list is 1."""
    out, start = [], None
    for i, v in enumerate(ids + [0]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    return out


# Requests ------------------------------------------------------------------------------------

STATE = "Look at the photo."
HOTDOG_QUESTIONS = {"hotdog": {"type": "noul", "instructions": "The photo shows a hot dog"},
                    "cat": {"type": "noul", "instructions": "The photo shows a cat"}}
# tests/test_live.py's QUESTIONS, the README's example.
README_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds",
                          "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "furious"]},
}
# name -> (questions, state, images). The README read asks its questions of the hot dog photo.
REQUESTS = {
    "hotdog": (HOTDOG_QUESTIONS, STATE, ["hotdog"]),
    "readme_hotdog": (README_QUESTIONS, STATE, ["hotdog"]),
}
# (request, canvas index): index k is read k of upstream's default policy, at seed + 7919k.
SELECTION = [("hotdog", 0), ("hotdog", 1), ("readme_hotdog", 0), ("readme_hotdog", 1)]
# Issue #124: the states of upstream_tables.py's trim_states(), each the text of a prompt with
# this image. mlx-vlm strips the user's text with Python's str.strip()
# (prompt_utils.extract_text_from_content) and the template's `trim`, jinja2's, strips it again,
# so the characters str.isspace() accepts (U+001C to U+001F among them) go at either end and
# U+200B stays.
TRIM_IMAGE = "gradients"


def trim_states():
    rows = [("trim_none", STATE)]
    rows += [(f"trim_u{ord(c):04x}_end", STATE + c) for c in "\x1c\x1d\x1e\x1f"]
    rows += [(f"trim_u{ord(c):04x}_start", c + STATE) for c in "\x1c\x1d\x1e\x1f"]
    rows += [(f"trim_u{ord(c):04x}_end", STATE + c) for c in "\x0b\x85\xa0\u200b"]
    rows.append(("trim_whitespace_only", "".join(c for c in map(chr, range(sys.maxunicode + 1)) if c.isspace())))
    rows.append(("trim_empty", ""))
    return rows


def request_inputs(name, settings, eng, images):
    """What upstream's API and Engine build for a request: the seed (api.py:256-263, the image
    URLs included), the one group, its system text, template, slots, and the ImagePrompt
    MlxEngine.one_read makes from the content Engine.decide puts together."""
    questions, state, names = REQUESTS[name]
    body = {"model": "openjev-latest", "state": state, "questions": questions,
            "images": [data_url(*images[n]) for n in names]}
    req = SystemOneRequest.model_validate(body)
    dumped = {k: q.model_dump() for k, q in req.questions.items()}
    parts = image_parts(req.images, settings)
    key = [req.state, dumped, [p["image_url"]["url"] for p in parts]]
    seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()[:4], "big")
    schema = eng.build_schema(dumped)
    fmt = schema["format"]
    groups = eng.groups(schema["questions"], fmt)
    if len(groups) != 1:
        raise SystemExit(f"{name}: {len(groups)} groups, expected 1")
    group = groups[0]
    sys_text = eng.system_text(group, fmt, False)
    template, slots = eng.resolve_template(group, fmt)
    content = list(parts) + [{"type": "text", "text": state}]
    *image_content, text = content
    prompt = ImagePrompt(sys_text, text["text"], [p["image_url"]["url"] for p in image_content])
    return {"seed": seed, "format": fmt, "questions": [q["id"] for q in group], "system": sys_text,
            "template": template, "slots": slots, "prompt": prompt, "images": names}


# Preprocessing -------------------------------------------------------------------------------

def preprocess_all(processor, eng, settings, images, order):
    """Every prompt through upstream's _inputs, in the given order. Returns the image records,
    the prompt records and the full tensors."""
    sys_text = request_inputs("hotdog", settings, eng, images)["system"]
    image_records, prompt_records, tensors = {}, {}, {}
    for key in order:
        names = PROMPTS[key]
        prompt = ImagePrompt(sys_text, STATE, [data_url(*images[n]) for n in names])
        _, kwargs, n = upstream_inputs(processor, prompt)
        ids = [int(t) for t in kwargs["input_ids"].tolist()[0]]
        mm = [int(t) for t in np.array(kwargs["mm_token_type_ids"]).reshape(-1)]
        pixel_values = kwargs["pixel_values"]
        per_image = ([np.array(p) for p in pixel_values] if isinstance(pixel_values, list)
                     else [np.array(pixel_values)[i] for i in range(np.array(pixel_values).shape[0])])
        ip = processor.image_processor
        soft = [(int(p.shape[-2]) // ip.patch_size) * (int(p.shape[-1]) // ip.patch_size)
                // ip.pooling_kernel_size ** 2 for p in per_image]
        prompt_records[key] = {
            "images": names, "system": sys_text, "state": STATE, "ids": ids, "tokens": n,
            "mm_token_type_ids": mm, "image_runs": runs(mm), "soft_tokens": soft,
            "pixel_values": ({"stacked": True, "shape": list(np.array(pixel_values).shape)}
                             if not isinstance(pixel_values, list)
                             else {"stacked": False, "shapes": [list(p.shape) for p in per_image]}),
        }
        if len(names) == 1:
            name = names[0]
            data, content_type = images[name]
            pil = prompt.pil()[0]
            rgb = np.ascontiguousarray(np.array(pil), dtype=np.uint8)
            opened = Image.open(io.BytesIO(data))
            pixels = np.array(pixel_values, dtype=np.float32)
            image_records[name] = {
                "file": IMAGES[name][0], "content_type": content_type, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "format": opened.format,
                "mode": opened.mode, "frames": getattr(opened, "n_frames", 1),
                "decoded": {"width": pil.size[0], "height": pil.size[1],
                            "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()},
                "resized": {"width": int(pixels.shape[-1]), "height": int(pixels.shape[-2])},
                "soft_tokens": soft[0],
                "pixel_values": tensor_record(name, pixels),
            }
            tensors[name] = (rgb, pixels)
    # A prompt with several images carries each image's own pixels, unchanged.
    for key, record in prompt_records.items():
        if len(record["images"]) > 1:
            for i, name in enumerate(record["images"]):
                if record["soft_tokens"][i] != image_records[name]["soft_tokens"]:
                    raise SystemExit(f"{key}: image {i} differs from {name} alone")
    return image_records, prompt_records, tensors


def state_prompt_records(processor, sys_text, images, order):
    """Each of trim_states(), in the given order, as the text of a prompt with TRIM_IMAGE through
    upstream's _inputs: the ids, mm_token_type_ids and soft tokens, recorded as `prompts` are."""
    states = dict(trim_states())
    records = {}
    for name in order:
        prompt = ImagePrompt(sys_text, states[name], [data_url(*images[TRIM_IMAGE])])
        _, kwargs, n = upstream_inputs(processor, prompt)
        ids = [int(t) for t in kwargs["input_ids"].tolist()[0]]
        mm = [int(t) for t in np.array(kwargs["mm_token_type_ids"]).reshape(-1)]
        records[name] = {"images": [TRIM_IMAGE], "system": sys_text, "state": states[name], "ids": ids,
                         "tokens": n, "mm_token_type_ids": mm, "image_runs": runs(mm),
                         "soft_tokens": [end - start for start, end in runs(mm)]}
    return {name: records[name] for name, _ in trim_states()}


def preprocessing_payload(processor, eng, settings, images):
    order = list(PROMPTS)
    a = preprocess_all(processor, eng, settings, images, order)
    b = preprocess_all(processor, eng, settings, images, list(reversed(order)))
    same = a[0] == b[0] and a[1] == b[1]
    ip = processor.image_processor
    tok = processor.tokenizer
    sys_text = a[1]["hotdog"]["system"]
    payload = {
        "generator": generator(),
        "processor": {
            "class": type(processor).__name__, "image_processor": type(ip).__name__,
            "max_soft_tokens": ip.max_soft_tokens, "patch_size": ip.patch_size,
            "pooling_kernel_size": ip.pooling_kernel_size, "rescale_factor": ip.rescale_factor,
            "do_normalize": ip.do_normalize, "size_ignored": ip.size, "resample": int(ip.resample),
            "image_seq_length": processor.image_seq_length,
            "image_token": processor.image_token, "boi_token": processor.boi_token,
            "eoi_token": processor.eoi_token, "image_token_id": processor.image_token_id,
            "boi_token_id": tok.convert_tokens_to_ids(processor.boi_token),
            "eoi_token_id": tok.convert_tokens_to_ids(processor.eoi_token),
        },
        "budget_rule": budget_rule(processor),
        "text_prompt": {"system": sys_text, "state": STATE,
                        "ids": eng.chat_prompt_ids(sys_text, STATE)},
        "images": {name: a[0][name] for name in IMAGES},
        "prompts": {key: a[1][key] for key in PROMPTS},
        "state_prompts": state_prompt_records(processor, sys_text, images, [n for n, _ in trim_states()]),
        "gif_cases": gif_case_records(sys_text),
    }
    same = same and payload["gif_cases"] == gif_case_records(sys_text)
    same = same and payload["state_prompts"] == state_prompt_records(
        processor, sys_text, images, [n for n, _ in reversed(trim_states())])
    return payload, a[2], same


def write_tensors(tensors):
    TENSORS_OUT.mkdir(parents=True, exist_ok=True)
    for name, (rgb, pixels) in tensors.items():
        (TENSORS_OUT / f"{name}.rgb.u8").write_bytes(rgb.tobytes())
        (TENSORS_OUT / f"{name}.pixel_values.f32").write_bytes(pixels.astype("<f4").tobytes())
    print(f"wrote {len(tensors) * 2} tensors to {TENSORS_OUT.relative_to(ROOT)}", file=sys.stderr)


# Reads ---------------------------------------------------------------------------------------

def build_reads(eng, settings, images):
    reads, prompts, inputs = [], {}, {}
    for name, ci in SELECTION:
        if name not in inputs:
            inputs[name] = request_inputs(name, settings, eng, images)
        r = inputs[name]
        canvas_seed = r["seed"] + 7919 * ci
        canvas = eng.build_canvas(r["template"], r["slots"], canvas_seed)
        prompts.setdefault(name, {"system": r["system"], "state": STATE, "images": r["images"]})
        reads.append({
            "id": f"{name}/g0/c{ci}/steps1", "request": name, "group": 0, "format": r["format"],
            "questions": r["questions"], "prompt": name, "seed": r["seed"], "canvas_index": ci,
            "canvas_seed": canvas_seed, "width": len(canvas), "canvas": canvas,
            "slots": [{"pos": s["pos"], "label_ids": list(s["label_ids"])} for s in r["slots"]],
            "steps": 1,
        })
    return reads, prompts, {name: r["prompt"] for name, r in inputs.items()}


def run_pass(rt, reads, image_prompts, settings, order):
    results, timings = {}, {}
    for i in order:
        r = reads[i]
        prompt = image_prompts[r["prompt"]]
        cached = prompt.key in rt.prefills
        started = time.perf_counter()
        tops, n = rt.pool.submit(rt.read, prompt, r["canvas"], r["slots"], settings.mlx_max_prompt,
                                 r["steps"]).result()
        timings[r["id"]] = {"seconds": time.perf_counter() - started, "prefill_cached": cached}
        results[r["id"]] = {
            "prompt_tokens": n,
            "logprobs": [[[int(t), float(v)] for t, v in top.items()] for top in tops],
            "distributions": [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, r["slots"])],
        }
    return results, timings


def reads_payload(args, model_path, eng, settings, images, processor, run):
    reads, prompts, image_prompts = build_reads(eng, settings, images)
    started = time.perf_counter()
    rt = MlxRuntime(model_path)
    run["load_seconds"] = time.perf_counter() - started
    rt.set_cache_limit(args.cache_limit_gb)
    # The runtime's own processor must expand the prompts as the weightless one did.
    for name, prompt in image_prompts.items():
        _, mine, _ = rt.pool.submit(rt._inputs, prompt).result()
        _, theirs, _ = upstream_inputs(processor, prompt)
        if mine["input_ids"].tolist() != theirs["input_ids"].tolist():
            raise SystemExit(f"{name}: the runtime's processor expands the prompt differently")
        prompts[name]["ids"] = [int(t) for t in mine["input_ids"].tolist()[0]]
        prompts[name]["tokens"] = len(prompts[name]["ids"])
    order = list(range(len(reads)))
    results_a, timings_a = run_pass(rt, reads, image_prompts, settings, order)
    rt.init_prefill_cache()
    gc.collect()
    results_b, timings_b = run_pass(rt, reads, image_prompts, settings, list(reversed(order)))
    differing = [rid for rid in results_a if results_a[rid] != results_b[rid]]
    run["reads_differing_between_passes"] = differing
    run["timings"] = [{"id": r["id"], "pass_1": timings_a[r["id"]], "pass_2": timings_b[r["id"]]}
                      for r in reads]
    rt.close()
    payload = {
        "generator": generator(),
        "settings": {"topk": TOPK, "vocab": VOCAB, "canvas": settings.canvas,
                     "canvas_step": settings.canvas_step, "mlx_max_prompt": settings.mlx_max_prompt,
                     "auto_threshold": settings.auto_threshold},
        "requests": {name: {"questions": q, "state": s, "images": n} for name, (q, s, n) in REQUESTS.items()},
        "prompts": prompts,
        "reads": [dict(r, **results_a[r["id"]]) for r in reads],
    }
    return payload, not differing


# Output --------------------------------------------------------------------------------------

def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, payload):
    """One top-level key per line and one entry per line, as the other fixtures."""
    lines = []
    for key, value in payload.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        elif isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            items = ",\n".join(f" {json.dumps(k)}: {dumps(v)}" for k, v in value.items())
            lines.append(f"{json.dumps(key)}: {{\n{items}\n}}")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"wrote {shown}: {len(text.encode('utf-8'))} bytes", file=sys.stderr)


def differences(payload, path):
    """The top-level keys (and for dicts of records, the record keys) where a run differs from
    the committed file. Empty when the run reproduces it."""
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = json.loads(dumps(payload))
    out = {}
    for key in sorted(set(fresh) | set(committed)):
        mine, theirs = fresh.get(key), committed.get(key)
        if mine == theirs:
            continue
        if isinstance(mine, dict) and isinstance(theirs, dict):
            out[key] = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)) or ["order"]
        elif key == "reads" and isinstance(mine, list) and isinstance(theirs, list):
            a, b = {r["id"]: r for r in mine}, {r["id"]: r for r in theirs}
            out[key] = sorted(i for i in set(a) | set(b) if a.get(i) != b.get(i)) or ["order"]
        else:
            out[key] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing them")
    ap.add_argument("--only", choices=["preprocessing"], help="skip the reads, which load the model")
    ap.add_argument("--cache-limit-gb", type=float, default=None,
                    help="MlxRuntime.set_cache_limit before the reads, as OPENJEV_MLX_CACHE_LIMIT_GB does")
    ap.add_argument("--run-out", default=str(RUN_OUT), help="where timings go")
    args = ap.parse_args()

    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    model_path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    settings = Settings()
    eng = Engine(settings, AutoTokenizer.from_pretrained(model_path))
    processor = load_processor(model_path)
    images = image_bytes()
    problems = []

    # The drawn images: written, or compared with the committed files.
    for name, (file, _, draw) in IMAGES.items():
        if draw is None:
            continue
        path = VISION / file
        if args.check:
            if not path.exists() or path.read_bytes() != images[name][0]:
                problems.append(f"Fixtures/vision/{file} differs from the drawn image")
        else:
            VISION.mkdir(parents=True, exist_ok=True)
            path.write_bytes(images[name][0])
            print(f"wrote Fixtures/vision/{file}: {len(images[name][0])} bytes", file=sys.stderr)

    run = {"generator": generator(), "cache_limit_gb": args.cache_limit_gb,
           "machine": {"platform": platform.platform(), "machine": platform.machine(), **device_info()}}
    payload, tensors, same = preprocessing_payload(processor, eng, settings, images)
    run["preprocessing_passes_agree"] = same
    print(f"preprocessing: two passes agree: {same}", file=sys.stderr)
    for key, p in payload["prompts"].items():
        print(f"  {key}: {p['tokens']} tokens, soft tokens {p['soft_tokens']}, {p['pixel_values']}",
              file=sys.stderr)
    if not same:
        problems.append("the two preprocessing passes disagree")
    elif args.check:
        diff = differences(payload, PREPROCESSING_OUT)
        if diff:
            problems.append(f"preprocessing differs from the committed file: {diff}")
    else:
        write_json(PREPROCESSING_OUT, payload)
    if same:
        write_tensors(tensors)

    if args.only != "preprocessing":
        reads, deterministic = reads_payload(args, model_path, eng, settings, images, processor, run)
        run["reads_deterministic"] = deterministic
        print(f"reads: pass 1 == pass 2: {deterministic}", file=sys.stderr)
        if not deterministic:
            problems.append("the two read passes disagree; reads.json was not written")
        elif args.check:
            diff = differences(reads, READS_OUT)
            if diff:
                problems.append(f"reads differ from the committed file: {diff}")
        else:
            write_json(READS_OUT, reads)

    run["problems"] = problems
    write_json(Path(args.run_out), run)
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
