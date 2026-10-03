# JevK5 tools (issue #55)

The conversion of JevK5 v0.2 to MLX that the `jevk5` backend reads, and the fixture its tests
replay. [docs/10-other-models.md](../../docs/10-other-models.md#jevk5-jevk5-02) describes the
model and what the port does; D-052 in [docs/06-decisions.md](../../docs/06-decisions.md) gives
the choices.

| Path | What it is |
|---|---|
| `convert.py` | Converts the checkpoint with `mlx_lm.convert` (4 or 8 bits, or 16 for the unquantized bfloat16 reference) and checks a folder against the pinned output (`--check`) |
| `reference.py` | Records `Fixtures/jevk5/reads.json` through upstream's `JevK5Engine` and the `jevk5` package, with the 4-bit conversion on MLX in place of vLLM |
| `requirements.txt` | The complete lock of the environment both scripts run in |
| `requirements-jevk5.txt` | The `jevk5` package at its v0.2.2 tag, installed without its dependencies |

## The environment

Outside the repository, as model weights and virtual environments never go in it:

```bash
/usr/local/bin/python3.12 -m venv ~/Library/Caches/OpenJevSwift/jevk5/venv
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python -m pip install -r Tools/jevk5/requirements.txt
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python -m pip install --no-deps -r Tools/jevk5/requirements-jevk5.txt
```

`mlx` and `mlx-metal` are 0.32.2, the MLX that mlx-swift 0.32.2 builds, and `mlx-lm` is 0.32.0,
the newest release that takes them. The `jevk5` package is not on PyPI; only its prompt module,
which needs nothing beyond the standard library, is used.

## The conversion

```bash
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/convert.py --bits 8
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/convert.py --bits 4
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/convert.py --bits 16
```

The 8-bit conversion is the one the server loads by default, since it gives the author's published
top answer on 230 of JevBench's 231 items (D-052); the 4-bit one is half its size, the default of an iOS app; the bfloat16 one keeps
the checkpoint's weights unquantized, a reference that tells the quantization's effect from the
port's in [docs/quality.md](../../docs/quality.md#jevk5), and is not meant for publishing.

The source is `alibiserikbay/JevK5` at the author's `v0.2` tag, commit `ea4804e`, which the
script downloads into the Hugging Face cache on first use (8.4 GB) and checks file by file. Not
`main`: since 2026-09-25 it holds JevK5 v0.3, other weights with another temperature, which
upstream does not serve. The output goes to `~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-8bit`
(or `-4bit`, or `-bf16`), 11 files (12 for bfloat16, whose weights mlx-lm writes in two shards):

| File | Where it comes from |
|---|---|
| `model.safetensors`, `model.safetensors.index.json` | mlx-lm's affine quantization, group size 64: 2.37 GB at 4 bits (4.503 bits per weight), 4.47 GB at 8 bits (8.502); in bfloat16, 8.41 GB in `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` |
| `config.json` | The source's as mlx-lm writes it back: indented its own way, `rope_parameters.rope_type` named `type`, and with its `quantization` entries (none in bfloat16) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `jevk5_config.json` | The source's files, unchanged; `jevk5_config.json` holds the temperature, 1.532 |
| `README.md` | The model card: the attribution, the license and how the files were made |
| `LICENSE`, `NOTICE` | JevK5's own, from github.com/allebee/jevk5 at v0.2.2 |

mlx-lm 0.32.0 has no module for the checkpoint's model type, `qwen3_5_text`, and the checkpoint
names its tensors `model.language_model.*`; the script registers mlx-lm's Qwen3.5 text model under
that type and maps the prefix, so the output keeps the model type and the weight names mlx-swift-lm's
`Qwen35TextModel` loads. A conversion takes 12 to 20 seconds and 6 to 8 GB on an M3 Max. Two
conversions with the pinned versions gave the same bytes, at each of the three sizes, and the
script compares a new one with the digests in its `OUTPUTS`, which `JevK5Checkpoint` in
`Sources/OpenJevLetterReadout` repeats for the two quantized ones (`JevK5CheckpointTests` keeps
them in agreement):

```bash
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/convert.py --check ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-4bit --bits 4
```

`--check` takes a snapshot of a published conversion too, such as the one the server downloads into
the Hugging Face cache. The Hub adds a `.gitattributes` to every repository, which the check leaves
out and says so; the commit pins it, and the server's downloader checks it as it checks every file.

## The fixture

```bash
make upstream
~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python Tools/jevk5/reference.py
```

It needs the 4-bit conversion and takes about seven minutes on an M3 Max. The 26 requests of
`Fixtures/encoders/corpus.json`, with the choices cut there for Verdict restored whole, and two
requests whose states take several prefill chunks, go through upstream's own
`EncoderEngine.build_schema` and `JevK5Engine.read_question`, whose call to vLLM is replaced by
the model on MLX through mlx-lm, read as the Swift backend reads it: 204 questions, 324 passes.
The file keeps every pass's option texts, its prompt (its text when it has at most 6,000
characters, its SHA-256 always), its token count and the SHA-256 of its ids, and the letters'
logits; each question's distribution and token count, and each request's; `spread` over generated
logits in 16 cases (ties at every cut, more than 256 options, the tree method); and the
tokenizer's letter ids and longest entry. The model-free tests replay it bit for bit; the opt-in
live tests (`OPENJEV_JEVK5_MODEL`) hold the Swift tokenizer and model to it.

Nothing here uploads anything. The maintainer published the two quantized conversions on
2026-10-03 as [Algorythm-Canada/jevk5-0.2-mlx-8bit](https://huggingface.co/Algorythm-Canada/jevk5-0.2-mlx-8bit)
and [Algorythm-Canada/jevk5-0.2-mlx-4bit](https://huggingface.co/Algorythm-Canada/jevk5-0.2-mlx-4bit),
byte for byte the pinned output, and `JevK5Checkpoint` pins their commits (D-052).
