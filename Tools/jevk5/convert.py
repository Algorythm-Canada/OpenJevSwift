#!/usr/bin/env python3
"""Convert JevK5 v0.2 to MLX with mlx_lm.convert, reproducing the conversions OpenJevSwift pins.

    PY=~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python
    $PY Tools/jevk5/convert.py --bits 4             # writes ~/Library/Caches/OpenJevSwift/jevk5/jevk5-0.2-mlx-4bit
    $PY Tools/jevk5/convert.py --bits 8             # writes .../jevk5-0.2-mlx-8bit
    $PY Tools/jevk5/convert.py --bits 16            # writes .../jevk5-0.2-mlx-bf16, unquantized
    $PY Tools/jevk5/convert.py --check DIR --bits 4 # checks a directory against the pinned digests

The source is alibiserikbay/JevK5 at the author's `v0.2` tag (SOURCE_REVISION), the weights
upstream OpenJev serves as `jevk5-0.2` and the ones JevK5's published v0.2 JevBench run read. The
repository's `main` has held v0.3 since 2026-09-25 (new weights, temperature 1.22), so the
revision is pinned rather than taken from `main`. The snapshot is read from the Hugging Face cache
(downloaded on first use, 8.4 GB) and every file is checked against SOURCE_FILES before mlx-lm
reads it.

mlx-lm 0.32.0 has no module for the checkpoint's model type, `qwen3_5_text`, the text-only
Qwen3.5 that mlx-swift-lm's `Qwen35TextModel` serves. Its `qwen3_5` module holds that model as
`TextModel` (parameters `model.*`, the head tied to the embeddings), so the script registers it
under `qwen3_5_text` before converting. The checkpoint stores its 426 tensors under
`model.language_model.`, the multimodal model's prefix, although it declares
`Qwen3_5ForCausalLM`; the registered model renames that prefix to `model.` before mlx-lm's own
sanitizing (the conv1d layout and the RMSNorm offset), as transformers' key mapping does. The
output keeps the checkpoint's `config.json` as mlx-lm writes it back (indented its own way, with
`rope_parameters.rope_type` named `type` and the quantization entries added) and the text model's
weight names, which is what mlx-swift-lm loads for that model type. Quantization is
mlx-lm's default affine scheme with a group size of 64, at 4 or 8 bits; `--bits 16` keeps the
checkpoint's bfloat16, a reference for telling quantization from the rest in a comparison, not a
conversion meant for publishing.

After mlx-lm has written the weights and `config.json`, the tokenizer files, the chat template,
`generation_config.json` and `jevk5_config.json` (the calibration temperature, 1.532) are copied
from the source byte for byte over what mlx-lm and transformers wrote, every other file mlx-lm
wrote is removed, and `README.md` (the attribution and how the files were made), `LICENSE` and
`NOTICE` (JevK5's own, from github.com/allebee/jevk5 at v0.2.2) are added. The output is
deterministic: two conversions with the pinned versions gave the same bytes, and OUTPUTS holds
their digests; the script says whether a new conversion matches them.

Nothing is uploaded. Publishing the output is the maintainer's step (docs/06-decisions.md).
Model weights never go into the repository: the default output folder is outside it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import types
import urllib.request
from pathlib import Path

SOURCE_REPO = "alibiserikbay/JevK5"
# The author's `v0.2` tag. Its files are those of the v0.2 commits on `main` (844e4d0 to 27d2d6b);
# the tag adds a README that points at v0.3.
SOURCE_REVISION = "ea4804e93a3db07c2250315c400f59683f54db6f"
# name -> (bytes, SHA-256) of the source files the conversion reads or copies. The two LFS files'
# digests are the Hub's own (its tree lists their SHA-256); the others were computed from the
# snapshot.
SOURCE_FILES = {
    "README.md": (7332, "e726479c05f264df8703b7b27a7a05a256d3ba929a8dd3fe5b9fdeed7b044f1e"),
    "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    "config.json": (1979, "63f47812d0f11118e4d252d2b3ad488707eb9287a11589f4fd382a1d31182724"),
    "generation_config.json": (116, "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73"),
    "jevk5_config.json": (23, "39d6574650b365c77425fc87ffd13ab389ea6508bc1e09ac1289a76faea62419"),
    "model.safetensors": (8411558400, "0fba3bba5d60b95b8de299ded905b01cb166dc33e23b04e4605f919daf2934f1"),
    "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1125, "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"),
}
# Copied into the output unchanged.
COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "jevk5_config.json")
# JevK5's license and notice, from the author's repository at the v0.2.2 tag.
JEVK5_REPO = "allebee/jevk5"
JEVK5_COMMIT = "0571ef373722dd10cace82c81580d7b675fe6b53"
LEGAL_FILES = {
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "NOTICE": (1401, "c6dc9c346b2b516da42b80902916bb6f07b90139d7aa7543420f0674474f31d7"),
}
# The versions the pinned outputs were made with. mlx and mlx-metal are the MLX that mlx-swift
# 0.32.2 builds; the quantized bytes depend on them and on mlx-lm.
PINNED_VERSIONS = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.32.0"}
GROUP_SIZE = 64
# The repositories the conversions are meant to be published as (docs/06-decisions.md).
PUBLISHED_AS = {4: "Algorythm-Canada/jevk5-0.2-mlx-4bit", 8: "Algorythm-Canada/jevk5-0.2-mlx-8bit"}
# bits -> {name: (bytes, SHA-256)} of a conversion made with PINNED_VERSIONS.
OUTPUTS: dict[int, dict[str, tuple[int, str]]] = {
    4: {
        "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        "NOTICE": (1401, "c6dc9c346b2b516da42b80902916bb6f07b90139d7aa7543420f0674474f31d7"),
        "README.md": (2190, "281143de695eff0af6408c8560425cfdd88ae1e24540a5817c650821d2dd860b"),
        "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        "config.json": (2430, "35fb2a84659b33e4d54af1a9d6cc3c179e40c95f3af00b3dc7dc9135d358ca9a"),
        "generation_config.json": (116, "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73"),
        "jevk5_config.json": (23, "39d6574650b365c77425fc87ffd13ab389ea6508bc1e09ac1289a76faea62419"),
        "model.safetensors": (2367223295, "3fd5171001a72b8879422ed2fe944c3bd8b5edf92b393cb2cdd843e4d409f1f8"),
        "model.safetensors.index.json": (67148, "f47a8c0ec8aa8d28fde00373f4f6ada5e203ca47c3d73a3133631983d1c37bfc"),
        "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        "tokenizer_config.json": (1125, "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"),
    },
    8: {
        "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        "NOTICE": (1401, "c6dc9c346b2b516da42b80902916bb6f07b90139d7aa7543420f0674474f31d7"),
        "README.md": (2190, "42cddae7f1c370442433c5a939f8c4a6f776c91de15c1b2986342a3b42f5df07"),
        "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        "config.json": (2430, "8ac5be312381d497966eca8877c0fea5d7f3760461a3713e026d838aef050072"),
        "generation_config.json": (116, "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73"),
        "jevk5_config.json": (23, "39d6574650b365c77425fc87ffd13ab389ea6508bc1e09ac1289a76faea62419"),
        "model.safetensors": (4469618681, "a928abc748a29dd1a853c11d21ae0c0c1fc51f97f11c2a66c43fb0e781285a56"),
        "model.safetensors.index.json": (67148, "7cc82135d26900005b834c016445b3989789174ff1a841214e2fc4f9c588c72a"),
        "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        "tokenizer_config.json": (1125, "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"),
    },
    16: {
        "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
        "NOTICE": (1401, "c6dc9c346b2b516da42b80902916bb6f07b90139d7aa7543420f0674474f31d7"),
        "README.md": (2121, "83edb60bdd0f4fe235c19da871c92974b576deda22d43053774157dbe48802ff"),
        "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
        "config.json": (2225, "d7d5cfcd5e6137a3efd09df123510d4429e3ffb26e39259da83fb66633d6483d"),
        "generation_config.json": (116, "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73"),
        "jevk5_config.json": (23, "39d6574650b365c77425fc87ffd13ab389ea6508bc1e09ac1289a76faea62419"),
        "model-00001-of-00002.safetensors": (5356480151, "00c61808fbf45a376d084d3fc01b3090af173ded162289abc5f58fdf68b82a8b"),
        "model-00002-of-00002.safetensors": (3055071588, "52c27623938c41ba411e866925f128bd67c80e450de929bbb22858df7aa799c4"),
        "model.safetensors.index.json": (37322, "8fa5319cd20676ac7ae8c7c66c3a1873ee2c3f6f0de28e832725c9741c7cb5f9"),
        "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        "tokenizer_config.json": (1125, "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"),
    },
}
DEFAULT_ROOT = Path.home() / "Library" / "Caches" / "OpenJevSwift" / "jevk5"
# MLX keeps freed buffers for reuse up to its memory limit; a cap keeps the pool small, as the
# OpenJevSwift MLX runs do (OPENJEV_MLX_CACHE_LIMIT_GB=4).
CACHE_LIMIT_BYTES = 4 << 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def listing(folder: Path) -> dict[str, tuple[int, str]]:
    """{name: (bytes, SHA-256)} of every file in `folder`, which must hold no subfolder."""
    found = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            sys.exit(f"{path} is not a file; a conversion holds files only")
        found[path.name] = (path.stat().st_size, sha256_file(path))
    return found


def check_versions(allow_other: bool) -> dict[str, str]:
    found = {}
    for name, pinned in PINNED_VERSIONS.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            sys.exit(f"{name} is not installed; install Tools/jevk5/requirements.txt")
    differ = {name: version for name, version in found.items() if version != PINNED_VERSIONS[name]}
    if differ and not allow_other:
        sys.exit(f"pinned {PINNED_VERSIONS}, found {found}; install Tools/jevk5/requirements.txt, "
                 "or pass --allow-other-versions (the output will not match OUTPUTS)")
    return found


def source_snapshot(explicit: str | None) -> Path:
    """The source folder: `--source`, or the pinned revision from the Hugging Face cache, which is
    downloaded only when the cache lacks it."""
    if explicit:
        return Path(explicit).expanduser()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        print(f"downloading {SOURCE_REPO} at {SOURCE_REVISION} (8.4 GB) into the Hugging Face cache",
              flush=True)
        return Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION))


def verify_source(folder: Path) -> None:
    for name, (size, digest) in SOURCE_FILES.items():
        path = folder / name
        if not path.is_file():
            sys.exit(f"{path} is missing")
        if path.stat().st_size != size or sha256_file(path) != digest:
            sys.exit(f"{path} is not the pinned file ({size} bytes, SHA-256 {digest})")
    print(f"source {folder}: {len(SOURCE_FILES)} files match the pinned digests", flush=True)


def legal_files(cache: Path) -> dict[str, Path]:
    """LICENSE and NOTICE from the author's repository, downloaded once and checked."""
    paths = {}
    for name, (size, digest) in LEGAL_FILES.items():
        path = cache / f"jevk5-{JEVK5_COMMIT[:7]}-{name}"
        if not (path.is_file() and path.stat().st_size == size and sha256_file(path) == digest):
            url = f"https://raw.githubusercontent.com/{JEVK5_REPO}/{JEVK5_COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                sys.exit(f"{url} is not the pinned file ({size} bytes, SHA-256 {digest})")
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        paths[name] = path
    return paths


# The checkpoint's prefix for the text model's parameters.
CHECKPOINT_PREFIX = "model.language_model."


def register_text_model() -> None:
    """Serve mlx-lm's Qwen3.5 text model under the checkpoint's model type, `qwen3_5_text`, with
    the checkpoint's `model.language_model.` prefix read as the text model's `model.`."""
    import mlx_lm.models.qwen3_5 as qwen3_5

    class TextModel(qwen3_5.TextModel):
        def sanitize(self, weights):
            renamed = {("model." + key[len(CHECKPOINT_PREFIX):]
                        if key.startswith(CHECKPOINT_PREFIX) else key): value
                       for key, value in weights.items()}
            return super().sanitize(renamed)

    module = types.ModuleType("mlx_lm.models.qwen3_5_text")
    module.Model = TextModel
    module.ModelArgs = qwen3_5.TextModelArgs
    sys.modules["mlx_lm.models.qwen3_5_text"] = module


def folder_name(bits: int) -> str:
    return "jevk5-0.2-mlx-bf16" if bits == 16 else f"jevk5-0.2-mlx-{bits}bit"


def readme(bits: int, versions: dict[str, str]) -> str:
    """The model card of the converted repository: the attribution, the license and how the files
    were made. Deterministic, so the output's digests are."""
    relation = "" if bits == 16 else "base_model_relation: quantized\n"
    if bits == 16:
        title, form = "bfloat16", "without quantization, in bfloat16"
        weights = (f"- The weights were converted to MLX unquantized by mlx-lm {versions['mlx-lm']}"
                   f" with\n  MLX {versions['mlx']}. `config.json` is the source's as mlx-lm writes"
                   " it back, with\n  `rope_parameters.rope_type` named `type`.")
    else:
        title, form = f"{bits}-bit", f"with {bits}-bit affine quantization (group\nsize {GROUP_SIZE})"
        weights = (f"- The weights were quantized to {bits} bits by mlx-lm {versions['mlx-lm']} with"
                   f" MLX\n  {versions['mlx']}. `config.json` is the source's as mlx-lm writes it "
                   "back, with\n  `rope_parameters.rope_type` named `type` and the `quantization` "
                   "entries added.")
    return f"""---
license: apache-2.0
base_model: {SOURCE_REPO}
{relation}library_name: mlx
pipeline_tag: text-generation
language:
- en
tags:
- mlx
- jevk5
- jev
- system-one
- typed-decisions
---

# JevK5 v0.2, MLX {title}

[JevK5](https://huggingface.co/{SOURCE_REPO}) v0.2 is Alibi Serikbay's model
([github.com/{JEVK5_REPO}](https://github.com/{JEVK5_REPO})); the credit is theirs. It is
Qwen3.5-4B with a LoRA distilled from Qwen3.6-27B, merged into the weights, and it answers a
typed decision with a softmax over its answer letters' next-token logits under one calibration
temperature, the readout of SemIf (github.com/TheoLeeCJ/SemIf).

This repository holds the same model converted to MLX {form}, for the `jevk5` backend of
[OpenJevSwift](https://github.com/Algorythm-Canada/OpenJevSwift), which serves it as `jevk5-0.2`
on Apple silicon. It is not the author's release, and it is not affiliated with TypeSafe AI.

## Source and changes

- Source: `{SOURCE_REPO}` at its `v0.2` tag, commit `{SOURCE_REVISION}` (`model.safetensors`
  SHA-256 `{SOURCE_FILES["model.safetensors"][1]}`).
{weights}
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` and
  `jevk5_config.json` (temperature 1.532) are the source's files, unchanged.
- `Tools/jevk5/convert.py` in OpenJevSwift reproduces every file of this repository byte for byte.

## Use

With OpenJevSwift: `OPENJEV_BACKEND=jevk5 openjev serve`, with `OPENJEV_JEVK5_MODEL` naming this
repository or a local copy. Read the model the way JevK5's own runtime does
(github.com/{JEVK5_REPO}, `jevk5/prompt.py`): the prompt is fixed and the answer is read from the
letter logits, never generated.

## License

Apache-2.0, as the source. `LICENSE` and `NOTICE` are JevK5's own, from github.com/{JEVK5_REPO} at
its v0.2.2 tag. Qwen3.5-4B is the Qwen team's, under Apache-2.0.
"""


def convert(bits: int, out: Path, source: Path, versions: dict[str, str], cache: Path) -> None:
    import mlx.core as mx
    from mlx_lm.convert import convert as mlx_convert

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    register_text_model()
    if out.exists():
        sys.exit(f"{out} exists; remove it or pass another --out")
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    legal = legal_files(cache)
    if bits == 16:
        mlx_convert(hf_path=str(source), mlx_path=str(staging))
    else:
        mlx_convert(hf_path=str(source), mlx_path=str(staging), quantize=True, q_bits=bits,
                    q_group_size=GROUP_SIZE, q_mode="affine")
    # What mlx-lm wrote beyond the weights and config.json (transformers' tokenizer files and a
    # template model card) goes; the source's own files replace it.
    for path in staging.iterdir():
        weights = path.name.startswith("model") and path.name.endswith((".safetensors", ".json"))
        if path.name != "config.json" and not weights:
            path.unlink()
    for name in COPIED:
        shutil.copyfile(source / name, staging / name)
    for name, path in legal.items():
        shutil.copyfile(path, staging / name)
    (staging / "README.md").write_text(readme(bits, versions), encoding="utf-8")
    staging.rename(out)


def report(folder: Path, bits: int) -> int:
    found = listing(folder)
    print(json.dumps({"folder": str(folder), "bits": bits,
                      "files": {name: {"bytes": size, "sha256": digest}
                                for name, (size, digest) in found.items()}}, indent=1))
    pinned = OUTPUTS.get(bits)
    if pinned is None:
        print(f"no pinned {bits}-bit conversion to compare with")
        return 0
    if found == pinned:
        print(f"{folder} is the pinned {bits}-bit conversion ({len(found)} files)")
        return 0
    for name in sorted(set(found) | set(pinned)):
        if found.get(name) != pinned.get(name):
            print(f"  {name}: found {found.get(name)}, pinned {pinned.get(name)}")
    print(f"{folder} differs from the pinned {bits}-bit conversion")
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), required=True,
                        help="4 or 8 for a quantized conversion, 16 for the unquantized reference")
    parser.add_argument("--out", help="the output folder (default: jevk5-0.2-mlx-4bit, -8bit or "
                                      f"-bf16 in {DEFAULT_ROOT})")
    parser.add_argument("--source", help="a local copy of the source revision instead of the "
                                         "Hugging Face cache; it is checked all the same")
    parser.add_argument("--check", metavar="DIR",
                        help="check DIR against the pinned conversion and convert nothing")
    parser.add_argument("--allow-other-versions", action="store_true",
                        help="convert with versions other than the pinned ones")
    args = parser.parse_args(argv)
    if args.check:
        return report(Path(args.check).expanduser(), args.bits)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    versions = check_versions(args.allow_other_versions)
    out = Path(args.out).expanduser() if args.out else DEFAULT_ROOT / folder_name(args.bits)
    source = source_snapshot(args.source)
    verify_source(source)
    convert(args.bits, out, source, versions, DEFAULT_ROOT / "downloads")
    return report(out, args.bits)


if __name__ == "__main__":
    sys.exit(main())
