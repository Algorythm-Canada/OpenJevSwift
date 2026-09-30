#!/usr/bin/env python3
"""Record the DiffusionGemma checkpoint's small JSON files as fixtures for OpenJevSwift (issue #23).

The script downloads three files of the pinned checkpoint (mlx-community/diffusiongemma-26B-A4B-it-4bit
at revision a7a81407613811e8ba63af92ac0d852b809e191f) into the Hugging Face cache and writes:

    model/config.json        config.json and generation_config.json verbatim, with the SHA-256 and
                             size of each file read
    model/weight_map.json    model.safetensors.index.json's total size, shard names and weight map

No weights are downloaded, and nothing of upstream OpenJev is involved, so the generator object
records the checkpoint (model_repo, model_revision) and not the upstream commit. Every file starts
with a "generator" object that names this script and its version and the Python and
huggingface_hub versions that wrote it.

Run it from the repository root (make fixtures runs it with the other scripts):

    PYTHONHASHSEED=0 Tools/fixtures/.venv/bin/python Tools/fixtures/checkpoint_tables.py

Running it twice with the same versions gives identical files.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import huggingface_hub  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Fixtures"
SCRIPT = "Tools/fixtures/checkpoint_tables.py"
# Bump when the shape of a file this script writes changes.
GENERATOR_VERSION = 1
MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
FILES = ("config.json", "generation_config.json", "model.safetensors.index.json")


# Output ------------------------------------------------------------------------------------

def generator():
    return {
        "script": SCRIPT,
        "version": GENERATOR_VERSION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "python": sys.version.split()[0],
        "huggingface_hub": huggingface_hub.__version__,
    }


def dumps(value):
    """Strict JSON: non-finite numbers are refused, so every file parses with an RFC 8259 parser."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write(relpath, payload):
    """One top-level key per line and one list entry per line: readable diffs, small files."""
    path = FIXTURES / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in {"generator": generator(), **payload}.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.write_text(text, encoding="utf-8")
    print(f"wrote Fixtures/{relpath}: {len(text.encode('utf-8'))} bytes")


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Tables ------------------------------------------------------------------------------------

def main():
    paths = {name: huggingface_hub.hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION)
             for name in FILES}
    files = {name: {"sha256": file_digest(path), "bytes": os.path.getsize(path)}
             for name, path in paths.items()}
    documents = {}
    for name, path in paths.items():
        with open(path, encoding="utf-8") as f:
            documents[name] = json.load(f)

    write("model/config.json", {
        "files": files,
        "config": documents["config.json"],
        "generation_config": documents["generation_config.json"],
    })

    index = documents["model.safetensors.index.json"]
    weight_map = index["weight_map"]
    write("model/weight_map.json", {
        "total_size": index["metadata"]["total_size"],
        "shards": sorted(set(weight_map.values())),
        "weight_map": weight_map,
    })


if __name__ == "__main__":
    main()
