# Checkpoint fixtures

The small JSON files of the pinned checkpoint (`mlx-community/diffusiongemma-26B-A4B-it-4bit` at
revision `a7a81407613811e8ba63af92ac0d852b809e191f`), copied verbatim. No weights. The checkpoint's
own files cannot be committed as they are, because every fixture starts with a `generator`
object, so each is wrapped. `Tools/fixtures/checkpoint_tables.py` writes these files; see
[../README.md](../README.md) for regenerating them.

## Files

| File | Contents |
|---|---|
| `config.json` | `files`: SHA-256 and size of `config.json`, `generation_config.json` and `model.safetensors.index.json`. `config`: the checkpoint's `config.json` object. `generation_config`: the `generation_config.json` object. |
| `weight_map.json` | `total_size` (16,542,844,632 bytes), `shards` (the four shard names, sorted) and `weight_map` (tensor name to shard, 1,647 tensors) from `model.safetensors.index.json` |

`generator` holds `script`, `version`, `model_repo`, `model_revision`, `python` and
`huggingface_hub`. The configuration tests (issue #23) decode `config` and `generation_config`
with `DiffusionGemmaConfiguration`; the weight coverage test (issue #27) reads `weight_map.json`.

What `config` holds that the Swift type does not read: `quantization_config`, a duplicate of
`quantization` that mlx-vlm also ignores, and `transformers_version`.
