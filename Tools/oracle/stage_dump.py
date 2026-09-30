"""Dump mlx-vlm's encoder intermediates for one oracle prompt (spike #22 bisection).

Wraps the __call__ of mlx-vlm's Attention, MLP, Router, Experts and DecoderLayer so their
outputs are recorded while upstream's own prefill (model.diffusion_prefill_cache) runs, and
saves them to one safetensors file. The Swift transliteration's --stages option records the same
points and reports the first tensor that differs.

    Tools/oracle/.venv/bin/python Tools/oracle/stage_dump.py PROMPT_KEY OUT.safetensors
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Upstream" / "openjev"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import huggingface_hub  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx_vlm.models.diffusion_gemma.language as language  # noqa: E402
from mlx_vlm import load  # noqa: E402

records = {}
calls = {}


def wrap(cls, name):
    original = cls.__call__

    def call(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        layer = getattr(self, "layer_idx", None)
        if layer is None:  # MLP, Router and Experts: the n-th call is layer n's
            layer = calls.get(name, 0)
            calls[name] = layer + 1
        key = f"{name}.{layer}"
        if name in ("router",):
            records[key + ".indices"] = out[0]
            records[key + ".weights"] = out[1]
        else:
            records[key] = out
        return out

    cls.__call__ = call


def main():
    key, out = sys.argv[1], sys.argv[2]
    oracle = json.loads((ROOT / "Fixtures" / "oracle" / "reads.json").read_text())
    ids = oracle["prompts"][key]["ids"]
    path = huggingface_hub.snapshot_download("mlx-community/diffusiongemma-26B-A4B-it-4bit",
                                             revision="a7a81407613811e8ba63af92ac0d852b809e191f",
                                             local_files_only=True)
    model, _ = load(path)
    for cls, name in [(language.Attention, "attn"), (language.MLP, "mlp"), (language.Router, "router"),
                      (language.Experts, "experts"), (language.DecoderLayer, "layer")]:
        wrap(cls, name)
    # Inside attention: the inputs and output of every SDPA call, in layer order.
    sdpa = language.scaled_dot_product_attention

    def recorded_sdpa(queries, keys, values, cache, scale, mask, **kwargs):
        out = sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, **kwargs)
        n = calls.get("sdpa", 0)
        calls["sdpa"] = n + 1
        records[f"sdpa.{n}.queries"], records[f"sdpa.{n}.keys"] = queries, keys
        records[f"sdpa.{n}.values"], records[f"sdpa.{n}.out"] = values, out
        return out

    language.scaled_dot_product_attention = recorded_sdpa

    # Layers 0 (sliding) and 5 (the first full-attention layer): every projection, norm and RoPE
    # in their attention, as a0.* and a5.*.
    import mlx.nn as nn
    from mlx_vlm.models.rope_utils import ProportionalRoPE
    from mlx_vlm.models.gemma4.language import RMSNormNoScale
    inside = [None]
    attention_call = language.Attention.__call__

    def attention(self, *args, **kwargs):
        inside[0] = self.layer_idx if self.layer_idx in (0, 5) else None
        order["n"] = 0
        try:
            return attention_call(self, *args, **kwargs)
        finally:
            inside[0] = None

    language.Attention.__call__ = attention
    order = {"n": 0}

    def trace(cls, name):
        original = cls.__call__

        def call(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            if inside[0] is not None:
                records[f"a{inside[0]}.{order['n']:02d}.{name}"] = out
                order["n"] += 1
            return out

        cls.__call__ = call

    for cls, name in [(nn.QuantizedLinear, "linear"), (nn.RMSNorm, "rmsnorm"), (nn.RoPE, "rope"),
                      (ProportionalRoPE, "rope"), (RMSNormNoScale, "rmsnorm_noscale")]:
        trace(cls, name)
    cache = model.diffusion_prefill_cache(mx.array([ids]))
    records["embeddings"] = model.model.encoder._embed_inputs(mx.array([ids]))
    records["a5.freqs"] = model.model.decoder.layers[5].self_attn.rope._freqs
    for i in (0, len(cache) - 1):
        k, v = language._cache_state(cache[i])
        records[f"cache.{i}.keys"], records[f"cache.{i}.values"] = k, v
    mx.eval(list(records.values()))
    mx.save_safetensors(out, {k: v for k, v in records.items()})
    print(f"saved {len(records)} tensors to {out}")


if __name__ == "__main__":
    main()
