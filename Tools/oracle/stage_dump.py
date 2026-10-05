"""Dump mlx-vlm's encoder intermediates for one oracle prompt (spike #22 bisection).

Wraps the __call__ of mlx-vlm's Attention, MLP, Router, Experts and DecoderLayer so their
outputs are recorded while upstream's own prefill (model.diffusion_prefill_cache) runs, and
saves them to one safetensors file. The Swift transliteration's --stages option records the same
points and reports the first tensor that differs.

    Tools/oracle/.venv/bin/python Tools/oracle/stage_dump.py PROMPT_KEY OUT.safetensors

With --image, PROMPT_KEY is a prompt of Fixtures/vision/reads.json (hotdog or readme_hotdog),
built through upstream's own ImagePrompt and MlxRuntime._inputs, and the dump also holds the
vision tower's stages (#47): pixel_values, vision.patches (the patch embedder's output),
vision.layer.N (each block's output), vision.pooled (the pooler's soft tokens), vision.out (the
tower's output), image_features (embed_vision's), inputs_embeds (after masked_scatter), the
encoder masks of layers 0 and 5 (mask.0, mask.5), and layer.N for every encoder layer.
OpenJevDiffusionGemmaTests' ImageStageTests compares the port with it.

    Tools/oracle/.venv/bin/python Tools/oracle/stage_dump.py --cache-limit-gb 4 --image hotdog \
        Tools/oracle/results/vision/hotdog.stages.safetensors

The first vision block is also recorded one operation at a time with mlx-vlm's own modules
(b0.normed, b0.q, b0.q_norm, b0.rope_timescale, b0.q_rope, b0.sdpa, b0.o_proj, b0.mlp and so on).
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


def image_main(key, out):
    import base64

    from mlx_vlm.models.gemma4 import gemma4, vision
    from openjev.mlx_backend import ImagePrompt, MlxRuntime

    fixture = json.loads((ROOT / "Fixtures" / "vision" / "reads.json").read_text())
    prompt = fixture["prompts"][key]
    path = huggingface_hub.snapshot_download("mlx-community/diffusiongemma-26B-A4B-it-4bit",
                                             revision="a7a81407613811e8ba63af92ac0d852b809e191f",
                                             local_files_only=True)
    model, processor = load(path)
    # A stand-in runtime: _inputs only needs mx and the processor, as vision_oracle.py does.
    runtime = MlxRuntime.__new__(MlxRuntime)
    runtime.mx, runtime.model, runtime.processor = mx, model, processor
    data = (ROOT / "Upstream" / "openjev" / "tests" / "data" / "hotdog.jpg").read_bytes()
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    _, kwargs, n = runtime._inputs(ImagePrompt(prompt["system"], prompt["state"], [url]))
    assert kwargs["input_ids"][0].tolist() == prompt["ids"], "the expanded ids differ from the fixture"

    def record(cls, name, pick=lambda out: out, counted=False):
        original = cls.__call__

        def call(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            if counted:
                index = calls.get(name, 0)
                calls[name] = index + 1
                records[f"{name}.{index}"] = pick(out)
            else:
                records[name] = pick(out)
            return out

        cls.__call__ = call

    block_call = vision.VisionTransformerBlock.__call__

    def block(self, x, positions, mask=None):
        # The first block, one operation at a time with mlx-vlm's own modules, as b0.*.
        if "b0.out" not in records:
            from mlx_vlm.models.base import ensure_fused_sdpa
            a = self.self_attn
            B, L, _ = x.shape
            r = records
            r["b0.positions"], r["b0.mask"] = positions, mask
            r["b0.normed"] = normed = self.input_layernorm(x)
            r["b0.q"] = q = a.q_proj(normed).reshape(B, L, a.num_heads, a.head_dim)
            r["b0.k"] = k = a.k_proj(normed).reshape(B, L, a.num_kv_heads, a.head_dim)
            r["b0.v"] = v = a.v_proj(normed).reshape(B, L, a.num_kv_heads, a.head_dim)
            r["b0.q_norm"], r["b0.k_norm"], r["b0.v_norm"] = q, k, v = a.q_norm(q), a.k_norm(k), a._v_norm(v)
            half = a.head_dim // 4
            r["b0.rope_timescale"] = mx.power(
                a.rope_base_frequency, (2.0 / (2 * half)) * mx.arange(0, half).astype(mx.float32))
            r["b0.q_rope"] = q = vision.apply_multidimensional_rope(q, positions, a.rope_base_frequency)
            r["b0.k_rope"] = k = vision.apply_multidimensional_rope(k, positions, a.rope_base_frequency)
            q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
            r["b0.sdpa"] = o = ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask)
            r["b0.o_proj"] = o = a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
            r["b0.post_attention"] = o = self.post_attention_layernorm(o)
            r["b0.h"] = h = x + o
            r["b0.pre_feedforward"] = n = self.pre_feedforward_layernorm(h)
            r["b0.gate"], r["b0.up"] = self.mlp.gate_proj(n), self.mlp.up_proj(n)
            r["b0.mlp"] = m = self.mlp(n)
            r["b0.post_feedforward"] = m = self.post_feedforward_layernorm(m)
            r["b0.out"] = h + m
        return block_call(self, x, positions, mask)

    vision.VisionTransformerBlock.__call__ = block
    record(vision.VisionPatchEmbedder, "vision.patches")
    record(vision.VisionTransformerBlock, "vision.layer", counted=True)
    record(vision.VisionPooler, "vision.pooled", pick=lambda out: out[0])
    record(vision.VisionModel, "vision.out")
    record(gemma4.MultimodalEmbedder, "image_features")
    wrap(language.DecoderLayer, "layer")
    embed = language.EncoderModel._embed_inputs

    def embed_inputs(self, *args, **kwargs):
        out = embed(self, *args, **kwargs)
        records["inputs_embeds"] = out
        return out

    language.EncoderModel._embed_inputs = embed_inputs
    make_masks = language.EncoderModel._make_encoder_masks

    def masks(self, *args, **kwargs):
        out = make_masks(self, *args, **kwargs)
        records["mask.0"], records["mask.5"] = out[0], out[5]
        return out

    language.EncoderModel._make_encoder_masks = masks
    records["pixel_values"] = kwargs["pixel_values"]
    cache = model.diffusion_prefill_cache(**kwargs)
    for i in (0, len(cache) - 1):
        k, v = language._cache_state(cache[i])
        records[f"cache.{i}.keys"], records[f"cache.{i}.values"] = k, v
    mx.eval(list(records.values()))
    mx.save_safetensors(out, {k: v for k, v in records.items()})
    print(f"{key}: {n} prompt tokens; saved {len(records)} tensors to {out}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cache-limit-gb":
        # As the oracle's other scripts: cap MLX's buffer pool for a long run.
        mx.set_cache_limit(int(float(sys.argv[2]) * 1024 ** 3))
        del sys.argv[1:3]
    if sys.argv[1] == "--image":
        image_main(sys.argv[2], sys.argv[3])
    else:
        main()
