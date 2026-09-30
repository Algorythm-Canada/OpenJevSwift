"""Run mlx-vlm's decoder on the Layr-Labs fork's encoder output (spike #22).

The Swift probe (`Probe --dump DIR --dump-reads ...`) writes each prompt's request cache, every
layer in temporal order, and its own slot maps. This script loads those caches into mlx-vlm's
KVCache and RotatingKVCache objects and runs upstream's read loop (mlx_backend.py:185-208) on
them. Comparing the three answers separates the two halves of the pass:

- mlx-vlm decoder on the fork's cache against the fork (Swift decoder, same cache): the decoders.
- mlx-vlm decoder on the fork's cache against the oracle (mlx-vlm decoder, mlx-vlm cache): the
  encoders.

A control does the same round trip with mlx-vlm's own cache, cut to the last 1,024 sliding
positions as the fork's ring keeps them. It must reproduce the oracle bit for bit.

    PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/oracle/crossfeed.py DIR
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Upstream" / "openjev"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import huggingface_hub  # noqa: E402
import mlx.core as mx  # noqa: E402

from mlx_vlm.models.diffusion_gemma.language import _cache_state  # noqa: E402
from openjev.engine import TOPK, slot_distribution  # noqa: E402
from openjev.mlx_backend import MlxRuntime  # noqa: E402

MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"


def build_cache(model, layers, prompt_tokens):
    """mlx-vlm cache objects holding the given (keys, values) per layer. A sliding row holds at
    most 1,024 positions; its offset is set to the prompt length so the canvas RoPE positions
    and mlx-vlm's window slicing see the real prompt."""
    cache = model.make_cache()
    for c, (keys, values) in zip(cache, layers):
        c.update_and_fetch(keys, values)
        c.offset = prompt_tokens
    return cache


def decode(model, cache, canvas, slots, steps):
    """mlx_backend.py:187-208 on a given cache."""
    x = mx.array([canvas])
    masks = model.diffusion_decoder_masks(x, cache, None)
    pos = mx.array([s["pos"] for s in slots])
    sc, sc_ctx, written = None, None, []
    for step in range(steps):
        logits = model.diffusion_decoder_logits(x, cache=cache, self_conditioning=sc, decoder_attention_mask=masks)
        if step + 1 == steps:
            break
        x[0, pos] = mx.argmax(logits[0, pos], axis=-1).astype(x.dtype)
        if sc_ctx is None:
            sc_ctx = model.diffusion_prepare_self_conditioning()
        sc = model.diffusion_self_conditioning(logits, sc_ctx)
        mx.eval(x, sc)
        written.append([x[0, s["pos"]].item() for s in slots])
    out = []
    for s in slots:
        row = logits[0, s["pos"]].astype(mx.float32)
        lp = row - mx.logsumexp(row)
        keep = sorted(set(mx.argpartition(-lp, TOPK)[:TOPK].tolist()) | set(s["label_ids"]))
        out.append([[int(k), float(v)] for k, v in zip(keep, lp[mx.array(keep)].tolist())])
    return out, written


def compare(got, want, slots):
    """max |dp| over labels, top-label agreement, max |dH|, max label |dlogprob|."""
    max_dp, agree, max_dh, max_dlp = 0.0, 0, 0.0, 0.0
    for a_pairs, b_pairs, slot in zip(got, want, slots):
        a = {int(t): v for t, v in a_pairs}
        b = {int(t): v for t, v in b_pairs}
        da, db = slot_distribution(a, slot["label_ids"]), slot_distribution(b, slot["label_ids"])
        max_dp = max(max_dp, max(abs(x - y) for x, y in zip(da["probs"], db["probs"])))
        agree += max(range(len(da["probs"])), key=da["probs"].__getitem__) == \
            max(range(len(db["probs"])), key=db["probs"].__getitem__)
        max_dh = max(max_dh, abs(da["entropy"] - db["entropy"]))
        max_dlp = max(max_dlp, max(abs(a[i] - b[i]) for i in slot["label_ids"]))
    return {"max_probability_difference": max_dp, "top_label_agreement": agree, "slots": len(slots),
            "max_entropy_difference": max_dh, "max_label_logprob_difference": max_dlp}


def main():
    directory = Path(sys.argv[1])
    swift = json.loads((directory / "swift_reads.json").read_text())
    oracle = json.loads((ROOT / "Fixtures" / "oracle" / "reads.json").read_text())
    reads = [r for r in oracle["reads"] if r["id"] in swift]
    path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    rt = MlxRuntime(path)
    model = rt.model
    window = model.config.text_config.sliding_window
    rows = []

    def one(r):
        prompt = oracle["prompts"][r["prompt"]]
        n = len(prompt["ids"])
        arrays = mx.load(str(directory / (r["prompt"].replace("/", "_") + ".safetensors")))
        fork_layers = [(arrays[f"layers.{i}.keys"], arrays[f"layers.{i}.values"])
                       for i in range(len(model.config.text_config.layer_types))]
        on_fork, written_fork = decode(model, build_cache(model, fork_layers, n), r["canvas"], r["slots"], r["steps"])
        # control: mlx-vlm's own prefill, cut the way the fork's ring cuts it
        own = model.diffusion_prefill_cache(mx.array([prompt["ids"]]))
        own_layers = []
        for c, kind in zip(own, model.config.text_config.layer_types):
            keys, values = _cache_state(c)
            if kind == "sliding_attention" and keys.shape[2] > window:
                keys, values = keys[:, :, -window:, :], values[:, :, -window:, :]
            own_layers.append((keys, values))
        control, written_control = decode(model, build_cache(model, own_layers, n), r["canvas"], r["slots"], r["steps"])
        return on_fork, written_fork, control, written_control

    for r in reads:
        on_fork, written_fork, control, written_control = rt.pool.submit(one, r).result()
        row = {
            "id": r["id"],
            "control_bit_identical_to_oracle": control == r["logprobs"] and written_control == r["written"],
            "mlxvlm_decoder_on_fork_cache_vs_fork": compare(on_fork, swift[r["id"]]["logprobs"], r["slots"]),
            "mlxvlm_decoder_on_fork_cache_vs_oracle": compare(on_fork, r["logprobs"], r["slots"]),
            "fork_vs_oracle": compare(swift[r["id"]]["logprobs"], r["logprobs"], r["slots"]),
            "written_on_fork_cache": written_fork, "written_fork": swift[r["id"]]["written"],
            "written_oracle": r["written"],
        }
        rows.append(row)
        d, e, f = (row["mlxvlm_decoder_on_fork_cache_vs_fork"], row["mlxvlm_decoder_on_fork_cache_vs_oracle"],
                   row["fork_vs_oracle"])
        print(f"{r['id']:34s} control exact {row['control_bit_identical_to_oracle']!s:5s} | decoders |dp| "
              f"{d['max_probability_difference']:.2e} top {d['top_label_agreement']}/{d['slots']} | encoders |dp| "
              f"{e['max_probability_difference']:.2e} top {e['top_label_agreement']}/{e['slots']} | fork vs oracle "
              f"{f['max_probability_difference']:.2e}", flush=True)

    def worst(key, field):
        return max(row[key][field] for row in rows)

    summary = {
        "reads": len(rows),
        "control_bit_identical": sum(row["control_bit_identical_to_oracle"] for row in rows),
        "decoders": {"max_probability_difference": worst("mlxvlm_decoder_on_fork_cache_vs_fork", "max_probability_difference"),
                     "top_label_agreement": sum(row["mlxvlm_decoder_on_fork_cache_vs_fork"]["top_label_agreement"] for row in rows),
                     "max_label_logprob_difference": worst("mlxvlm_decoder_on_fork_cache_vs_fork", "max_label_logprob_difference")},
        "encoders": {"max_probability_difference": worst("mlxvlm_decoder_on_fork_cache_vs_oracle", "max_probability_difference"),
                     "top_label_agreement": sum(row["mlxvlm_decoder_on_fork_cache_vs_oracle"]["top_label_agreement"] for row in rows),
                     "max_label_logprob_difference": worst("mlxvlm_decoder_on_fork_cache_vs_oracle", "max_label_logprob_difference")},
        "slots": sum(row["fork_vs_oracle"]["slots"] for row in rows),
    }
    print(json.dumps(summary, indent=1))
    out = ROOT / "Tools" / "oracle" / "results" / "crossfeed.json"
    out.write_text(json.dumps({"summary": summary, "reads": rows}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    rt.close()


if __name__ == "__main__":
    main()
