"""How far upstream's own read moves under changes that leave the mathematics alone (spike #22).

Each variant repeats MlxRuntime.read (Upstream/openjev/openjev/mlx_backend.py:175-208) on the
inputs in Fixtures/oracle/reads.json, with one change that is exact in real arithmetic but not
in bfloat16:

- baseline: no change. Must reproduce reads.json bit for bit.
- chunked_prefill: the prompt prefilled in chunks of 64 tokens (mlx-vlm's own chunk_prefill path).
- unsorted_decoder_experts: the decoder's expert gathers without mlx-vlm's sort (the path it
  takes below 64 assignments), the encoder unchanged.
- explicit_masks: the decoder given all-true boolean masks where mlx-vlm passes None.

Each variant is compared with the oracle using the Swift probe's metrics, and the result goes to
Tools/oracle/results/sensitivity.json, or --out. --cache-limit-gb caps MLX's buffer pool as
OPENJEV_MLX_CACHE_LIMIT_GB does, which leaves every read bit-identical (spike #22). Run from the
repository root:

    PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/oracle/sensitivity.py [--cache-limit-gb 4] [--out PATH]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Upstream" / "openjev"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import huggingface_hub  # noqa: E402
import mlx.core as mx  # noqa: E402

import mlx_vlm.models.diffusion_gemma.language as language  # noqa: E402
from openjev.engine import TOPK, slot_distribution  # noqa: E402
from openjev.mlx_backend import MlxRuntime  # noqa: E402

MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
CHUNK = 64


def unsorted_experts_call(self, x, top_k_indices, top_k_weights):
    """language.Experts.__call__ with do_sort forced off."""
    x = mx.expand_dims(x, (-2, -3))
    gate_up = self.gate_up_proj(x, top_k_indices, sorted_indices=False)
    gate = gate_up[..., : self.hidden_dims]
    up = gate_up[..., self.hidden_dims:]
    y = self.down_proj(language.geglu(gate, up), top_k_indices, sorted_indices=False)
    y = y.squeeze(-2)
    return (y * top_k_weights[..., None]).sum(axis=-2)


def read(rt, ids, canvas, slots, steps, variant):
    """MlxRuntime.read without its prefill cache, with one variant applied."""
    model = rt.model
    input_ids = mx.array([ids])
    if variant == "chunked_prefill":
        cache = model.diffusion_prefill_cache(input_ids, chunk_prefill=True, prefill_step_size=CHUNK)
    else:
        cache = model.diffusion_prefill_cache(input_ids)
    x = mx.array([canvas])
    masks = model.diffusion_decoder_masks(x, cache, None)
    if variant == "explicit_masks":
        width = x.shape[1]
        filled = {}
        for layer_type, mask in masks.items():
            if mask is None:
                layer = next(i for i, t in enumerate(model.config.text_config.layer_types) if t == layer_type)
                state = language._cache_state(cache[layer])
                key_len = state[0].shape[2] + width
                mask = mx.ones((1, 1, width, key_len), dtype=mx.bool_)
            filled[layer_type] = mask
        masks = filled
    saved = language.Experts.__call__
    if variant == "unsorted_decoder_experts":
        mx.eval([c.state for c in cache])  # the prefill runs with the sorted path
        language.Experts.__call__ = unsorted_experts_call
    try:
        pos = mx.array([s["pos"] for s in slots])
        sc, sc_ctx = None, None
        written = []
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
            out.append(dict(zip(keep, lp[mx.array(keep)].tolist())))
        mx.eval(logits)
    finally:
        language.Experts.__call__ = saved
    return out, written


def compare(tops, oracle_read):
    """The probe's metrics: max |dp| over labels, top-label agreement, max |dH|; per slot too."""
    max_dp, agree, max_dh, max_dlp = 0.0, 0, 0.0, 0.0
    per_slot = []
    for top, slot, want, pairs in zip(tops, oracle_read["slots"], oracle_read["distributions"],
                                      oracle_read["logprobs"]):
        got = slot_distribution(top, slot["label_ids"])
        dp = max(abs(a - b) for a, b in zip(got["probs"], want["probs"]))
        same = max(range(len(got["probs"])), key=got["probs"].__getitem__) == \
            max(range(len(want["probs"])), key=want["probs"].__getitem__)
        dh = abs(got["entropy"] - want["entropy"])
        max_dp, agree, max_dh = max(max_dp, dp), agree + same, max(max_dh, dh)
        oracle_lp = {int(t): v for t, v in pairs}
        max_dlp = max(max_dlp, max(abs(top[i] - oracle_lp[i]) for i in slot["label_ids"]))
        per_slot.append({"probs": got["probs"], "entropy": got["entropy"], "max_probability_difference": dp,
                         "top_label_agrees": same, "entropy_difference": dh})
    return {"max_probability_difference": max_dp, "top_label_agreement": agree,
            "slots": len(oracle_read["slots"]), "max_entropy_difference": max_dh,
            "max_label_logprob_difference": max_dlp, "per_slot": per_slot}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-limit-gb", type=float, default=None,
                    help="MlxRuntime.set_cache_limit before the reads, as OPENJEV_MLX_CACHE_LIMIT_GB does")
    ap.add_argument("--out", default=str(ROOT / "Tools" / "oracle" / "results" / "sensitivity.json"))
    args = ap.parse_args()
    oracle = json.loads((ROOT / "Fixtures" / "oracle" / "reads.json").read_text())
    path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    rt = MlxRuntime(path)
    rt.set_cache_limit(args.cache_limit_gb)
    variants = ["baseline", "chunked_prefill", "unsorted_decoder_experts", "explicit_masks"]
    results = {}
    for variant in variants:
        started = time.perf_counter()
        rows = []
        for r in oracle["reads"]:
            ids = oracle["prompts"][r["prompt"]]["ids"]
            tops, written = rt.pool.submit(read, rt, ids, r["canvas"], r["slots"], r["steps"], variant).result()
            row = {"id": r["id"], **compare(tops, r), "written_equal": written == r["written"],
                   "bit_identical": [[[int(t), float(v)] for t, v in top.items()] for top in tops] == r["logprobs"]}
            rows.append(row)
        overall = {
            "max_probability_difference": max(x["max_probability_difference"] for x in rows),
            "top_label_agreement": sum(x["top_label_agreement"] for x in rows),
            "slots": sum(x["slots"] for x in rows),
            "max_entropy_difference": max(x["max_entropy_difference"] for x in rows),
            "max_label_logprob_difference": max(x["max_label_logprob_difference"] for x in rows),
            "reads_bit_identical": sum(x["bit_identical"] for x in rows),
            "reads": len(rows),
            "written_mismatches": [x["id"] for x in rows if not x["written_equal"]],
            "seconds": time.perf_counter() - started,
        }
        if overall["reads_bit_identical"] == overall["reads"]:
            # A variant that reproduces the oracle exactly adds nothing per slot.
            for row in rows:
                row.pop("per_slot", None)
        results[variant] = {"overall": overall, "reads": rows}
        print(f"{variant:26s} max|dp| {overall['max_probability_difference']:.3e}  top "
              f"{overall['top_label_agreement']}/{overall['slots']}  max|dH| {overall['max_entropy_difference']:.3e}  "
              f"max|dlp| {overall['max_label_logprob_difference']:.3e}  bit-identical reads "
              f"{overall['reads_bit_identical']}/{overall['reads']}  written mismatches {overall['written_mismatches']}",
              flush=True)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"chunk": CHUNK, "cache_limit_gb": args.cache_limit_gb, "variants": results},
                              indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    rt.close()


if __name__ == "__main__":
    main()
