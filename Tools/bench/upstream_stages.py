#!/usr/bin/env python3
"""Times upstream's MlxRuntime (mlx-vlm 0.6.15, MLX 0.32.2) the way openjev-bench's prefill and
profile modes time the Swift port (issue #32): the prefill evaluated on its own, then a one-step
read over it, for the oracle's 182-token quickstart prompt (made unique per run, same length) and
states of about 1,000 and 10,000 tokens. Prints one JSON line per size. From the repository root,
with upstream's environment (Tools/jevbench/README.md) and the pinned snapshot:

    HF_HUB_OFFLINE=1 Tools/jevbench/.venv/bin/python Tools/bench/upstream_stages.py \
        ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/a7a81407613811e8ba63af92ac0d852b809e191f
"""
import json
import sys
import time

sys.path.insert(0, "Upstream/openjev")
from openjev.mlx_backend import MlxRuntime

SNAP = sys.argv[1]
rt = MlxRuntime(SNAP)
mx, model, tok = rt.mx, rt.model, rt.processor.tokenizer
oracle = json.load(open("Fixtures/oracle/reads.json"))
q = next(r for r in oracle["reads"] if r["prompt"] == "quickstart/g0" and r["steps"] == 1)
slots = [{"pos": s["pos"], "label_ids": s["label_ids"]} for s in q["slots"]]
sentence = "The checkout page times out after the card form is submitted. "

def prompts(size, index):
    if size == 0:
        base = oracle["prompts"]["quickstart/g0"]["ids"]
        return base[:-20] + [(base[-20] + index) % 1000 + 1000] + base[-19:]  # unique, same length
    return tok.encode(f"Ticket u-{index}. " + sentence * max(1, size // 12))

def run(size, runs, warm=2):
    pre, rd, n = [], [], 0
    def work(index):
        ids = prompts(size, index)
        rt.prefills.clear()
        t0 = time.perf_counter()
        cache = model.diffusion_prefill_cache(input_ids=mx.array([ids]))
        mx.eval([c.state for c in cache])
        t1 = time.perf_counter()
        rt.prefills[tuple(ids)] = (cache, len(ids))
        rt.read(ids, q["canvas"], slots, 10 ** 6, 1)
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, len(ids)
    rt.pool.submit(lambda: [work(10_000 + i) for i in range(warm)]).result()
    for i in range(runs):
        a, b, n = rt.pool.submit(work, i).result()
        pre.append(a); rd.append(b)
    pre.sort(); rd.sort()
    med = lambda v: (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2
    print(json.dumps({"size": size, "prompt_tokens": n, "runs": runs,
                      "prefill_p50_ms": round(med(pre), 1), "read_p50_ms": round(med(rd), 1),
                      "prefill_tokens_per_s": round(n / (med(pre) / 1e3))}), flush=True)

run(0, 20)
run(1000, 5)
run(10000, 5)
