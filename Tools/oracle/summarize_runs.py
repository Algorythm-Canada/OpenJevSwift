"""Latency and memory tables for the spike #22 report, from the run files in Tools/oracle/results.

Python (the oracle, MlxRuntime.read), the Layr-Labs fork (the Swift probe) and the Swift
transliteration, each without and with a 4 GB MLX cache limit. A cold read includes the prefill
of its prompt; a cached read reuses the prompt's prefill and runs only the decoder passes and the
slot log-softmax. Every read ran twice per process, so each figure is the median of the samples.

    Tools/oracle/.venv/bin/python Tools/oracle/summarize_runs.py
"""
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
GIB = 1024 ** 3
PROMPTS = [("single_noul/g0", "1 question, 78 tokens"), ("quickstart/g0", "3 questions, 182 tokens"),
           ("indexed_12_mixed/g0", "12 questions, 1,572 tokens"), ("long_state/g0", "3 questions, 2,939 tokens")]


def python_samples(run):
    out = []
    for t in run["timings"]:
        for key in ("pass_1", "pass_2"):
            p = t[key]
            out.append((t["id"], p["prefill_cached"], p["seconds"]))
    return out


def probe_samples(run):
    out = []
    for r in run["reads"]:
        for key in ("timing_pass_1", "timing_pass_2"):
            p = r[key]
            out.append((r["id"], p["prefill_cached"], p["total_seconds"]))
    return out


def transliteration_samples(run):
    out = []
    for r in run["per_read"]:
        for key in ("timing_pass_1", "timing_pass_2"):
            p = r[key]
            out.append((r["id"], p["prefill_cached"], p["seconds"]))
    return out


def latency(samples, prompt, cached, steps=1):
    values = [s for rid, c, s in samples
              if rid.startswith(prompt + "/") and rid.endswith(f"steps{steps}") and c == cached]
    return statistics.median(values) * 1000 if values else None


def memory(run, key):
    m = run["memory"].get(key, {})
    return m.get("phys_footprint_bytes", 0) / GIB, m.get("mlx_cache_bytes", 0) / GIB


def main():
    runs = [
        ("Python oracle", "oracle_run.json", python_samples),
        ("Python oracle, 4 GB limit", "oracle_run_cache_limit_4gb.json", python_samples),
        ("Fork probe", "probe_run.json", probe_samples),
        ("Fork probe, 4 GB limit", "probe_run_cache_limit_4gb.json", probe_samples),
        ("Transliteration", "transliteration_run.json", transliteration_samples),
        ("Transliteration, 4 GB limit", "transliteration_run_cache_limit_4gb.json", transliteration_samples),
    ]
    loaded = [(name, json.loads((RESULTS / f).read_text()), fn) for name, f, fn in runs if (RESULTS / f).exists()]

    print("| Run | Load s | Footprint after load GiB | After 2 x 27 reads GiB | MLX cache after reads GiB | Peak MLX GiB |")
    print("|---|---|---|---|---|---|")
    for name, run, _ in loaded:
        after_load, _ = memory(run, "after_load")
        after, cache = memory(run, "after_pass_2")
        peak = run["memory"]["after_pass_2"].get("mlx_peak_bytes", 0) / GIB
        print(f"| {name} | {run['load_seconds']:.1f} | {after_load:.2f} | {after:.2f} | {cache:.2f} | {peak:.2f} |")

    print()
    print("| Run | " + " | ".join(f"{label}: cold / cached ms" for _, label in PROMPTS) + " |")
    print("|---|" + "---|" * len(PROMPTS))
    for name, run, fn in loaded:
        samples = fn(run)
        cells = []
        for prompt, _ in PROMPTS:
            cold, warm = latency(samples, prompt, False), latency(samples, prompt, True)
            cells.append(f"{cold:.0f} / {warm:.0f}" if cold and warm else (f"{cold:.0f} / n/a" if cold else "n/a"))
        print(f"| {name} | " + " | ".join(cells) + " |")

    print()
    print("| Run | quickstart steps 1 / 2 / 3, cached ms | long_state steps 1 / 2 / 3, cached ms |")
    print("|---|---|---|")
    for name, run, fn in loaded:
        samples = fn(run)
        cells = []
        for prompt in ("quickstart/g0", "long_state/g0"):
            values = [latency(samples, prompt, True, steps) for steps in (1, 2, 3)]
            cells.append(" / ".join(f"{v:.0f}" if v else "n/a" for v in values))
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
