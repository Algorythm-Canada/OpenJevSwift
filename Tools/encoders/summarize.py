#!/usr/bin/env python3
"""Markdown tables from the harness results, for docs/spikes/encoder-runtime.md (spike #56).

    Tools/encoders/.venv/bin/python Tools/encoders/summarize.py docs/spikes/encoder-runtime/macos
    Tools/encoders/.venv/bin/python Tools/encoders/summarize.py docs/spikes/encoder-runtime/iphone
    Tools/encoders/.venv/bin/python Tools/encoders/summarize.py docs/spikes/encoder-runtime

Given a folder of harness results (<package>-<units>.json), it prints three tables: latency and
memory, latency by sequence length, and parity with the compute plan. Given the folder that holds
verdict-coreml.json and laya-coreml.json, it prints the conversion and Python parity tables. It
uses only the standard library.
"""

import json
import sys
from pathlib import Path

ORDER = ["verdict-m18-fp16", "verdict-m18-w8", "verdict-e17-fp16", "verdict-e17-fp32", "laya-m18-fp16", "laya-m18-w8",
         "laya-e17-fp16"] + [f"laya-f18-b1s{s}-fp16" for s in (128, 256, 512, 1024)]
UNITS = ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"]


def mb(n):
    return f"{n / 1e6:,.0f}"


def small(x):
    """Three decimals, or scientific notation below 0.001."""
    return f"{x:.3f}" if abs(x) >= 1e-3 else f"{x:.1e}"


def ms(stats, key="medianMs"):
    return f"{stats[key]:,.1f}" if stats and stats.get("count") else ""


def load(folder):
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        data = json.loads(path.read_text())
        if "package" in data and "units" in data:
            rows.append(data)
    rows.sort(key=lambda r: (ORDER.index(r["package"]) if r["package"] in ORDER else 99, UNITS.index(r["units"])))
    return rows


def plan_share(result):
    """The plan's estimated cost by device, or, where Core ML gives no estimate, its operations by
    device (constants, which run on no device, left out)."""
    plan = result.get("computePlan") or {}
    share = plan.get("costShare") or {}
    suffix = ""
    if not share:
        ops = {k: v for k, v in (plan.get("operations") or {}).items() if k != "none"}
        total = sum(ops.values())
        share = {k: v / total for k, v in ops.items()} if total else {}
        suffix = " of operations" if share else ""
    names = {"neuralEngine": "ANE", "gpu": "GPU", "cpu": "CPU"}
    parts = [f"{names.get(k, k)} {v * 100:.0f}%" for k, v in sorted(share.items(), key=lambda kv: -kv[1]) if v >= 0.005]
    return ", ".join(parts) + suffix


def conversion_tables(folder):
    for name in ("verdict-coreml.json", "laya-coreml.json"):
        report = json.loads((folder / name).read_text())
        laya = name.startswith("laya")
        print(f"\n{name}: wrapper against the reference in float32, largest logit difference "
              f"{report['wrapper_max_abs_logit_difference_float32']:.1e}\n")
        print("| Package | Converted in s | Size MB | Operations |")
        print("|---|---|---|---|")
        for package, entry in report["packages"].items():
            ops = entry.get("operations", {})
            print(f"| {package} | {entry.get('conversion_seconds', '')} | {mb(entry['package_bytes'])} | "
                  f"{sum(ops.values()):,} |")
        extra = " | Rounded answers identical" if laya else ""
        print(f"\n| Package | Units, batch | Max abs logit difference | Max abs probability difference "
              f"| Mean abs probability difference | Top answers kept{extra} |")
        print("|---|---|---|---|---|---|" + ("---|" if laya else ""))
        key = "max_abs_probability_difference_unrounded" if laya else "max_abs_probability_difference"
        mean = "mean_abs_probability_difference_unrounded" if laya else "mean_abs_probability_difference"
        for package, entry in report["packages"].items():
            for setting, p in entry["parity"].items():
                if "error" in p:
                    print(f"| {package} | {setting} | load failed: {p['error'].split('=')[-1].strip(' ;}')} "
                          f"| | | |" + (" |" if laya else ""))
                    continue
                rounded = f" | {p['rounded_answers_identical']}" if laya else ""
                print(f"| {package} | {setting} | {small(p['max_abs_logit_difference'])} | {p[key]:.2e} | "
                      f"{p[mean]:.1e} | {p['top_label_agreement']}{rounded} |")


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/spikes/encoder-runtime/macos")
    if (folder / "verdict-coreml.json").exists():
        conversion_tables(folder)
        return
    rows = load(folder)
    if not rows:
        sys.exit(f"no results in {folder}")
    device = rows[0]["device"]
    print(f"Device: {device['machine']} ({device['model']}), {device['operatingSystem']}, "
          f"{device['processorCount']} cores, {device['physicalMemoryBytes'] / 2**30:.0f} GiB\n")

    print("| Package | Units | Compile s | Load s (first, second) | Batch 1 median ms | Batch 1 p95 ms "
          "| Batch 16 per call median ms | Batch 16 p95 ms | Batch 16 per question ms "
          "| Peak footprint during reads MB | Process peak so far MB | Thermal |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['package']} | {r['units']} | {r['compileSeconds']:.1f} | {r['firstLoadSeconds']:.1f}, "
              f"{r['secondLoadSeconds']:.1f} | {ms(r['batch1'])} | {ms(r['batch1'], 'p95Ms')} | "
              f"{ms(r['batch16PerCall'])} | {ms(r['batch16PerCall'], 'p95Ms')} | {ms(r['batch16PerQuestion'])} | "
              f"{mb(r['peakFootprintDuringRunBytes'])} | {mb(r['lifetimePeakFootprintBytes'])} | "
              f"{r['thermalStart']} to {r['thermalEnd']} |")

    print("\n| Package | Units | Batch 1 median ms at 128, 256, 512, 1024 tokens | Batch 16 median ms at 128, 256, 512, 1024 "
          "| Function loads s | Footprint after first call MB |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        one = [ms(r["batch1ByLength"].get(f"s{s}")) or "none" for s in (128, 256, 512, 1024)]
        sixteen = [ms(r["batch16ByLength"].get(f"b16_s{s}")) or "none" for s in (128, 256, 512, 1024)]
        loads = r.get("functionLoadSeconds") or {}
        load_text = ", ".join(f"{k} {v:.1f}" for k, v in loads.items())
        prints = r.get("footprintAfterWarmupBytes") or {}
        print_text = f"{mb(min(prints.values()))} to {mb(max(prints.values()))}" if prints else ""
        print(f"| {r['package']} | {r['units']} | {', '.join(one)} | {', '.join(sixteen)} | {load_text} | {print_text} |")

    print("\n| Package | Units | Max abs probability difference (batch 1, 16) | Mean (batch 1) | Max abs logit difference "
          "| Top answers kept (batch 1, 16) | Non-finite | Tokenization | Planned cost by device |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p1, p16 = r["parityBatch1"], r.get("parityBatch16")
        d16 = f"{p16['maxAbsProbabilityDifference']:.2e}" if p16 else "not run"
        t16 = str(p16["topLabelAgreement"]) if p16 else "not run"
        n16 = str(p16["nonFinite"]) if p16 else "not run"
        print(f"| {r['package']} | {r['units']} | {p1['maxAbsProbabilityDifference']:.2e}, {d16} | "
              f"{p1['meanAbsProbabilityDifference']:.1e} | {small(p1['maxAbsLogitDifference'])} | "
              f"{p1['topLabelAgreement']}, {t16} of {p1['questions']} | {p1['nonFinite']}, {n16} | {r['tokenizationMatches']} | "
              f"{plan_share(r)} |")

    failures = folder / "failures.txt"
    if failures.exists() and failures.read_text().strip():
        print("\nFailures, as `failures.txt` records them:\n")
        print("```text")
        print(failures.read_text().rstrip("\n"))
        print("```")


if __name__ == "__main__":
    main()
