"""Statistics behind the D-014 tolerances (spike #22) and their long-prompt revision (D-048).

Every implementation below computes the same read in exact arithmetic; they differ only in the
last bits of some kernels or in the order of operations. For each, against the oracle
(Fixtures/oracle/reads.json): the label probability differences over every label of every slot,
the per-slot maximum and its mean over the slots, top-label agreement overall and by the oracle's
top-two margin, the entropy differences, and the argmaxes written between steps.

Inputs (all produced by the scripts in Tools/oracle and Tools/fixtures):
- mlx-vlm with a 64-token chunked prefill, and with unsorted decoder expert gathers
  (--sensitivity, by default Tools/oracle/results/sensitivity.json);
- the Layr-Labs fork through the Swift probe (Probe --dump writes swift_reads.json, Probe --maps
  the same maps for every read; pass its path);
- the Swift transliteration on mlx-swift's own kernels (--native, by default
  Tools/oracle/results/transliteration_run.json). A run with a cache limit is accepted: a limit
  leaves every read bit-identical (spike #22).

    Tools/oracle/.venv/bin/python Tools/oracle/tolerance_stats.py [--oracle PATH] [--sensitivity PATH]
        [--native PATH] [--out PATH] [SWIFT_READS_JSON] [NAME=RUN_JSON ...]

Each NAME=RUN_JSON adds a Transliteration run (for example one with TRANSLITERATION_BUG set) under
that name. Rows are computed over every read; over the reads whose prompt passes 1,024 tokens;
over those reads' slots with at most 4 labels, where the 55- to 255-option choices cannot dilute
the label mean (D-048); and over the JevBench reads alone.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Upstream" / "openjev"))
from openjev.engine import slot_distribution  # noqa: E402

RESULTS = ROOT / "Tools" / "oracle" / "results"
# The sliding window: past it the decoder's sliding layers read only the last 1,023 positions.
LONG_PROMPT = 1024
# The few-label slots: at most this many labels, as most JevBench and TypeSafe questions have (they
# have 2 to 6 and 2 to 8), against the fixture's 10- to 255-option choices.
FEW_LABELS = 4


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def mean(values):
    return sum(values) / len(values) if values else None


def distributions_from_maps(read, maps):
    out = []
    for pairs, slot in zip(maps, read["slots"]):
        top = {int(t): v for t, v in pairs}
        out.append(slot_distribution(top, slot["label_ids"]))
    return out


def compare(name, oracle, variant, reads=None, max_labels=None):
    """variant: {read id: {"distributions": [...], "written": [...]}}. `reads` keeps only those
    read ids, `max_labels` only the slots with at most that many labels."""
    diffs, slot_max, entropy, margins_disagree = [], [], [], []
    agree = slots = 0
    buckets = {0.0: [0, 0], 0.1: [0, 0], 0.2: [0, 0], 0.3: [0, 0], 0.5: [0, 0]}
    written_total = written_equal = 0
    for read in oracle["reads"]:
        got = variant.get(read["id"])
        if got is None or (reads is not None and read["id"] not in reads):
            continue
        for mine, want, slot in zip(got["distributions"], read["distributions"], read["slots"]):
            if max_labels is not None and len(slot["label_ids"]) > max_labels:
                continue
            d = [abs(a - b) for a, b in zip(mine["probs"], want["probs"])]
            diffs += d
            slot_max.append(max(d))
            entropy.append(abs(mine["entropy"] - want["entropy"]))
            ranked = sorted(want["probs"], reverse=True)
            margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            same = max(range(len(mine["probs"])), key=mine["probs"].__getitem__) == \
                max(range(len(want["probs"])), key=want["probs"].__getitem__)
            agree += same
            slots += 1
            for threshold, cell in buckets.items():
                if margin >= threshold:
                    cell[0] += same
                    cell[1] += 1
            if not same:
                margins_disagree.append(margin)
        if read["steps"] > 1 and "written" in got:
            written_total += 1
            written_equal += got["written"] == read["written"]
    if not slots:  # a run made before the fixture had these reads
        return None
    return {
        "name": name,
        "labels": len(diffs),
        "slots": slots,
        "mean_probability_difference": mean(diffs),
        "median_probability_difference": quantile(diffs, 0.5),
        "p90_probability_difference": quantile(diffs, 0.9),
        "p99_probability_difference": quantile(diffs, 0.99),
        "max_probability_difference": max(diffs),
        "mean_slot_max_probability_difference": mean(slot_max),
        "slots_within_0.02": sum(x <= 0.02 for x in slot_max),
        "slots_within_0.05": sum(x <= 0.05 for x in slot_max),
        "slots_within_0.10": sum(x <= 0.10 for x in slot_max),
        "top_label_agreement": agree,
        "top_label_agreement_by_oracle_margin": {f">={k}": f"{v[0]}/{v[1]}" for k, v in buckets.items()},
        "largest_oracle_margin_with_disagreement": max(margins_disagree) if margins_disagree else None,
        "mean_entropy_difference": mean(entropy),
        "p90_entropy_difference": quantile(entropy, 0.9),
        "max_entropy_difference": max(entropy),
        "written_argmaxes_equal": f"{written_equal}/{written_total}",
    }


def transliteration_maps(by_id, path, native=False):
    run = json.loads(Path(path).read_text())
    rows = run["per_read"]
    shown = Path(path).resolve().relative_to(ROOT) if Path(path).resolve().is_relative_to(ROOT) else path
    if native and (run.get("metallib") or run.get("rope_frequencies_from")):
        raise SystemExit(f"{shown} is not on mlx-swift's own kernels; rerun Transliteration without "
                         "--metallib, --oracle-rope or --freqs-from")
    if not rows or any("logprobs" not in r for r in rows):
        raise SystemExit(f"{shown} has no per-read maps; rerun Transliteration without --summary-only")
    return {r["id"]: {"distributions": distributions_from_maps(by_id[r["id"]], r["logprobs"]),
                      "written": r["written"]} for r in rows if r["id"] in by_id}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oracle", default=str(ROOT / "Fixtures" / "oracle" / "reads.json"))
    ap.add_argument("--sensitivity", default=str(RESULTS / "sensitivity.json"),
                    help="Tools/oracle/sensitivity.py's output")
    ap.add_argument("--native", default=str(RESULTS / "transliteration_run.json"),
                    help="a Transliteration run on mlx-swift's own kernels, with its per-read maps")
    ap.add_argument("--out", default=str(RESULTS / "tolerance_stats.json"))
    ap.add_argument("runs", nargs="*", help="the fork's maps (a path), then NAME=RUN_JSON runs")
    args = ap.parse_args()

    oracle = json.loads(Path(args.oracle).read_text())
    by_id = {r["id"]: r for r in oracle["reads"]}
    long_ids = {r["id"] for r in oracle["reads"] if oracle["prompts"][r["prompt"]]["tokens"] > LONG_PROMPT}
    jevbench_ids = {r["id"] for r in oracle["reads"] if r["request"] in oracle.get("jevbench", {})}
    variants = []

    sensitivity = json.loads(Path(args.sensitivity).read_text())
    for name in ("chunked_prefill", "unsorted_decoder_experts"):
        rows = sensitivity["variants"][name]["reads"]
        variants.append((f"mlx-vlm, {name.replace('_', ' ')}",
                         {r["id"]: {"distributions": r["per_slot"]} for r in rows}))

    extra = [a for a in args.runs if "=" in a]
    positional = [a for a in args.runs if "=" not in a]
    if positional:
        fork = json.loads(Path(positional[0]).read_text())
        variants.append(("Layr-Labs fork (Swift probe)", {
            rid: {"distributions": distributions_from_maps(by_id[rid], v["logprobs"]), "written": v["written"]}
            for rid, v in fork.items() if rid in by_id}))

    if Path(args.native).exists():
        variants.append(("Swift transliteration, mlx-swift kernels", transliteration_maps(by_id, args.native, True)))

    for spec in extra:
        name, path = spec.split("=", 1)
        variants.append((name, transliteration_maps(by_id, path)))

    subsets = [("", None, None), (f", prompts over {LONG_PROMPT:,} tokens", long_ids, None),
               (f", prompts over {LONG_PROMPT:,} tokens, slots with at most {FEW_LABELS} labels", long_ids,
                FEW_LABELS)]
    if jevbench_ids:
        subsets.append((", JevBench reads", jevbench_ids, None))
    out = [row for suffix, reads, max_labels in subsets for name, variant in variants
           if (row := compare(name + suffix, oracle, variant, reads, max_labels)) is not None]
    for row in out:
        print(f"\n{row['name']}")
        for key, value in row.items():
            if key != "name":
                print(f"  {key}: {value}")
    path = Path(args.out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
