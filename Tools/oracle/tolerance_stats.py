"""Statistics behind the D-014 tolerances (spike #22).

Every implementation below computes the same read in exact arithmetic; they differ only in the
last bits of some kernels or in the order of operations. For each, against the oracle
(Fixtures/oracle/reads.json): the label probability differences over every label of every slot,
the per-slot maximum, top-label agreement overall and by the oracle's top-two margin, the
entropy differences, and the argmaxes written between steps.

Inputs (all produced by the scripts in Tools/oracle and Tools/fixtures):
- mlx-vlm with a 64-token chunked prefill, and with unsorted decoder expert gathers
  (Tools/oracle/results/sensitivity.json);
- the Layr-Labs fork through the Swift probe (Probe --dump writes swift_reads.json; pass its path);
- the Swift transliteration on mlx-swift's own kernels (Tools/oracle/results/transliteration_run.json).

    Tools/oracle/.venv/bin/python Tools/oracle/tolerance_stats.py [SWIFT_READS_JSON] [NAME=RUN_JSON ...]

Each NAME=RUN_JSON adds a Transliteration run (for example one with TRANSLITERATION_BUG set) under
that name. Rows are also computed over the four reads whose prompt passes 1,024 tokens.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Upstream" / "openjev"))
from openjev.engine import slot_distribution  # noqa: E402

RESULTS = ROOT / "Tools" / "oracle" / "results"


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def distributions_from_maps(read, maps):
    out = []
    for pairs, slot in zip(maps, read["slots"]):
        top = {int(t): v for t, v in pairs}
        out.append(slot_distribution(top, slot["label_ids"]))
    return out


LONG = {"indexed_12_mixed/g0", "many_choices/g0", "widest_schema/g0", "long_state/g0"}


def compare(name, oracle, variant, only=None):
    """variant: {read id: {"distributions": [...], "written": [...]}}"""
    diffs, slot_max, entropy, margins_disagree, rows = [], [], [], [], []
    agree = slots = 0
    buckets = {0.0: [0, 0], 0.1: [0, 0], 0.2: [0, 0], 0.3: [0, 0], 0.5: [0, 0]}
    written_total = written_equal = 0
    for read in oracle["reads"]:
        got = variant.get(read["id"])
        if got is None or (only is not None and read["prompt"] not in only):
            continue
        for mine, want in zip(got["distributions"], read["distributions"]):
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
    return {
        "name": name,
        "labels": len(diffs),
        "slots": slots,
        "mean_probability_difference": sum(diffs) / len(diffs),
        "median_probability_difference": quantile(diffs, 0.5),
        "p90_probability_difference": quantile(diffs, 0.9),
        "p99_probability_difference": quantile(diffs, 0.99),
        "max_probability_difference": max(diffs),
        "slots_within_0.02": sum(x <= 0.02 for x in slot_max),
        "slots_within_0.05": sum(x <= 0.05 for x in slot_max),
        "slots_within_0.10": sum(x <= 0.10 for x in slot_max),
        "top_label_agreement": agree,
        "top_label_agreement_by_oracle_margin": {f">={k}": f"{v[0]}/{v[1]}" for k, v in buckets.items()},
        "largest_oracle_margin_with_disagreement": max(margins_disagree) if margins_disagree else None,
        "mean_entropy_difference": sum(entropy) / len(entropy),
        "p90_entropy_difference": quantile(entropy, 0.9),
        "max_entropy_difference": max(entropy),
        "written_argmaxes_equal": f"{written_equal}/{written_total}",
    }


def main():
    oracle = json.loads((ROOT / "Fixtures" / "oracle" / "reads.json").read_text())
    by_id = {r["id"]: r for r in oracle["reads"]}
    variants = []

    sensitivity = json.loads((RESULTS / "sensitivity.json").read_text())
    for name in ("chunked_prefill", "unsorted_decoder_experts"):
        rows = sensitivity["variants"][name]["reads"]
        variants.append((f"mlx-vlm, {name.replace('_', ' ')}",
                         {r["id"]: {"distributions": r["per_slot"]} for r in rows}))

    extra = [a for a in sys.argv[1:] if "=" in a]
    positional = [a for a in sys.argv[1:] if "=" not in a]
    if positional:
        fork = json.loads(Path(positional[0]).read_text())
        variants.append(("Layr-Labs fork (Swift probe)", {
            rid: {"distributions": distributions_from_maps(by_id[rid], v["logprobs"]), "written": v["written"]}
            for rid, v in fork.items()}))

    native = RESULTS / "transliteration_run.json"
    if native.exists():
        run = json.loads(native.read_text())
        rows = run["per_read"]
        if run.get("metallib") or run.get("rope_frequencies_from") or run.get("cache_limit_gb"):
            raise SystemExit(f"{native.relative_to(ROOT)} is not the native baseline; rerun Transliteration "
                             "with no options")
        if not rows or any("logprobs" not in r for r in rows):
            raise SystemExit(f"{native.relative_to(ROOT)} has no per-read maps; rerun Transliteration "
                             "without --summary-only")
        variants.append(("Swift transliteration, mlx-swift kernels", {
            r["id"]: {"distributions": distributions_from_maps(by_id[r["id"]], r["logprobs"]),
                      "written": r["written"]} for r in rows}))

    for spec in extra:
        name, path = spec.split("=", 1)
        rows = json.loads(Path(path).read_text())["per_read"]
        variants.append((name, {
            r["id"]: {"distributions": distributions_from_maps(by_id[r["id"]], r["logprobs"]),
                      "written": r["written"]} for r in rows}))

    out = [compare(name, oracle, variant) for name, variant in variants]
    out += [compare(name + ", prompts over 1,024 tokens", oracle, variant, only=LONG)
            for name, variant in variants]
    for row in out:
        print(f"\n{row['name']}")
        for key, value in row.items():
            if key != "name":
                print(f"  {key}: {value}")
    (RESULTS / "tolerance_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {(RESULTS / 'tolerance_stats.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
