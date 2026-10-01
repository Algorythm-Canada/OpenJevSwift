"""The tables of docs/quality.md, rendered from the result files under results/.

    python3 Tools/jevbench/harness.py report

Every number comes from a result file and the pinned datasets (`harness.py fetch`); nothing here
is typed in by hand. `harness.py compare` and `published` print the same comparisons item by item.
"""

from __future__ import annotations

import statistics
import textwrap
from pathlib import Path

import harness
from harness import markdown_table, num, pct

LARGEST = 5


def result_files(results: Path) -> list:
    return sorted(results.glob("*.json")) + sorted((results / "typesafe102").glob("*.json"))


def pairs(docs: list) -> list:
    """(dataset, model, swift run, upstream run) for every model run on both servers."""
    found = {}
    for doc in docs:
        found.setdefault((doc["dataset"]["name"], doc["model"]), {})[doc["server"]["name"]] = doc
    return [(dataset, model, runs["swift"], runs["upstream"])
            for (dataset, model), runs in sorted(found.items(), key=lambda pair: (
                pair[0][0] != "jevbench", pair[0][0], pair[0][1]))
            if "swift" in runs and "upstream" in runs]


def agreement_tables(docs: list) -> list:
    """Swift against upstream: one row per model and dataset, one per question type, the largest
    deviations, and the near ties and disagreements."""
    overall_rows, type_rows, largest_rows, tie_rows = [], [], [], []
    for dataset, model, swift, upstream in pairs(docs):
        result = harness.compare_docs(swift, upstream, top=LARGEST)
        overall, flips = result["overall"], result["correctness"]
        overall_rows.append([
            model, dataset, str(overall["items"]), f"{overall['agree']} of {overall['items']}",
            str(overall["identical"]), num(overall["mean_abs_diff"]),
            num(overall["max_abs_diff"]), f"{flips['a_only']} and {flips['b_only']}",
            f"{flips['mcnemar_p']:.3g}"])
        for kind, group in result["per_type"].items():
            type_rows.append([model, dataset, kind, str(group["items"]),
                              f"{group['agree']} of {group['items']}", num(group["mean_abs_diff"]),
                              num(group["max_abs_diff"]),
                              f"{group['max_item']} ({group['max_label']})"])
        for entry in result["largest"]:
            largest_rows.append([model, dataset, entry["id"], entry["type"], entry["label"],
                                 f"{entry['a']:.4f}", f"{entry['b']:.4f}", num(entry["diff"])])
        ties = result["near_ties"]
        closest = min(ties, key=lambda entry: entry["b_margin"], default=None)
        tie_rows.append([
            model, dataset, str(len(ties)), f"{sum(entry['agree'] for entry in ties)} of {len(ties)}",
            f"{closest['id']} ({num(closest['b_margin'])} upstream, {num(closest['a_margin'])} Swift)"
            if closest else "",
            ", ".join(entry["id"] for entry in result["disagreements"]) or "none"])
    return [
        markdown_table(["model", "dataset", "items", "top answer agrees", "identical answers",
                        "mean abs diff", "largest abs diff",
                        "right only on Swift, only upstream", "McNemar p"], overall_rows),
        markdown_table(["model", "dataset", "type", "items", "top answer agrees",
                        "mean abs diff", "largest abs diff", "largest at (label)"], type_rows),
        f"The {LARGEST} largest deviations of each model and dataset:",
        markdown_table(["model", "dataset", "item", "type", "label", "Swift", "upstream",
                        "abs diff"], largest_rows),
        "The items whose upstream top two are less than 0.01 apart, where the parity bound of "
        "D-034 and D-037 allows a changed top answer (`harness.py compare` lists each):",
        markdown_table(["model", "dataset", "near ties", "top answer kept", "closest (top-two margin)",
                        "disagreements"], tie_rows),
    ]


def published_tables(docs: list, cache: Path) -> list:
    rows, tiers, differ = [], [], []
    for doc in docs:
        result = harness.against_published(doc, cache)
        if not result:
            continue
        outcomes, board = result["outcomes"], result["board"] or {}
        rows.append([result["model"], result["server"], result["row"]["key"],
                     str(result["items"]), pct(result["ours_accuracy"]),
                     pct(result["published_accuracy"]), pct(outcomes["agreement"]),
                     str(outcomes["ours_only"]), str(outcomes["published_only"]),
                     f"{outcomes['mcnemar_p']:.3g}", pct(board.get("sealed_accuracy"))])
        for tier, counts in result["per_tier"].items():
            tiers.append([result["model"], result["server"], tier, str(counts["items"]),
                          pct(counts["ours"] / counts["items"]),
                          pct(counts["published"] / counts["items"])])
        if result["server"] == "upstream":
            by_type = {}
            for entry in result["differ"]:
                by_type.setdefault(entry["type"], []).append(entry["id"])
            for kind, ids in sorted(by_type.items()):
                listed = ", ".join(ids) if len(ids) <= 6 else f"{len(ids)} items"
                differ.append([result["model"], kind, str(len(ids)), listed])
    if not rows:
        return []
    return [
        markdown_table(["model", "server", "published row", "public items", "ours", "published",
                        "same outcome", "right only here", "right only there", "McNemar p",
                        "published sealed"], rows),
        markdown_table(["model", "server", "tier", "items", "ours", "published"], tiers),
        "Items whose outcome differs from the published row's (upstream's run; Swift's is the same):",
        markdown_table(["model", "type", "items", "which"], differ),
    ]


def typesafe_tables(docs: list) -> list:
    rows, published = [], None
    for doc in docs:
        if doc["dataset"]["name"] != "typesafe102":
            continue
        semif, overall = doc["summary"]["semif"], doc["summary"]["overall"]
        rows.append([doc["model"], doc["server"]["name"], f"{semif['rows']} of {semif['of']}",
                     str(semif.get("cases")), num(semif.get("equal_case_modal_agreement"), 3),
                     num(semif.get("equal_case_total_variation"), 3), pct(overall["accuracy"]),
                     num(overall["brier_mean"]), num((overall["ece"] or {}).get("ece"))])
        if semif["rows"] == semif["of"]:
            published = semif.get("published")
    if not rows:
        return []
    out = [markdown_table(["model", "server", "rows", "cases", "modal agreement (equal-case)",
                           "total variation (equal-case)", "accuracy (pooled)", "Brier", "ECE"],
                          rows)]
    if published:
        out += ["The answers TypeSafe's snapshots publish, over the same rows:", markdown_table(
            ["model", "rows", "cases", "modal agreement (equal-case)",
             "total variation (equal-case)"],
            [[f"{key} ({entry['model']})", str(entry["rows"]), str(entry["cases"]),
              num(entry["equal_case_modal_agreement"], 3),
              num(entry["equal_case_total_variation"], 3)] for key, entry in published.items()])]
    return out


# The Swift encoder on a Mac: one Core ML function per input shape (batch 1 or 16 by length), the
# least recently used released first once the server keeps as many as it may; the warm-up's three
# questions load the batch-16 function for 128 tokens. A one-question request reads through the
# batch-1 function of the smallest shape that holds it. A run records how many functions its server
# kept (`function_capacity`: a number, or "all", D-042); runs recorded before kept two (D-037 item 3).
RECORDED_FUNCTION_CAPACITY = 2
SHAPES = {"verdict-1.4": (128, 256, 512), "laya-1.0": (128, 256, 512, 1024)}


def function_capacity(doc: dict, shapes: tuple) -> int:
    """How many functions the run's server kept loaded; "all" is every shape at both batches."""
    kept = doc["server"].get("function_capacity", RECORDED_FUNCTION_CAPACITY)
    return 2 * len(shapes) if kept == "all" else int(kept)


def function_load_rows(docs: list) -> list:
    """Which Swift requests had to load a Core ML function, by simulating that cache over one
    server process's requests (JevBench, then TypeSafe), with the model time each took."""
    swift = {(doc["model"], doc["dataset"]["name"]): doc for doc in docs
             if doc["server"]["name"] == "swift"}
    rows = []
    for model, shapes in SHAPES.items():
        runs = [swift.get((model, name)) for name in ("jevbench", "typesafe102")]
        if not all(runs):
            continue
        capacity = function_capacity(runs[0], shapes)
        loaded, seen, first, again, ordinary = [("b16", shapes[0])], set(), [], [], {}
        for doc in runs:
            for item in doc["items"]:
                timing = ((item.get("timing") or {}).get("server") or {}).get("model")
                if item["status"] != "answered" or timing is None:
                    continue
                tokens = min(item["usage"]["input_tokens"], shapes[-1])
                key = ("b1", min(shape for shape in shapes if shape >= tokens))
                if key in loaded:
                    loaded.remove(key)
                    ordinary.setdefault(key[1], []).append(timing)
                else:
                    (again if key in seen else first).append(timing)
                    seen.add(key)
                loaded.append(key)
                del loaded[:-capacity]
        loads = first + again
        medians = ", ".join(f"{shape}: {statistics.median(ordinary[shape]):.0f}"
                            for shape in shapes if shape in ordinary)
        rows.append([model, str(sum(len(doc["items"]) for doc in runs)), str(len(first)),
                     str(len(again)),
                     f"{min(loads):.0f} to {max(loads):.0f}" if loads else "",
                     medians])
    return rows


def environment_rows(docs: list) -> list:
    """One row per model and server: both datasets ran on one server process."""
    rows, seen = [], {}
    for doc in docs:
        server, hardware = doc["server"], doc["hardware"]
        if server.get("implementation") == "OpenJevSwift":
            code = f"OpenJevSwift {server.get('version')} at {str(server.get('commit'))[:7]}"
            runtime = server.get("runtime") or ""
        else:
            packages = server.get("packages") or {}
            code = (f"openjev {server.get('version')} at {str(server.get('commit'))[:7]}, "
                    f"Python {server.get('python')}")
            runtime = (f"PyTorch {packages.get('torch')} on the {server.get('device')}, "
                       f"{server.get('dtype')}, {server.get('torch_threads')} threads")
        key = (doc["model"], server["name"])
        if key in seen:
            seen[key][0] += f", {doc['dataset']['name']}"
            continue
        seen[key] = [doc["dataset"]["name"], doc["model"], server["name"], code, runtime,
                     f"{hardware.get('cpu')}, {hardware.get('memory_gb')} GB, {hardware.get('os')}",
                     doc["started_utc"][:10]]
        rows.append(seen[key])
    return rows


def render(results: Path, cache: Path) -> str:
    docs = harness.load_docs(result_files(results), cache)
    out = ["### The runs", markdown_table(harness.SUMMARY_HEADER, harness.summary_rows(docs))]
    out += ["### Swift against upstream"] + agreement_tables(docs)
    published = published_tables(docs, cache)
    if published:
        out += ["### Against JevBench's published rows"] + published
    typesafe = typesafe_tables(docs)
    if typesafe:
        out += ["### The TypeSafe subset with SemIf's metrics"] + typesafe
    tiers = [row for doc in docs if doc["dataset"]["name"] == "jevbench"
             for row in harness.tier_rows(doc)]
    if tiers:
        out += ["### JevBench by tier", markdown_table(
            ["model", "server", "tier", "items", "accuracy", "Brier", "ECE", "ordinal MAE"], tiers)]
    loads = function_load_rows(docs)
    if loads:
        kept = sorted({str(doc["server"].get("function_capacity", RECORDED_FUNCTION_CAPACITY))
                       for doc in docs if doc["server"]["name"] == "swift"
                       and doc["model"] in SHAPES})
        cache = ("its two-function cache" if kept == ["2"] else
                 "its function cache (" + " or ".join(kept) + " functions kept, as each run "
                 "records)")
        out += ["### The Swift server's Core ML function loads",
                "Requests that had to load a Core ML function on the Swift server, from a simulation "
                f"of {cache} over each server's requests, and the median model time "
                "of the other requests by input shape:",
                markdown_table(["model", "requests", "first loads", "loaded again", "load ms",
                                "other requests' median ms by shape"], loads)]
    out += ["### Machines and versions", markdown_table(
        ["datasets", "model", "server", "code", "runtime", "machine", "run on"],
        environment_rows(docs))]
    # prose wraps at 100 columns as the repository's documents do; tables and headings stay whole
    out = [block if block.startswith(("|", "#"))
           else textwrap.fill(block, 100, break_on_hyphens=False) for block in out]
    return "\n\n".join(out) + "\n"
