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
# Models whose runs' timings the tables leave out, and why. The DiffusionGemma runs shared the Mac's
# GPU with other work, and the JevK5 runs were not taken under a benchmark's protocol, so their
# timings are not a measurement; docs/benchmarks.md is (D-044). JevK5's reference was timed on the
# author's GPU, which says nothing about a Mac either.
UNTIMED = {"openjev-0.1": "not reported", "jevk5-0.2": "not reported"}
# D-014's bounds on the aggregates the wire answers allow (the entropy bounds need the top-k
# entropy of each read, which an answer does not carry). D-048 bounds the long prompts by each
# read slot's largest difference, a measure of reads that an answer averages, so the long prompts'
# figures are shown and not bounded.
D014 = {"mean": 0.02, "agree": 0.90, "confident_agree": 0.97}


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


def author_pairs(docs: list) -> list:
    """(dataset, model, swift run, the model author's run) for every model run on the Swift
    server whose reference is its author's published run (harness.AUTHOR_RUNS)."""
    found = {}
    for doc in docs:
        found.setdefault((doc["dataset"]["name"], doc["model"]), {})[doc["server"]["name"]] = doc
    return [(dataset, model, runs["swift"], runs[harness.AUTHOR_SERVER])
            for (dataset, model), runs in sorted(found.items())
            if "swift" in runs and harness.AUTHOR_SERVER in runs]


def author_tables(docs: list) -> list:
    """Swift against the model author's published run: the top answers, the billed tokens, which
    are equal exactly when the prompts are, and the probability differences, overall and by
    question type, with the largest."""
    rows, type_rows, largest_rows = [], [], []
    for dataset, model, swift, author in author_pairs(docs):
        result = harness.compare_docs(swift, author, top=LARGEST)
        overall, flips, tokens = result["overall"], result["correctness"], result["input_tokens"]
        rows.append([
            model, dataset, str(overall["items"]), f"{overall['agree']} of {overall['items']}",
            f"{tokens['equal']} of {tokens['items']}", num(overall["mean_abs_diff"]),
            num(overall["median_max_abs_diff"]), num(overall["max_abs_diff"]),
            f"{flips['a_only']} and {flips['b_only']}", f"{flips['mcnemar_p']:.3g}"])
        for kind, group in result["per_type"].items():
            type_rows.append([model, dataset, kind, str(group["items"]),
                              f"{group['agree']} of {group['items']}", num(group["mean_abs_diff"]),
                              num(group["max_abs_diff"]),
                              f"{group['max_item']} ({group['max_label']})"])
        for entry in result["largest"]:
            largest_rows.append([model, dataset, entry["id"], entry["type"], entry["label"],
                                 f"{entry['a']:.4f}", f"{entry['b']:.4f}", num(entry["diff"]),
                                 "yes" if entry["agree"] else "no"])
    if not rows:
        return []
    return [
        markdown_table(["model", "dataset", "items", "top answer agrees", "input tokens equal",
                        "mean abs diff", "median of each item's largest abs diff",
                        "largest abs diff", "right only on Swift, only the author's",
                        "McNemar p"], rows),
        markdown_table(["model", "dataset", "type", "items", "top answer agrees",
                        "mean abs diff", "largest abs diff", "largest at (label)"], type_rows),
        f"The {LARGEST} largest deviations:",
        markdown_table(["model", "dataset", "item", "type", "label", "Swift", "author",
                        "abs diff", "top answer agrees"], largest_rows),
    ]


# JevK5's other conversions' runs, `results/jevk5-conversions/<conversion>/` with a `typesafe102/`
# folder as `results/` has. The main result files are the server's default conversion's; only the
# table of conversions reads these.
CONVERSIONS = "jevk5-conversions"
# A conversion's top answer is expected to be the author's where the author's top two are at least
# this far apart; closer is a near-tie, which bfloat16 rounding alone can turn (D-052).
CLEAR_MARGIN = 0.05


def conversion_files(results: Path) -> list:
    folder = results / CONVERSIONS
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*/*.json")) + sorted(folder.glob("*/typesafe102/*.json"))


def conversion_tables(docs: list, others: list) -> list:
    """Every conversion of a model against its author's published run, the default conversion's
    run (in `docs`) first and the others' (`others`) after it: the top answers, overall and where
    the author's top two are clear of a tie, the billed tokens, the probability differences, and
    the accuracy on the author's dataset and on TypeSafe's."""
    rows = []
    for dataset, model, default, author in author_pairs(docs):
        def runs(group, name):
            return [doc for doc in group if doc["server"]["name"] == "swift"
                    and doc["model"] == model and doc["dataset"]["name"] == name]

        # an invalid answer has no top answer to agree with, as compare_docs counts it
        clear = {item["id"] for item in author["items"] if item["status"] == "answered"
                 and item.get("valid") and harness.top_two(item["probabilities"])[2] >= CLEAR_MARGIN}
        typesafe = {doc["server"].get("conversion"): doc
                    for doc in runs(docs, "typesafe102") + runs(others, "typesafe102")}
        others_here = sorted(runs(others, dataset),
                             key=lambda doc: doc["server"].get("conversion") or "")
        if not others_here:
            continue
        for doc in [default] + others_here:
            result = harness.compare_docs(doc, author)
            overall, tokens = result["overall"], result["input_tokens"]
            answered = {item["id"] for item in doc["items"]
                        if item["status"] == "answered" and item.get("valid")}
            judged = clear & answered
            missed = sum(1 for entry in result["disagreements"] if entry["id"] in judged)
            name = doc["server"].get("conversion") or "not recorded"
            other = typesafe.get(doc["server"].get("conversion"))
            rows.append([
                model, name + (", the default" if doc is default else ""),
                f"{overall['agree']} of {overall['items']}",
                f"{len(judged) - missed} of {len(judged)}",
                f"{tokens['equal']} of {tokens['items']}", num(overall["mean_abs_diff"]),
                num(overall["median_max_abs_diff"]), num(overall["max_abs_diff"]),
                pct(doc["summary"]["overall"].get("accuracy")),
                pct(other["summary"]["overall"].get("accuracy")) if other else "not run"])
        rows.append([model, "the author's published run", "", "", "", "", "", "",
                     pct(author["summary"]["overall"].get("accuracy")), "not published"])
    if not rows:
        return []
    return [markdown_table(
        ["model", "conversion", "top answer agrees",
         f"top answer agrees where the author's top two are at least {CLEAR_MARGIN} apart",
         "input tokens equal", "mean abs diff", "median of each item's largest abs diff",
         "largest abs diff", "accuracy", "TypeSafe accuracy"], rows)]


def is_diffusiongemma(doc: dict) -> bool:
    return doc["server"].get("backend") == "mlx"


def within_d014(overall: dict, bounds: dict) -> bool:
    """Whether the answers meet every D-014 bound the wire allows. A subset with no item (no
    confident reference) has nothing to fail."""
    if not overall["items"]:
        return False
    checks = [overall["mean_abs_diff"] <= D014["mean"],
              overall["agree"] >= D014["agree"] * overall["items"]]
    if bounds["confident_items"]:
        checks.append(bounds["confident_agree"] >= D014["confident_agree"]
                      * bounds["confident_items"])
    return all(checks)


def agreement_tables(docs: list) -> list:
    """Swift against upstream: one row per model and dataset, one per question type, the largest
    deviations, the encoders' near ties and disagreements (D-034, D-037) and DiffusionGemma's
    aggregates against D-014."""
    overall_rows, type_rows, largest_rows, tie_rows, d014_rows = [], [], [], [], []
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
        disagreements = ", ".join(entry["id"] for entry in result["disagreements"]) or "none"
        if is_diffusiongemma(upstream):
            bounds = result["bounds"]
            d014_rows.append([
                model, dataset, str(overall["items"]), num(overall["mean_abs_diff"]),
                f"{num(bounds['long_mean_max_abs_diff'])} and {num(bounds['long_mean_abs_diff'])} "
                f"over {bounds['long_items']}",
                f"{overall['agree']} of {overall['items']} "
                f"({pct(overall['agree'] / overall['items'] if overall['items'] else None)})",
                f"{bounds['confident_agree']} of {bounds['confident_items']}"
                + (f" ({pct(bounds['confident_agree'] / bounds['confident_items'])})"
                   if bounds["confident_items"] else ""),
                "yes" if within_d014(overall, bounds) else "no"])
            continue
        ties = result["near_ties"]
        closest = min(ties, key=lambda entry: entry["b_margin"], default=None)
        tie_rows.append([
            model, dataset, str(len(ties)), f"{sum(entry['agree'] for entry in ties)} of {len(ties)}",
            f"{closest['id']} ({num(closest['b_margin'])} upstream, {num(closest['a_margin'])} Swift)"
            if closest else "",
            disagreements])
    d014 = []
    if d014_rows:
        d014 = [
            "DiffusionGemma's answers against D-014's aggregate bounds, which the 63 oracle reads "
            "are held to (here over answers, each the mean of up to four reads, with upstream's as "
            "the reference; `harness.py compare` lists every disagreement). Past 1,024 tokens D-048 "
            "bounds each read slot's largest difference, which an answer averages, so the long "
            "prompts' figures are shown and not bounded:",
            markdown_table(["model", "dataset", "items",
                            "mean abs diff, every label (at most 0.02)",
                            "over prompts of more than 1,024 tokens: mean of each item's largest "
                            "abs diff, and mean abs diff over every label (not bounded)",
                            "top answer agrees (at least 90%)",
                            "where upstream's top two are at least 0.5 apart (at least 97%)",
                            "within every bound"], d014_rows)]
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
    ] + d014


def published_tables(docs: list, cache: Path) -> list:
    rows, tiers, differ, lists = [], [], [], {}
    for doc in docs:
        result = harness.against_published(doc, cache)
        if not result:
            continue
        lists.setdefault(result["model"], {})[result["server"]] = result["differ"]
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
    for model, by_server in lists.items():
        upstream = by_server.get("upstream")
        for server, entries in sorted(by_server.items(), key=lambda pair: pair[0] != "upstream"):
            # Swift's list is shown only where it is not upstream's
            if server != "upstream" and upstream is not None and entries == upstream:
                continue
            by_type = {}
            for entry in entries:
                by_type.setdefault(entry["type"], []).append(entry["id"])
            for kind, ids in sorted(by_type.items()):
                listed = ", ".join(ids) if len(ids) <= 6 else f"{len(ids)} items"
                differ.append([model, server, kind, str(len(ids)), listed])
    if not rows:
        return []
    # the two lists are compared only where both runs exist; an unpaired model says which it has
    status = {}
    for model, by_server in lists.items():
        if "swift" in by_server and "upstream" in by_server:
            status[model] = "the same" if by_server["swift"] == by_server["upstream"] else "differs"
        else:
            status[model] = "only " + " and ".join(f"{server}'s run"
                                                   for server in sorted(by_server))
    note = ("upstream's run; the Swift run's list is the same"
            if all(value == "the same" for value in status.values()) else
            "upstream's run, and the Swift run's where it is not the same: "
            + ", ".join(f"{model} {status[model]}" for model in lists))
    return [
        markdown_table(["model", "server", "published row", "public items", "ours", "published",
                        "same outcome", "right only here", "right only there", "McNemar p",
                        "published sealed"], rows),
        markdown_table(["model", "server", "tier", "items", "ours", "published"], tiers),
        f"Items whose outcome differs from the published row's ({note}):",
        markdown_table(["model", "server", "type", "items", "which"], differ),
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
    server process's requests (JevBench, then TypeSafe), with the model time each took.

    The two files of a model must record the same capacity: one cache cannot span runs whose
    servers kept different numbers of functions, so such a pair stops the report."""
    swift = {(doc["model"], doc["dataset"]["name"]): doc for doc in docs
             if doc["server"]["name"] == "swift"}
    rows = []
    for model, shapes in SHAPES.items():
        runs = [swift.get((model, name)) for name in ("jevbench", "typesafe102")]
        if not all(runs):
            continue
        recorded = [doc["server"].get("function_capacity", RECORDED_FUNCTION_CAPACITY)
                    for doc in runs]
        if len({str(kept) for kept in recorded}) > 1:
            raise SystemExit(
                f"the Swift {model} runs record different function capacities (JevBench "
                f"{recorded[0]}, TypeSafe {recorded[1]}); the simulation replays one server's "
                "cache over both, so run both datasets on one server with servers.py")
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


# The settings servers.py gives every server of a backend; a result file records them too.
DEFAULT_SETTINGS = {"OPENJEV_BACKEND", "OPENJEV_ENCODER_MODELS"}


def environment_rows(docs: list) -> list:
    """One row per model and server: both datasets ran on one server process."""
    rows, seen = [], {}
    for doc in docs:
        server, hardware = doc["server"], doc["hardware"]
        if server.get("implementation") == "OpenJevSwift":
            code = f"OpenJevSwift {server.get('version')} at {str(server.get('commit'))[:7]}"
            runtime = server.get("runtime") or ""
        elif server["name"] == harness.AUTHOR_SERVER:
            code = (f"{server.get('implementation')} {server.get('version')} at "
                    f"{str(server.get('commit'))[:7]}, published")
            runtime = server.get("runtime") or ""
        else:
            packages = server.get("packages") or {}
            code = (f"openjev {server.get('version')} at {str(server.get('commit'))[:7]}, "
                    f"Python {server.get('python')}")
            if server.get("backend") == "mlx":
                runtime = (f"MLX {packages.get('mlx')} and mlx-vlm {packages.get('mlx-vlm')} on "
                           f"the GPU, {server.get('dtype')} weights")
            else:
                runtime = (f"PyTorch {packages.get('torch')} on the {server.get('device')}, "
                           f"{server.get('dtype')}, {server.get('torch_threads')} threads")
        # the settings a run gave its server beyond the defaults (servers.py --setting)
        extra = [f"{name}={value}" for name, value in (server.get("settings") or {}).items()
                 if name.startswith("OPENJEV_") and name not in DEFAULT_SETTINGS]
        if extra:
            runtime += ", " + ", ".join(extra)
        key = (doc["model"], server["name"])
        if key in seen:
            seen[key][0] += f", {doc['dataset']['name']}"
            continue
        machine = (f"{hardware.get('cpu')}, {hardware.get('memory_gb')} GB, {hardware.get('os')}"
                   if hardware.get("cpu") else hardware.get("platform") or "not recorded")
        seen[key] = [doc["dataset"]["name"], doc["model"], server["name"], code, runtime, machine,
                     (doc["started_utc"] or "")[:10]]
        rows.append(seen[key])
    return rows


def runs_rows(docs: list) -> list:
    """The summary rows, without the timings of an UNTIMED model's runs."""
    rows = harness.summary_rows(docs)
    for doc, row in zip(docs, rows):
        if doc["model"] in UNTIMED:
            row[-3:] = [UNTIMED[doc["model"]]] * 3
    return rows


def render(results: Path, cache: Path) -> str:
    docs = harness.load_docs(result_files(results), cache)
    out = ["### The runs", markdown_table(harness.SUMMARY_HEADER, runs_rows(docs))]
    out += ["### Swift against upstream"] + agreement_tables(docs)
    author = author_tables(docs)
    if author:
        out += ["### Swift against the model author's published run",
                "JevK5's reference is its author's own published v0.2 run, as it is upstream's: "
                "upstream's server reads JevK5's letters from vLLM, which needs an NVIDIA GPU "
                "(D-052). Equal input tokens mean equal prompts."] + author
    conversions = conversion_tables(docs, harness.load_docs(conversion_files(results), cache))
    if conversions:
        out += ["### JevK5's conversions against the author's run",
                "The same runs with each MLX conversion of JevK5 v0.2 that Tools/jevk5/convert.py "
                f"pins, from {CONVERSIONS}/: the default is the main result files' conversion, "
                "and bfloat16 is the unquantized weights. A near-tie, the author's top two less "
                f"than {CLEAR_MARGIN} apart, can turn on bfloat16 rounding alone (D-052)."]
        out += conversions
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
