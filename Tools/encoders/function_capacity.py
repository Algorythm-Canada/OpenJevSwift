#!/usr/bin/env python3
"""How many Core ML functions a Mac's encoder backend should keep loaded (D-042).

    function_capacity.py replay laya-1.0 RESULT.json [RESULT.json ...]
    function_capacity.py serve --backend laya --functions 2 --order shuffled --out DIR
    function_capacity.py report DIR [--backend laya]

`replay` reads JevBench harness result files (Tools/jevbench, issue #61), given in the order one
server answered them, and replays each request's function through CoreMLEncoderModel's
least-recently-used cache: a one-question request runs through the batch-1 function of the
smallest length that holds its `usage.input_tokens`, the warm-up's three questions load
`b16_s128` first, and a function that is not loaded is loaded after the cache is trimmed to one
fewer than its capacity. It prints the loads for every capacity in the recorded order, the reloads
in random orders of the same requests, and the reloads when a share of the requests are batches
(which run through a batch-16 function).

`serve` starts the release build's `openjev serve` for one backend with OPENJEV_ENCODER_FUNCTIONS
(`all` leaves it unset), runs JevBench and the TypeSafe rows through Tools/jevbench/harness.py, in
the datasets' order or shuffled with a fixed seed, reads the server's resident memory with vmmap
after the warm-up and after the reads, and writes the result files and a summary to DIR. `report`
compares the runs in DIR: model time per request, the requests slower than a load's threshold,
the memory, and whether the answers are identical across capacities.

The resident memory is the physical footprint plus the resident pages of the files Core ML maps
each loaded GPU function's weights from (`payload-*.bin`), which the footprint does not count.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JEVBENCH = ROOT / "Tools" / "jevbench"
LENGTHS = {"laya-1.0": [128, 256, 512, 1024], "verdict-1.4": [128, 256, 512]}
MODELS = {"laya": "laya-1.0", "verdict": "verdict-1.4"}
WARM_UP = ["b16_s128"]


def function(batch: int, tokens: int, lengths: list) -> str:
    for length in lengths:
        if length >= tokens:
            return f"b{batch}_s{length}"
    raise ValueError(f"{tokens} tokens: longer than every function")


def replay_loads(calls: list, capacity: int, loaded: list) -> list:
    """For each call, None when its function was loaded, else "first" or "reload"."""
    cache, seen, events = list(loaded), set(loaded), []
    for name in calls:
        if name in cache:
            cache.remove(name)
            cache.append(name)
            events.append(None)
            continue
        del cache[: max(0, len(cache) - (capacity - 1))]
        cache.append(name)
        events.append("reload" if name in seen else "first")
        seen.add(name)
    return events


def answered(paths: list) -> list:
    """The answered items of result files, in order, each with its file's dataset name."""
    items = []
    for path in paths:
        doc = json.loads(Path(path).read_text())
        for item in doc["items"]:
            if item.get("status") == "answered":
                items.append({**item, "dataset": doc["dataset"].get("name")})
    return items


def replay(args) -> None:
    lengths = LENGTHS[args.model]
    items = answered(args.results)
    calls = [function(1, item["usage"]["input_tokens"], lengths) for item in items]
    counts = {name: calls.count(name) for name in sorted(set(calls), key=lambda n: int(n[4:]))}
    every = 2 * len(lengths)
    print(f"{args.model}: {len(calls)} one-question requests; by function: {counts}")
    print("\nRecorded order, warm-up first:\n")
    print("| capacity | first loads | reloads | requests that waited for a load |")
    print("|---|---|---|---|")
    for capacity in range(1, every + 1):
        events = replay_loads(calls, capacity, WARM_UP)
        first, again = events.count("first"), events.count("reload")
        print(f"| {capacity} | {first} | {again} | {100 * (first + again) / len(calls):.1f}% |")

    rng = random.Random(args.seed)
    print(f"\nThe same requests in {args.orders} random orders: reloads per 100 requests, mean "
          "(5th to 95th percentile):\n")
    print("| capacity | reloads |")
    print("|---|---|")
    for capacity in range(1, every + 1):
        shares = []
        for _ in range(args.orders):
            order = calls[:]
            rng.shuffle(order)
            shares.append(100 * replay_loads(order, capacity, WARM_UP).count("reload") / len(order))
        shares.sort()
        low, high = shares[int(0.05 * len(shares))], shares[int(0.95 * len(shares))]
        print(f"| {capacity} | {statistics.mean(shares):.1f} ({low:.1f} to {high:.1f}) |")

    buckets = [int(name.split("_s")[1]) for name in calls]
    print(f"\nStreams of {args.stream} requests drawn from the same lengths, a share of them one "
          f"question and the rest batches; reloads per 100 requests over {args.streams} streams:\n")
    print("| one-question share | " + " | ".join(f"{c}" for c in range(1, every + 1)) + " |")
    print("|---|" + "---|" * every)
    for share in (1.0, 0.75, 0.5, 0.25, 0.0):
        cells = []
        for capacity in range(1, every + 1):
            reloads = 0
            for _ in range(args.streams):
                stream = [f"b{1 if rng.random() < share else 16}_s{rng.choice(buckets)}"
                          for _ in range(args.stream)]
                reloads += replay_loads(stream, capacity, WARM_UP).count("reload")
            cells.append(f"{100 * reloads / (args.streams * args.stream):.1f}")
        print(f"| {share:.2f} | " + " | ".join(cells) + " |")


def resident(pid: int) -> dict:
    text = subprocess.run(["vmmap", str(pid)], capture_output=True, text=True).stdout
    unit = {"K": 1 / 1024, "M": 1, "G": 1024}
    found = re.search(r"Physical footprint:\s+([\d.]+)([KMG])", text)
    footprint = float(found.group(1)) * unit[found.group(2)] if found else 0.0
    copies = 0.0
    for line in text.splitlines():
        if line.startswith("mapped file") and "/payload-" in line:
            sizes = re.search(r"\[\s*([\d.]+)([KMG])\s+([\d.]+)([KMG])", line)
            copies += float(sizes.group(3)) * unit[sizes.group(4)]
    return {"footprint": round(footprint), "weight_copies": round(copies),
            "resident": round(footprint + copies)}


def serve(args) -> None:
    sys.path.insert(0, str(JEVBENCH))
    try:
        import harness
    except ImportError:
        sys.exit(f"{JEVBENCH}/harness.py is missing: serve needs the JevBench harness (issue #61)")
    if args.binary:
        binary = Path(args.binary)
    else:
        # where the release build is, as servers.py finds it
        shown = subprocess.run(["swift", "build", "-c", "release", "--show-bin-path"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        binary = Path(shown or ROOT / ".build" / "release") / "openjev"
    if not binary.is_file():
        sys.exit(f"{binary} is missing; run swift build -c release --product openjev, or --binary")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = f"{args.backend}-{args.functions}-{args.order}"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENJEV_")}
    env.update({"OPENJEV_HOST": "127.0.0.1", "OPENJEV_PORT": str(port),
                "OPENJEV_BACKEND": args.backend, "OPENJEV_LOG_LEVEL": "info"})
    if args.encoder_models:
        env["OPENJEV_ENCODER_MODELS"] = str(Path(args.encoder_models).expanduser())
    if args.functions != "all":
        env["OPENJEV_ENCODER_FUNCTIONS"] = args.functions
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    with open(out / f"{label}.log", "w") as log:
        process = subprocess.Popen([str(binary), "serve"], env=env, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT)
    try:
        while True:
            if process.poll() is not None:
                sys.exit(f"the server exited with status {process.returncode}; see {log.name}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        summary = {"label": label, "backend": args.backend, "functions": args.functions,
                   "order": args.order, "seed": args.seed,
                   "startup_s": round(time.monotonic() - started, 2),
                   "memory_mb": {"after_warm_up": resident(process.pid)}}
        cache = Path(args.cache) if args.cache else harness.default_cache()
        rng = random.Random(args.seed)
        for name in ("jevbench", "typesafe102"):
            dataset = harness.load_dataset(name, cache)
            if args.order == "shuffled":
                tasks = list(dataset.tasks)
                rng.shuffle(tasks)
                dataset.tasks = tasks
            doc = harness.run_dataset(dataset, base, MODELS[args.backend], label, cache,
                                      server_info={"functions": args.functions,
                                                   "order": args.order},
                                      progress=lambda line: None)
            harness.write_result(out / f"{label}-{name}.json", doc)
        summary["memory_mb"]["after_reads"] = resident(process.pid)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    (out / f"{label}-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


def report(args) -> None:
    out = Path(args.out)
    model_name = MODELS[args.backend]
    rows, runs = [], {}
    for path in sorted(out.glob(f"{args.backend}-*-summary.json")):
        summary = json.loads(path.read_text())
        label = summary["label"]
        items = answered([out / f"{label}-{name}.json" for name in ("jevbench", "typesafe102")])
        runs[label] = items
        model = sorted(item["timing"]["server"]["model"] for item in items)

        def rank(q: float) -> float:
            # nearest rank: the smallest value with at least q of the values at or below it
            return model[max(0, math.ceil(q * len(model)) - 1)]

        memory = summary["memory_mb"]
        rows.append([label, len(model), sum(m > args.threshold for m in model),
                     f"{statistics.median(model):.1f}", f"{rank(0.95):.1f}", f"{rank(0.99):.1f}",
                     f"{statistics.mean(model):.1f}", f"{sum(model) / 1000:.1f}",
                     f"{memory['after_warm_up']['resident']:,}",
                     f"{memory['after_reads']['footprint']:,}",
                     f"{memory['after_reads']['weight_copies']:,}",
                     f"{memory['after_reads']['resident']:,}"])
    print(f"| run | requests | model time over {args.threshold:.0f} ms | median ms | p95 ms | "
          "p99 ms | mean ms | total model s | resident after warm-up MB | footprint after MB | "
          "weight copies after MB | resident after MB |")
    print("|" + "---|" * 12)
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")

    # Each order served with a cap and with every function: the same requests in the same order,
    # so the cache's replay says which requests waited for a load under the cap.
    for capped in sorted(runs):
        backend, functions, order = capped.split("-", 2)
        kept = f"{backend}-all-{order}"
        if functions == "all" or kept not in runs:
            continue
        a, b = runs[capped], runs[kept]
        if [(i["dataset"], i["id"]) for i in a] != [(i["dataset"], i["id"]) for i in b]:
            print(f"\n{capped} and {kept} answered different requests or orders; not compared")
            continue
        same = sum(x["answer"] == y["answer"] for x, y in zip(a, b))
        calls = [function(1, i["usage"]["input_tokens"], LENGTHS[model_name]) for i in a]
        events = replay_loads(calls, int(functions), WARM_UP)
        again = [k for k, event in enumerate(events) if event == "reload"]
        print(f"\n{order}: {same} of {len(a)} answers identical with {functions} functions and "
              f"with every function; at {functions} the replay has {events.count('first')} first "
              f"loads and {len(again)} reloads")
        if again:
            capped_ms = [a[k]["timing"]["server"]["model"] for k in again]
            kept_ms = [b[k]["timing"]["server"]["model"] for k in again]
            print(f"  the {len(again)} requests that reload: median {statistics.median(capped_ms):.0f}"
                  f" ms ({min(capped_ms):.0f} to {max(capped_ms):.0f}) with {functions} "
                  f"functions, {statistics.median(kept_ms):.0f} ms with every function")
        total_a = sum(i["timing"]["server"]["model"] for i in a) / 1000
        total_b = sum(i["timing"]["server"]["model"] for i in b) / 1000
        print(f"  model time {total_a:.1f} s with {functions} functions, {total_b:.1f} s with "
              "every function")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("replay", help="replay result files through the cache")
    one.add_argument("model", choices=tuple(LENGTHS))
    one.add_argument("results", nargs="+")
    one.add_argument("--orders", type=int, default=2000)
    one.add_argument("--streams", type=int, default=300)
    one.add_argument("--stream", type=int, default=1000)
    one.add_argument("--seed", type=int, default=61)
    two = commands.add_parser("serve", help="one server run through the JevBench harness")
    two.add_argument("--backend", choices=tuple(MODELS), default="laya")
    two.add_argument("--functions", default="all", help="OPENJEV_ENCODER_FUNCTIONS, or all")
    two.add_argument("--order", choices=("dataset", "shuffled"), default="dataset")
    two.add_argument("--seed", type=int, default=61)
    two.add_argument("--binary", help="the openjev binary (default: the release build's)")
    two.add_argument("--encoder-models", help="OPENJEV_ENCODER_MODELS for the server")
    two.add_argument("--cache", help="the JevBench datasets' cache (default: the harness's)")
    two.add_argument("--out", required=True)
    three = commands.add_parser("report", help="compare the serve runs in a folder")
    three.add_argument("out")
    three.add_argument("--backend", choices=tuple(MODELS), default="laya")
    three.add_argument("--threshold", type=float, default=300.0,
                       help="model time in ms above which a request counts as a load")
    args = parser.parse_args()
    {"replay": replay, "serve": serve, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
