#!/usr/bin/env python3
"""JevBench harness for OpenJevSwift (issue #61): run, score and compare /v1/systemone servers.

Two datasets, downloaded at pinned revisions into a cache outside the repository and checked by
size and SHA-256 before use:

- ``jevbench``: JevBench v1's 231 public items, fstandhartinger/jevbench at JEVBENCH_COMMIT.
- ``typesafe102``: the 102 rows of TypeSafe's public evaluations that SemIf compares with Jev
  (TheoLeeCJ/SemIf-OpenJev at SEMIF_COMMIT), rebuilt by SemIf's own builder from the four case
  snapshots that evals.typesafe.ai publishes.

Every item becomes one ``/v1/systemone`` request through JevBench's own ``typesafe`` adapter, and
the answers are scored by JevBench's own ``scoring``, ``metrics`` and ``summarize`` code. Both are
vendored unchanged under vendor/ (vendor/README.md). The TypeSafe rows also get SemIf's own metrics
(``evaluate_external.type_safe``). Standard library only; README.md gives the commands that wrote
results/.

Commands: fetch, run, summary, compare, published, report, calibration.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor"
RESULTS = HERE / "results"
sys.path.insert(0, str(VENDOR / "jevbench"))

from jevbench import scoring as jb_scoring  # noqa: E402
from jevbench import summarize as jb_summarize  # noqa: E402
from jevbench import tasks as jb_tasks  # noqa: E402
from jevbench.adapters import base as jb_base  # noqa: E402
from jevbench.adapters import typesafe as jb_typesafe  # noqa: E402

SCHEMA = "openjevswift-jevbench-result/1"
USER_AGENT = "OpenJevSwift-jevbench/1 (+https://github.com/Algorythm-Canada/OpenJevSwift)"

# JevBench v1. The three split files hash as the benchmark's own datasets/manifest.json says.
JEVBENCH_REPO = "fstandhartinger/jevbench"
JEVBENCH_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
# path in the repository -> (tier, size in bytes, SHA-256). The tiers are the names the
# benchmark's per-task results use: the original items are its "standard" tier.
JEVBENCH_SPLITS = {
    "datasets/public/easy.jsonl": (
        "easy", 37220, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "datasets/public/original.jsonl": (
        "standard", 57237, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "datasets/public/hard.jsonl": (
        "hard", 651848, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
JEVBENCH_ITEMS = 231
# The published outcome (correct or wrong) of every public item per system (JevBench v1.3.0), and
# the newest published board (v1.4.2.2) with each system's tier, hard-tier and sealed aggregates.
PUBLISHED_PER_TASK = ("results/v1.2/jevbench-v1.2-per-task.json", 908122,
                      "c772b3b85809f483449ec2a6ae0272347371ed5f40f7faff8394ecf4ee22ae7c")
PUBLISHED_BOARD = ("results/v1.4.2.2/jevbench-v1.4.2.2-results.json", 622975,
                   "7f39b2f742a69ded7384fb7eb4c54daa9cf67b26e72e25133c6da1f8e49cf570")

# SemIf's TypeSafe subset. The selection manifest names the 102 rows and pins the parsed payload of
# each case snapshot; the snapshots themselves are TypeSafe's and are fetched from its site.
SEMIF_REPO = "TheoLeeCJ/SemIf-OpenJev"
SEMIF_COMMIT = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
SEMIF_SELECTION = ("benchmarks/manifests/source-selection.jsonl", 192604,
                   "81d98a4195971e1fe046673928730b7112a6a7677baf046505761f2504b424c7")
TYPESAFE_EVALS = "https://evals.typesafe.ai/"
# workflow -> (the file its viewer page loads, size, SHA-256), as served on 2026-10-01.
TYPESAFE_SNAPSHOTS = {
    "invoice_processing": ("invoice_processing-cases.js?v=8c2f8869", 884891,
                           "8c2f886978a30e637d577cd0713b3cf12bb622ef7210fcc9a4be47540a6df27f"),
    "agent_trace_observability": ("agent_trace_observability-cases.js?v=4a3821a2", 202830,
                                  "4a3821a24366dc9830e12b481a23c96d0ab70e3c473b59551b88e8d0c4b615b3"),
    "customer_service": ("customer_service-cases.js?v=066f789b", 225834,
                         "066f789bcc17ae906a17fc37ad4fd2f2bb1e881245aeb76b524715bc962a2493"),
    "security_incidents": ("security_incidents-cases.js?v=6c96b19f", 89160,
                           "6c96b19f192d07004613afc281a31712aa7d04bd75e31a3afe2322174ade0645"),
}
TYPESAFE_ROWS = 102
# What SemIf's builder writes from the pinned selection and snapshots: the rows themselves are
# pinned, so a stale or edited cached copy is rebuilt rather than used.
TYPESAFE_BUILT = ("typesafe102.jsonl", 1431505,
                  "734bfa7c56a1e4616e3dee56a71b3dd44b2717def0644622f00c6006d3e80f75")
# The models TypeSafe's snapshots publish answers for, as SemIf's builder keys them.
TYPESAFE_PUBLISHED = ("typesafe", "opus", "sol")

# Files under vendor/: path there -> (repository, commit, path in the repository, SHA-256). Each is
# an unchanged copy; `fetch` compares them with the pinned commits and the smoke test with these
# digests. vendor/jevbench/jevbench/adapters/__init__.py is this project's own package marker.
VENDORED = {
    "jevbench/LICENSE": (JEVBENCH_REPO, JEVBENCH_COMMIT, "LICENSE",
                         "3e5beed774bb0bcbfb2fcf24ba9554212c6ae112937d308040989465ab0c5784"),
    "jevbench/jevbench/__init__.py": (JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/__init__.py",
                                      "64c28dc1534502c429a3b67eee79ab7909fab995b7d116c9b089f0f3bb8652f7"),
    "jevbench/jevbench/tasks.py": (JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/tasks.py",
                                   "b9c2f8ba9a7301519656d79d47c31b8ab0542a5545498ce3b339d5e93b976812"),
    "jevbench/jevbench/scoring.py": (JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/scoring.py",
                                     "6aa17d212ea20a3eeb6876149367842e8462c11dcb54d73df062fb406368331e"),
    "jevbench/jevbench/metrics.py": (JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/metrics.py",
                                     "4038c8423ad1cf28601afb6babe777b71f08e379b9ca125b09b52c0ac7d60515"),
    "jevbench/jevbench/summarize.py": (JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/summarize.py",
                                       "aa5f24da407d004cd80bad1cd957b6987aba152591084c096d1de51dd9775cbe"),
    "jevbench/jevbench/adapters/base.py": (
        JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/adapters/base.py",
        "fdddc42e4f71d774096f806c4ee844cc4c4b428385f01a017cb64153d7c089f4"),
    "jevbench/jevbench/adapters/typesafe.py": (
        JEVBENCH_REPO, JEVBENCH_COMMIT, "jevbench/adapters/typesafe.py",
        "780f879416479b645ee519bb85d0cce722d0df7aa2b41a029de92b76f2593e51"),
    "semif/LICENSE": (SEMIF_REPO, SEMIF_COMMIT, "LICENSE",
                      "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"),
    "semif/build_typesafe.py": (SEMIF_REPO, SEMIF_COMMIT, "benchmarks/build_typesafe.py",
                                "0e7a22842c879de34e2f7f657974d1907c696d0df6c720972b7c60bd7b8f1efb"),
    "semif/evaluate_external.py": (SEMIF_REPO, SEMIF_COMMIT, "benchmarks/evaluate_external.py",
                                   "9a459430882949b7ad8e4baae5c5d2c9127d9eef1e7e12daedb225115c9a3f47"),
}

# The benchmark's published row for each model this port serves, and how that row was produced
# where it differs from a run of this harness against upstream's server.
PUBLISHED_ROWS = {
    "verdict-1.4": {
        "key": "openjev-verdict-1.4",
        "setup": "the same weights (heman10x/rlcd-modernbert-151m 8af2496) through the author's v1.4 "
                 "engine (Heman10x-NGU/Verdict-open-jev 30f1556) and JevBench's verdict_local adapter, "
                 "PyTorch on 4 threads of a Ryzen 5 3600",
    },
    "laya-1.0": {
        "key": "laya",
        "setup": "another checkpoint: convaiinnovations/laya (the repository root), not "
                 "laya-typed-decisions, through the laya package with a 512-token budget and "
                 "JevBench's laya_local adapter, PyTorch on 4 threads of a Ryzen 5 3600",
    },
    "openjev-0.1": {
        "key": "openjev-razorback16",
        "setup": "upstream on vLLM with the NVFP4 weights, on a RunPod RTX PRO 4500 Blackwell, "
                 "through JevBench's typesafe adapter",
    },
}

# A model author's own published run of the public items, for a model whose upstream server this
# Mac cannot run: JevK5's, the reference the Swift `jevk5` backend is compared with, as upstream
# compared its vLLM path with it (upstream's JevK5 server reads its letters from vLLM, which needs an
# NVIDIA GPU; D-052). `author-run` turns the file into a result file whose server is AUTHOR_SERVER.
AUTHOR_SERVER = "author"
# The run is the file as the commit that added it holds it, made by that commit's code, as the
# author's bench/SUBMISSION.md there describes: the package's in-process adapter, jevk5_direct,
# no server, on the checkpoint revision it names, whose files are the v0.2 tag's (D-052).
AUTHOR_RUNS = {
    "jevk5-0.2": {
        "repo": "allebee/jevk5", "commit": "85238d7be5527370c43206fe54cd752eb3134c1b",
        "version": "0.2.0", "path": "results/public231/jevk5-v0.2.jsonl", "bytes": 221596,
        "sha256": "571872b233bb48d26ebd735b4573aa8f28bd4828471bb89b4b9e1651dcb2e265",
        "checkpoint": "alibiserikbay/JevK5@3c673298eb7f2dc7cb98019262c55b1fac01a0bc",
        "runtime": "the jevk5 package 0.2.0's in-process adapter, jevk5_direct (transformers, "
                   "the bf16 weights of alibiserikbay/JevK5 at 3c67329, CUDA graphs), through "
                   "JevBench's runner at 0caa1d0, as the author published it",
        "hardware": "one NVIDIA H100, batch 1 (the author's bench/SUBMISSION.md)",
    },
}

# What each model accepts per question (upstream's build_schema): an item outside this is skipped
# for that model, never sent, and counted. Verdict's head has 25 logits, the last one for its
# "insufficient evidence" option (encoders.py, VerdictEngine.max_choices).
JEV_LIMITS = {"choice": 255, "score": 10}
SHAPE_LIMITS = {"verdict-1.4": {"choice": 24, "score": 10}}

FAILED_SCORE = {"valid": False, "strict_valid": False, "renormalized": False, "correct": False,
                "predicted": None}
COST_BASIS = "local_no_provider_tariff"


class PinError(RuntimeError):
    """A downloaded or vendored file does not have the pinned size and SHA-256."""


# Files, downloads and pins


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache() -> Path:
    if os.environ.get("JEVBENCH_CACHE"):
        return Path(os.environ["JEVBENCH_CACHE"]).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "OpenJevSwift" / "jevbench"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "OpenJevSwift" / "jevbench"


def display_path(path) -> str:
    """A path as a result file records it: relative to the repository or to ~, so that no file
    names the machine's home folder."""
    path = Path(path).expanduser().absolute()
    for base, prefix in ((ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def github_raw(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def is_pinned(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == digest


def require_pinned(path: Path, size: int, digest: str) -> Path:
    if not is_pinned(path, size, digest):
        raise PinError(f"{path} is missing or not the pinned file; run fetch")
    return path


def download(url: str, dest: Path, size: int, digest: str) -> Path:
    """Download url to dest unless dest already holds the pinned bytes; refuse any other bytes."""
    if is_pinned(dest, size, digest):
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(size + 1)
    if len(data) != size or sha256_bytes(data) != digest:
        raise PinError(f"{url} changed: expected {size} bytes with SHA-256 {digest}, received "
                       f"{len(data)} bytes with SHA-256 {sha256_bytes(data)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, dest)
    return dest


def check_vendored() -> dict:
    """The SHA-256 of every vendored file, after checking each against its pin."""
    found = {}
    for path, (_, _, _, digest) in VENDORED.items():
        actual = sha256_file(VENDOR / path)
        if actual != digest:
            raise PinError(f"vendor/{path} has SHA-256 {actual}, pinned {digest}")
        found[path] = actual
    return found


def compare_vendored_with_sources() -> None:
    """Download every vendored file at its pinned commit and require the same bytes."""
    for path, (repo, commit, source, digest) in VENDORED.items():
        request = urllib.request.Request(github_raw(repo, commit, source),
                                         headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if sha256_bytes(data) != digest or (VENDOR / path).read_bytes() != data:
            raise PinError(f"vendor/{path} differs from {repo}@{commit[:7]}:{source}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semif_builder():
    return load_module("semif_build_typesafe", VENDOR / "semif" / "build_typesafe.py")


def semif_evaluator():
    return load_module("semif_evaluate_external", VENDOR / "semif" / "evaluate_external.py")


def jevbench_root(cache: Path) -> Path:
    return cache / f"jevbench-{JEVBENCH_COMMIT[:7]}"


def typesafe_root(cache: Path) -> Path:
    return cache / f"typesafe102-semif-{SEMIF_COMMIT[:7]}"


def fetch_jevbench(cache: Path) -> None:
    root = jevbench_root(cache)
    for path, (_, size, digest) in JEVBENCH_SPLITS.items():
        download(github_raw(JEVBENCH_REPO, JEVBENCH_COMMIT, path), root / path, size, digest)
    for path, size, digest in (PUBLISHED_PER_TASK, PUBLISHED_BOARD):
        download(github_raw(JEVBENCH_REPO, JEVBENCH_COMMIT, path), root / path, size, digest)


def author_run_path(cache: Path, model: str) -> Path:
    run = AUTHOR_RUNS[model]
    return cache / f"{run['repo'].replace('/', '--')}-{run['commit'][:7]}" / run["path"]


def fetch_author_runs(cache: Path) -> None:
    for model, run in AUTHOR_RUNS.items():
        download(github_raw(run["repo"], run["commit"], run["path"]), author_run_path(cache, model),
                 run["bytes"], run["sha256"])


def fetch_typesafe(cache: Path) -> Path:
    """Download SemIf's selection and TypeSafe's four snapshots; rebuild the 102 rows with SemIf's
    own builder, which checks each snapshot's parsed payload against the selection's hash, unless
    the cache already holds the pinned rows."""
    root = typesafe_root(cache)
    path, size, digest = SEMIF_SELECTION
    selection = download(github_raw(SEMIF_REPO, SEMIF_COMMIT, path), root / "source-selection.jsonl",
                         size, digest)
    for workflow, (name, size, digest) in TYPESAFE_SNAPSHOTS.items():
        download(TYPESAFE_EVALS + name, root / "sources" / f"typesafe-{workflow}-cases.js", size,
                 digest)
    name, size, digest = TYPESAFE_BUILT
    output = root / name
    if not is_pinned(output, size, digest):
        building = root / (name + ".part")
        building.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(VENDOR / "semif" / "build_typesafe.py"),
                        "--source-dir", str(root / "sources"), "--selection", str(selection),
                        "--output", str(building)], check=True, stdout=subprocess.DEVNULL)
        if not is_pinned(building, size, digest):
            raise PinError(f"SemIf's builder wrote {sha256_file(building)} from the pinned "
                           f"snapshots, where {digest} is pinned")
        os.replace(building, output)
    return output


# Datasets


class Dataset:
    """The items of one dataset as JevBench Task objects, with what a result file records of it.

    ``tiers`` maps an item to its tier (JevBench's easy, standard and hard; "typesafe102" for the
    TypeSafe rows), and ``gold`` holds SemIf's built row for each TypeSafe item.
    """

    def __init__(self, name: str, tasks: list, tiers: dict, meta: dict, gold: dict | None = None):
        self.name = name
        self.tasks = tasks
        self.tiers = tiers
        self.meta = meta
        self.gold = gold or {}
        self.by_id = {task.id: task for task in tasks}
        if len(self.by_id) != len(tasks):
            raise ValueError(f"{name}: duplicate item ids")


def load_jevbench(cache: Path, fetch: bool = True) -> Dataset:
    if fetch:
        fetch_jevbench(cache)
    root = jevbench_root(cache)
    tasks, tiers, files = [], {}, []
    for path, (tier, size, digest) in JEVBENCH_SPLITS.items():
        require_pinned(root / path, size, digest)
        for task in jb_tasks.load_jsonl(str(root / path)):
            tasks.append(task)
            tiers[task.id] = tier
        files.append({"path": path, "bytes": size, "sha256": digest})
    if len(tasks) != JEVBENCH_ITEMS:
        raise ValueError(f"expected {JEVBENCH_ITEMS} public items, found {len(tasks)}")
    meta = {"name": "jevbench", "title": "JevBench v1, public items",
            "source": f"https://github.com/{JEVBENCH_REPO}", "commit": JEVBENCH_COMMIT,
            "files": files, "items": len(tasks),
            "license": "MIT, Copyright (c) 2026 Florian Standhartinger and contributors"}
    return Dataset("jevbench", tasks, tiers, meta)


def load_item_file(path: Path) -> Dataset:
    """A JSONL file of items in JevBench's task format, for trial runs and the smoke test. A line
    may carry a "tier"; the others go to the tier "custom"."""
    tasks, tiers = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            task = jb_tasks.Task.from_dict(raw)
            tasks.append(task)
            tiers[task.id] = raw.get("tier", "custom")
    meta = {"name": "items", "title": f"items from {path.name}", "source": display_path(path),
            "files": [{"path": path.name, "bytes": path.stat().st_size,
                       "sha256": sha256_file(path)}], "items": len(tasks), "license": None}
    return Dataset("items", tasks, tiers, meta)


def typesafe_task(row: dict, question: dict, document) -> object:
    """One of SemIf's TypeSafe rows as a JevBench Task with TypeSafe's own question and document.

    The request carries the snapshot's question (type, instructions, criteria) and its document,
    as TypeSafe asked Jev, rather than SemIf's prompt rendering of them. A noul's labels are
    JevBench's ["no", "yes"]; a choice's are its option ids in the criteria's order, which SemIf's
    builder checks against its selection.
    """
    option_ids = [option["id"] for option in row["options"]]
    if question["type"] != row["primitive"]:
        raise ValueError(f"{row['id']}: snapshot question is {question['type']}, row is "
                         f"{row['primitive']}")
    if row["primitive"] == "noul":
        labels, expected = ["no", "yes"], "yes" if option_ids[row["label"]] == "true" else "no"
    else:
        if list(question.get("criteria") or {}) != option_ids:
            raise ValueError(f"{row['id']}: snapshot options differ from the row's")
        labels, expected = option_ids, option_ids[row["label"]]
    spec = {"type": question["type"], "instructions": question["instructions"]}
    if question.get("criteria") is not None:
        spec["criteria"] = question["criteria"]
    task = jb_tasks.Task(id=row["id"], family=row["family"], state=document, question=spec,
                         labels=labels, expected=expected, split="public")
    task.validate()
    return task


def load_typesafe(cache: Path, fetch: bool = True) -> Dataset:
    root = typesafe_root(cache)
    if fetch:
        fetch_typesafe(cache)
    # the rows and the snapshots the questions and documents come from, checked on every load
    built = require_pinned(root / TYPESAFE_BUILT[0], *TYPESAFE_BUILT[1:])
    for workflow, (_, size, digest) in TYPESAFE_SNAPSHOTS.items():
        require_pinned(root / "sources" / f"typesafe-{workflow}-cases.js", size, digest)
    rows = [json.loads(line) for line in built.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != TYPESAFE_ROWS:
        raise ValueError(f"expected {TYPESAFE_ROWS} TypeSafe rows, found {len(rows)}")
    parse = semif_builder().parse_payload
    payloads = {workflow: parse(root / "sources" / f"typesafe-{workflow}-cases.js")["eval"]
                for workflow in TYPESAFE_SNAPSHOTS}
    tasks, tiers, gold = [], {}, {}
    for row in rows:
        source = row["provenance"]
        evaluation = payloads[source["workflow"]]
        task = typesafe_task(row, evaluation["questions"][source["question_index"]],
                             evaluation["documents"][source["document_index"]])
        tasks.append(task)
        tiers[task.id] = "typesafe102"
        gold[task.id] = row
    meta = {"name": "typesafe102",
            "title": "TypeSafe public evaluations, SemIf's 102-row subset",
            "source": f"https://github.com/{SEMIF_REPO}", "commit": SEMIF_COMMIT,
            "files": [{"path": SEMIF_SELECTION[0], "bytes": SEMIF_SELECTION[1],
                       "sha256": SEMIF_SELECTION[2]}]
                     + [{"path": TYPESAFE_EVALS + name, "bytes": size, "sha256": digest}
                        for name, size, digest in TYPESAFE_SNAPSHOTS.values()],
            "built_sha256": TYPESAFE_BUILT[2], "items": len(tasks),
            "license": "SemIf's selection and builder: MIT, Copyright (c) 2026 TheoLeeCJ. "
                       "TypeSafe's snapshots carry no license grant: no text, reference answer or "
                       "published distribution of theirs is stored in a result file."}
    return Dataset("typesafe102", tasks, tiers, meta, gold)


def load_dataset(name: str, cache: Path, fetch: bool = True) -> Dataset:
    if name == "jevbench":
        return load_jevbench(cache, fetch)
    if name == "typesafe102":
        return load_typesafe(cache, fetch)
    raise ValueError(f"unknown dataset {name!r}")


# Running a server


def shape_skip_reason(task, model: str) -> str | None:
    """Why `model` cannot be asked this item's question, or None when it can."""
    limits = SHAPE_LIMITS.get(model, JEV_LIMITS)
    kind, criteria = task.question["type"], task.question.get("criteria")
    if kind in ("choice", "score") and criteria is not None and len(criteria) > limits[kind]:
        unit = "options" if kind == "choice" else "levels"
        return f"{kind} with {len(criteria)} {unit}; {model} takes at most {limits[kind]}"
    return None


def parse_server_timing(value: str | None) -> dict | None:
    """`model;dur=41.2, server;dur=2.8, total;dur=44.0` as {"model": 41.2, ...} in milliseconds."""
    if not value:
        return None
    timing = {}
    for part in value.split(","):
        name, _, rest = part.strip().partition(";")
        for parameter in rest.split(";"):
            key, _, number = parameter.strip().partition("=")
            if key == "dur":
                try:
                    timing[name] = float(number)
                except ValueError:
                    pass
    return timing or None


class RecordingPost:
    """Stands in for ``http_post_json`` in JevBench's typesafe adapter (adapters/base.py).

    It posts the same bytes with the same headers and returns the same (status, parsed body,
    latency), so the adapter's request mapping and answer parsing run unchanged. It also keeps
    what the adapter drops: the digest and size of the body sent, the response's server-timing and
    x-request-id headers, and the HTTP time.
    """

    def __init__(self):
        self.exchange = None

    def __call__(self, url, body, headers, timeout_s=120.0):
        data = json.dumps(body).encode("utf-8")
        self.exchange = {"body_sha256": sha256_bytes(data), "body_bytes": len(data)}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status, payload, response_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            status, payload, response_headers = error.code, error.read(), error.headers
        except Exception as error:  # a network error has no HTTP status, as in base.http_post_json
            self.exchange["error"] = f"{type(error).__name__}: {error}"
            raise ConnectionError(f"{type(error).__name__}: {error}") from error
        latency = time.perf_counter() - started
        text = payload.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        self.exchange.update({
            "http_ms": round(latency * 1000, 3),
            "server_timing": parse_server_timing(response_headers.get("server-timing")),
            "request_id": response_headers.get("x-request-id"),
        })
        return status, parsed, latency


def state_digest(state) -> dict:
    """The digest and length of the state as the server serialises it for the model."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return {"sha256": sha256_bytes(text.encode("utf-8")), "chars": len(text)}


def request_record(dataset: str, body: dict, task) -> dict:
    """What a result file keeps of a request. JevBench's questions are kept whole (MIT); its states
    are long (480 KB over the hard tier), so only their digest is kept. Of a TypeSafe row only ids
    and digests are kept: TypeSafe's text carries no license grant."""
    record = {"model": body["model"], "state": state_digest(body["state"])}
    if dataset == "typesafe102":
        question = body["questions"]["decision"]
        record["question"] = {"type": question["type"], "options": task.labels}
    else:
        record["questions"] = body["questions"]
    return record


def answer_of(raw) -> dict | None:
    if isinstance(raw, dict) and isinstance(raw.get("answers"), dict):
        answer = raw["answers"].get("decision")
        return answer if isinstance(answer, dict) else None
    return None


def outcome(result) -> str:
    """answered, refused (a 4xx other than an authentication error or 429) or failed."""
    if result.ok:
        return "answered"
    if result.status is not None and 400 <= result.status < 500 and result.status not in (
            401, 403, 429):
        return "refused"
    return "failed"


def base_item(task, dataset: Dataset, published: dict | None) -> dict:
    """The fields every item of a result file carries, whatever happened to it. A TypeSafe item
    keeps no expected label: it is TypeSafe's reference answer."""
    item = {"id": task.id, "tier": dataset.tiers[task.id], "family": task.family,
            "type": task.question["type"], "labels": task.labels}
    if dataset.name != "typesafe102":
        item.update({"group": task.group, "expected": task.expected})
    if published is not None:
        item["published"] = published
    return item


def run_item(adapter, recorder: RecordingPost, task, dataset: Dataset, model: str,
             published: dict | None) -> dict:
    """Ask the server one item through JevBench's adapter and score the answer as its Runner does."""
    item = base_item(task, dataset, published)
    reason = shape_skip_reason(task, model)
    if reason:
        item.update({"status": "skipped", "skip_reason": reason})
        return item
    recorder.exchange = None
    started = time.perf_counter()
    try:
        result = adapter.run(task)
    except Exception as error:  # as jevbench.runner.Runner.run_task records a crash
        result = jb_base.DecisionResult(adapter.name, False, error=type(error).__name__)
    wall = time.perf_counter() - started
    exchange = recorder.exchange or {}
    scored = jb_scoring.score_task(result.probs or {}, task) if result.ok else FAILED_SCORE
    item.update({
        "status": outcome(result),
        "http_status": result.status,
        "request": {**request_record(dataset.name, adapter.build_request(task), task),
                    "body_sha256": exchange.get("body_sha256"),
                    "body_bytes": exchange.get("body_bytes")},
        "answer": answer_of(result.raw),
        "usage": result.usage or None,
        "probabilities": result.probs,
        "valid": scored["valid"],
        "strict_valid": scored.get("strict_valid", False),
        "renormalized": scored.get("renormalized", False),
        "predicted": scored.get("predicted"),
        "error": error_record(dataset.name, result, scored),
        "timing": {"wall_ms": round(wall * 1000, 3), "http_ms": exchange.get("http_ms"),
                   "server": exchange.get("server_timing")},
        "request_id": exchange.get("request_id"),
    })
    if dataset.name != "typesafe102":
        # a TypeSafe row's correctness would give away TypeSafe's reference answer; a reader
        # scores those rows against the cached dataset (with_correctness)
        item["correct"] = scored.get("correct")
    if item["status"] != "answered":
        item["response"] = response_record(dataset.name, result.raw)
    return item


def error_record(dataset: str, result, scored: dict) -> str | None:
    """The error a result file keeps. JevBench's adapter writes a refusal as `HTTP {status}:` and
    the first 300 characters of the body, and a 422 body echoes the request's values, so of a
    TypeSafe row only the status and a plain detail message are kept."""
    error = result.error or scored.get("error")
    if dataset != "typesafe102" or not error or not error.startswith("HTTP "):
        return error
    detail = result.raw.get("detail") if isinstance(result.raw, dict) else None
    return f"HTTP {result.status}" + (f": {detail}" if isinstance(detail, str) else "")


def response_record(dataset: str, raw):
    """What a result file keeps of a response that is not an answer. A 422 echoes the offending
    input, so of a TypeSafe row only a plain detail message is kept."""
    if dataset == "typesafe102":
        detail = raw.get("detail") if isinstance(raw, dict) else None
        return {"detail": detail} if isinstance(detail, str) else None
    if isinstance(raw, str):
        return raw[:500]
    return raw if isinstance(raw, dict) else None


def get_json(url: str, timeout: float = 30.0, key: str | None = None):
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def hardware() -> dict:
    """The machine the harness and a local server run on."""
    info = {"platform": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        def sysctl(name):
            try:
                return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                                      check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None

        def sw_vers(flag):
            try:
                return subprocess.run(["sw_vers", flag], capture_output=True, text=True,
                                      check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None

        memory = sysctl("hw.memsize")
        info.update({
            "model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"),
            "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
            "memory_gb": round(int(memory) / 2**30) if memory and memory.isdigit() else None,
            "os": f"macOS {sw_vers('-productVersion')} ({sw_vers('-buildVersion')})",
        })
    return info


def published_outcomes(cache: Path, model: str) -> tuple[dict | None, dict]:
    """The benchmark's published row for `model` and its outcome per public item."""
    row = PUBLISHED_ROWS.get(model)
    if row is None:
        return None, {}
    path = jevbench_root(cache) / PUBLISHED_PER_TASK[0]
    if not path.is_file() or sha256_file(path) != PUBLISHED_PER_TASK[2]:
        raise PinError(f"{path} is missing or not the pinned file; run fetch")
    system = json.loads(path.read_text(encoding="utf-8"))["systems"][row["key"]]
    outcomes = {item: {"outcome": code, "latency_s": latency}
                for item, (code, latency) in system["public_tasks"].items()}
    info = {"key": row["key"], "display": system["display"], "setup": row["setup"],
            "source": f"{JEVBENCH_REPO}@{JEVBENCH_COMMIT[:7]}:{PUBLISHED_PER_TASK[0]}"}
    return info, outcomes


def run_dataset(dataset: Dataset, base_url: str, model: str, server: str, cache: Path,
                key_env: str = "", server_info: dict | None = None, timeout: float = 120.0,
                ids: set | None = None, progress=print) -> dict:
    """Run every item of `dataset` against one server, one request at a time, and return the
    result document. The stop rule is the benchmark runner's: a 401, 403 or 429, or three failures
    in a row, ends the run and leaves the rest unattempted."""
    vendored = check_vendored()  # before the first request: a changed copy scores nothing
    adapter = jb_typesafe.TypeSafeAdapter(endpoint=base_url, model=model, key_env=key_env,
                                          timeout_s=timeout)
    recorder = RecordingPost()
    # the adapter's one HTTP call, replaced for this run; requests go one at a time
    jb_typesafe.http_post_json = recorder
    key = os.environ.get(key_env) if key_env else None
    try:
        listing = get_json(base_url.rstrip("/") + "/v1/models", key=key)
    except (OSError, ValueError) as error:
        listing = {"error": f"{type(error).__name__}: {error}"}
    published_row, outcomes = (published_outcomes(cache, model) if dataset.name == "jevbench"
                               else (None, {}))
    tasks = [task for task in dataset.tasks if ids is None or task.id in ids]
    started_at = datetime.datetime.now(datetime.timezone.utc)
    started = time.perf_counter()
    items, failures, stopped = [], 0, None
    for index, task in enumerate(tasks, 1):
        if stopped:
            items.append({**base_item(task, dataset, outcomes.get(task.id)),
                          "status": "unattempted"})
            continue
        item = run_item(adapter, recorder, task, dataset, model, outcomes.get(task.id))
        items.append(item)
        failures = failures + 1 if item["status"] == "failed" else 0
        if item.get("http_status") in (401, 403, 429) or failures >= 3:
            stopped = f"stopped after {task.id}: {item.get('error')}"
            progress(f"STOP: {stopped}")
        if index % 25 == 0 or index == len(tasks):
            progress(f"{index}/{len(tasks)} items")
    doc = {
        "schema": SCHEMA,
        "dataset": dataset.meta,
        "model": model,
        "server": {"name": server, "base_url": base_url, "models": listing, **(server_info or {})},
        "client": {"harness": "Tools/jevbench/harness.py",
                   "harness_sha256": sha256_file(Path(__file__)),
                   "python": platform.python_version(), "vendored": vendored,
                   "adapter": f"{JEVBENCH_REPO}@{JEVBENCH_COMMIT[:7]} jevbench/adapters/typesafe.py",
                   "concurrency": 1, "timeout_s": timeout},
        "hardware": hardware(),
        "published_row": published_row,
        "started_utc": started_at.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - started, 1),
        "stopped": stopped,
        "items": items,
    }
    try:
        doc["summary"] = summarize_doc(doc, dataset)
    except Exception as error:  # the answers are kept; `summary` scores the file again
        doc["summary"] = {"error": f"{type(error).__name__}: {error}"}
    return doc


def author_run_doc(model: str, cache: Path, fetch: bool = True) -> dict:
    """A model author's published run of JevBench's public items (AUTHOR_RUNS) as a result
    document, so `compare`, `summary` and `report` read it as they read a run of this harness."""
    run = AUTHOR_RUNS[model]
    dataset = load_jevbench(cache, fetch=fetch)
    path = author_run_path(cache, model)
    if fetch:
        fetch_author_runs(cache)
    require_pinned(path, run["bytes"], run["sha256"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return author_doc(dataset, rows, model, run)


def author_doc(dataset: Dataset, rows: list, model: str, run: dict) -> dict:
    """The result document of a published run's rows, JevBench's runner records.

    Each item takes the distribution the author's server returned (`probs_as_returned`) and its
    `usage`, scored again with JevBench's score_task as this harness scores its own runs; the
    published record's own outcome is kept beside it. The run's prompts were built in the author's
    process from JevBench's tasks, not sent to a server, so a token count compares with this
    harness's runs only as far as the prompts are the same; equal counts on an item show that
    they are."""
    by_id = {row["task_id"]: row for row in rows}
    items = []
    for task in dataset.tasks:
        item = base_item(task, dataset, None)
        row = by_id.get(task.id)
        if row is None:
            items.append({**item, "status": "unattempted"})
            continue
        probabilities = row.get("probs_as_returned") or row.get("probs")
        answered = bool(row.get("ok")) and probabilities is not None
        scored = jb_scoring.score_task(probabilities, task) if answered else FAILED_SCORE
        item.update({
            "status": "answered" if answered else "failed",
            "http_status": row.get("status_code"),
            "request": None, "answer": None,
            "usage": row.get("usage"),
            "probabilities": probabilities,
            "valid": scored["valid"],
            "strict_valid": scored.get("strict_valid", False),
            "renormalized": scored.get("renormalized", False),
            "predicted": scored.get("predicted"),
            "correct": scored.get("correct"),
            "error": row.get("error"),
            "timing": {"wall_ms": round((row.get("latency_s") or 0) * 1000, 3), "http_ms": None,
                       "server": None},
            "request_id": None,
            "author_record": {"predicted": row.get("predicted"), "correct": row.get("correct"),
                              "raw_sha256": row.get("raw_sha256")},
        })
        items.append(item)
    stamps = [row["ts"] for row in rows if isinstance(row.get("ts"), (int, float))]
    started = (datetime.datetime.fromtimestamp(min(stamps), datetime.timezone.utc)
               .isoformat(timespec="seconds") if stamps else None)
    doc = {
        "schema": SCHEMA,
        "dataset": dataset.meta,
        "model": model,
        "server": {"name": AUTHOR_SERVER, "implementation": run["repo"],
                   "version": run["version"], "commit": run["commit"], "runtime": run["runtime"],
                   "checkpoint": run.get("checkpoint"),
                   "source": f"{run['repo']}@{run['commit'][:7]}:{run['path']}",
                   "source_bytes": run["bytes"], "source_sha256": run["sha256"], "models": None},
        "client": {"harness": "Tools/jevbench/harness.py author-run",
                   "harness_sha256": sha256_file(Path(__file__)),
                   "python": platform.python_version(), "vendored": check_vendored()},
        "hardware": {"platform": run.get("hardware") or "not recorded by the published run"},
        "published_row": None,
        "started_utc": started,
        "duration_s": None,
        "stopped": None,
        "items": items,
    }
    doc["summary"] = summarize_doc(doc, dataset if dataset.name == "typesafe102" else None)
    return doc


# Result files


def write_result(path: Path, doc: dict) -> None:
    """The document with its items last, one per line, so a rerun's diff reads item by item. The
    file is ASCII: a dataset's own punctuation (JevBench's items use dashes this repository keeps
    out of its files) is written as JSON escapes, which read back as the same text."""
    head = json.dumps({key: value for key, value in doc.items() if key != "items"},
                      ensure_ascii=True, indent=1)
    items = ",\n".join(json.dumps(item, ensure_ascii=True, separators=(",", ":"))
                       for item in doc["items"])
    text = head[:-2] + ',\n "items": [\n' + items + "\n ]\n}\n"
    if json.loads(text) != {**json.loads(head), "items": doc["items"]}:
        raise AssertionError("the result file would not read back as the document")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_result(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} file")
    doc["_path"] = str(path)
    return doc


def default_output(dataset: str, model: str, server: str) -> Path:
    folder = RESULTS if dataset == "jevbench" else RESULTS / dataset
    return folder / f"{model}-{server}.json"


# Scoring


def task_of(item: dict, dataset: Dataset | None = None):
    """The JevBench Task an item was scored against: the dataset's own when given (a TypeSafe
    result keeps no expected label), else rebuilt from what the result file keeps."""
    if dataset is not None:
        return dataset.by_id[item["id"]]
    return jb_tasks.Task(id=item["id"], family=item["family"], state=None,
                         question={"type": item["type"]}, labels=item["labels"],
                         expected=item.get("expected"), split="public", group=item.get("group"))


def records_of(doc: dict, dataset: Dataset | None = None) -> tuple[list, list]:
    """Every planned Task and the benchmark Runner's record for each attempted item, rescored from
    the stored probabilities with JevBench's score_task. Skipped and unattempted items have no
    record, so the benchmark's coverage counts them and its accuracy leaves them out."""
    tasks, records = [], []
    for item in doc["items"]:
        task = task_of(item, dataset)
        tasks.append(task)
        if item["status"] in ("skipped", "unattempted"):
            continue
        answered = item["status"] == "answered"
        scored = (jb_scoring.score_task(item["probabilities"] or {}, task) if answered
                  else FAILED_SCORE)
        records.append({
            "task_id": task.id, "family": task.family, "split": task.split, "group": task.group,
            "ok": answered, "valid": scored["valid"], "correct": scored["correct"],
            "predicted": scored.get("predicted"), "probs": scored.get("probs"),
            "strict_valid": scored.get("strict_valid", False),
            "renormalized": scored.get("renormalized", False), "probs_source": "native",
            "model": doc["model"], "latency_s": (item.get("timing") or {}).get("wall_ms", 0) / 1000,
            "cost_usd": None, "cost_basis": COST_BASIS, "status_code": item.get("http_status"),
        })
    return tasks, records


def counts(doc: dict) -> dict:
    found = {"items": len(doc["items"])}
    for status in ("answered", "skipped", "refused", "failed", "unattempted"):
        found[status] = sum(1 for item in doc["items"] if item["status"] == status)
    found["invalid"] = sum(1 for item in doc["items"]
                           if item["status"] == "answered" and not item.get("valid"))
    return found


def model_time(doc: dict) -> dict:
    """The model part of the server-timing header (milliseconds), over every item whose response
    carried one, refusals included."""
    values = [item["timing"]["server"]["model"] for item in doc["items"]
              if ((item.get("timing") or {}).get("server") or {}).get("model") is not None]
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None}
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "p50_ms": statistics.median(values)}


def semif_scores(doc: dict, dataset: Dataset) -> dict:
    """SemIf's own TypeSafe metrics (evaluate_external.type_safe) over the answered rows, for this
    run and for the published answers in the snapshots: equal-case modal agreement with the
    reference and equal-case total variation from the reference distribution."""
    evaluate = semif_evaluator().type_safe
    answered = {item["id"]: item for item in doc["items"]
                if item["status"] == "answered" and item.get("valid")}
    gold = [dataset.gold[item_id] for item_id in dataset.gold if item_id in answered]

    def predictions(distribution_of):
        return [{"id": row["id"], "option_ids": [option["id"] for option in row["options"]],
                 "probabilities": distribution_of(row)} for row in gold]

    def ours(row):
        # the distribution JevBench scores, rescaled to sum to 1: JevBench takes a sum within
        # 1e-3 as it is, and SemIf's evaluator refuses one more than 1e-4 away
        probabilities = jb_scoring.score_task(answered[row["id"]]["probabilities"],
                                              dataset.by_id[row["id"]])["probs"]
        if row["primitive"] == "noul":
            values = [probabilities["yes"], probabilities["no"]]
        else:
            values = [probabilities[option["id"]] for option in row["options"]]
        total = sum(values)
        return [value / total for value in values]

    scores = {"rows": len(gold), "of": len(dataset.gold)}
    if gold:
        mine = evaluate(gold, predictions(ours), predictions(ours))
        scores.update({key: mine["direct"][key] for key in
                       ("cases", "equal_case_modal_agreement", "equal_case_total_variation")})
        # the published answers over the same rows, where the snapshot has the model's answer
        scores["published"] = {}
        for key in TYPESAFE_PUBLISHED:
            rows = [row for row in gold if key in row["published_models"]]
            if rows:
                theirs = [{"id": row["id"], "option_ids": [o["id"] for o in row["options"]],
                           "probabilities": row["published_models"][key]["distribution"]}
                          for row in rows]
                result = evaluate(rows, theirs, theirs)["direct"]
                scores["published"][key] = {
                    "model": rows[0]["published_models"][key]["model"], "rows": len(rows),
                    "cases": result["cases"],
                    "equal_case_modal_agreement": result["equal_case_modal_agreement"],
                    "equal_case_total_variation": result["equal_case_total_variation"]}
    return scores


def summarize_doc(doc: dict, dataset: Dataset | None = None) -> dict:
    """JevBench's own summary of a run: its metric() overall and per tier, and for JevBench items
    its summarize() (families, macro accuracy, paraphrase consistency); for TypeSafe rows SemIf's
    metrics as well."""
    typesafe = doc["dataset"]["name"] == "typesafe102"
    if typesafe and dataset is None:
        raise ValueError("a TypeSafe result is scored against the dataset; pass it")
    tasks, records = records_of(doc, dataset if typesafe else None)
    if typesafe:
        summary = {"overall": jb_summarize.metric(tasks, records)}
    else:
        full = jb_summarize.summarize(tasks, records)
        summary = {"overall": {key: value for key, value in full.items()
                               if key not in ("per_family", "splits")},
                   "per_family": full["per_family"]}
    tiers = {}
    for item, task in zip(doc["items"], tasks):
        tiers.setdefault(item["tier"], []).append(task)
    if len(tiers) > 1:
        summary["per_tier"] = {tier: jb_summarize.metric(members, records)
                               for tier, members in tiers.items()}
    summary["counts"] = counts(doc)
    summary["model_time"] = model_time(doc)
    if typesafe:
        summary["semif"] = semif_scores(doc, dataset)
    return summary


# Comparing runs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


# The reference's top-two margin under which a changed top answer is within the parity bound of
# spike #56, D-034 and D-037: the top answer must hold wherever the reference's top two are at
# least 0.01 apart.
NEAR_TIE = 0.01
# One of D-014's aggregate bounds for DiffusionGemma takes a subset: the slots where the reference's
# top two are at least 0.5 apart, where the top label must hold on 97% of them. Over the prompts
# past 1,024 tokens D-048 bounds each read slot's largest label probability difference, a measure
# of reads, which answers average; `compare` reports the long prompts' figures per answer, not
# against a bound.
CONFIDENT_MARGIN = 0.5
LONG_PROMPT = 1024


def top_two(probabilities: dict) -> tuple:
    ranked = sorted(probabilities.items(), key=lambda pair: (-pair[1], pair[0]))
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], ranked[0][1], ranked[0][1] - second


def compare_docs(a: dict, b: dict, top: int = 10) -> dict:
    """Agreement between two runs of one model on one dataset, `b` taken as the reference.

    Differences are over every label probability of every item both runs answered, as spike #56
    measured Core ML against PyTorch: the mean and the largest absolute difference."""
    if a["model"] != b["model"] or a["dataset"]["name"] != b["dataset"]["name"]:
        raise ValueError(f"compare needs one model and dataset: {a['model']}/"
                         f"{a['dataset']['name']} and {b['model']}/{b['dataset']['name']}")
    items_a = {item["id"]: item for item in a["items"]}
    items_b = {item["id"]: item for item in b["items"]}

    def usable(item):
        return item is not None and item["status"] == "answered" and item.get("valid")

    both = [item_id for item_id in items_a if usable(items_a[item_id])
            and usable(items_b.get(item_id))]
    # every item either run holds, so one run's missing coverage shows whichever run it is
    ids = list(items_a) + [item_id for item_id in items_b if item_id not in items_a]
    one_side = [{"id": item_id, "a": (items_a.get(item_id) or {}).get("status", "absent"),
                 "b": (items_b.get(item_id) or {}).get("status", "absent")}
                for item_id in ids
                if usable(items_a.get(item_id)) != usable(items_b.get(item_id))]
    groups, disagreements, deviations, near_ties = {}, [], [], []
    flips = {"both_correct": 0, "both_wrong": 0, "a_only": 0, "b_only": 0}
    # the prompt tokens each run billed, where both say: equal counts mean equal prompts
    tokens = {"items": 0, "equal": 0, "differ": []}
    bounds = {"confident_items": 0, "confident_agree": 0, "long_items": 0, "long_sum": 0.0,
              "long_entries": 0, "long_max_sum": 0.0}
    for item_id in both:
        x, y = items_a[item_id], items_b[item_id]
        kind = x["type"]
        billed = [(item.get("usage") or {}).get("input_tokens") for item in (x, y)]
        if None not in billed:
            tokens["items"] += 1
            if billed[0] == billed[1]:
                tokens["equal"] += 1
            else:
                tokens["differ"].append({"id": item_id, "a": billed[0], "b": billed[1]})
        group = groups.setdefault(kind, {"items": 0, "agree": 0, "identical": 0, "sum": 0.0,
                                         "entries": 0, "max": 0.0, "max_item": None,
                                         "max_label": None})
        px, py = x["probabilities"], y["probabilities"]
        diffs = {label: abs(float(px[label]) - float(py[label])) for label in x["labels"]}
        label, largest = max(diffs.items(), key=lambda pair: (pair[1], pair[0]))
        group["items"] += 1
        group["agree"] += x["predicted"] == y["predicted"]
        group["identical"] += all(float(px[k]) == float(py[k]) for k in x["labels"])
        group["sum"] += sum(diffs.values())
        group["entries"] += len(diffs)
        if largest >= group["max"]:
            group.update({"max": largest, "max_item": item_id, "max_label": label})
        deviations.append({"id": item_id, "type": kind, "family": x["family"], "label": label,
                           "a": float(px[label]), "b": float(py[label]), "diff": largest,
                           "agree": x["predicted"] == y["predicted"]})
        _, top_b, margin_b = top_two(py)
        if margin_b >= CONFIDENT_MARGIN:
            bounds["confident_items"] += 1
            bounds["confident_agree"] += x["predicted"] == y["predicted"]
        if ((y.get("usage") or {}).get("input_tokens") or 0) > LONG_PROMPT:
            bounds["long_items"] += 1
            bounds["long_sum"] += sum(diffs.values())
            bounds["long_entries"] += len(diffs)
            bounds["long_max_sum"] += largest
        if margin_b < NEAR_TIE:
            near_ties.append({"id": item_id, "type": kind, "a": x["predicted"],
                              "b": y["predicted"], "a_margin": top_two(px)[2],
                              "b_margin": margin_b, "agree": x["predicted"] == y["predicted"]})
        if x["predicted"] != y["predicted"]:
            disagreements.append({"id": item_id, "type": kind, "family": x["family"],
                                  "a": x["predicted"], "b": y["predicted"],
                                  "a_p": float(px[x["predicted"]]), "b_p": top_b,
                                  "b_margin": margin_b, "diff": largest,
                                  "a_correct": x.get("correct"), "b_correct": y.get("correct")})
        if x.get("correct") is not None and y.get("correct") is not None:
            key = ("both_correct" if x["correct"] and y["correct"] else
                   "both_wrong" if not x["correct"] and not y["correct"] else
                   "a_only" if x["correct"] else "b_only")
            flips[key] += 1
    per_type = {}
    for kind, group in sorted(groups.items()):
        per_type[kind] = {"items": group["items"], "agree": group["agree"],
                          "identical": group["identical"],
                          "mean_abs_diff": group["sum"] / group["entries"],
                          "max_abs_diff": group["max"], "max_item": group["max_item"],
                          "max_label": group["max_label"]}
    total_entries = sum(group["entries"] for group in groups.values())
    overall = {"items": len(both), "agree": sum(g["agree"] for g in groups.values()),
               "identical": sum(g["identical"] for g in groups.values()),
               "mean_abs_diff": (sum(g["sum"] for g in groups.values()) / total_entries
                                 if total_entries else None),
               "max_abs_diff": max((g["max"] for g in groups.values()), default=None),
               # the median over items of each item's largest difference
               "median_max_abs_diff": (statistics.median(entry["diff"] for entry in deviations)
                                       if deviations else None)}
    deviations.sort(key=lambda entry: (-entry["diff"], entry["id"]))
    return {
        "model": a["model"], "dataset": a["dataset"]["name"],
        "a": {"server": a["server"]["name"], "file": a.get("_path")},
        "b": {"server": b["server"]["name"], "file": b.get("_path")},
        "overall": overall, "per_type": per_type,
        "correctness": {**flips, "mcnemar_p": mcnemar_exact(flips["a_only"], flips["b_only"])},
        "disagreements": disagreements, "largest": deviations[:top],
        "near_ties": near_ties,
        # over the items both answered, by the reference's margin and prompt length (D-014, D-048):
        # past 1,024 tokens the mean over every label, which items with many labels dilute, and
        # the mean of each item's largest difference, which they cannot (both informational)
        "bounds": {"confident_items": bounds["confident_items"],
                   "confident_agree": bounds["confident_agree"],
                   "long_items": bounds["long_items"],
                   "long_mean_abs_diff": (bounds["long_sum"] / bounds["long_entries"]
                                          if bounds["long_entries"] else None),
                   "long_mean_max_abs_diff": (bounds["long_max_sum"] / bounds["long_items"]
                                              if bounds["long_items"] else None)},
        "answered_by_one_only": one_side,
        "input_tokens": tokens,
    }


def against_published(doc: dict, cache: Path) -> dict | None:
    """A JevBench run against the benchmark's published row for the same model: the outcome of
    every public item, the public accuracy, and the board's aggregates."""
    row = doc.get("published_row")
    if doc["dataset"]["name"] != "jevbench" or not row:
        return None
    table = {"both_correct": 0, "both_wrong": 0, "ours_only": 0, "published_only": 0}
    differ, tiers = [], {}
    for item in doc["items"]:
        published = (item.get("published") or {}).get("outcome")
        if item["status"] in ("skipped", "unattempted") or published is None:
            continue
        ours, theirs = bool(item.get("correct")), published == "c"
        tier = tiers.setdefault(item["tier"], {"items": 0, "ours": 0, "published": 0})
        tier["items"] += 1
        tier["ours"] += ours
        tier["published"] += theirs
        key = ("both_correct" if ours and theirs else "both_wrong" if not ours and not theirs
               else "ours_only" if ours else "published_only")
        table[key] += 1
        if ours != theirs:
            differ.append({"id": item["id"], "tier": item["tier"], "type": item["type"],
                           "family": item["family"], "ours": ours, "published": theirs})
    n = sum(table.values())
    board = jevbench_root(cache) / PUBLISHED_BOARD[0]
    aggregates = None
    if board.is_file() and sha256_file(board) == PUBLISHED_BOARD[2]:
        systems = json.loads(board.read_text(encoding="utf-8"))["systems"]
        system = next((entry for entry in systems if entry.get("key") == row["key"]), None)
        if system:
            aggregates = {"public_accuracy": system.get("public_accuracy"),
                          "sealed_accuracy": system.get("sealed_accuracy"),
                          "tiers_with_heldout": system.get("tiers"),
                          "hard_brier_220": (system.get("hard") or {}).get("brier_mean"),
                          "hard_ece_220": (system.get("hard") or {}).get("ece"),
                          "source": f"{JEVBENCH_REPO}@{JEVBENCH_COMMIT[:7]}:{PUBLISHED_BOARD[0]}"}
    return {"model": doc["model"], "server": doc["server"]["name"], "row": row, "items": n,
            "ours_accuracy": (table["both_correct"] + table["ours_only"]) / n if n else None,
            "published_accuracy": (table["both_correct"] + table["published_only"]) / n if n else None,
            "outcomes": {**table, "agreement": (table["both_correct"] + table["both_wrong"]) / n
                         if n else None,
                         "mcnemar_p": mcnemar_exact(table["ours_only"], table["published_only"])},
            "per_tier": tiers, "differ": differ, "board": aggregates}


# Printing


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def num(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if value != 0 and abs(value) < 10 ** -(digits - 1):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def ms(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def summary_rows(docs: list) -> list:
    """One row per run: the counts and JevBench's accuracy, Brier score and ECE."""
    rows = []
    for doc in docs:
        summary = doc["summary"]
        if "error" in summary:
            rows.append([doc["dataset"]["name"], doc["model"], doc["server"]["name"],
                         str(len(doc["items"])), f"scoring failed: {summary['error']}"]
                        + [""] * (len(SUMMARY_HEADER) - 5))
            continue
        overall, found = summary["overall"], summary["counts"]
        latency = overall.get("latency") or {}

        def wall(key):
            return ms(None if latency.get(key) is None else latency[key] * 1000)

        skipped = str(found["skipped"]) + (f" (+{found['unattempted']} not run)"
                                           if found["unattempted"] else "")
        rows.append([doc["dataset"]["name"], doc["model"], doc["server"]["name"],
                     str(found["items"]), str(found["answered"]), skipped,
                     str(found["refused"] + found["failed"] + found["invalid"]),
                     pct(overall.get("accuracy")), num(overall.get("brier_mean")),
                     num((overall.get("ece") or {}).get("ece")), wall("p50_s"), wall("p95_s"),
                     ms(summary["model_time"]["p50_ms"])])
    return rows


SUMMARY_HEADER = ["dataset", "model", "server", "items", "answered", "skipped",
                  "refused, failed or invalid", "accuracy", "Brier", "ECE", "p50 ms", "p95 ms",
                  "model p50 ms"]


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def tier_rows(doc: dict) -> list:
    rows = []
    for tier, metric in (doc["summary"].get("per_tier") or {}).items():
        rows.append([doc["model"], doc["server"]["name"], tier, str(metric["n_planned"]),
                     pct(metric["accuracy"]), num(metric["brier_mean"]),
                     num((metric["ece"] or {}).get("ece")),
                     num(metric["ordinal_mae"], 3) if metric["ordinal_mae"] is not None else "n/a"])
    return rows


def compare_text(result: dict, items: int = 10) -> str:
    out = [f"{result['model']} on {result['dataset']}: {result['a']['server']} against "
           f"{result['b']['server']} (reference)"]
    overall = result["overall"]
    out.append(f"top answers agree on {overall['agree']} of {overall['items']} items answered by "
               f"both; identical answers on {overall['identical']}")
    rows = [[kind, str(g["items"]), f"{g['agree']}/{g['items']}", str(g["identical"]),
             num(g["mean_abs_diff"]), num(g["max_abs_diff"]), f"{g['max_item']} ({g['max_label']})"]
            for kind, g in result["per_type"].items()]
    rows.append(["all", str(overall["items"]), f"{overall['agree']}/{overall['items']}",
                 str(overall["identical"]), num(overall["mean_abs_diff"]),
                 num(overall["max_abs_diff"]), ""])
    out.append(markdown_table(["type", "items", "top answer agrees", "identical",
                               "mean abs diff", "largest abs diff", "largest at"], rows))
    bounds = result["bounds"]
    out.append(f"where {result['b']['server']}'s top two are at least {CONFIDENT_MARGIN} apart the "
               f"top answer agrees on {bounds['confident_agree']} of {bounds['confident_items']}; "
               f"over the {bounds['long_items']} prompts longer than {LONG_PROMPT} tokens the mean "
               f"of each item's largest abs diff is {num(bounds['long_mean_max_abs_diff'])} and the "
               f"mean abs diff over their labels {num(bounds['long_mean_abs_diff'])} (informational: "
               f"D-048 bounds reads, which an answer averages)")
    tokens = result["input_tokens"]
    out.append(f"input tokens equal on {tokens['equal']} of the {tokens['items']} items both "
               f"billed; the median of each item's largest abs diff is "
               f"{num(overall['median_max_abs_diff'])}"
               + ("" if not tokens["differ"] else "; they differ on " + ", ".join(
                   f"{d['id']} ({d['a']} and {d['b']})" for d in tokens["differ"][:items])))
    flips = result["correctness"]
    out.append(f"correct in both {flips['both_correct']}, wrong in both {flips['both_wrong']}, "
               f"only {result['a']['server']} {flips['a_only']}, only {result['b']['server']} "
               f"{flips['b_only']} (exact McNemar p = {flips['mcnemar_p']:.3g})")
    if result["disagreements"]:
        out.append(markdown_table(
            ["item", "type", result["a"]["server"], result["b"]["server"],
             f"{result['b']['server']} top-two margin", "largest abs diff"],
            [[d["id"], d["type"], f"{d['a']} ({d['a_p']:.4f})", f"{d['b']} ({d['b_p']:.4f})",
              num(d["b_margin"]), num(d["diff"])] for d in result["disagreements"]]))
    out.append(markdown_table(
        ["item", "type", "label", result["a"]["server"], result["b"]["server"], "abs diff",
         "top answer agrees"],
        [[d["id"], d["type"], d["label"], f"{d['a']:.4f}", f"{d['b']:.4f}", num(d["diff"]),
          "yes" if d["agree"] else "no"] for d in result["largest"][:items]]))
    if result["near_ties"]:
        out.append(f"{len(result['near_ties'])} items where {result['b']['server']}'s top two are "
                   f"less than {NEAR_TIE} apart, where the parity bound allows a changed top answer:")
        out.append(markdown_table(
            ["item", "type", result["a"]["server"], result["b"]["server"],
             f"{result['a']['server']} margin", f"{result['b']['server']} margin"],
            [[d["id"], d["type"], d["a"], d["b"], num(d["a_margin"]), num(d["b_margin"])]
             for d in result["near_ties"]]))
    if result["answered_by_one_only"]:
        out.append("answered by one run only: " + ", ".join(
            f"{entry['id']} ({entry['a']} / {entry['b']})"
            for entry in result["answered_by_one_only"]))
    return "\n\n".join(out)


def published_text(result: dict) -> str:
    row, outcomes = result["row"], result["outcomes"]
    out = [f"{result['model']} ({result['server']}) against JevBench's published row "
           f"{row['key']} ({row['display']}): {row['setup']}",
           f"public items {result['items']}: ours {pct(result['ours_accuracy'])}, published "
           f"{pct(result['published_accuracy'])}; same outcome on {pct(outcomes['agreement'])} "
           f"(both correct {outcomes['both_correct']}, both wrong {outcomes['both_wrong']}, only "
           f"ours {outcomes['ours_only']}, only published {outcomes['published_only']}, exact "
           f"McNemar p = {outcomes['mcnemar_p']:.3g})"]
    out.append(markdown_table(
        ["tier", "items", "ours", "published"],
        [[tier, str(t["items"]), pct(t["ours"] / t["items"]), pct(t["published"] / t["items"])]
         for tier, t in result["per_tier"].items()]))
    if result["board"]:
        board = result["board"]
        out.append(f"board v1.4.2.2: public accuracy {pct(board['public_accuracy'])}, sealed "
                   f"{pct(board['sealed_accuracy'])}; hard tier with its held-out half (220 items): "
                   f"Brier {num(board['hard_brier_220'])}, ECE {num(board['hard_ece_220'])}")
    return "\n\n".join(out)


# Commands


def command_fetch(args) -> int:
    cache = Path(args.cache)
    check_vendored()
    if not args.offline:
        compare_vendored_with_sources()
    for name in ("jevbench", "typesafe102") if args.dataset == "all" else (args.dataset,):
        dataset = load_dataset(name, cache, fetch=not args.offline)
        print(f"{name}: {len(dataset.tasks)} items in {cache}")
    if args.dataset != "typesafe102":
        if not args.offline:
            fetch_author_runs(cache)
        for model, run in AUTHOR_RUNS.items():
            require_pinned(author_run_path(cache, model), run["bytes"], run["sha256"])
            print(f"{model}: the author's published run, {run['repo']}@{run['commit'][:7]}")
    print("vendored files match their pins" + ("" if args.offline else " and their sources"))
    return 0


def command_run(args) -> int:
    cache = Path(args.cache)
    if args.items:
        dataset = load_item_file(Path(args.items))
    else:
        dataset = load_dataset(args.dataset, cache)
    server_info = json.loads(Path(args.server_info).read_text()) if args.server_info else None
    output = Path(args.output) if args.output else default_output(dataset.name, args.model,
                                                                  args.server)
    if output.exists() and not args.force:
        print(f"{output} exists; pass --force to replace it", file=sys.stderr)
        return 2
    ids = set(args.ids.split(",")) if args.ids else None
    doc = run_dataset(dataset, args.base_url, args.model, args.server, cache,
                      key_env=args.api_key_env or "", server_info=server_info,
                      timeout=args.timeout, ids=ids)
    write_result(output, doc)
    print(f"wrote {output} ({output.stat().st_size // 1024} KB)")
    print(markdown_table(SUMMARY_HEADER, summary_rows([doc])))
    return 0 if not doc["stopped"] else 1


def with_correctness(doc: dict, dataset: Dataset) -> dict:
    """A TypeSafe result with each answered item's correctness, scored in memory against the
    cached dataset, since the file does not keep it."""
    for item in doc["items"]:
        if item["status"] == "answered":
            item["correct"] = jb_scoring.score_task(item["probabilities"] or {},
                                                    dataset.by_id[item["id"]]).get("correct")
        elif item["status"] in ("refused", "failed"):
            item["correct"] = False
    return doc


def load_docs(paths: list, cache: Path) -> list:
    """Result files, each scored again from its answers: what `summary`, `compare` and `report`
    read, after checking the vendored files they score with."""
    check_vendored()
    docs, datasets = [], {}
    for path in paths:
        doc = read_result(Path(path))
        name = doc["dataset"]["name"]
        if name == "typesafe102":
            datasets.setdefault(name, load_dataset(name, cache))
            with_correctness(doc, datasets[name])
        doc["summary"] = summarize_doc(doc, datasets.get(name))
        docs.append(doc)
    return docs


def command_summary(args) -> int:
    docs = load_docs(args.files, Path(args.cache))
    print(markdown_table(SUMMARY_HEADER, summary_rows(docs)))
    tiers = [row for doc in docs for row in tier_rows(doc)]
    if tiers:
        print()
        print(markdown_table(["model", "server", "tier", "items", "accuracy", "Brier", "ECE",
                              "ordinal MAE"], tiers))
    return 0


def command_compare(args) -> int:
    a, b = load_docs([args.a, args.b], Path(args.cache))
    result = compare_docs(a, b, top=args.top)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(compare_text(result, items=args.top))
    return 0


def command_published(args) -> int:
    cache = Path(args.cache)
    check_vendored()
    fetch_jevbench(cache)
    for path in args.files:
        result = against_published(read_result(Path(path)), cache)
        print(published_text(result) if result else f"{path}: no published row for this model")
        print()
    return 0


def command_author_run(args) -> int:
    output = Path(args.output) if args.output else default_output("jevbench", args.model,
                                                                  AUTHOR_SERVER)
    if output.exists() and not args.force:
        print(f"{output} exists; pass --force to replace it", file=sys.stderr)
        return 2
    doc = author_run_doc(args.model, Path(args.cache))
    write_result(output, doc)
    print(f"wrote {output} ({output.stat().st_size // 1024} KB)")
    print(markdown_table(SUMMARY_HEADER, summary_rows([doc])))
    return 0


def command_report(args) -> int:
    from report import render  # the tables of docs/quality.md

    print(render(Path(args.results), Path(args.cache)))
    return 0


def command_calibration(args) -> int:
    from calibration import render  # the calibration tables of docs/quality.md

    print(render(Path(args.results), Path(args.cache), model=args.model))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default=str(default_cache()),
                        help="where datasets are downloaded (default %(default)s, or JEVBENCH_CACHE)")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download and check the datasets and vendored files")
    fetch.add_argument("--dataset", choices=("jevbench", "typesafe102", "all"), default="all")
    fetch.add_argument("--offline", action="store_true",
                       help="download nothing: check the vendored files against their pins and "
                            "the cached datasets against theirs")
    fetch.set_defaults(func=command_fetch)

    run = commands.add_parser("run", help="run a dataset against a /v1/systemone server")
    run.add_argument("--base-url", default="http://127.0.0.1:8080")
    run.add_argument("--model", required=True, help="the served model name, as the request sends it")
    run.add_argument("--server", required=True, help="a short name for the server, such as swift")
    run.add_argument("--dataset", choices=("jevbench", "typesafe102"), default="jevbench")
    run.add_argument("--items", help="a JSONL file of JevBench-format items instead of a dataset")
    run.add_argument("--api-key-env", help="the variable that holds the API key, if one is needed")
    run.add_argument("--server-info", help="a JSON file of server versions to record")
    run.add_argument("--output", help="the result file (default results/{model}-{server}.json)")
    run.add_argument("--ids", help="only these comma-separated item ids")
    run.add_argument("--timeout", type=float, default=120.0, help="seconds per request")
    run.add_argument("--force", action="store_true", help="replace an existing result file")
    run.set_defaults(func=command_run)

    summary = commands.add_parser("summary", help="print the summary tables of result files")
    summary.add_argument("files", nargs="+")
    summary.set_defaults(func=command_summary)

    compare = commands.add_parser("compare", help="agreement between two runs of one model")
    compare.add_argument("a", help="the run to check, for example the Swift server's")
    compare.add_argument("b", help="the reference run, for example upstream's")
    compare.add_argument("--top", type=int, default=10, help="largest deviations to list")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=command_compare)

    published = commands.add_parser("published", help="a JevBench run against the published row")
    published.add_argument("files", nargs="+")
    published.set_defaults(func=command_published)

    author = commands.add_parser(
        "author-run", help="a model author's published JevBench run as a result file")
    author.add_argument("--model", choices=tuple(AUTHOR_RUNS), default="jevk5-0.2")
    author.add_argument("--output", help="the result file (default results/{model}-author.json)")
    author.add_argument("--force", action="store_true", help="replace an existing result file")
    author.set_defaults(func=command_author_run)

    report = commands.add_parser("report", help="the tables of docs/quality.md")
    report.add_argument("--results", default=str(RESULTS))
    report.set_defaults(func=command_report)

    calibration = commands.add_parser("calibration",
                                      help="the calibration tables of docs/quality.md")
    calibration.add_argument("--results", default=str(RESULTS))
    calibration.add_argument("--model", default="openjev-0.1",
                             help="the model whose runs to calibrate (default %(default)s)")
    calibration.set_defaults(func=command_calibration)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
