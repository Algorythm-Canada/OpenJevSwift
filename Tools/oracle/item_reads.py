"""The reads of JevBench and TypeSafe items, upstream's against the port's, bit for bit.

D-014 holds the port to mlx-vlm's 27 oracle reads bit for bit in its exact tier: the mlx-metal
wheel's metallib and the oracle's RoPE table (D-036). This tool takes that check to any item the
JevBench harness asks. It rebuilds the request bodies as Tools/jevbench/harness.py sends them,
records every read upstream's own MlxEngine.decide makes on them, and compares those reads with
the ones the port's DecisionEngine and model make (the UpstreamProbe target ItemReads), read by
read and prefill layer by layer. It settled PR #105's four wide-margin long-prompt flips
(docs/quality.md, "A disagreement on DiffusionGemma").

Run from the repository root, one model process at a time, with a Python that has MLX 0.32.2 and
mlx-vlm 0.6.15 (Tools/jevbench/.venv or Tools/oracle/.venv) and the datasets fetched by
`harness.py fetch`:

    PY=Tools/jevbench/.venv/bin/python
    $PY Tools/oracle/item_reads.py bodies jevbench:hard-sol-b-long_policy-06 typesafe102:e2e58201a90c11192f70edbf
    $PY Tools/oracle/item_reads.py upstream --variants chunked_prefill
    swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads --metallib "$WHEEL_METALLIB" --oracle-rope
    swift run --package-path Tools/oracle/UpstreamProbe -c release ItemReads
    $PY Tools/oracle/item_reads.py compare --summary Tools/oracle/results/NAME.json
    $PY Tools/oracle/item_reads.py long-slots

- bodies: each item's body from the harness's dataset loader and JevBench's typesafe adapter, its
  SHA-256 checked against `body_sha256` in every DiffusionGemma result file that holds the item.
- upstream: upstream's MlxEngine.decide on each body, as its /v1/systemone route calls it (the
  route's seed, OPENJEV_BACKEND=mlx, the pinned snapshot, the MLX cache limit), with pass-through
  spies on MlxRuntime.read and Engine.build_canvas that record each read's prompt ids, canvas,
  seed, slots, steps and maps, and digests of every layer of each prompt's prefill cache. Each
  name in --variants then replays every recorded read through Tools/oracle/sensitivity.py's
  read() with that variant (chunked_prefill is spike #22's 64-token chunked prefill).
- compare: every port_*.json in the work folder against upstream.json, and the answers against the
  result files. It prints, per item and configuration, whether the port's engine built upstream's
  reads, whether each read is identical (token ids and float32 logprob bits of the top 20 and the
  labels), each prefill layer's keys, values and sliding decoder view by SHA-256, and the answers;
  per configuration, the mean and largest |dp| over the reads' label probabilities and the answers
  whose top label differs from upstream's. --summary writes that as JSON.
- long-slots: D-014's long-prompt figure over the oracle fixture's slots with few labels, from
  committed files only (Fixtures/oracle/reads.json and two of spike #22's runs).

The work files (bodies.json, upstream.json, port_*.json) go to ~/Library/Caches/OpenJevSwift/
item-reads unless --work or OPENJEV_ITEM_READS names another folder. Keep them out of the
repository: they hold TypeSafe's text, and prompt ids that decode to it, which carry no license
grant, and about 0.5 MB per run. The summary keeps digests, equality flags and the model's outputs,
and no state text, prompt ids or reference answers.
"""
import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "Tools" / "oracle"
UPSTREAM = ROOT / "Upstream" / "openjev"
JEVBENCH = ROOT / "Tools" / "jevbench"
RESULTS = HERE / "results"

MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
MODEL = "openjev-0.1"
DATASETS = ("jevbench", "typesafe102")
SERVERS = ("swift", "upstream")


def default_work() -> Path:
    if os.environ.get("OPENJEV_ITEM_READS"):
        return Path(os.environ["OPENJEV_ITEM_READS"]).expanduser()
    return Path.home() / "Library" / "Caches" / "OpenJevSwift" / "item-reads"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids) -> str:
    """The SHA-256 of a token id list written as a JSON array, as `json.dumps(ids)` writes it."""
    return sha256_hex(json.dumps([int(i) for i in ids]).encode())


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent) + ("\n" if indent is not None else ""),
                    encoding="utf-8")


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def shown(path: Path) -> str:
    """A path as an output file records it: relative to the repository or to ~."""
    path = Path(path).expanduser().absolute()
    for base, prefix in ((ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(path.relative_to(base))
        except ValueError:
            continue
    return path.name


def import_harness():
    sys.path.insert(0, str(JEVBENCH))
    import harness  # noqa: E402

    return harness


def import_upstream_engine():
    """Upstream's engine module, for slot_distribution; its imports need no model."""
    sys.path.insert(0, str(UPSTREAM))
    from openjev import engine  # noqa: E402

    return engine


def result_file(dataset: str, server: str) -> Path:
    folder = JEVBENCH / "results" if dataset == "jevbench" else JEVBENCH / "results" / dataset
    return folder / f"{MODEL}-{server}.json"


def result_items(dataset: str, server: str) -> dict:
    path = result_file(dataset, server)
    if not path.exists():
        return {}
    return {item["id"]: item for item in read_json(path)["items"]}


# bodies


def command_bodies(args) -> int:
    harness = import_harness()
    cache = harness.default_cache()
    # the harness's adapter; the endpoint and key never reach the body
    adapter = harness.jb_typesafe.TypeSafeAdapter(endpoint="http://127.0.0.1:1", model=MODEL)
    datasets, out = {}, []
    for spec in args.items:
        dataset, _, item_id = spec.partition(":")
        if dataset not in DATASETS or not item_id:
            raise SystemExit(f"{spec}: name an item as jevbench:ID or typesafe102:ID")
        if dataset not in datasets:
            datasets[dataset] = harness.load_dataset(dataset, cache, fetch=False)
        task = datasets[dataset].by_id.get(item_id)
        if task is None:
            raise SystemExit(f"{spec}: {dataset} has no item {item_id}")
        text = json.dumps(adapter.build_request(task))  # as harness.RecordingPost sends it
        digest = sha256_hex(text.encode("utf-8"))
        checked = []
        for server in SERVERS:
            item = result_items(dataset, server).get(item_id)
            if item is None:
                continue
            recorded = (item.get("request") or {}).get("body_sha256")
            if recorded is None:  # skipped there, so never sent
                continue
            if recorded != digest:
                raise SystemExit(f"{spec}: the body's SHA-256 is {digest}; "
                                 f"{shown(result_file(dataset, server))} recorded {recorded}")
            checked.append(server)
        note = (f"equal to body_sha256 in the {' and '.join(checked)} result file"
                f"{'s' if len(checked) > 1 else ''}" if checked else "no result file holds it")
        print(f"{spec}: {len(text.encode('utf-8')):,} bytes, SHA-256 {digest[:16]}, {note}")
        out.append({"dataset": dataset, "id": item_id, "type": task.question["type"],
                    "labels": list(task.labels), "body_text": text, "body_sha256": digest,
                    "result_files_checked": checked})
    write_json(args.work / "bodies.json", out)
    print(f"wrote {args.work / 'bodies.json'}")
    return 0


# upstream


def gigabytes(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def tensor_digest(mx, np, array) -> dict:
    """What Fixtures/oracle/reads.json records of a cached tensor: the raw bytes' SHA-256 (C
    order), and the sum and the sum of squares in float64."""
    raw = np.array(mx.view(array, mx.uint16)) if array.dtype == mx.bfloat16 else np.array(array)
    values = np.array(array.astype(mx.float32)).astype(np.float64)
    return {"dtype": str(array.dtype).replace("mlx.core.", ""), "shape": list(array.shape),
            "sha256": sha256_hex(np.ascontiguousarray(raw).tobytes()),
            "sum": float(values.sum()), "sum_of_squares": float((values * values).sum())}


def machine() -> dict:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.machine()
    return {"chip": chip, "macos": platform.mac_ver()[0]}


def command_upstream(args) -> int:
    bodies = read_json(args.work / "bodies.json")
    # Upstream's Settings reads OPENJEV_*: only the settings below apply, as servers.py starts it.
    for name in [n for n in os.environ if n.startswith("OPENJEV_")]:
        del os.environ[name]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import huggingface_hub  # noqa: E402

    snapshot = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                                                 local_files_only=True)
    os.environ["OPENJEV_BACKEND"] = "mlx"
    os.environ["OPENJEV_MLX_MODEL"] = snapshot
    os.environ["OPENJEV_MLX_CACHE_LIMIT_GB"] = gigabytes(args.cache_limit_gb)
    sys.path.insert(0, str(UPSTREAM))

    import asyncio  # noqa: E402
    import importlib.metadata as metadata  # noqa: E402

    import mlx.core as mx  # noqa: E402
    import numpy as np  # noqa: E402
    from mlx_vlm.models.diffusion_gemma.language import _cache_state  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    from openjev.api import SystemOneRequest  # noqa: E402
    from openjev.config import Settings  # noqa: E402
    from openjev.mlx_backend import MlxEngine  # noqa: E402

    settings = Settings()
    started = time.perf_counter()
    engine = MlxEngine(settings, AutoTokenizer.from_pretrained(settings.mlx_model))
    print(f"loaded in {time.perf_counter() - started:.1f} s, MLX cache limit "
          f"{settings.mlx_cache_limit_gb} GB", flush=True)
    rt = engine.runtime
    recorded, seeds = [], {}
    original_read, original_canvas = rt.read, engine.build_canvas

    def canvas_spy(template, slots, seed):
        canvas = original_canvas(template, slots, seed)
        seeds[tuple(canvas)] = seed
        return canvas

    def read_spy(prompt, canvas, slots, max_tokens, steps=1):
        cached = tuple(prompt) in rt.prefills
        tops, prompt_tokens = original_read(prompt, canvas, slots, max_tokens, steps)
        recorded.append({
            "seed": seeds.get(tuple(canvas)), "prompt_ids": [int(i) for i in prompt],
            "canvas": [int(t) for t in canvas],
            "slots": [{"pos": s["pos"], "label_ids": list(s["label_ids"])} for s in slots],
            "steps": steps, "prompt_tokens": prompt_tokens, "prefill_cached": cached,
            # MlxRuntime.read's maps: float32 log-softmax values widened to Python floats
            "logprobs": [[[int(t), float(v)] for t, v in top.items()] for top in tops]})
        return tops, prompt_tokens

    engine.build_canvas = canvas_spy
    rt.read = read_spy

    def cache_digests(prompt_ids):
        cache, _ = rt.prefills[tuple(prompt_ids)]
        text = rt.model.config.text_config
        rows = []
        for layer, entry in enumerate(cache):
            keys, values = _cache_state(entry)
            row = {"layer": layer, "kind": text.layer_types[layer], "offset": int(entry.offset),
                   "keys": tensor_digest(mx, np, keys), "values": tensor_digest(mx, np, values)}
            if text.layer_types[layer] == "sliding_attention":
                # what a decoder pass reads: the last sliding_window - 1 positions
                start = max(0, int(keys.shape[2]) - (text.sliding_window - 1))
                row["decoder_view"] = {"start": start,
                                       "keys": tensor_digest(mx, np, keys[:, :, start:, :]),
                                       "values": tensor_digest(mx, np, values[:, :, start:, :])}
            rows.append(row)
        return rows

    async def decide_all():
        items = []
        loop = asyncio.get_running_loop()
        for body in bodies:
            req = SystemOneRequest.model_validate(json.loads(body["body_text"]))
            # the /v1/systemone route's arguments (api.py), text states only
            questions = {k: q.model_dump() for k, q in req.questions.items()}
            options = {"steps": req.steps, "samples": req.samples, "think": req.think,
                       "sequential": req.sequential}
            key = [req.state, questions]
            seed = int.from_bytes(
                hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()[:4], "big")
            first = len(recorded)
            t0 = time.perf_counter()
            answers, input_tokens, thought_tokens = await engine.decide(
                questions, req.state, seed, None, options)
            # in seed order (group, then re-read), which every replay of them keeps
            reads = sorted(recorded[first:], key=lambda r: r["seed"])
            caches = []
            for prompt_ids in dict.fromkeys(tuple(r["prompt_ids"]) for r in reads):
                caches.append({"prompt_ids_sha256": ids_digest(prompt_ids),
                               "layers": await loop.run_in_executor(rt.pool, cache_digests,
                                                                    prompt_ids)})
            print(f"{body['id']}: {len(reads)} reads, prompt tokens "
                  f"{sorted({r['prompt_tokens'] for r in reads})}, "
                  f"{time.perf_counter() - t0:.1f} s, answers {json.dumps(answers)[:200]}",
                  flush=True)
            # each read's slots are its group's questions in this order, their labels these options
            schema = engine.build_schema(questions)["questions"]
            items.append({"id": body["id"], "dataset": body["dataset"], "seed": seed,
                          "questions": [{"key": q["key"], "type": q["type"],
                                         "options": [name for name, _ in q["choices"]]}
                                        for q in schema],
                          "answers": answers,
                          "usage": {"input_tokens": input_tokens, "output_tokens": thought_tokens},
                          "reads": reads, "caches": caches})
        return items

    items = asyncio.run(decide_all())

    variants = {}
    if args.variants:
        sys.path.insert(0, str(HERE))
        import sensitivity  # noqa: E402  (MlxRuntime.read without the prefill cache, one change)

        for variant in args.variants.split(","):
            t0 = time.perf_counter()
            rows = []
            for item in items:
                reads = []
                for r in item["reads"]:
                    tops, written = rt.pool.submit(sensitivity.read, rt, r["prompt_ids"],
                                                   r["canvas"], r["slots"], r["steps"],
                                                   variant).result()
                    reads.append({"logprobs": [[[int(t), float(v)] for t, v in top.items()]
                                               for top in tops], "written": written})
                rows.append({"id": item["id"], "reads": reads})
            variants[variant] = {"items": rows, "seconds": time.perf_counter() - t0}
            if variant == "chunked_prefill":
                variants[variant]["chunk"] = sensitivity.CHUNK
            print(f"variant {variant}: {sum(len(r['reads']) for r in rows)} reads replayed in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)

    commit = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    metallib = Path(mx.__file__).parent / "lib" / "mlx.metallib"
    payload = {
        "run_on": time.strftime("%Y-%m-%d"), "machine": machine(),
        "upstream_commit": commit, "python": platform.python_version(),
        "packages": {name: metadata.version(name)
                     for name in ("mlx", "mlx-metal", "mlx-vlm", "transformers")},
        "metallib": {"wheel": f"mlx-metal {metadata.version('mlx-metal')}",
                     "path": "mlx/lib/mlx.metallib", "sha256": sha256_hex(metallib.read_bytes())},
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "settings": {"mlx_cache_limit_gb": settings.mlx_cache_limit_gb,
                     "auto_threshold": settings.auto_threshold, "auto_max": settings.auto_max,
                     "canvas": settings.canvas, "canvas_step": settings.canvas_step},
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "items": items, "variants": variants,
    }
    write_json(args.work / "upstream.json", payload)
    print(f"wrote {args.work / 'upstream.json'}; peak MLX memory {payload['peak_mlx_gib']:.1f} GiB")
    rt.close()
    return 0


# compare


def read_key(read: dict) -> tuple:
    return (tuple(read["prompt_ids"]), tuple(read["canvas"]),
            tuple((s["pos"], tuple(s["label_ids"])) for s in read["slots"]), read["steps"])


def same_maps(want: list, got: list) -> bool:
    """One read's maps, slot by slot: the same token ids in the same order, each logprob with the
    same float32 bits and the same value widened, as ReadOracleTests compares them. `got` rows may
    carry the bits as a third element (the port's), which must agree."""
    if len(want) != len(got):
        return False
    for w, g in zip(want, got):
        if [int(t) for t, *_ in w] != [int(t) for t, *_ in g]:
            return False
        for (_, wv, *_), (_, gv, *bits) in zip(w, g):
            if wv != gv or f32_bits(wv) != f32_bits(gv) or (bits and int(bits[0]) != f32_bits(gv)):
                return False
    return True


def paired(want: list, got: list, what: str) -> list:
    """want and got side by side. Lists of different lengths are refused, never cut short: a
    missing or extra item, read, slot or label means the runs do not match."""
    if len(want) != len(got):
        raise SystemExit(f"{what}: {len(got)} against upstream's {len(want)}; the runs do not "
                         "match, so nothing is compared")
    return list(zip(want, got))


def label_probabilities(slot_distribution, read: dict, maps: list) -> list:
    """slot_distribution's label probabilities of each slot, as the engine averages them."""
    return [slot_distribution({int(t): v for t, v, *_ in m}, s["label_ids"])["probs"]
            for s, m in paired(read["slots"], maps, "the slots of a read")]


def means(per_read: list) -> list:
    """Per slot, the mean over the reads, in read order, as read_group sums them."""
    return [[sum(p[slot][label] for p in per_read) / len(per_read)
             for label in range(len(per_read[0][slot]))] for slot in range(len(per_read[0]))]


def argmax(values: list) -> int:
    return max(range(len(values)), key=values.__getitem__)


def tensor_type(digest: dict) -> tuple:
    return digest["dtype"], tuple(digest["shape"])


def layer_structure(want: dict | None, got: dict | None) -> list:
    """How one layer of two prefill caches differs before any hash is compared: one side lacks
    it, or its index, kind or offset, a tensor's dtype or shape, or the decoder view (on one side
    only, another start, another dtype or shape) differ."""
    if want is None or got is None:
        return ["missing in " + ("upstream's run" if want is None else "the port's run")]
    problems = [f"{key} {want.get(key)!r} against {got.get(key)!r}"
                for key in ("layer", "kind", "offset") if want.get(key) != got.get(key)]
    problems += [f"{part} {tensor_type(want[part])} against {tensor_type(got[part])}"
                 for part in ("keys", "values")
                 if tensor_type(want[part]) != tensor_type(got[part])]
    view, other = want.get("decoder_view"), got.get("decoder_view")
    if (view is None) != (other is None):
        problems.append("a decoder view on one side only")
    elif view is not None:
        if view.get("start") != other.get("start"):
            problems.append(f"decoder view start {view.get('start')} against {other.get('start')}")
        problems += [f"decoder view {part} {tensor_type(view[part])} against "
                     f"{tensor_type(other[part])}" for part in ("keys", "values")
                     if tensor_type(view[part]) != tensor_type(other[part])]
    return problems


def compare_caches(want: list, got: list) -> dict:
    """Layer by layer, the structure first: a layer one side lacks, or whose index, kind, offset,
    dtypes, shapes or decoder view differ, is unequal whatever its hashes, as ReadOracleTests
    checks the offset, dtype and shape before the digests. Then the keys and values by SHA-256,
    and the decoder view of each of upstream's sliding layers. Equal layers are listed by index."""
    rows = []
    for index in range(max(len(want), len(got))):
        w = want[index] if index < len(want) else None
        g = got[index] if index < len(got) else None
        problems = layer_structure(w, g)
        rows.append({
            "problems": problems,
            "keys": not problems and w["keys"]["sha256"] == g["keys"]["sha256"],
            "values": not problems and w["values"]["sha256"] == g["values"]["sha256"],
            # each sliding layer of upstream's has a view the port's must match
            "view": None if w is None or "decoder_view" not in w else (
                not problems and all(w["decoder_view"][part]["sha256"]
                                     == g["decoder_view"][part]["sha256"]
                                     for part in ("keys", "values")))})
    views = [r["view"] for r in rows if r["view"] is not None]
    first = next((i for i, r in enumerate(rows) if not (r["keys"] and r["values"])), None)
    out = {"layers": len(want), "layers_in_port": len(got),
           "layers_equal": sum(r["keys"] and r["values"] for r in rows),
           "keys_equal_layers": [i for i, r in enumerate(rows) if r["keys"]],
           "values_equal_layers": [i for i, r in enumerate(rows) if r["values"]],
           "sliding_views_equal": sum(views), "sliding_views": len(views),
           "structure_differences": [{"layer": i, "problems": r["problems"]}
                                     for i, r in enumerate(rows) if r["problems"]],
           "first_differing_layer": None}
    if first is not None:
        r = rows[first]
        out["first_differing_layer"] = {"layer": first, "keys_equal": r["keys"],
                                        "values_equal": r["values"]}
        if r["problems"]:
            out["first_differing_layer"]["structure"] = r["problems"]
        else:
            # how far a differing part is off; equal parts' sums still differ in the last bits,
            # since numpy and the Swift digest add in different orders
            w, g = want[first], got[first]
            for part in ("keys", "values"):
                if not r[part]:
                    out["first_differing_layer"][f"{part}_relative_sum_of_squares_difference"] = (
                        abs(w[part]["sum_of_squares"] - g[part]["sum_of_squares"])
                        / max(abs(w[part]["sum_of_squares"]), 1e-300))
    return out


def configuration_name(port: dict, wheel_sha256: str) -> str:
    wheel = port["metallib"]["sha256"] == wheel_sha256
    table = port.get("rope_table") == "oracle"
    return {(True, True): "exact", (True, False): "wheel_metallib",
            (False, True): "oracle_rope", (False, False): "native"}[(wheel, table)]


DESCRIPTIONS = {
    "exact": "the exact tier: the wheel's metallib and the oracle's RoPE table",
    "native": "the port's own kernels and RoPE table, as the Swift server reads",
    "wheel_metallib": "the wheel's metallib with the port's own RoPE table",
    "oracle_rope": "the port's own kernels with the oracle's RoPE table",
    "chunked_prefill": "mlx-vlm itself with a 64-token chunked prefill (sensitivity.py)",
}
ORDER = ["exact", "native", "wheel_metallib", "oracle_rope", "chunked_prefill"]


def command_compare(args) -> int:
    slot_distribution = import_upstream_engine().slot_distribution
    bodies = {b["id"]: b for b in read_json(args.work / "bodies.json")}
    up = read_json(args.work / "upstream.json")
    wheel = up["metallib"]["sha256"]
    port_paths = args.port or sorted(args.work.glob("port_*.json"))
    ports = [(Path(p), read_json(p)) for p in port_paths]
    files = {(d, s): result_items(d, s) for d in DATASETS for s in SERVERS}

    print(f"upstream {up['upstream_commit']}: Python {up['python']}, MLX {up['packages']['mlx']}, "
          f"mlx-vlm {up['packages']['mlx-vlm']}, metallib {wheel[:8]}, run {up['run_on']} on "
          f"{up['machine']['chip']}")
    summary_items, reference = [], {}
    for item in up["items"]:
        body = bodies[item["id"]]
        reads = item["reads"]  # in seed order
        if not reads or len({read_key(r)[2] for r in reads}) != 1:
            raise SystemExit(f"{item['id']}: {len(reads)} reads over more than one group or none; "
                             "compare averages one group per item, as a JevBench or TypeSafe "
                             "item asks one question")
        probs = [label_probabilities(slot_distribution, r, r["logprobs"]) for r in reads]
        reference[item["id"]] = {"reads": reads, "probs": probs, "item": item,
                                 "caches": {c["prompt_ids_sha256"]: c["layers"]
                                            for c in item["caches"]}}
        answer = item["answers"]
        in_files = {}
        for server in SERVERS:
            recorded = files[(item["dataset"], server)].get(item["id"])
            if recorded is not None:
                in_files[server] = recorded["answer"]
        record = {
            "dataset": item["dataset"], "id": item["id"], "type": body["type"],
            # the options of each slot's label probabilities, in their order
            "questions": item.get("questions"), "body_sha256": body["body_sha256"],
            "result_files_checked": body["result_files_checked"], "seed": item["seed"],
            "prompts": [{"prompt_ids_sha256": ids_digest(p), "tokens": len(p)}
                        for p in dict.fromkeys(tuple(r["prompt_ids"]) for r in reads)],
            "prompt_tokens": [r["prompt_tokens"] for r in reads],
            "read_seed_offsets": [r["seed"] - item["seed"] for r in reads],
            "steps": [r["steps"] for r in reads],
            "label_probabilities": probs,
            "answers": answer,
            "answers_equal_result_files": {s: answer.get("decision") == a
                                           for s, a in in_files.items()},
            "result_file_answers": {s: a for s, a in in_files.items()},
        }
        summary_items.append(record)
        decision = answer.get("decision", {})
        print(f"\n{item['id']} ({item['dataset']}, {body['type']}, prompt tokens "
              f"{sorted(set(record['prompt_tokens']))}, {len(reads)} reads at seed offsets "
              f"{record['read_seed_offsets']})")
        print(f"  upstream's engine: {json.dumps(decision)[:150]}")
        for server, equal in record["answers_equal_result_files"].items():
            print(f"    {'equal to' if equal else 'DIFFERS from'} the {server} result file's"
                  + ("" if equal else f" {json.dumps(in_files[server])[:120]}"))

    configurations = []
    for path, port in ports:
        name = configuration_name(port, wheel)
        configurations.append(compare_port(name, path, port, reference, files, slot_distribution))
    for variant, data in up.get("variants", {}).items():
        configurations.append(compare_variant(variant, data, reference, slot_distribution))
    configurations.sort(key=lambda c: ORDER.index(c["name"]) if c["name"] in ORDER else 99)

    print("\nper configuration (|dp| over the label probabilities of every read; answers whose "
          "top label differs from upstream's):")
    for c in configurations:
        t = c["totals"]
        engine = c.get("engine_totals")
        extra = (f"; engine: same reads {engine['items_with_upstream_reads']}/{engine['items']}, "
                 f"reads identical {engine['reads_identical']}/{engine['reads']}, answers equal "
                 f"{engine['answers_equal_upstream']}/{engine['items']}") if engine else ""
        print(f"  {c['name']:15s} reads identical {t['reads_identical']}/{t['reads']}, mean |dp| "
              f"{t['mean_abs_dp']:.4f}, max {t['max_abs_dp']:.4f} over {t['labels']}, answers "
              f"flipped {t['answers_flipped']}/{t['items']}{extra}")

    if args.summary:
        summary = {
            "what": "PR #105 follow-up: upstream's reads of JevBench and TypeSafe items against "
                    "the port's, by Tools/oracle/item_reads.py and UpstreamProbe ItemReads. "
                    "Bit-for-bit flags compare token ids and float32 logprob bits (reads) and "
                    "SHA-256 (prefill layers); label probabilities are slot_distribution's, per "
                    "read in read order. No state text, prompt ids or reference answers.",
            "upstream": {k: up[k] for k in ("run_on", "machine", "upstream_commit", "python",
                                            "packages", "metallib", "model", "settings")},
            "items": summary_items,
            "configurations": configurations,
        }
        write_json(Path(args.summary), summary, indent=1)
        print(f"\nwrote {shown(Path(args.summary))}")
    return 0


def dp_totals(per_item, reference) -> dict:
    """Mean and largest |dp| over every label probability of every read, and flipped answers."""
    diffs, flipped, identical, reads = [], 0, 0, 0
    for item_id, got in per_item.items():
        want = reference[item_id]["probs"]
        for w, g in paired(want, got["label_probabilities"], f"{item_id}: reads"):
            for ws, gs in paired(w, g, f"{item_id}: the slots of a read"):
                diffs += [abs(a - b) for a, b in paired(ws, gs, f"{item_id}: labels")]
        identical += sum(x is True for x in got["reads_identical"])
        reads += len(got["reads_identical"])
        flipped += any(argmax(a) != argmax(b) for a, b in
                       paired(means(want), means(got["label_probabilities"]),
                              f"{item_id}: slots"))
    return {"reads": reads, "reads_identical": identical, "labels": len(diffs),
            "mean_abs_dp": sum(diffs) / len(diffs) if diffs else 0.0,
            "max_abs_dp": max(diffs, default=0.0), "items": len(per_item),
            "answers_flipped": flipped}


def compare_port(name, path, port, reference, files, slot_distribution) -> dict:
    print(f"\n== {name}: {DESCRIPTIONS.get(name, '')} ({shown(path)}; metallib "
          f"{port['metallib']['sha256'][:8]}, RoPE table {port.get('rope_table')}, run "
          f"{port.get('run_on')})")
    replays = {r["id"]: r for r in port.get("replays", [])}
    engine_items = {r["id"]: r for r in port.get("items", [])}
    if not replays and not engine_items:
        raise SystemExit(f"{shown(path)} holds no reads")
    for part, ids in (("replays", replays), ("engine runs", engine_items)):
        if ids and set(ids) != set(reference):
            raise SystemExit(f"{shown(path)}: its {part} cover {sorted(ids)}, upstream.json's "
                             f"{sorted(reference)}; rerun ItemReads on this work folder")
    per_item, out_items = {}, []
    engine_totals = {"items": 0, "items_with_upstream_reads": 0, "reads": 0,
                     "reads_identical": 0, "answers_equal_upstream": 0}
    for item_id, ref in reference.items():
        want_reads = ref["reads"]
        row = {"id": item_id}
        replay = replays.get(item_id)
        if replay is not None:
            pairs = paired(want_reads, replay["reads"], f"{shown(path)}: {item_id}'s replays")
            identical = [same_maps(w["logprobs"], g["logprobs"]) for w, g in pairs]
            probs = [label_probabilities(slot_distribution, w, g["logprobs"]) for w, g in pairs]
            row["replay"] = {
                "reads_identical": identical,
                "prompt_tokens_equal": all(w["prompt_tokens"] == g["prompt_tokens"]
                                           for w, g in pairs),
                "label_probabilities": probs}
            # upstream's prompts, prefilled by the port: every one of them
            prompts = {c["prompt_ids_sha256"]: c["layers"] for c in replay["caches"]}
            if set(prompts) != set(ref["caches"]):
                raise SystemExit(f"{shown(path)}: {item_id}'s replay digests other prompts than "
                                 "upstream's")
            row["prefill"] = [compare_caches(ref["caches"][d], prompts[d]) for d in ref["caches"]]
            per_item[item_id] = {"reads_identical": identical, "label_probabilities": probs}
        engine = engine_items.get(item_id)
        if engine is not None:
            want = {read_key(r): r for r in want_reads}
            got = {read_key(r): r for r in engine["reads"]}
            same = set(want) == set(got) and len(engine["reads"]) == len(want_reads)
            # upstream's reads in seed order; None for one the engine did not make
            identical = [same_maps(want[k]["logprobs"], got[k]["logprobs"]) if k in got else None
                         for k in (read_key(r) for r in want_reads)]
            answer = engine["answers"]
            upstream_answer = ref["item"]["answers"]
            equals = {"upstream run": answer == upstream_answer}
            for server in SERVERS:
                recorded = files[(ref["item"]["dataset"], server)].get(item_id)
                if recorded is not None:
                    equals[f"{server} result file"] = answer.get("decision") == recorded["answer"]
            row["engine"] = {
                "same_reads_as_upstream": same,
                "read_seed_offsets": sorted(r["seed"] - ref["item"]["seed"]
                                            for r in engine["reads"]),
                "reads_identical": identical,
                "prompt_tokens_equal": sorted(r["prompt_tokens"] for r in engine["reads"])
                == sorted(r["prompt_tokens"] for r in want_reads),
                "answers": answer, "answers_equal": equals,
                "usage": engine["usage"]}
            if replay is None:
                # upstream's prompts that the engine also read; the count says when one is not
                prompts = {c["prompt_ids_sha256"]: c["layers"] for c in engine["caches"]}
                row["prefill"] = [compare_caches(ref["caches"][d], prompts[d])
                                  for d in ref["caches"] if d in prompts]
                row["prefill_prompts"] = f"{len(row['prefill'])} of {len(ref['caches'])}"
            engine_totals["items"] += 1
            engine_totals["items_with_upstream_reads"] += same
            engine_totals["reads"] += len(want_reads)
            engine_totals["reads_identical"] += sum(x is True for x in identical)
            engine_totals["answers_equal_upstream"] += equals["upstream run"]
            if replay is None and same:
                per_item[item_id] = {
                    "reads_identical": identical,
                    "label_probabilities": [label_probabilities(slot_distribution, w,
                                                                got[read_key(w)]["logprobs"])
                                            for w in want_reads]}
        out_items.append(row)
        print_port_item(row)
    result = {"name": name, "description": DESCRIPTIONS.get(name, ""),
              "metallib": port["metallib"], "rope_table": port.get("rope_table"),
              "own_rope_entries_differing_from_oracle":
                  port.get("own_rope_entries_differing_from_oracle"),
              "run_on": port.get("run_on"), "items": out_items,
              "totals": dp_totals(per_item, reference)}
    if engine_totals["items"]:
        result["engine_totals"] = engine_totals
    return result


def print_port_item(row: dict) -> None:
    parts = []
    if "engine" in row:
        e = row["engine"]
        answer = e["answers"].get("decision", {})
        equals = ", ".join(f"{'=' if v else '!='} {k}" for k, v in e["answers_equal"].items())
        parts.append(f"engine: same reads as upstream {e['same_reads_as_upstream']}, identical "
                     f"{sum(x is True for x in e['reads_identical'])}/"
                     f"{len(e['reads_identical'])}, prompt tokens equal "
                     f"{e['prompt_tokens_equal']}; answer {json.dumps(answer)[:110]} ({equals})")
    if "replay" in row:
        r = row["replay"]
        parts.append(f"replay of upstream's reads: identical {sum(r['reads_identical'])}/"
                     f"{len(r['reads_identical'])}, prompt tokens equal {r['prompt_tokens_equal']}")
    if "prefill_prompts" in row:
        parts.append(f"prefill: {row['prefill_prompts']} of upstream's prompts read by the engine")
    for cache in row.get("prefill", []):
        if cache["structure_differences"]:
            parts.append(f"prefill structure differs in {len(cache['structure_differences'])} "
                         f"layers, first: {cache['structure_differences'][0]}")
        first = cache["first_differing_layer"]
        where = "none" if first is None else f"layer {first['layer']} (structure)" if (
            "structure" in first) else (
            f"layer {first['layer']} (keys {'equal' if first['keys_equal'] else 'differ'}, "
            f"values {'equal' if first['values_equal'] else 'differ'}"
            + "".join(f", {part}' sum of squares "
                      f"{first[part + '_relative_sum_of_squares_difference']:.1e} relative"
                      for part in ("keys", "values")
                      if part + "_relative_sum_of_squares_difference" in first) + ")")
        parts.append(f"prefill: {cache['layers_equal']}/{cache['layers']} layers equal, sliding "
                     f"views {cache['sliding_views_equal']}/{cache['sliding_views']}, first "
                     f"difference {where}")
    print(f"  {row['id']}")
    for part in parts:
        print(f"    {part}")


def compare_variant(variant, data, reference, slot_distribution) -> dict:
    name = variant
    print(f"\n== {name}: {DESCRIPTIONS.get(name, 'sensitivity.py variant ' + variant)}")
    per_item, out_items = {}, []
    rows = {row["id"]: row for row in data["items"]}
    if set(rows) != set(reference):
        raise SystemExit(f"upstream.json's {variant} variant covers {sorted(rows)}, its engine "
                         f"run {sorted(reference)}")
    for item_id, ref in reference.items():
        row = rows[item_id]
        # the variant replayed upstream's reads in their order
        pairs = paired(ref["reads"], row["reads"], f"{variant}: {item_id}'s reads")
        identical = [same_maps(w["logprobs"], g["logprobs"]) for w, g in pairs]
        probs = [label_probabilities(slot_distribution, w, g["logprobs"]) for w, g in pairs]
        flipped = [argmax(a) != argmax(b)
                   for a, b in paired(means(ref["probs"]), means(probs), f"{item_id}: slots")]
        per_item[row["id"]] = {"reads_identical": identical, "label_probabilities": probs}
        out_items.append({"id": row["id"], "reads_identical": identical,
                          "label_probabilities": probs, "means": means(probs),
                          "top_label_differs": flipped})
        print(f"  {row['id']}: identical {sum(identical)}/{len(identical)}, means "
              f"{[[round(p, 4) for p in m] for m in means(probs)]} against upstream's "
              f"{[[round(p, 4) for p in m] for m in means(ref['probs'])]}"
              f"{', top label differs' if any(flipped) else ''}")
    return {"name": name, "description": DESCRIPTIONS.get(name, ""),
            "chunk": data.get("chunk"), "items": out_items,
            "totals": dp_totals(per_item, reference)}


# long-slots


def command_long_slots(args) -> int:
    """From committed files: D-014's long-prompt mean |dp| over all labels and over slots with at
    most --max-labels labels, for spike #22's chunked-prefill mlx-vlm and the transliteration on
    native kernels, and the oracle's top-two margins of those slots."""
    sys.path.insert(0, str(HERE))
    import tolerance_stats  # noqa: E402  (its LONG prompts and upstream's slot_distribution)

    oracle = read_json(ROOT / "Fixtures" / "oracle" / "reads.json")
    by_id = {r["id"]: r for r in oracle["reads"]}
    runs = {}
    sensitivity = read_json(RESULTS / "sensitivity.json")["variants"]["chunked_prefill"]["reads"]
    runs["mlx-vlm, 64-token chunked prefill (sensitivity.json)"] = {
        r["id"]: r["per_slot"] for r in sensitivity}
    native = read_json(RESULTS / "transliteration_run.json")["per_read"]
    runs["transliteration, mlx-swift kernels (transliteration_run.json)"] = {
        r["id"]: tolerance_stats.distributions_from_maps(by_id[r["id"]], r["logprobs"])
        for r in native}
    long_reads = [r for r in oracle["reads"] if r["prompt"] in tolerance_stats.LONG]
    slots = [d for r in long_reads for d in r["distributions"]]
    few = [d for d in slots if len(d["probs"]) <= args.max_labels]
    margins = []
    for d in slots:
        ranked = sorted(d["probs"], reverse=True)
        margins.append(ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0))
    print(f"oracle reads over 1,024 prompt tokens: {len(long_reads)} reads, {len(slots)} slots, "
          f"{sum(len(d['probs']) for d in slots):,} labels; {len(slots) - len(few)} slots with "
          f"more than {args.max_labels} labels hold "
          f"{sum(len(d['probs']) for d in slots) - sum(len(d['probs']) for d in few):,} of them, "
          f"{len(few)} slots with at most {args.max_labels} hold "
          f"{sum(len(d['probs']) for d in few)}")
    sizes = {}
    for d in slots:
        sizes[len(d["probs"])] = sizes.get(len(d["probs"]), 0) + 1
    print("slots by label count: " + ", ".join(f"{n} with {k}" for k, n in sorted(sizes.items())))
    print(f"slots whose oracle top-two margin is under 0.5: {sum(m < 0.5 for m in margins)} of "
          f"{len(margins)}")
    for name, run in runs.items():
        every, small = [], []
        for r in long_reads:
            if r["id"] not in run:
                raise SystemExit(f"{name} has no read {r['id']}")
            for want, got in paired(r["distributions"], run[r["id"]], f"{name}: {r['id']}"):
                d = [abs(a - b) for a, b in paired(want["probs"], got["probs"], f"{r['id']}")]
                every += d
                if len(want["probs"]) <= args.max_labels:
                    small += d
        print(f"{name}: mean |dp| {sum(every) / len(every):.4f} over all {len(every):,} labels, "
              f"{sum(small) / len(small):.4f} over the {len(small)} labels of the slots with at "
              f"most {args.max_labels}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=default_work(),
                        help="the folder for the work files (default %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bodies", help="rebuild and check the items' request bodies")
    p.add_argument("items", nargs="+", help="jevbench:ID or typesafe102:ID")
    p = sub.add_parser("upstream", help="upstream's engine on the bodies, every read recorded")
    p.add_argument("--cache-limit-gb", type=float, default=4.0,
                   help="OPENJEV_MLX_CACHE_LIMIT_GB (default 4)")
    p.add_argument("--variants", default="",
                   help="sensitivity.py variants to replay the reads with, comma-separated "
                        "(chunked_prefill, unsorted_decoder_experts, explicit_masks, baseline)")
    p = sub.add_parser("compare", help="the port's runs against upstream's")
    p.add_argument("--port", nargs="*", type=Path,
                   help="the port's run files (default: every port_*.json in the work folder)")
    p.add_argument("--summary", help="write the comparison, without text or prompt ids, here")
    p = sub.add_parser("long-slots", help="D-014's long-prompt figure over few-label slots")
    p.add_argument("--max-labels", type=int, default=4)
    args = parser.parse_args(argv)
    return {"bodies": command_bodies, "upstream": command_upstream, "compare": command_compare,
            "long-slots": command_long_slots}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
