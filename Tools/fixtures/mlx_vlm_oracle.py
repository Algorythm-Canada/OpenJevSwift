"""Record DiffusionGemma slot logprobs from upstream's own MLX read, as the layer 2 oracle.

Spike #22. Upstream's `openjev.mlx_backend.MlxRuntime(model_path).read(prompt, canvas, slots,
max_tokens, steps)` runs mlx-vlm 0.6.15 on the pinned 4-bit checkpoint. This script feeds it a
fixed set of reads built by upstream's `Engine` from the fixture requests, checks every input
against the committed fixtures, and writes what the read returns to Fixtures/oracle/reads.json:

- per slot, the `{token id: logprob}` map exactly as `MlxRuntime.read` returns it (top 20 plus
  every label, float32 log-softmax), as `[token id, logprob]` pairs in the returned order;
- the label probabilities and entropy from `openjev.engine.slot_distribution`;
- for `steps` above 1, the argmaxes the read writes back at the slot positions between passes,
  seen by a pass-through spy on the model's decoder call;
- the prompt token count, and per prompt a digest of the prefill cache of the first and the last
  layer in temporal order (SHA-256 of the raw bfloat16 bytes, plus float64 sums);
- the full-attention layers' proportional RoPE frequency table as mlx-vlm computed it, and the
  SHA-256 of the mlx-metal wheel's mlx.metallib. A Swift port reproduces every read bit for bit
  when it loads that metallib (MLX.GPU.metallib) and uses that table (spike #22); mlx-swift's own
  kernels round pow, sin and cos differently in the last bit.

Besides the fixture requests, the reads include long JevBench states with few labels (D-048):
each item of JEVBENCH_ITEMS as the one-question request Tools/jevbench/harness.py sends it,
built by JevBench's own typesafe adapter and checked against the body digest both servers'
result files recorded, read as upstream's default policy reads it (the request seed and the
three re-reads at + 7919k). Their states are JevBench's public items, MIT; the dashes in them,
which this repository keeps out of its files, are written as JSON escapes.

Every read runs twice, the second time from an emptied prefill cache and in reverse order, so a
read whose prefill was cached the first time is computed from scratch the second time and the
other way round. The two passes must agree bit for bit, or nothing is written.

Timings and memory depend on the machine and on whatever else runs on it, so they go to a
separate file (by default Tools/oracle/results/oracle_run.json), never into the fixture. With
`--check` nothing is written to Fixtures: the run is compared with the committed reads.json,
every top-level key of it (the pins, settings, fixture checks, RoPE table, prompts and reads),
and the script fails unless both passes agree and nothing differs. That is how
`--cache-limit-gb 4` is shown to leave the oracle unchanged.

Usage, from the repository root (Tools/oracle/requirements.txt pins the environment):

    make upstream
    python3 Tools/jevbench/harness.py fetch
    python3.14 -m venv Tools/oracle/.venv
    Tools/oracle/.venv/bin/pip install -r Tools/oracle/requirements.txt
    Tools/oracle/.venv/bin/python Tools/oracle/fetch_checkpoint.py
    PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/mlx_vlm_oracle.py

The script lives in Tools/fixtures because Tests/OpenJevCoreTests/Fixtures/FixturePinTests.swift
requires every fixture's generator to be a script there. It runs from Tools/oracle/.venv, not
Tools/fixtures/.venv, because it needs MLX and the 16.6 GB weights.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
FIXTURES = ROOT / "Fixtures"
JEVBENCH = ROOT / "Tools" / "jevbench"
OUT = FIXTURES / "oracle" / "reads.json"
RUN_OUT = ROOT / "Tools" / "oracle" / "results" / "oracle_run.json"
SCRIPT = "Tools/fixtures/mlx_vlm_oracle.py"
# Bump when the shape of reads.json changes. 2: the JevBench reads and their `jevbench` records.
GENERATOR_VERSION = 2
UPSTREAM_COMMIT = "dcd2094"
MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"

# Settings read the environment; clear it so the defaults are upstream's own.
for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Everything is read from the cache; fetch_checkpoint.py downloads it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(JEVBENCH))

import harness as jevbench_harness  # noqa: E402  Tools/jevbench/harness.py, standard library only
import huggingface_hub  # noqa: E402
import jinja2  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402
from importlib.metadata import version as package_version  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openjev.api import SystemOneRequest  # noqa: E402
from openjev.config import Settings  # noqa: E402
from openjev.engine import TOPK, VOCAB, Engine, slot_distribution  # noqa: E402
from openjev.mlx_backend import MlxRuntime  # noqa: E402


# Pins ----------------------------------------------------------------------------------------

def upstream_head():
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_info():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return {"device_name": info.get("device_name"), "architecture": info.get("architecture"),
            "memory_size": info.get("memory_size")}


def metallib():
    """The Metal library the wheel ships: its path and SHA-256."""
    path = Path(mx.__file__).parent / "lib" / "mlx.metallib"
    return {"path": str(path.relative_to(Path(mx.__file__).parents[1])),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rope_tables(rt):
    """The frequencies mlx-vlm's ProportionalRoPE passes to mx.fast.rope for full attention,
    as float32 bit patterns (the unrotated pairs are infinite) and as values."""
    text = rt.model.config.text_config
    layer = text.layer_types.index("full_attention")
    freqs = np.array(rt.model.model.decoder.layers[layer].self_attn.rope._freqs).astype(np.float32)
    finite = [float(x) for x in freqs if np.isfinite(x)]
    return {"layer_type": "full_attention", "dims": int(text.global_head_dim), "count": int(freqs.size),
            "finite": len(finite), "float32_bits": [int(b) for b in freqs.view(np.uint32)],
            "finite_values": finite}


def generator():
    """What reads.json depends on. The device is recorded because Metal kernels may round
    differently on another GPU family; nothing here changes between runs on one machine."""
    return {
        "script": SCRIPT,
        "version": GENERATOR_VERSION,
        "upstream": "razorback16/openjev",
        "upstream_commit": UPSTREAM_COMMIT,
        "tokenizer_repo": MODEL_REPO,
        "tokenizer_revision": MODEL_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "python": sys.version.split()[0],
        "mlx": package_version("mlx"),
        "mlx_metal": package_version("mlx-metal"),
        "mlx_vlm": package_version("mlx-vlm"),
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "jinja2": jinja2.__version__,
        "numpy": np.__version__,
        "device": device_info()["device_name"],
        "gpu_architecture": device_info()["architecture"],
        "metallib_sha256": metallib()["sha256"],
        "jevbench": jevbench_harness.JEVBENCH_REPO,
        "jevbench_commit": jevbench_harness.JEVBENCH_COMMIT,
    }


# Requests ------------------------------------------------------------------------------------

def load_fixture(relpath):
    return json.loads((FIXTURES / relpath).read_text(encoding="utf-8"))


def long_state():
    """A support thread long enough that the prompt passes the 1,024-token sliding window almost
    three times over, so the decoder's sliding layers see only the last 1,023 of about 3,000
    encoder positions (R15). Deterministic: the text is a pure function of the line number."""
    lines = []
    for i in range(1, 85):
        code = f"{(i * 7919) % 10000:04d}"
        hour, minute, count = 8 + i % 10, (i * 13) % 60, i % 5 + 1
        lines.append([
            f"Customer ({i}): I tried to connect my Stripe account again at {hour}:{minute:02d} "
            f"and the dashboard still answers with a 403 error.",
            f"Agent ({i}): Thanks for the update. Could you confirm the account ID ending in {code} "
            f"and whether two-factor authentication is on?",
            f"Customer ({i}): The account ID ends in {code}. Two-factor is on, and I regenerated "
            f"the API keys {count} times this week.",
            f"Agent ({i}): I see {count} failed OAuth handshakes in our logs; the last one "
            f"reported that the read_write scope was missing.",
        ][i % 4])
    return "\n".join(lines)


def jevbench_requests(cache):
    """item id -> (request body, record) for JEVBENCH_ITEMS. The body is the one
    Tools/jevbench/harness.py sends for the item: built by JevBench's typesafe adapter (vendored
    unchanged) for JEVBENCH_MODEL, and checked against the body digest that both servers' result
    files recorded, so the reads are those of the runs quality.md compares."""
    try:
        dataset = jevbench_harness.load_dataset("jevbench", cache, fetch=False)
    except jevbench_harness.PinError as error:
        raise SystemExit(f"{error}; run python3 Tools/jevbench/harness.py fetch")
    adapter = jevbench_harness.jb_typesafe.TypeSafeAdapter(endpoint="http://127.0.0.1", model=JEVBENCH_MODEL,
                                                           key_env="")
    sent = {}
    for name in JEVBENCH_RESULTS:
        doc = jevbench_harness.read_result(JEVBENCH / "results" / name)
        if doc["model"] != JEVBENCH_MODEL:
            raise SystemExit(f"Tools/jevbench/results/{name} is a run of {doc['model']}, not {JEVBENCH_MODEL}")
        sent[name] = {item["id"]: (item.get("request") or {}).get("body_sha256") for item in doc["items"]}
    files = {tier: path for path, (tier, _, _) in jevbench_harness.JEVBENCH_SPLITS.items()}
    out = {}
    for item_id in JEVBENCH_ITEMS:
        task = dataset.by_id[item_id]
        body = adapter.build_request(task)
        digest = jevbench_harness.sha256_bytes(json.dumps(body).encode("utf-8"))
        for name, digests in sent.items():
            if digests.get(item_id) != digest:
                raise SystemExit(f"{item_id}: the request body differs from the one Tools/jevbench/results/"
                                 f"{name} recorded")
        out[item_id] = (body, {"file": files[dataset.tiers[item_id]], "family": task.family,
                               "type": task.question["type"], "body_sha256": digest})
    return out


def requests(jevbench_cache):
    """name -> request body. The fixture requests come from groups-and-canvases (canvas 64);
    long_state is the quickstart's questions over long_state(); a JevBench item's name is its id.
    Also returns the fixture cases and the JevBench items' records."""
    cases = {c["name"]: c for c in load_fixture("groups-and-canvases/groups_and_canvases.json")["cases"]
             if c["settings"] == {}}
    out = {name: case["request"] for name, case in cases.items()}
    out["long_state"] = dict(out["quickstart"], state=long_state())
    jevbench = jevbench_requests(jevbench_cache)
    out.update({name: body for name, (body, _) in jevbench.items()})
    return out, cases, {name: record for name, (_, record) in jevbench.items()}


# JevBench items with prompts over 1,024 tokens and 2 to 4 labels (D-048): one question with a few
# labels, as JevBench and TypeSafe ask (2 to 6 and 2 to 8), which the fixture's long reads above
# hold in few slots. The
# first three are PR #105's wide-margin long-prompt flips, which the exact tier showed to be
# native-kernel noise; both servers answered the others alike, at upstream top-two margins from
# 0.005 to 0.99 (Tools/jevbench/results/openjev-0.1-*.json).
JEVBENCH_ITEMS = [
    "hard-opus-c-long_policy-04",  # noul, 3,264 prompt tokens, upstream margin 0.45, flipped
    "hard-sol-b-long_policy-06",   # choice of 4, 2,647 tokens, 0.27, flipped
    "hard-opus-a-long_policy-19",  # noul, 2,318 tokens, 0.25, flipped
    "hard-opus-b-multi_hop-03",    # choice of 4, 2,989 tokens, 0.005
    "hard-sol-a-multi_hop-08",     # choice of 4 over a JSON state, 2,335 tokens, 0.45
    "hard-opus-c-long_policy-08",  # score of 4 levels, 3,322 tokens, 0.55
    "hard-opus-a-long_policy-08",  # choice of 4, 2,396 tokens, 0.80
    "hard-opus-c-long_policy-10",  # choice of 4, 3,643 tokens, 0.96
    "hard-opus-b-multi_hop-08",    # noul, 2,406 tokens, 0.99
]
JEVBENCH_MODEL = "openjev-0.1"
JEVBENCH_RESULTS = ["openjev-0.1-upstream.json", "openjev-0.1-swift.json"]
# Upstream's default read policy (OPENJEV_AUTO_MAX 4): the first read and three re-reads.
POLICY_READS = 4

# Which reads: (request, group index, canvas index, steps). For a fixture request, canvas index 0
# is the group seed, 1 the second sample seed (+ 7919), 2 seed 0, as groups-and-canvases lists
# them. For a JevBench item, canvas index k is read k of upstream's default policy, at the group
# seed + 7919k (read_group's `seed + k * 7919`).
SELECTION = [
    ("single_noul", 0, 0, 1), ("single_noul", 0, 2, 1),              # width 16, one question
    ("quickstart", 0, 0, 1), ("quickstart", 0, 1, 1), ("quickstart", 0, 2, 1),  # width 32
    ("nouls_8", 0, 0, 1), ("nouls_8", 0, 1, 1),                      # width 48
    ("lines_10_mixed", 0, 0, 1), ("lines_10_mixed", 0, 1, 1),        # width 64
    ("six_questions", 0, 0, 1),
    ("non_ascii", 0, 0, 1),
    ("indexed_12_mixed", 0, 0, 1), ("indexed_12_mixed", 0, 1, 1),    # 12 mixed questions
    ("many_choices", 0, 0, 1), ("many_choices", 0, 1, 1),            # 100, 100 and 55 options
    ("widest_schema", 0, 0, 1), ("widest_schema", 0, 1, 1),          # the 255-option choice
    ("nouls_24", 0, 0, 1), ("nouls_24", 1, 0, 1),                    # two groups: chunked text
    ("long_state", 0, 0, 1), ("long_state", 0, 1, 1),                # prompt over 1,024 tokens
    ("quickstart", 0, 0, 2), ("quickstart", 0, 0, 3),                # steps above 1
    ("lines_10_mixed", 0, 0, 2), ("lines_10_mixed", 0, 0, 3),
    ("long_state", 0, 0, 2), ("long_state", 0, 0, 3),
] + [(item, 0, k, 1) for item in JEVBENCH_ITEMS for k in range(POLICY_READS)]
FIXED_SEEDS = [0, 2**32 - 1, 2**32 + 104729]


def seed_of(body, settings):
    """api.py:260-263, as Tools/fixtures/upstream_tables.py records it in seeds.json."""
    req = SystemOneRequest.model_validate(body)
    questions = {k: q.model_dump() for k, q in req.questions.items()}
    text = json.dumps([req.state, questions], sort_keys=True)
    return req, questions, int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def build_reads(tok, jevbench_cache):
    """Every read's inputs, from upstream's Engine, checked against the committed fixtures."""
    settings = Settings()
    if settings.auto_max != POLICY_READS:
        raise SystemExit(f"upstream's auto_max is {settings.auto_max}, expected {POLICY_READS}")
    eng = Engine(settings, tok)
    bodies, cases, jevbench = requests(jevbench_cache)
    prompt_rows = {}
    for row in load_fixture("chat-prompts/prompts.json")["cases"]:
        key = (row["messages"][0]["content"], row["messages"][1]["content"])
        prompt_rows[key] = row["thinking_off"]["ids"]
    # The reads upstream's engine made for the policy requests that MlxEngine would prefill
    # from chat_prompt_ids(sys_text, content): the same prompt and canvas count as recorded.
    policy_reads = {}
    for case in load_fixture("policies/policies.json")["cases"]:
        for r in case["reads"]:
            if r.get("mlx_prompt") == "chat_prompt_ids" and isinstance(r["content"], str):
                policy_reads[(r["sys_text"], r["content"], tuple(r["canvas"]))] = case["name"]

    prompts, reads, checks = {}, [], {"canvases": 0, "prompts_in_fixture": 0, "policy_reads": 0}
    for name, k, ci, steps in SELECTION:
        req, questions, seed = seed_of(bodies[name], settings)
        schema = eng.build_schema(questions)
        qs, fmt = schema["questions"], schema["format"]
        groups = eng.groups(qs, fmt)
        group = groups[k]
        if name in jevbench:
            # the slot's labels in upstream's order (a noul's yes first), as its probabilities read
            jevbench[name]["options"] = [choice for choice, _ in qs[0]["choices"]]
        sys_text = eng.system_text(group, fmt, len(groups) > 1)
        state = req.state
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        template, slots = eng.resolve_template(group, fmt)
        group_seed = seed + 104729 * k
        if name in jevbench:
            canvas_seed = group_seed + 7919 * ci
        else:
            canvas_seed = ([group_seed, group_seed + 7919] + FIXED_SEEDS)[ci]
        canvas = eng.build_canvas(template, slots, canvas_seed)
        ids = eng.chat_prompt_ids(sys_text, state_text)

        if name in cases:  # the fixture recorded this group: every input must match it
            fg = cases[name]["groups"][k]
            fc = fg["canvases"][ci]
            if (fg["template"] != list(template) or fg["slots"] != [
                    {"pos": s["pos"], "label_ids": list(s["label_ids"])} for s in slots]
                    or fg["width"] != len(canvas) or fc["seed"] != canvas_seed or fc["canvas"] != canvas
                    or cases[name]["seed"] != seed):
                raise SystemExit(f"{name} group {k}: inputs differ from groups_and_canvases.json")
            checks["canvases"] += 1
        fixture_ids = prompt_rows.get((sys_text, state_text))
        if fixture_ids is not None:
            if fixture_ids != ids:
                raise SystemExit(f"{name} group {k}: prompt ids differ from chat-prompts/prompts.json")
            checks["prompts_in_fixture"] += 1
        if (sys_text, state_text, tuple(canvas)) in policy_reads:
            checks["policy_reads"] += 1

        prompt_key = f"{name}/g{k}"
        if prompt_key not in prompts:
            prompts[prompt_key] = {"system": sys_text, "user": state_text, "ids": ids,
                                   "in_chat_prompts_fixture": fixture_ids is not None}
        reads.append({
            "id": f"{name}/g{k}/c{ci}/steps{steps}",
            "request": name, "group": k, "format": fmt, "questions": [q["id"] for q in group],
            "prompt": prompt_key, "canvas_index": ci, "canvas_seed": canvas_seed,
            "width": len(canvas), "canvas": canvas,
            "slots": [{"pos": s["pos"], "label_ids": list(s["label_ids"])} for s in slots],
            "steps": steps,
        })
    return reads, prompts, checks, settings, jevbench


# Memory --------------------------------------------------------------------------------------

class RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(35)]


def process_memory():
    """Resident size, physical footprint and its lifetime peak (what Activity Monitor shows),
    from proc_pid_rusage(RUSAGE_INFO_V4), plus MLX's own allocator counters."""
    info = RUsageInfoV4()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    rc = libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    out = {}
    if rc == 0:
        out = {"resident_bytes": info.f6, "phys_footprint_bytes": info.f7,
               "lifetime_max_phys_footprint_bytes": info.f28}
    out.update({"mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(),
                "mlx_peak_bytes": mx.get_peak_memory()})
    return out


# Prefill cache digests -----------------------------------------------------------------------

def tensor_digest(a):
    """SHA-256 of the raw bytes in C order, the dtype and shape, and float64 sums, so a later
    test can compare exactly (the hash) or within a tolerance (the sums)."""
    raw = np.array(mx.view(a, mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)
    f = np.array(a.astype(mx.float32)).astype(np.float64)
    return {"dtype": str(a.dtype).replace("mlx.core.", ""), "shape": list(a.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            "sum": float(f.sum()), "sum_of_squares": float((f * f).sum()),
            "max_abs": float(np.abs(f).max())}


def cache_digests(rt, prompt_ids):
    """The first layer (sliding) and the last layer (full) of the cached prefill, in temporal
    order. For the sliding layer both the whole state mlx-vlm holds and the last 1,023
    positions (the decoder's view) are digested: a one-shot prefill keeps every position in
    mlx-vlm's RotatingKVCache, while a ring of 1,024 slots keeps only the tail."""
    from mlx_vlm.models.diffusion_gemma.language import _cache_state

    cache, n = rt.prefills[tuple(prompt_ids)]
    text = rt.model.config.text_config
    window = text.sliding_window - 1
    out = []
    for layer in (0, len(cache) - 1):
        keys, values = _cache_state(cache[layer])
        kind = text.layer_types[layer]
        row = {"layer": layer, "kind": kind, "offset": int(cache[layer].offset),
               "positions": [0, int(keys.shape[2])],
               "keys": tensor_digest(keys), "values": tensor_digest(values)}
        if kind == "sliding_attention":
            start = max(0, int(keys.shape[2]) - window)
            row["decoder_view"] = {"positions": [start, int(keys.shape[2])],
                                   "keys": tensor_digest(keys[:, :, start:, :]),
                                   "values": tensor_digest(values[:, :, start:, :])}
        out.append(row)
    return out


# Reads ---------------------------------------------------------------------------------------

class DecoderSpy:
    """Pass-through wrapper on model.diffusion_decoder_logits that notes the canvas each pass
    receives. MlxRuntime.read evaluates the canvas before every pass, so reading it here adds
    no work and changes nothing."""

    def __init__(self, model):
        self.original = model.diffusion_decoder_logits
        self.seen = []
        model.diffusion_decoder_logits = self

    def __call__(self, canvas_ids, cache=None, self_conditioning=None, decoder_attention_mask=None):
        self.seen.append(canvas_ids.tolist()[0])
        return self.original(canvas_ids, cache=cache, self_conditioning=self_conditioning,
                             decoder_attention_mask=decoder_attention_mask)


def run_pass(rt, spy, reads, prompts, settings, order):
    """One read per row, in the given order. Returns {read id: result}, {read id: timing} and
    {prompt key: cache digests}. A prompt's digests are taken right after its first read of the
    pass, while upstream's prefill cache, bounded in entries and in tokens, still holds it."""
    results, timings, digests = {}, {}, {}
    for i in order:
        r = reads[i]
        ids = prompts[r["prompt"]]["ids"]
        cached = tuple(ids) in rt.prefills
        spy.seen = []
        started = time.perf_counter()
        tops, n = rt.pool.submit(rt.read, ids, r["canvas"], r["slots"], settings.mlx_max_prompt,
                                 r["steps"]).result()
        seconds = time.perf_counter() - started
        seen = spy.seen
        if len(seen) != r["steps"] or seen[0] != r["canvas"]:
            raise SystemExit(f"{r['id']}: the decoder saw {len(seen)} canvases, expected {r['steps']}")
        positions = {s["pos"] for s in r["slots"]}
        for later in seen[1:]:
            if any(a != b for p, (a, b) in enumerate(zip(later, r["canvas"])) if p not in positions):
                raise SystemExit(f"{r['id']}: a pinned canvas position changed between passes")
        results[r["id"]] = {
            "prompt_tokens": n,
            "written": [[c[s["pos"]] for s in r["slots"]] for c in seen[1:]],
            "logprobs": [[[int(t), float(v)] for t, v in top.items()] for top in tops],
            "distributions": [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, r["slots"])],
        }
        timings[r["id"]] = {"seconds": seconds, "prefill_cached": cached, "prompt_tokens": n}
        if r["prompt"] not in digests:
            if tuple(ids) not in rt.prefills:
                raise SystemExit(f"{r['prompt']}: the prefill cache dropped the prompt it just read")
            digests[r["prompt"]] = rt.pool.submit(cache_digests, rt, ids).result()
    return results, timings, digests


# Output --------------------------------------------------------------------------------------

# The dashes this repository keeps out of its files, which JevBench's states use, are written as
# JSON escapes and read back as the same text. Nothing else is escaped, so the reads recorded
# before the JevBench ones keep their bytes.
DASHES = {code: f"\\u{code:04x}" for code in range(0x2012, 0x2016)}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).translate(DASHES)


def compare_with_committed(payload, committed):
    """What differs between a run's payload and the committed reads.json, key by key: the names
    of differing generator pins, the ids of differing reads, the keys of differing prompts, and
    True for any other top-level key that differs. Empty when the run reproduces the file."""
    fresh = json.loads(dumps(payload))
    differences = {}
    for key in sorted(set(fresh) | set(committed)):
        mine, theirs = fresh.get(key), committed.get(key)
        if mine == theirs:
            continue
        if key == "reads" and isinstance(mine, list) and isinstance(theirs, list):
            by_mine = {r.get("id"): r for r in mine}
            by_theirs = {r.get("id"): r for r in theirs}
            ids = sorted(i for i in set(by_mine) | set(by_theirs) if by_mine.get(i) != by_theirs.get(i))
            differences[key] = ids or ["order"]
        elif key in ("prompts", "generator", "jevbench") and isinstance(mine, dict) and isinstance(theirs, dict):
            names = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
            differences[key] = names or ["order"]
        else:
            differences[key] = True
    return differences


def write_json(path, payload):
    """One top-level key per line and one list entry per line (the layout of the other
    fixtures): readable diffs, small files."""
    lines = []
    for key, value in payload.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        elif isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            items = ",\n".join(f" {json.dumps(k)}: {dumps(v)}" for k, v in value.items())
            lines.append(f"{json.dumps(key)}: {{\n{items}\n}}")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"wrote {shown}: {len(text.encode('utf-8'))} bytes", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed reads.json instead of writing it")
    ap.add_argument("--cache-limit-gb", type=float, default=None,
                    help="MlxRuntime.set_cache_limit before the reads, as OPENJEV_MLX_CACHE_LIMIT_GB does")
    ap.add_argument("--run-out", default=str(RUN_OUT), help="where timings and memory go")
    ap.add_argument("--jevbench-cache", default=str(jevbench_harness.default_cache()),
                    help="the cache Tools/jevbench/harness.py fetch downloaded JevBench into")
    args = ap.parse_args()

    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    model_path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    reads, prompts, checks, settings, jevbench = build_reads(tok, Path(args.jevbench_cache))
    print(f"{len(reads)} reads over {len(prompts)} prompts; fixture checks {checks}", file=sys.stderr)
    for key, p in prompts.items():
        print(f"  {key}: {len(p['ids'])} prompt tokens", file=sys.stderr)

    memory = {"before_load": process_memory()}
    started = time.perf_counter()
    rt = MlxRuntime(model_path)
    load_seconds = time.perf_counter() - started
    # More entries than prompts, so that only upstream's 16,384-token budget evicts: the fixture
    # requests' prompts stay cached for each other's reads, the long JevBench ones in turn.
    rt.prompt_cache_entries = max(rt.prompt_cache_entries, len(prompts) + 1)
    rt.set_cache_limit(args.cache_limit_gb)
    memory["after_load"] = process_memory()
    spy = DecoderSpy(rt.model)

    # One unrecorded read compiles the Metal kernels, so the first recorded read is not charged
    # for it; the prefill cache is emptied afterwards.
    first = reads[0]
    warm_started = time.perf_counter()
    rt.pool.submit(rt.read, prompts[first["prompt"]]["ids"], first["canvas"], first["slots"],
                   settings.mlx_max_prompt, 1).result()
    warmup_seconds = time.perf_counter() - warm_started
    rt.init_prefill_cache()
    rt.prompt_cache_entries = max(rt.prompt_cache_entries, len(prompts) + 1)

    order = list(range(len(reads)))
    results_a, timings_a, digests_a = run_pass(rt, spy, reads, prompts, settings, order)
    memory["after_pass_1"] = process_memory()

    rt.init_prefill_cache()
    rt.prompt_cache_entries = max(rt.prompt_cache_entries, len(prompts) + 1)
    gc.collect()
    results_b, timings_b, digests_b = run_pass(rt, spy, reads, prompts, settings, list(reversed(order)))
    memory["after_pass_2"] = process_memory()

    same_reads = [rid for rid in results_a if results_a[rid] != results_b[rid]]
    same_digests = digests_a == digests_b
    deterministic = not same_reads and same_digests
    print(f"pass 1 == pass 2: {deterministic} (reads differing: {same_reads}, digests equal: {same_digests})",
          file=sys.stderr)
    # Upstream re-reads a group only when a slot of its first read is above the threshold, so the
    # JevBench re-reads are its reads only then.
    not_reread = [name for name in jevbench
                  if max(d["entropy"] for d in results_a[f"{name}/g0/c0/steps1"]["distributions"])
                  <= settings.auto_threshold]
    if not_reread:
        print(f"first read at or below the re-read threshold {settings.auto_threshold}: {not_reread}",
              file=sys.stderr)

    payload = {
        "generator": generator(),
        "settings": {"topk": TOPK, "vocab": VOCAB, "canvas": settings.canvas,
                     "canvas_step": settings.canvas_step, "mlx_max_prompt": settings.mlx_max_prompt},
        "fixture_checks": checks,
        "rope": rope_tables(rt),
        "jevbench": jevbench,
        "prompts": {k: dict(v, tokens=len(v["ids"]), cache=digests_a.get(k)) for k, v in prompts.items()},
        "reads": [dict(r, **results_a[r["id"]]) for r in reads],
    }

    comparison = None
    if args.check:
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        comparison = {"differences_from_committed": compare_with_committed(payload, committed)}
        print(f"against the committed reads.json: {comparison}", file=sys.stderr)
    elif deterministic and not not_reread:
        write_json(OUT, payload)

    run = {
        "generator": {k: v for k, v in generator().items()},
        "machine": {"platform": platform.platform(), "mac_ver": platform.mac_ver()[0],
                    "machine": platform.machine(), **device_info()},
        "cache_limit_gb": args.cache_limit_gb,
        "load_seconds": load_seconds,
        "warmup_read_seconds": warmup_seconds,
        "memory": memory,
        "deterministic": deterministic,
        "reads_differing_between_passes": same_reads,
        "jevbench_first_reads_not_reread": not_reread,
        "check": comparison,
        "timings": [{"id": r["id"], "pass_1": timings_a[r["id"]], "pass_2": timings_b[r["id"]]} for r in reads],
    }
    write_json(Path(args.run_out), run)
    rt.close()
    # The run file is written either way, so a failed run leaves its evidence behind.
    if not deterministic:
        raise SystemExit("the two passes disagree" + ("" if args.check else "; reads.json was not written"))
    if not_reread:
        raise SystemExit(f"upstream would not re-read {not_reread}; take their re-reads out of the selection"
                         + ("" if args.check else "; reads.json was not written"))
    if comparison and comparison["differences_from_committed"]:
        raise SystemExit(f"the run differs from the committed reads.json: {comparison['differences_from_committed']}")


if __name__ == "__main__":
    main()
