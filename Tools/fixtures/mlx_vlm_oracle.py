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

Every read runs twice, the second time from an emptied prefill cache and in reverse order, so a
read whose prefill was cached the first time is computed from scratch the second time and the
other way round. The two passes must agree bit for bit, or nothing is written.

Timings and memory depend on the machine and on whatever else runs on it, so they go to a
separate file (by default Tools/oracle/results/oracle_run.json), never into the fixture. With
`--check` nothing is written to Fixtures: the run is compared with the committed reads.json,
which is how `--cache-limit-gb 4` is shown to leave every logprob unchanged.

Usage, from the repository root (Tools/oracle/requirements.txt pins the environment):

    make upstream
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
OUT = FIXTURES / "oracle" / "reads.json"
RUN_OUT = ROOT / "Tools" / "oracle" / "results" / "oracle_run.json"
SCRIPT = "Tools/fixtures/mlx_vlm_oracle.py"
# Bump when the shape of reads.json changes.
GENERATOR_VERSION = 1
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


def requests():
    """name -> request body. The fixture requests come from groups-and-canvases (canvas 64);
    long_state is the quickstart's questions over long_state()."""
    cases = {c["name"]: c for c in load_fixture("groups-and-canvases/groups_and_canvases.json")["cases"]
             if c["settings"] == {}}
    out = {name: case["request"] for name, case in cases.items()}
    out["long_state"] = dict(out["quickstart"], state=long_state())
    return out, cases


# Which reads: (request, group index, canvas index, steps). Canvas index 0 is the group seed,
# 1 the second sample seed (+ 7919), 2 seed 0, as groups-and-canvases lists them.
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
]
FIXED_SEEDS = [0, 2**32 - 1, 2**32 + 104729]


def seed_of(body, settings):
    """api.py:260-263, as Tools/fixtures/upstream_tables.py records it in seeds.json."""
    req = SystemOneRequest.model_validate(body)
    questions = {k: q.model_dump() for k, q in req.questions.items()}
    text = json.dumps([req.state, questions], sort_keys=True)
    return req, questions, int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def build_reads(tok):
    """Every read's inputs, from upstream's Engine, checked against the committed fixtures."""
    settings = Settings()
    eng = Engine(settings, tok)
    bodies, cases = requests()
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
        sys_text = eng.system_text(group, fmt, len(groups) > 1)
        state = req.state
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        template, slots = eng.resolve_template(group, fmt)
        group_seed = seed + 104729 * k
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
    return reads, prompts, checks, settings


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
    """One read per row, in the given order. Returns {read id: result} and {read id: timing}."""
    results, timings = {}, {}
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
    return results, timings


def digests_for(rt, reads, prompts):
    """Cache digests for every prompt of the pass that just ran (each is still cached if the
    cache holds at least as many entries as there are prompts)."""
    out = {}
    for r in reads:
        key = r["prompt"]
        if key not in out and tuple(prompts[key]["ids"]) in rt.prefills:
            out[key] = rt.pool.submit(cache_digests, rt, prompts[key]["ids"]).result()
    return out


# Output --------------------------------------------------------------------------------------

def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


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
    args = ap.parse_args()

    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    model_path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    reads, prompts, checks, settings = build_reads(tok)
    print(f"{len(reads)} reads over {len(prompts)} prompts; fixture checks {checks}", file=sys.stderr)
    for key, p in prompts.items():
        print(f"  {key}: {len(p['ids'])} prompt tokens", file=sys.stderr)

    memory = {"before_load": process_memory()}
    started = time.perf_counter()
    rt = MlxRuntime(model_path)
    load_seconds = time.perf_counter() - started
    # More entries than prompts, so every prefill of a pass stays cached for the digests.
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
    results_a, timings_a = run_pass(rt, spy, reads, prompts, settings, order)
    digests_a = digests_for(rt, reads, prompts)
    memory["after_pass_1"] = process_memory()

    rt.init_prefill_cache()
    rt.prompt_cache_entries = max(rt.prompt_cache_entries, len(prompts) + 1)
    gc.collect()
    results_b, timings_b = run_pass(rt, spy, reads, prompts, settings, list(reversed(order)))
    digests_b = digests_for(rt, reads, prompts)
    memory["after_pass_2"] = process_memory()

    same_reads = [rid for rid in results_a if results_a[rid] != results_b[rid]]
    same_digests = digests_a == digests_b
    deterministic = not same_reads and same_digests
    print(f"pass 1 == pass 2: {deterministic} (reads differing: {same_reads}, digests equal: {same_digests})",
          file=sys.stderr)

    payload = {
        "generator": generator(),
        "settings": {"topk": TOPK, "vocab": VOCAB, "canvas": settings.canvas,
                     "canvas_step": settings.canvas_step, "mlx_max_prompt": settings.mlx_max_prompt},
        "fixture_checks": checks,
        "rope": rope_tables(rt),
        "prompts": {k: dict(v, tokens=len(v["ids"]), cache=digests_a.get(k)) for k, v in prompts.items()},
        "reads": [dict(r, **results_a[r["id"]]) for r in reads],
    }

    comparison = None
    if args.check:
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        want = {r["id"]: r for r in committed["reads"]}
        mismatched = [r["id"] for r in payload["reads"] if want.get(r["id"]) != json.loads(dumps(r))]
        if committed.get("rope") != json.loads(dumps(payload["rope"])):
            mismatched.append("rope")
        prompt_mismatch = [k for k, v in payload["prompts"].items()
                           if committed["prompts"].get(k) != json.loads(dumps(v))]
        comparison = {"reads_differing_from_committed": mismatched,
                      "prompts_differing_from_committed": prompt_mismatch}
        print(f"against the committed reads.json: {comparison}", file=sys.stderr)
    elif deterministic:
        write_json(OUT, payload)
    else:
        raise SystemExit("the two passes disagree; reads.json was not written")

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
        "check": comparison,
        "timings": [{"id": r["id"], "pass_1": timings_a[r["id"]], "pass_2": timings_b[r["id"]]} for r in reads],
    }
    write_json(Path(args.run_out), run)
    rt.close()
    if comparison and (comparison["reads_differing_from_committed"] or comparison["prompts_differing_from_committed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
