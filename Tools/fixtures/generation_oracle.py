"""Record DiffusionGemma text generation from upstream's own MLX path, as the generation oracle.

Issues #50, #51 and #52. Upstream's `openjev.mlx_backend.MlxRuntime.generate(prompt, max_tokens,
stop_ids, emit, skip_special)` runs mlx-vlm 0.6.15's `stream_diffusion_generate` greedily
(`temperature=0.0`, as upstream calls it) on the pinned 4-bit checkpoint, and `MlxEngine.think`
runs it for a thought before a read. This script writes Fixtures/generation/generation.json:

- `sampler`: test vectors for the sampling functions of mlx-vlm's `generate/diffusion.py` (lines
  285 to 505) on small synthetic logits, computed on the CPU: `_diffusion_initialize_canvas`
  after `mx.random.seed`, `_diffusion_linear_temperature`, `_diffusion_sample_canvas` at
  temperature 0 and above after `mx.random.seed`, `_diffusion_token_probability`,
  `_diffusion_token_entropy`, `_diffusion_confidence_transfer_mask`,
  `_diffusion_entropy_transfer_mask` and `_diffusion_stable_and_confident`. Float arrays are
  float32 bit patterns.
- `generations`: chat replies through upstream's own chat prompt (`MlxGenerator.prompt_ids`) and
  `MlxRuntime.generate`, with the thought-channel markers skipped as `MlxGenerator.generate` skips
  them: a short answer, a list, a JSON reply, a reply of several blocks, one ended by an extra stop
  id, one cut by `max_tokens`, and one after a prompt longer than the sliding window. Per block: the canvas length, the initial canvas, the denoising
  steps taken, why the block ended, the final canvas, and the tokens the block committed. Per reply:
  the generated ids, the prompt tokens, the finish reason, the stop token, and every `emit(text,
  token)` call in order.
- `think`: `MlxEngine.decide` with `think` for the README example and for a sequential request of
  24 questions: per thought the prompt, the ids `MlxRuntime.generate` returned, the thought ids
  after the cut, the prefix, and the billing; then the answers, the billed input and the thought
  tokens of the whole request.

Seeding. `_diffusion_initialize_canvas` draws the initial canvas and every re-noised position
with `mx.random.randint` from MLX's global generator, which upstream never seeds, so upstream's
replies are not reproducible from one process to the next. This script seeds MLX's generator with
`mx.random.seed(seed)` before each generation and each `decide` and records the seed, so a port
that draws from a generator seeded the same way (mlx-swift's `MLXRandom.RandomState(seed:)`
splits keys as MLX's global `KeySequence` does) reproduces every reply.

Every generation runs twice, the second pass in reverse order from emptied prefill caches. The
two passes must agree bit for bit, or nothing is written. Timings go to a separate file (by
default Tools/oracle/results/generation_run.json). With `--check` nothing is written to
Fixtures: the run is compared with the committed generation.json key by key.

Usage, from the repository root, in the environment of Tools/fixtures/mlx_vlm_oracle.py:

    PYTHONHASHSEED=0 Tools/oracle/.venv/bin/python Tools/fixtures/generation_oracle.py \\
        --cache-limit-gb 4
"""
import argparse
import asyncio
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
FIXTURES = ROOT / "Fixtures"
OUT = FIXTURES / "generation" / "generation.json"
RUN_OUT = ROOT / "Tools" / "oracle" / "results" / "generation_run.json"
SCRIPT = "Tools/fixtures/generation_oracle.py"
# Bump when the shape of generation.json changes. 2: the `long_prompt` reply and dashes escaped.
GENERATOR_VERSION = 2
UPSTREAM_COMMIT = "dcd2094"
MODEL_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
MODEL_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"

for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(UPSTREAM))

import huggingface_hub  # noqa: E402
import jinja2  # noqa: E402
import mlx.core as mx  # noqa: E402
from importlib import import_module  # noqa: E402
import numpy as np  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402
from importlib.metadata import version as package_version  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openjev.api import SystemOneRequest  # noqa: E402
from openjev.chat import MlxGenerator  # noqa: E402
from openjev.config import Settings  # noqa: E402
from openjev.engine import Engine  # noqa: E402
from openjev.mlx_backend import MlxEngine, MlxRuntime  # noqa: E402

# By module path: the package's `generate` attribute is the function, which `import a.b.c as d`
# would resolve through.
diffusion = import_module("mlx_vlm.generate.diffusion")


# Pins ----------------------------------------------------------------------------------------

def upstream_head():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def device_info():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return {"device_name": info.get("device_name"), "architecture": info.get("architecture"),
            "memory_size": info.get("memory_size")}


def metallib_sha256():
    path = Path(mx.__file__).parent / "lib" / "mlx.metallib"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator():
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
        "metallib_sha256": metallib_sha256(),
    }


# Sampler test vectors ------------------------------------------------------------------------

def bits(a):
    """A float32 array as its shape and uint32 bit patterns in C order."""
    a = np.array(a.astype(mx.float32) if isinstance(a, mx.array) else a, dtype=np.float32)
    return {"shape": list(a.shape), "float32_bits": [int(b) for b in a.reshape(-1).view(np.uint32)]}


def ints(a):
    a = np.array(a)
    return {"shape": list(a.shape), "values": [int(v) for v in a.reshape(-1)]}


def bools(a):
    a = np.array(a)
    return {"shape": list(a.shape), "values": [bool(v) for v in a.reshape(-1)]}


def synthetic_logits(seed, shape, scale):
    """Deterministic float32 logits: numpy's PCG64 normals times `scale`."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def peaked_logits(seed, shape, peaks):
    """Logits whose rows are near one-hot: row r has a peak of height peaks[r] at a random index,
    so their entropies run from about 0 to well above 0.1."""
    rng = np.random.default_rng(seed)
    out = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    flat = out.reshape(-1, shape[-1])
    for r in range(flat.shape[0]):
        flat[r, rng.integers(shape[-1])] += np.float32(peaks[r % len(peaks)])
    return flat.reshape(shape)


def sampler_vectors():
    """The sampling functions of diffusion.py lines 285 to 505 on synthetic inputs, on the CPU."""
    out = {}
    vocab = 262144
    with mx.stream(mx.cpu):
        # The initial canvas and the re-noise draws: two draws after one seed, of each length the
        # block loop uses.
        draws = []
        for seed in (0, 1234, 2**32 + 7):
            mx.random.seed(seed)
            first = diffusion._diffusion_initialize_canvas(1, 64, vocab, mx.int32)
            second = diffusion._diffusion_initialize_canvas(1, 256, vocab, mx.int32)
            mx.eval(first, second)
            draws.append({"seed": seed, "vocab_size": vocab,
                          "draws": [ints(first), ints(second)]})
        out["initialize_canvas"] = draws

        schedule = {"t_min": 0.4, "t_max": 0.8}
        out["linear_temperature"] = {
            "schedule": schedule, "max_denoising_steps": 48,
            "values": [diffusion._diffusion_linear_temperature(s, 48, schedule) for s in range(48, 0, -1)],
            "float32_bits": [int(np.float32(diffusion._diffusion_linear_temperature(s, 48, schedule)).view(np.uint32))
                             for s in range(48, 0, -1)],
            "none_schedule": diffusion._diffusion_linear_temperature(48, 48, None),
        }

        logits = synthetic_logits(50, (1, 6, 10), 2.0)
        samples = []
        for temperature, seed in ((0.0, None), (0.7, 1234), (1.0, 99), (1.6, 7)):
            if seed is not None:
                mx.random.seed(seed)
            s = diffusion._diffusion_sample_canvas(mx.array(logits), mx.int32, temperature)
            mx.eval(s)
            samples.append({"temperature": temperature, "seed": seed, "ids": ints(s)})
        out["sample_canvas"] = {"logits": bits(logits), "cases": samples}

        tokens = np.array([[3, 0, 9, 4, 4, 7]], dtype=np.int32)
        p = diffusion._diffusion_token_probability(mx.array(logits), mx.array(tokens))
        out["token_probability"] = {"logits": bits(logits), "token_ids": ints(tokens), "probability": bits(p)}

        entropy_logits = [synthetic_logits(51, (1, 8, 16), 3.0),
                          peaked_logits(52, (2, 12, 32), [0.0, 4.0, 8.0, 12.0, 16.0, 20.0])]
        out["token_entropy"] = [{"logits": bits(x), "entropy": bits(diffusion._diffusion_token_entropy(mx.array(x)))}
                                for x in entropy_logits]

        entropy_masks = []
        for x in entropy_logits:
            e = diffusion._diffusion_token_entropy(mx.array(x))
            for bound in (0.1, 0.5, 0.0):
                m = diffusion._diffusion_entropy_transfer_mask(e, bound)
                entropy_masks.append({"entropy": bits(e), "entropy_bound": bound, "mask": bools(m)})
        # ties and zeros: equal entropies sort stably by index
        tied = np.array([[0.05, 0.05, 0.0, 0.2, 0.05, 0.0, 0.01, 0.04]], dtype=np.float32)
        entropy_masks.append({"entropy": bits(tied), "entropy_bound": 0.1,
                              "mask": bools(diffusion._diffusion_entropy_transfer_mask(mx.array(tied), 0.1))})
        out["entropy_transfer_mask"] = entropy_masks

        confidence = np.array([[0.95, 0.2, 0.91, 0.5, 0.99, 0.3],
                               [0.1, 0.4, 0.3, 0.2, 0.05, 0.35],
                               [0.9, 0.89, 0.1, 0.92, 0.2, 0.93]], dtype=np.float32)
        unrevealed = np.array([[True, True, True, False, False, True],
                               [True, True, False, True, True, True],
                               [False, False, False, False, False, False]])
        cases = []
        for threshold, force_all in ((0.9, False), (0.9, True), (0.5, False), (1.0, False)):
            m = diffusion._diffusion_confidence_transfer_mask(mx.array(confidence), mx.array(unrevealed),
                                                              threshold, force_all=force_all)
            cases.append({"threshold": threshold, "force_all": force_all, "mask": bools(m)})
        out["confidence_transfer_mask"] = {"confidence": bits(confidence), "unrevealed": bools(unrevealed),
                                           "cases": cases}

        # A sequence of calls sharing one history, as one block's steps share it: canvases that
        # change, repeat, and logits that are or are not confident.
        confident = peaked_logits(53, (1, 6, 32), [40.0])
        unsure = synthetic_logits(54, (1, 6, 32), 1.0)
        a = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
        b = np.array([[1, 2, 3, 4, 5, 7]], dtype=np.int32)
        sequence = [(a, unsure), (a, unsure), (a, confident), (b, confident), (b, confident),
                    (a, confident)]
        stable = []
        for config in ({"confidence_threshold": 0.005, "stability_threshold": 1},
                       {"confidence_threshold": 0.005, "stability_threshold": 2},
                       {"confidence_threshold": 1.0, "stability_threshold": 1}, None):
            history, results = [], []
            for canvas, x in sequence:
                results.append(diffusion._diffusion_stable_and_confident(mx.array(canvas), mx.array(x), history,
                                                                         config))
            stable.append({"config": config, "results": results})
        out["stable_and_confident"] = {
            "logits": {"confident": bits(confident), "unsure": bits(unsure)},
            "mean_entropy": {k: float(mx.mean(diffusion._diffusion_token_entropy(mx.array(v))).item())
                             for k, v in (("confident", confident), ("unsure", unsure))},
            "canvases": {"a": ints(a), "b": ints(b)},
            "sequence": [["a" if c is a else "b", "confident" if x is confident else "unsure"] for c, x in sequence],
            "cases": stable,
        }
    return out


# Generations ---------------------------------------------------------------------------------

# name -> (messages, max_tokens, stop strings, MLX seed). The first three are #51's three oracle
# prompts; `story` runs past two full blocks to a partial third; `stop_comma` ends at its first
# comma through an extra stop id; `story_cut` is cut by max_tokens inside its only block;
# `long_prompt`'s prompt is longer than the sliding window, so its first commit trims the sliding
# layers' caches.
def support_thread():
    """A support thread of about 1,200 tokens, so its prompt is longer than the sliding layers'
    1,023-position window and a reply's first commit trims them. Deterministic: the text is a pure
    function of the line number."""
    lines = []
    for i in range(1, 37):
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


GENERATIONS = [
    ("short_answer", [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
     64, None, 0),
    ("list", [{"role": "user", "content": "List five fruits, one per line, with no other text."}], 128, None, 0),
    ("json", [{"role": "user", "content": "Reply with only a JSON object with the keys name and age for a "
                                          "person called Ada who is 36."}], 128, None, 0),
    ("story", [{"role": "user", "content": "Write a story of about 900 words about a lighthouse keeper who "
                                           "finds a message in a bottle."}], 640, None, 0),
    ("stop_comma", [{"role": "user", "content": "Name three colours, separated by commas, and nothing else."}], 64,
     [","], 0),
    ("story_cut", [{"role": "user", "content": "Write a story of about 400 words about a lighthouse keeper who "
                                               "finds a message in a bottle."}], 40, None, 0),
    ("long_prompt", [{"role": "user", "content": "Here is a support thread.\n\n" + support_thread()
                      + "\n\nWrite a detailed report of about 400 words on what went wrong, what was "
                        "tried, and what the agent should do next."}], 320, None, 0),
    ("short_answer_seed_7", [{"role": "user", "content": "What is the capital of France? Answer in one short "
                                                         "sentence."}], 64, None, 7),
]


class Probe:
    """Pass-through spies on what stream_diffusion_generate calls, which change nothing: the
    decoder masks (one call per block), the decoder logits (one per step), the step temperature,
    the canvas draws and the stability check. The final canvas of a block is the argmax of its
    last step's logits over that step's temperature, exactly as the loop computes it."""

    def __init__(self, model):
        self.model = model
        self.blocks = []
        self.originals = {"masks": model.diffusion_decoder_masks, "logits": model.diffusion_decoder_logits,
                          "temperature": diffusion._diffusion_linear_temperature,
                          "initialize": diffusion._diffusion_initialize_canvas,
                          "stable": diffusion._diffusion_stable_and_confident,
                          "stream": diffusion.stream_diffusion_generate}
        probe = self

        def masks(canvas_ids, cache, decoder_attention_mask=None):
            # The block's initial canvas was drawn just before this call, so the previous block
            # counted it; it is this block's.
            if probe.blocks:
                probe.blocks[-1]["draws"] -= 1
            probe.blocks.append({"canvas_length": int(canvas_ids.shape[1]), "initial": None, "steps": 0,
                                 "draws": 1, "stable": [], "last_logits": None, "last_t": None, "results": []})
            return probe.originals["masks"](canvas_ids, cache, decoder_attention_mask)

        def logits(canvas_ids, cache=None, self_conditioning=None, decoder_attention_mask=None):
            block = probe.blocks[-1]
            if block["steps"] == 0:
                block["initial"] = [int(t) for t in canvas_ids.tolist()[0]]
            block["steps"] += 1
            out = probe.originals["logits"](canvas_ids, cache=cache, self_conditioning=self_conditioning,
                                            decoder_attention_mask=decoder_attention_mask)
            block["last_logits"] = out
            return out

        def temperature(cur_step, max_denoising_steps, schedule_config):
            t = probe.originals["temperature"](cur_step, max_denoising_steps, schedule_config)
            if probe.blocks:
                probe.blocks[-1]["last_t"] = t
            return t

        def initialize(batch_size, canvas_length, vocab_size, dtype):
            if probe.blocks:
                probe.blocks[-1]["draws"] += 1
            return probe.originals["initialize"](batch_size, canvas_length, vocab_size, dtype)

        def stable(accepted_canvas, processed_logits, history, stopping_config):
            result = probe.originals["stable"](accepted_canvas, processed_logits, history, stopping_config)
            probe.blocks[-1]["stable"].append(result)
            return result

        def stream(*args, **kwargs):
            probe.stream_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, list))}
            for r in probe.originals["stream"](*args, **kwargs):
                probe.results.append(r)
                yield r

        model.diffusion_decoder_masks = masks
        model.diffusion_decoder_logits = logits
        diffusion._diffusion_linear_temperature = temperature
        diffusion._diffusion_initialize_canvas = initialize
        diffusion._diffusion_stable_and_confident = stable
        diffusion.stream_diffusion_generate = stream
        self.results = []

    def reset(self):
        self.blocks, self.results = [], []

    def block_records(self, max_steps):
        """The blocks of the last generation, with the committed tokens of each block from the
        stream's results (diffusion_canvas_index)."""
        out = []
        for index, block in enumerate(self.blocks, start=1):
            final = mx.argmax(block["last_logits"] / block["last_t"], axis=-1).astype(mx.int32)
            committed = [int(r.token) for r in self.results
                         if r.diffusion_canvas_index == index and not r.diffusion_block_complete
                         and not r.is_draft and r.finish_reason is None]
            if block["stable"] and block["stable"][-1]:
                ended = "stable_and_confident"
            elif block["steps"] == max_steps:
                ended = "max_denoising_steps"
            else:
                ended = "all_revealed"
            out.append({"canvas_length": block["canvas_length"], "initial_canvas": block["initial"],
                        "steps": block["steps"], "canvas_draws": block["draws"], "ended": ended,
                        "final_canvas": [int(t) for t in final.tolist()[0]], "committed": committed})
            block["last_logits"] = None
        return out


def run_generation(rt, probe, gen, case, max_steps):
    name, messages, max_tokens, stop, seed = case
    upstream = {"messages": messages, "max_tokens": max_tokens}
    if stop is not None:
        upstream["stop"] = stop
    prompt = gen.prompt_ids(upstream)
    stop_ids = gen.stop_ids(upstream)
    skip = gen.engine.thought_open + gen.engine.thought_close
    pieces = []

    def emit(text, token):
        pieces.append([text, token])
        return True

    probe.reset()

    def work():
        mx.random.seed(seed)
        return rt.generate(prompt, max_tokens, stop_ids, emit, skip)

    started = time.perf_counter()
    ids, prompt_tokens, finish = rt.pool.submit(work).result()
    seconds = time.perf_counter() - started
    terminal = probe.results[-1] if probe.results else None
    stop_token = int(terminal.token) if terminal is not None and finish == "stop" else None
    blocks = rt.pool.submit(probe.block_records, max_steps).result()
    record = {
        "name": name, "messages": messages, "max_tokens": max_tokens, "stop": stop, "stop_ids": stop_ids,
        "skip_special_token_ids": skip, "seed": seed, "prompt": prompt,
        "prompt_tokens": prompt_tokens, "finish_reason": finish, "stop_token": stop_token,
        "generated": ids, "text": "".join(t for t, _ in pieces), "pieces": pieces, "blocks": blocks,
    }
    committed = [t for b in blocks for t in b["committed"]]
    if committed != ids:
        raise SystemExit(f"{name}: the blocks committed {len(committed)} tokens, generate returned {len(ids)}")
    return record, {"seconds": seconds, "blocks": len(blocks), "tokens": len(ids)}


# Think ---------------------------------------------------------------------------------------

README_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?",
             "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
    "tone": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "furious"]},
}
STATE = "Everything is down and we have a demo with our biggest client at noon."
THINKS = [
    ("readme", {"model": "openjev-latest", "state": STATE, "questions": README_QUESTIONS, "think": 128}, 0),
    ("sequential_24", {"model": "openjev-latest", "state": STATE, "sequential": True, "think": 64,
                       "questions": {f"k{i}": {"type": "noul", "instructions": f"Is statement {i} about an outage?"}
                                     for i in range(24)}}, 0),
]


def mlx_engine(settings, tok, rt):
    """An MlxEngine over the already loaded runtime, rather than a second 16 GB load."""
    eng = MlxEngine.__new__(MlxEngine)
    Engine.__init__(eng, settings, tok)
    eng.runtime = rt
    return eng


def run_think(eng, case):
    name, body, seed = case
    req = SystemOneRequest.model_validate(body)
    questions = {k: q.model_dump() for k, q in req.questions.items()}
    text = json.dumps([req.state, questions], sort_keys=True)
    request_seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    thoughts = []
    rt = eng.runtime
    original_generate = rt.generate

    def generate(prompt, max_tokens, stop_ids, emit, skip_special=None):
        ids, billed, finish = original_generate(prompt, max_tokens, stop_ids, emit, skip_special)
        thoughts.append({"prompt": list(prompt), "budget": max_tokens, "stop_ids": list(stop_ids),
                         "skip_special_token_ids": list(skip_special or ()), "generated": ids,
                         "prompt_tokens": billed, "finish_reason": finish})
        return ids, billed, finish

    original_think = eng.think

    async def think(sys_text, state_text, budget):
        prefix, thought, billed = await original_think(sys_text, state_text, budget)
        thoughts[-1].update({"system": sys_text, "user": state_text, "thought_tokens": thought,
                             "billed": billed, "prefix": prefix,
                             "thought": prefix[len(thoughts[-1]["prompt"]):len(prefix) - len(eng.thought_close)]})
        return prefix, thought, billed

    rt.generate = generate
    eng.think = think
    options = {"think": req.think, "sequential": req.sequential, "steps": req.steps, "samples": req.samples}

    async def decide():
        return await eng.decide(questions, req.state, request_seed, None, options)

    try:
        rt.pool.submit(mx.random.seed, seed).result()
        started = time.perf_counter()
        answers, billed, thought = asyncio.run(decide())
        seconds = time.perf_counter() - started
    finally:
        rt.generate = original_generate
        eng.think = original_think
    return ({"name": name, "body": body, "request_seed": request_seed, "seed": seed, "thoughts": thoughts,
             "answers": answers, "input_tokens": billed, "output_tokens": thought},
            {"seconds": seconds, "thoughts": len(thoughts)})


# Output --------------------------------------------------------------------------------------

# The dashes this repository keeps out of its files, which the model writes in its replies, are
# written as JSON escapes and read back as the same text, as Tools/fixtures/mlx_vlm_oracle.py does.
DASHES = {code: f"\\u{code:04x}" for code in range(0x2012, 0x2016)}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).translate(DASHES)


def write_json(path, payload):
    """One top-level key per line and one list entry per line, as the other fixtures."""
    lines = []
    for key, value in payload.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        elif isinstance(value, dict) and value and all(isinstance(v, (dict, list)) for v in value.values()):
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


def compare_with_committed(payload, committed):
    fresh = json.loads(dumps(payload))
    differences = {}
    for key in sorted(set(fresh) | set(committed)):
        mine, theirs = fresh.get(key), committed.get(key)
        if mine == theirs:
            continue
        if isinstance(mine, list) and isinstance(theirs, list):
            names = sorted({r.get("name") for r in mine + theirs if isinstance(r, dict)}
                           - {r.get("name") for r in mine if r in theirs})
            differences[key] = names or ["order"]
        elif isinstance(mine, dict) and isinstance(theirs, dict):
            differences[key] = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        else:
            differences[key] = True
    return differences


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed generation.json instead of writing it")
    ap.add_argument("--cache-limit-gb", type=float, default=None,
                    help="MlxRuntime.set_cache_limit before the runs, as OPENJEV_MLX_CACHE_LIMIT_GB does")
    ap.add_argument("--run-out", default=str(RUN_OUT), help="where timings go")
    args = ap.parse_args()

    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    model_path = huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    settings = Settings()

    sampler = sampler_vectors()
    sampler_again = sampler_vectors()
    if json.loads(dumps(sampler)) != json.loads(dumps(sampler_again)):
        raise SystemExit("the sampler vectors differ between two runs")

    started = time.perf_counter()
    rt = MlxRuntime(model_path)
    load_seconds = time.perf_counter() - started
    rt.prompt_cache_entries = settings.mlx_prompt_cache
    rt.set_cache_limit(args.cache_limit_gb)
    eng = mlx_engine(settings, tok, rt)
    gen = MlxGenerator(settings, eng)

    config = rt.model.config
    generation_config = getattr(config, "generation_config", None)
    if not isinstance(generation_config, dict) or not generation_config:
        raise SystemExit("mlx-vlm did not load the checkpoint's generation_config.json")
    crit = getattr(rt.processor, "tokenizer", rt.processor).stopping_criteria
    detokenizer = type(rt.processor.detokenizer).__name__
    trim_space = getattr(rt.processor.detokenizer, "trim_space", None)
    max_steps = int(generation_config.get("max_denoising_steps") or diffusion.DEFAULT_DIFFUSION_MAX_DENOISING_STEPS)
    probe = Probe(rt.model)

    # One unrecorded generation compiles the kernels.
    rt.pool.submit(lambda: (mx.random.seed(0), rt.generate(gen.prompt_ids(
        {"messages": GENERATIONS[0][1], "max_tokens": 16}), 16, [], lambda t, k: True,
        eng.thought_open + eng.thought_close))).result()
    eos_after = list(crit.eos_token_ids)

    def one_pass(order):
        rt.init_prefill_cache()
        rt.prompt_cache_entries = settings.mlx_prompt_cache
        gc.collect()
        records, timings = {}, {}
        for i in order:
            case = GENERATIONS[i]
            records[case[0]], timings[case[0]] = run_generation(rt, probe, gen, case, max_steps)
            print(f"  {case[0]}: {records[case[0]]['finish_reason']}, {len(records[case[0]]['generated'])} tokens, "
                  f"{len(records[case[0]]['blocks'])} blocks: {records[case[0]]['text'][:120]!r}", file=sys.stderr)
        thinks, think_timings = {}, {}
        for i in (range(len(THINKS)) if order[0] == 0 else reversed(range(len(THINKS)))):
            rt.init_prefill_cache()
            rt.prompt_cache_entries = settings.mlx_prompt_cache
            case = THINKS[i]
            thinks[case[0]], think_timings[case[0]] = run_think(eng, case)
            print(f"  think {case[0]}: {thinks[case[0]]['output_tokens']} thought tokens, "
                  f"{thinks[case[0]]['input_tokens']} input tokens", file=sys.stderr)
        return records, timings, thinks, think_timings

    print("pass 1", file=sys.stderr)
    records_a, timings_a, thinks_a, think_timings_a = one_pass(list(range(len(GENERATIONS))))
    print("pass 2", file=sys.stderr)
    records_b, timings_b, thinks_b, think_timings_b = one_pass(list(reversed(range(len(GENERATIONS)))))
    differing = [n for n in records_a if dumps(records_a[n]) != dumps(records_b[n])]
    differing += [f"think/{n}" for n in thinks_a if dumps(thinks_a[n]) != dumps(thinks_b[n])]
    deterministic = not differing
    print(f"pass 1 == pass 2: {deterministic} (differing: {differing})", file=sys.stderr)

    problems = []
    stop_case = records_a["stop_comma"]
    if stop_case["finish_reason"] != "stop" or stop_case["stop_token"] not in stop_case["stop_ids"]:
        problems.append("stop_comma did not end at its extra stop id")
    if records_a["story_cut"]["finish_reason"] != "length":
        problems.append("story_cut was not cut by max_tokens")
    if len(records_a["story"]["blocks"]) < 3:
        problems.append("story did not run past two blocks")
    if len(records_a["long_prompt"]["prompt"]) <= 1100 or len(records_a["long_prompt"]["blocks"]) < 2:
        problems.append("long_prompt does not commit a block after a prompt of more than 1,100 tokens")

    payload = {
        "generator": generator(),
        "settings": {
            "generation_config": generation_config,
            "eos_token_ids": eos_after,
            "canvas_length": int(config.canvas_length),
            "min_canvas_length": diffusion.DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
            "max_denoising_steps": max_steps,
            "sampler": "confidence-threshold",
            "confidence_threshold": diffusion.DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD,
            "temperature": 0.0,
            "detokenizer": detokenizer, "trim_space": trim_space,
            "thought_open": eng.thought_open, "thought_close": eng.thought_close, "scaffold": eng.scaffold,
            "mlx_max_prompt": settings.mlx_max_prompt,
        },
        "sampler": sampler,
        "generations": [records_a[c[0]] for c in GENERATIONS],
        "think": [thinks_a[c[0]] for c in THINKS],
    }

    comparison = None
    if args.check:
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        comparison = {"differences_from_committed": compare_with_committed(payload, committed)}
        print(f"against the committed generation.json: {comparison}", file=sys.stderr)
    elif deterministic and not problems:
        write_json(OUT, payload)

    run = {
        "generator": generator(),
        "machine": {"platform": platform.platform(), "mac_ver": platform.mac_ver()[0],
                    "machine": platform.machine(), **device_info()},
        "cache_limit_gb": args.cache_limit_gb,
        "load_seconds": load_seconds,
        "deterministic": deterministic,
        "differing_between_passes": differing,
        "problems": problems,
        "check": comparison,
        "timings": {"pass_1": timings_a, "pass_2": timings_b,
                    "think_pass_1": think_timings_a, "think_pass_2": think_timings_b},
    }
    write_json(Path(args.run_out), run)
    rt.close()
    if not deterministic:
        raise SystemExit("the two passes disagree" + ("" if args.check else "; generation.json was not written"))
    if problems:
        raise SystemExit(f"{problems}" + ("" if args.check else "; generation.json was not written"))
    if comparison and comparison["differences_from_committed"]:
        raise SystemExit(f"the run differs from the committed generation.json: {comparison['differences_from_committed']}")


if __name__ == "__main__":
    main()
