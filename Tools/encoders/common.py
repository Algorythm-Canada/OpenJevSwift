"""Pins, paths and helpers shared by the spike #56 scripts in Tools/encoders.

Importing this module puts the pinned upstream checkout (Upstream/openjev, `make upstream`) on the
import path, clears upstream's OPENJEV_* settings from the environment so that its defaults apply,
and quietens the libraries' progress output.
"""

import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
FIXTURES = ROOT / "Fixtures" / "encoders"
RESULTS = ROOT / "docs" / "spikes" / "encoder-runtime"
UPSTREAM_COMMIT = "dcd2094"
VERDICT_REPO = "heman10x/rlcd-modernbert-151m"
VERDICT_REVISION = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
LAYA_REPO = "convaiinnovations/laya-typed-decisions"
LAYA_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
# Where converted Core ML packages go: outside the repository, as they are model weights.
MODELS = Path(os.environ.get("OPENJEV_ENCODER_MODELS", Path.home() / "Library" / "Caches" / "OpenJevSwift" / "encoders"))
# PyTorch's CPU kernels split reductions by thread, so the thread count is pinned to keep reruns
# identical. Eight threads keep to the performance cores of the reference machine.
TORCH_THREADS = 8

for _name in [n for n in os.environ if n.startswith("OPENJEV_") and n != "OPENJEV_ENCODER_MODELS"]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not (UPSTREAM / ".git").exists():
    sys.exit("Upstream/openjev is missing; run make upstream")
sys.path.insert(0, str(UPSTREAM))

# laya warns once, at load, that it clamps the shipped choice:11+ temperature; laya.json records it.
warnings.filterwarnings("ignore", message="laya: this checkpoint ships temperatures")
# gliclass loads its checkpoint through a transformers class of another model type, by design.
warnings.filterwarnings("ignore", message=".*You are using a model of type.*")


def upstream_head():
    try:
        out = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def require_upstream():
    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        sys.exit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")


def cpu_name():
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def versions(*modules):
    """The Python version and each named module's version, for a generator object."""
    import importlib

    out = {"python": sys.version.split()[0]}
    for name in modules:
        out[name] = importlib.import_module(name).__version__
    return out


def generator(script, version, *modules):
    return {
        "script": script,
        "version": version,
        "upstream": "razorback16/openjev",
        "upstream_commit": UPSTREAM_COMMIT,
        "verdict_repo": VERDICT_REPO,
        "verdict_revision": VERDICT_REVISION,
        "laya_repo": LAYA_REPO,
        "laya_revision": LAYA_REVISION,
        **versions(*modules),
        "cpu": cpu_name(),
    }


def dumps(value):
    """Strict JSON: non-finite numbers are refused, so every file parses with an RFC 8259 parser."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write(path, gen, payload):
    """One top-level key per line and one list entry per line, as Tools/fixtures writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in {"generator": gen, **payload}.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(text.encode('utf-8'))} bytes", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f32(values):
    """float32 values as the Python floats they are exactly equal to."""
    import numpy as np

    return [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1).tolist()]


def snapshot(repo, revision, patterns):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=patterns)


def verdict_snapshot():
    return snapshot(VERDICT_REPO, VERDICT_REVISION,
                    ["config.json", "calibrator.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])


def laya_snapshot():
    return snapshot(LAYA_REPO, LAYA_REVISION, ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])


def settings(**kw):
    """Upstream's Settings for an encoder backend on the CPU, without the warmup read."""
    from openjev.config import Settings

    return Settings(device="cpu", warmup=False, **kw)


def verdict_engine(encoder_batch=16):
    """Upstream's VerdictEngine, loaded exactly as its load() does, from the pinned snapshot."""
    from openjev.encoders import VerdictEngine

    eng = VerdictEngine.__new__(VerdictEngine)
    eng.s = settings(backend="verdict", verdict_model=verdict_snapshot(), encoder_batch=encoder_batch)
    eng.load()
    return eng


def laya_engine(encoder_batch=16):
    """Upstream's LayaEngine, loaded exactly as its load() does, from the pinned snapshot."""
    from openjev.encoders import LayaEngine

    eng = LayaEngine.__new__(LayaEngine)
    eng.s = settings(backend="laya", laya_model=laya_snapshot(), encoder_batch=encoder_batch)
    eng.load()
    return eng


def context_of(state):
    """The state as both models read it: text as is, anything else as json.dumps(ensure_ascii=False)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class Recorder:
    """Calls through to a callable and keeps what it was given and what it returned."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *args, **kw):
        out = self.fn(*args, **kw)
        self.calls.append((args, kw, out))
        return out


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
