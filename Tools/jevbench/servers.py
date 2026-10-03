#!/usr/bin/env python3
"""Start an OpenJev server for one backend, run the JevBench harness against it, then stop it.

    python3 Tools/jevbench/servers.py --server swift --backend verdict
    python3 Tools/jevbench/servers.py --server upstream --backend laya
    python3 Tools/jevbench/servers.py --server upstream --backend mlx --command \\
        openjev-bench reads --url {url} --server upstream
    python3 Tools/jevbench/servers.py --server swift --backend jevk5 \\
        --setting OPENJEV_MLX_CACHE_LIMIT_GB=4

`swift` runs this repository's release build (`.build/release/openjev serve`). `upstream` runs
razorback16/openjev at the commit the Makefile pins (`make upstream`) with `python -m openjev`,
from the virtual environment README.md describes, upstream's checkout on PYTHONPATH and its
checkpoint read from the pinned snapshot in the Hugging Face cache. Upstream's `jevk5` reads its
letters from a vLLM server, which needs an NVIDIA GPU, so only the Swift server runs it here; its
reference is the model author's published run (`harness.py author-run`). Both listen on 127.0.0.1 on a
free port with warm-up on, as their defaults have it. The server's log goes beside the harness's
cache; each dataset's result goes to results/ (README.md). Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import harness

ROOT = harness.ROOT
UPSTREAM = ROOT / "Upstream" / "openjev"
DEFAULT_PYTHON = harness.HERE / ".venv" / "bin" / "python"

# backend -> the model it serves
MODELS = {"verdict": "verdict-1.4", "laya": "laya-1.0", "mlx": "openjev-0.1", "jevk5": "jevk5-0.2"}
# The JevK5 conversion the Swift server reads by default: Tools/jevk5/convert.py --bits 8's folder.
JEVK5_MODEL = Path.home() / "Library" / "Caches" / "OpenJevSwift" / "jevk5" / "jevk5-0.2-mlx-8bit"
# The checkpoints upstream loads, at the revisions Fixtures/encoders and THIRD_PARTY.md pin, and
# the files of each its loader reads (Tools/encoders/common.py).
CHECKPOINTS = {
    "verdict": ("heman10x/rlcd-modernbert-151m", "8af2496eb63c7fa66d7d234e1f62629380030eb4",
                ["config.json", "calibrator.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"], "OPENJEV_VERDICT_MODEL"),
    "laya": ("convaiinnovations/laya-typed-decisions", "1a793eb568e6718f15941d08f85432581df534e3",
             ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
             "OPENJEV_LAYA_MODEL"),
    "mlx": ("mlx-community/diffusiongemma-26B-A4B-it-4bit",
            "a7a81407613811e8ba63af92ac0d852b809e191f", None, "OPENJEV_MLX_MODEL"),
}
# What the Swift server runs on a Mac for each encoder backend (D-011, D-034, D-037).
SWIFT_RUNTIME = {
    "verdict": ("verdict-m18-fp16", "Core ML, float16 multifunction package, .cpuAndGPU, up to 16 "
                                    "questions per call"),
    "laya": ("laya-m18-fp16", "Core ML, float16 multifunction package, .cpuAndGPU, up to 16 "
                              "questions per call"),
}
UPSTREAM_PACKAGES = ("torch", "transformers", "tokenizers", "gliclass", "laya", "numpy",
                     "huggingface_hub", "safetensors", "fastapi", "starlette", "pydantic",
                     "uvicorn", "mlx", "mlx-vlm")
# What the Swift server is built from.
PACKAGE_PATHS = ("Sources", "Package.swift", "Package.resolved")
# Settings each child gets on top of the environment, which loses every other OPENJEV_ variable so
# that the servers' defaults apply (and no API key is required).
COMMON_SETTINGS = {"OPENJEV_HOST": "127.0.0.1", "OPENJEV_LOG_LEVEL": "info"}
# What --setting may not replace: the settings this script chooses itself.
RESERVED_SETTINGS = ({"OPENJEV_PORT", "OPENJEV_BACKEND", "OPENJEV_ENCODER_MODELS",
                      "OPENJEV_JEVK5_MODEL"}
                     | set(COMMON_SETTINGS) | {entry[3] for entry in CHECKPOINTS.values()})
# What --setting refuses because it can hold a credential, which a result file would record: the
# API key and origin secret, the model routes and upstream's vLLM URL (either URL can carry a user
# and password), and any name that says it is a key, secret, token, password or credential. The
# harness sends neither a key nor an origin secret, so a server given one would refuse its requests.
SECRET_SETTINGS = {"OPENJEV_API_KEY", "OPENJEV_ORIGIN_SECRET", "OPENJEV_MODEL_ROUTES",
                   "OPENJEV_UPSTREAM"}
SECRET_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def extra_settings(pairs: list) -> dict:
    """--setting OPENJEV_NAME=VALUE, repeated: server settings beyond the defaults, which a result
    file records with the others. A run made with one says so; the JevBench runs of D-041 used
    none, and the DiffusionGemma runs cap MLX's buffer pool (docs/quality.md)."""
    settings = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.startswith("OPENJEV_") or name in RESERVED_SETTINGS:
            sys.exit(f"--setting takes OPENJEV_NAME=VALUE for a setting this script does not set "
                     f"itself, not {name!r}")
        if name in SECRET_SETTINGS or any(word in name for word in SECRET_WORDS):
            sys.exit(f"--setting refuses {name}: it can hold a credential, which the result file "
                     "would record")
        settings[name] = value
    return settings


def output(command: list, env: dict | None = None, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, env=env,
                              cwd=cwd).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def first_line(text: str | None) -> str | None:
    return text.splitlines()[0] if text else None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def pinned_upstream_commit() -> str:
    match = re.search(r"^UPSTREAM_OPENJEV_COMMIT := (\w+)$", (ROOT / "Makefile").read_text(),
                      re.MULTILINE)
    return match.group(1)


def child_environment(settings: dict) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENJEV_")}
    env.update(COMMON_SETTINGS)
    env.update(settings)
    return env


def swift_binary(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    folder = output(["swift", "build", "-c", "release", "--show-bin-path"], cwd=ROOT)
    return Path(folder or ROOT / ".build" / "release") / "openjev"


def swift_server(backend: str, binary: Path, encoder_models: str | None,
                 jevk5_model: str | None = None) -> tuple:
    """The command, environment and version record of the Swift server for `backend`."""
    if not binary.is_file():
        sys.exit(f"{binary} is missing; run swift build -c release --product openjev")
    settings = {"OPENJEV_BACKEND": backend}
    package, runtime = SWIFT_RUNTIME.get(backend, (None, None))
    info = {
        "implementation": "OpenJevSwift", "backend": backend,
        "version": output([str(binary), "--version"]),
        # the last commit that changed the package, which a rebase of the branch keeps
        "commit": output(["git", "log", "-1", "--format=%H", "--", *PACKAGE_PATHS], cwd=ROOT),
        # untracked files included: SwiftPM builds every source file under Sources/
        "source_changes": bool(output(["git", "status", "--porcelain", "--untracked-files=all",
                                       "--", *PACKAGE_PATHS], cwd=ROOT)),
        "binary": os.path.relpath(binary, ROOT), "binary_sha256": harness.sha256_file(binary),
        "toolchain": first_line(output(["swift", "--version"])),
        "xcode": " ".join((output(["xcodebuild", "-version"]) or "").split("\n")) or None,
        "runtime": runtime, "package": package,
    }
    if encoder_models:
        folder = Path(encoder_models).expanduser()
        settings["OPENJEV_ENCODER_MODELS"] = str(folder)
        info["package_source"] = f"OPENJEV_ENCODER_MODELS={harness.display_path(folder)}"
        if package:
            check = subprocess.run(
                [sys.executable, str(ROOT / "Tools" / "encoders" / "manifest.py"), "--model",
                 backend, "--check", "--models", str(folder)],
                capture_output=True, text=True, cwd=ROOT)
            lines = (check.stdout.strip() or check.stderr.strip()).splitlines()
            info["package_check"] = (lines[-1].replace(str(folder), harness.display_path(folder))
                                     if lines else None)
            if check.returncode != 0:
                sys.exit(f"{package} in {harness.display_path(folder)} is not the published "
                         f"package: {info['package_check']}; convert it again, or leave out "
                         "--encoder-models to download the published one")
            manifest = folder / f"{package}.mlpackage" / "Manifest.json"
            if manifest.is_file():
                info["package_manifest_sha256"] = harness.sha256_file(manifest)
    elif package:
        info["package_source"] = "downloaded from Algorythm-Canada/openjev-models (D-033)"
    if backend == "mlx":
        repo, revision = CHECKPOINTS["mlx"][:2]
        info["runtime"] = "MLX on the GPU (D-039)"
        info["model_source"] = (f"OPENJEV_MLX_MODEL's default, {repo} at {revision[:7]}, through "
                                "the Hugging Face cache")
    if backend == "jevk5":
        folder = Path(jevk5_model or JEVK5_MODEL).expanduser()
        settings["OPENJEV_JEVK5_MODEL"] = str(folder)
        info["runtime"] = "MLX on the GPU, Qwen3.5 through mlx-swift-lm (D-052)"
        info["model_source"] = f"OPENJEV_JEVK5_MODEL={harness.display_path(folder)}"
        # the folder must be one of the pinned conversions, as convert.py checks: 4-bit, 8-bit, or
        # the unquantized reference that tells quantization from the rest in a comparison
        checks = {}
        for bits in (4, 8, 16):
            check = subprocess.run(
                [sys.executable, str(ROOT / "Tools" / "jevk5" / "convert.py"), "--check",
                 str(folder), "--bits", str(bits)], capture_output=True, text=True, cwd=ROOT)
            lines = (check.stdout.strip() or check.stderr.strip()).splitlines()
            checks[bits] = (lines[-1].replace(str(folder), harness.display_path(folder))
                            if lines else None)
            if check.returncode == 0:
                info["model_check"] = checks[bits]
                info["conversion"] = ("jevk5-0.2-mlx-bf16" if bits == 16
                                      else f"jevk5-0.2-mlx-{bits}bit")
                break
        else:
            # the 8-bit check's reason, since that is the conversion the server takes by default
            sys.exit(f"{harness.display_path(folder)} is not a pinned conversion: {checks[8]}; "
                     "run Tools/jevk5/convert.py --bits 8")
    info["settings"] = {key: harness.display_path(value)
                        if key in ("OPENJEV_ENCODER_MODELS", "OPENJEV_JEVK5_MODEL") else value
                        for key, value in settings.items()}
    return [str(binary), "serve"], settings, info


def upstream_server(backend: str, python: Path) -> tuple:
    """The command, environment and version record of upstream's server for `backend`."""
    if backend == "jevk5":
        sys.exit("upstream's jevk5 backend reads its letters from a vLLM server, which needs an "
                 "NVIDIA GPU; compare the Swift run with the author's published run instead "
                 "(python3 Tools/jevbench/harness.py author-run)")
    if not python.is_file():
        sys.exit(f"{python} is missing; create it as Tools/jevbench/README.md describes")
    head = output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"])
    pinned = pinned_upstream_commit()
    if not head or not head.startswith(pinned):
        sys.exit(f"Upstream/openjev is at {head}, the Makefile pins {pinned}; run make upstream")
    repo, revision, patterns, variable = CHECKPOINTS[backend]
    resolve = ("import sys; from huggingface_hub import snapshot_download; "
               "print(snapshot_download(sys.argv[1], revision=sys.argv[2], "
               "allow_patterns=sys.argv[3].split(',') if sys.argv[3] else None))")
    arguments = [str(python), "-c", resolve, repo, revision, ",".join(patterns or [])]
    # the cached snapshot without asking the Hub first; a download only when it is missing
    snapshot = (output(arguments, env={**os.environ, "HF_HUB_OFFLINE": "1"})
                or output(arguments))
    if not snapshot:
        sys.exit(f"could not resolve {repo} at {revision} with {python}")
    probe = ("import importlib.metadata as m, json, platform, sys\n"
             "names = sys.argv[1].split(',')\n"
             "found = {}\n"
             "for name in names:\n"
             "    try: found[name] = m.version(name)\n"
             "    except m.PackageNotFoundError: pass\n"
             "import openjev\n"
             "record = {'python': platform.python_version(), 'openjev': openjev.__version__,"
             " 'openjev_file': openjev.__file__, 'packages': found}\n"
             "try:\n"
             "    import torch\n"
             "    record.update(torch_threads=torch.get_num_threads(), cuda=torch.cuda.is_available())\n"
             "except ImportError:\n"
             "    pass\n"
             "print(json.dumps(record))\n")
    env = child_environment({"PYTHONPATH": str(UPSTREAM)})
    versions = json.loads(output([str(python), "-c", probe, ",".join(UPSTREAM_PACKAGES)],
                                 env=env) or "{}")
    if not str(versions.get("openjev_file", "")).startswith(str(UPSTREAM)):
        sys.exit(f"{python} imports openjev from {versions.get('openjev_file')}, not {UPSTREAM}")
    settings = {"OPENJEV_BACKEND": backend, variable: snapshot, "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(UPSTREAM)}
    info = {
        "implementation": "razorback16/openjev", "backend": backend, "commit": head,
        "version": versions.get("openjev"), "python": versions.get("python"),
        "interpreter": os.path.relpath(python, ROOT), "packages": versions.get("packages"),
        "torch_threads": versions.get("torch_threads"),
        "device": ("mlx, the GPU" if backend == "mlx" else
                   "cuda" if versions.get("cuda") else "cpu"),
        "dtype": ("the checkpoint's" if backend == "mlx" else
                  "bfloat16" if versions.get("cuda") else "float32"),
        "checkpoint": {"repo": repo, "revision": revision},
        "settings": {key: harness.display_path(value) if key == "PYTHONPATH" else value
                     for key, value in settings.items() if key != variable},
    }
    return [str(python), "-m", "openjev"], settings, info


def swift_function_capacity(log: Path):
    """How many Core ML functions the Swift server keeps loaded, from the settings line it logs at
    startup: `encoder_functions=all` or a number (D-042). A binary from before D-042 logs none and
    kept two (D-037 item 3)."""
    found = re.search(r"\bencoder_functions=(\S+)", log.read_text(errors="replace"))
    if not found:
        return 2
    return int(found.group(1)) if found.group(1).isdigit() else found.group(1)


def wait_until_healthy(base_url: str, process: subprocess.Popen, log: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
            sys.exit(f"the server exited with status {process.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    process.terminate()
    sys.exit(f"the server was not healthy after {timeout:.0f} s; see {log}")


def stop(process: subprocess.Popen, timeout: float = 60.0) -> int:
    """SIGTERM, the graceful shutdown both servers implement, then SIGKILL after `timeout`."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return process.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", choices=("swift", "upstream"), required=True)
    parser.add_argument("--backend", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=("jevbench", "typesafe102", "all"), default="all")
    parser.add_argument("--binary", help="the Swift openjev binary (default: the release build)")
    parser.add_argument("--encoder-models",
                        help="OPENJEV_ENCODER_MODELS for the Swift server; unset downloads the "
                             "published packages")
    parser.add_argument("--jevk5-model", default=str(JEVK5_MODEL),
                        help="OPENJEV_JEVK5_MODEL for the Swift jevk5 server, a pinned "
                             "conversion's folder (default %(default)s, the 8-bit one)")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON),
                        help="the interpreter of upstream's environment (default %(default)s)")
    parser.add_argument("--cache", default=str(harness.default_cache()))
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--ids", help="only these comma-separated item ids (trial runs)")
    parser.add_argument("--output-dir", help="where the result files go (default results/)")
    parser.add_argument("--force", action="store_true", help="replace existing result files")
    parser.add_argument("--setting", action="append", default=[], metavar="OPENJEV_NAME=VALUE",
                        help="a server setting beyond the defaults, recorded in the result file "
                             "(repeatable), such as OPENJEV_MLX_CACHE_LIMIT_GB=4")
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="run this command against the server instead of the harness, with "
                             "{url} replaced by the server's base URL; everything after it is "
                             "the command (openjev-bench --url, docs/benchmarks.md)")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    extra = extra_settings(args.setting)
    names = ("jevbench", "typesafe102") if args.dataset == "all" else (args.dataset,)
    datasets = [] if args.command else [harness.load_dataset(name, cache) for name in names]
    model = MODELS[args.backend]
    outputs = []
    for dataset in datasets:
        path = harness.default_output(dataset.name, model, args.server)
        if args.output_dir:
            path = Path(args.output_dir) / path.relative_to(harness.RESULTS)
        if path.exists() and not args.force:
            sys.exit(f"{path} exists; pass --force to replace it")
        outputs.append(path)

    if args.server == "swift":
        command, settings, info = swift_server(args.backend, swift_binary(args.binary),
                                               args.encoder_models, args.jevk5_model)
    else:
        command, settings, info = upstream_server(args.backend, Path(args.python))
    settings.update(extra)
    info["settings"].update(extra)
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log = cache / "logs" / f"{model}-{args.server}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = child_environment({**settings, "OPENJEV_PORT": str(port)})
    print(f"starting {args.server} {args.backend} on {base_url} (log {log})", flush=True)
    started = time.monotonic()
    with open(log, "w") as log_file:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log_file,
                                   stderr=subprocess.STDOUT)
    try:
        wait_until_healthy(base_url, process, log, args.startup_timeout)
        info["startup_s"] = round(time.monotonic() - started, 1)
        if args.server == "swift" and args.backend in SWIFT_RUNTIME:
            info["function_capacity"] = swift_function_capacity(log)
        if args.command:
            print(json.dumps({"server": info}, indent=1), flush=True)
            command = [part.replace("{url}", base_url) for part in args.command]
            print(f"running {' '.join(command)}", flush=True)
            return subprocess.run(command, cwd=ROOT).returncode
        ids = set(args.ids.split(",")) if args.ids else None
        for dataset, path in zip(datasets, outputs):
            print(f"{dataset.name}: {len(dataset.tasks)} items", flush=True)
            doc = harness.run_dataset(dataset, base_url, model, args.server, cache,
                                      server_info=info, ids=ids,
                                      progress=lambda line: print(line, flush=True))
            harness.write_result(path, doc)
            print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
            print(harness.markdown_table(harness.SUMMARY_HEADER, harness.summary_rows([doc])))
    finally:
        status = stop(process)
        # uvicorn re-raises SIGTERM once its graceful shutdown is over, so upstream ends with -15
        print(f"server stopped with status {status}"
              + (f" (signal {-status})" if status is not None and status < 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
