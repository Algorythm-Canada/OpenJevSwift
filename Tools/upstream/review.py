#!/usr/bin/env python3
"""Reviews the pins of THIRD_PARTY.md that it follows against their projects' state (issue #66).

    python3 Tools/upstream/review.py                 # the review as Markdown, on stdout
    python3 Tools/upstream/review.py --json          # the same review as JSON
    python3 Tools/upstream/review.py --from FILE     # render a saved --json review; no network

For each GitHub project it follows (razorback16/openjev, ml-explore/mlx-swift,
ml-explore/mlx-swift-lm, Blaizzy/mlx-vlm, the Layr-Labs fork of mlx-swift-lm and
huggingface/swift-transformers) it prints the pinned revision, the head of the default branch and
the latest release, the commits and releases in between with their dates and first lines, and the
commits that touch the paths this repository depends on. For upstream OpenJev it also lists the
files changed under openjev/ and tests/ by area (wire, engine and read policy, models, settings,
tests), and the mlx-vlm requirement and backends that upstream's pyproject.toml declares. For each
Hugging Face checkpoint it follows it prints the revision that `main` points at now, since a moved
`main` changes what a fresh download gets, and the commits since the pin. It also checks that the
Makefile, Package.resolved and Tools/oracle/requirements.txt agree with THIRD_PARTY.md.

It only reads. Pins come from THIRD_PARTY.md and the files above. A project's history comes from
`git` when a local clone already holds the default branch's head (`--clone`; by default
Upstream/openjev, which `make upstream` creates and fetches), else from the GitHub REST API through
`--gh` (default ghp, the wrapper this repository's maintainers use; CI passes gh). The head of each
default branch comes from `git ls-remote`, checkpoints from huggingface.co/api/models. It never
fetches into a clone, writes a file or changes anything on GitHub. Standard library only, Python 3.9
or later.

The JSON form is the input of tracking_issue.py, which the Upstream review workflow runs monthly to
open or update one tracking issue (docs/upstream-log.md). Exit status 0 when the review was
produced, also when a project could not be read: that project carries an `error`, which the report
shows. Exit status 1 when the review could not start (THIRD_PARTY.md unreadable, no GitHub CLI).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THIRD_PARTY = ROOT / "THIRD_PARTY.md"
MAKEFILE = ROOT / "Makefile"
PACKAGE_RESOLVED = ROOT / "Package.resolved"
ORACLE_REQUIREMENTS = ROOT / "Tools" / "oracle" / "requirements.txt"

FORMAT_VERSION = 1
HUB = "https://huggingface.co"
USER_AGENT = "OpenJevSwift-upstream-review (https://github.com/Algorythm-Canada/OpenJevSwift)"
ATTEMPTS = 3  # each network read, against connection resets and 5xx answers
COMMAND_TIMEOUT = 120  # seconds for one gh or git ls-remote call
MAX_COMMITS = 30  # newest commits listed per project; watched paths list all of theirs
MAX_WATCHED = 100  # commits listed per watched path
HUB_COMMIT_LIMIT = 500  # commits of a checkpoint's main read before the search stops
COMPARE_FILE_CAP = 300  # GitHub's compare lists at most this many files


# ---------------------------------------------------------------------------------------------
# What is reviewed


@dataclass(frozen=True)
class Watch:
    """Paths in an upstream project whose commits the review lists in full."""

    why: str  # what this repository takes from them
    path: str = ""  # a file or a directory, as git and the commits API take it
    pattern: str = ""  # or a regular expression over changed file paths


@dataclass(frozen=True)
class Area:
    """A group of upstream OpenJev's files, by what they decide for compatibility."""

    name: str
    prefixes: tuple
    why: str


@dataclass(frozen=True)
class Fact:
    """One line of a file, shown at the pin and at the head: a requirement or a pin of theirs."""

    label: str
    path: str
    pattern: str  # a regular expression; the first matching line is shown, or the first group


@dataclass(frozen=True)
class GitHubProject:
    repo: str  # owner/name, as THIRD_PARTY.md links it
    watches: tuple = ()
    areas: tuple = ()
    facts: tuple = ()
    clone: str = ""  # the default local clone, relative to the repository root
    pulls: str = ""  # open pull requests whose title matches are listed ("" lists none)


@dataclass(frozen=True)
class HubCheckpoint:
    repo: str


# The compatibility areas of upstream OpenJev's files. A file under openjev/ that none names is a
# new module, often a new backend. Changed lines that mention the re-read policy add "read policy".
OPENJEV_AREAS = (
    Area("wire", ("openjev/api.py", "openjev/chat.py"),
         "routes, request validation, error bodies and the chat endpoint"),
    Area("engine", ("openjev/engine.py",),
         "schemas, templates, canvases, seeds, slot distributions and the re-read loop"),
    Area("settings", ("openjev/config.py",),
         "every OPENJEV_* setting, the re-read threshold and count among them"),
    Area("models", ("openjev/mlx_backend.py", "openjev/encoders.py"),
         "the MLX DiffusionGemma backend and the encoder backends' read paths"),
    Area("server", ("openjev/__init__.py", "openjev/__main__.py", "openjev/warmup.py"),
         "start-up, warm-up and the version"),
    Area("dependencies", ("pyproject.toml",), "the mlx-vlm pin and each backend's packages"),
    Area("vLLM image", ("docker/", "docker-compose.yml"),
         "the images and the vLLM patch of the vLLM backend"),
    Area("tests", ("tests/",),
         "upstream's own tests, which the fixtures and the live suite mirror"),
)
OPENJEV_NEW_MODULE = "new module"
OPENJEV_OTHER = "other"
# Lines of engine.py and config.py that name the re-read policy: the threshold and count settings,
# the seed step of a re-read, the read loop and the entropy that drives it.
READ_POLICY = re.compile(r"auto_threshold|auto_max|OPENJEV_AUTO_|7919|read_group|one_read|"
                         r"slot_distribution|entropy|samples")
READ_POLICY_FILES = ("openjev/engine.py", "openjev/config.py")

GITHUB_PROJECTS = (
    GitHubProject(
        "razorback16/openjev",
        watches=(Watch("upstream's server: the compatibility target", path="openjev"),
                 Watch("upstream's tests", path="tests"),
                 Watch("upstream's dependencies, mlx-vlm's pin among them", path="pyproject.toml")),
        areas=OPENJEV_AREAS,
        facts=(Fact("Upstream's mlx-vlm requirement", "pyproject.toml", r"mlx-vlm==[^;\"'\s]+"),
               Fact("Upstream's backends (pyproject.toml extras)", "pyproject.toml", "")),
        clone="Upstream/openjev",
        pulls=".",
    ),
    GitHubProject(
        "ml-explore/mlx-swift",
        watches=(Watch("MLX's C++ core, which computes every kernel the port runs",
                       path="Source/Cmlx/mlx"),
                 Watch("the Swift API the port calls", path="Source/MLX"),
                 Watch("MLXNN's modules", path="Source/MLXNN"),
                 Watch("MLXFast's RoPE, RMSNorm and attention", path="Source/MLXFast"),
                 Watch("targets, platforms and requirements", path="Package.swift")),
    ),
    GitHubProject(
        "ml-explore/mlx-swift-lm",
        watches=(Watch("SwitchLinear, QuantizedSwitchLinear, gatherSort and scatterUnsort "
                       "(Experts.swift)", path="Libraries/MLXLMCommon/SwitchLayers.swift"),
                 Watch("loadWeights (WeightLoading.swift)",
                       path="Libraries/MLXLMCommon/Load.swift"),
                 Watch("BaseConfiguration's quantization types (Configuration.swift)",
                       path="Libraries/MLXLMCommon/BaseConfiguration.swift"),
                 Watch("the BaseLanguageModel protocol (ModelTree.swift)",
                       path="Libraries/MLXLMCommon/LanguageModel.swift"),
                 Watch("Gemma4VisionConfiguration (Configuration.swift)",
                       path="Libraries/MLXVLM/Models/Gemma4.swift"),
                 Watch("a DiffusionGemma model upstream", pattern=r"(?i)diffusion"),
                 Watch("its mlx-swift requirement, which bounds this package's",
                       path="Package.swift")),
        facts=(Fact("Its mlx-swift requirement", "Package.swift",
                    r"\.package\(url: \"https://github\.com/ml-explore/mlx-swift\"[^\n]*"),),
        pulls=r"(?i)diffusion",
    ),
    GitHubProject(
        "Blaizzy/mlx-vlm",
        watches=(Watch("DiffusionGemma itself, the port's reference",
                       path="mlx_vlm/models/diffusion_gemma"),
                 Watch("the diffusion generation loop", path="mlx_vlm/generate/diffusion.py"),
                 Watch("the Gemma 4 modules the read imports", path="mlx_vlm/models/gemma4"),
                 Watch("KVCache and RotatingKVCache", path="mlx_vlm/models/cache.py"),
                 Watch("SwitchLinear, _gather_sort and _scatter_unsort",
                       path="mlx_vlm/models/switch_layers.py"),
                 Watch("initialize_rope", path="mlx_vlm/models/rope_utils.py"),
                 Watch("the shared attention and output types", path="mlx_vlm/models/base.py")),
        pulls=r"(?i)diffusion",
    ),
    GitHubProject(
        "Layr-Labs/mlx-swift-lm",
        watches=(Watch("the fork's DiffusionGemma implementation, the second reference",
                       pattern=r"^Libraries/.*DiffusionGemma"),
                 Watch("its DiffusionGemma tests and oracles", pattern=r"^Tests/.*DiffusionGemma")),
    ),
    GitHubProject(
        "huggingface/swift-transformers",
        watches=(Watch("the tokenizers (BPE, normalizers, pre-tokenizers, decoders)",
                       path="Sources/Tokenizers"),
                 Watch("the Hub client and the configuration loader", path="Sources/Hub"),
                 Watch("requirements", path="Package.swift")),
    ),
)

HUB_CHECKPOINTS = (
    HubCheckpoint("google/diffusiongemma-26B-A4B-it"),
    HubCheckpoint("mlx-community/diffusiongemma-26B-A4B-it-4bit"),
    HubCheckpoint("mlx-community/diffusiongemma-26B-A4B-it-8bit"),
    HubCheckpoint("mlx-community/diffusiongemma-26B-A4B-it-bf16"),
    HubCheckpoint("convaiinnovations/laya-typed-decisions"),
    HubCheckpoint("heman10x/rlcd-modernbert-151m"),
)


# ---------------------------------------------------------------------------------------------
# Parsing the pins


class ReviewError(Exception):
    """The review cannot start, or one project cannot be read."""


class NotFound(ReviewError):
    pass


def table_rows(markdown):
    """The rows of the Markdown tables in a document, each a list of stripped cells.

    A `|` inside a code span does not split a cell; header and separator rows are dropped.
    """
    rows = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells, cell, in_code = [], "", False
        for char in line[1:-1]:
            if char == "`":
                in_code = not in_code
            if char == "|" and not in_code:
                cells.append(cell.strip())
                cell = ""
            else:
                cell += char
        cells.append(cell.strip())
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def third_party_pins(markdown):
    """Maps each GitHub `owner/name` and Hugging Face `hf:owner/name` that a THIRD_PARTY.md row
    links in its first cell to that row's `Pinned revision` cell."""
    rows = table_rows(markdown)
    header = next((r for r in rows if r and r[0] == "Project"), None)
    if header is None or "Pinned revision" not in header:
        raise ReviewError("THIRD_PARTY.md has no table with a Pinned revision column")
    column = header.index("Pinned revision")
    pins = {}
    for row in rows:
        if row is header or len(row) <= column:
            continue
        for host, name in re.findall(
                r"\]\(https://(github\.com|huggingface\.co)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\)",
                row[0]):
            key = name if host == "github.com" else "hf:" + name
            pins.setdefault(key, row[column])
    return pins


HEX = re.compile(r"^[0-9a-f]{7,40}$")


def pinned_revision(cell):
    """The revision a `Pinned revision` cell names: its first code span that is a commit or a
    version, else the first bare version number. None when the cell names neither."""
    for span in re.findall(r"`([^`]+)`", cell):
        if HEX.match(span) or re.match(r"^v?\d+(\.\d+)+", span):
            return span
    match = re.search(r"(?<![\w.])v?\d+\.\d+(?:\.\d+)*(?![\w.])", cell)
    return match.group(0) if match else None


def pinned_checkpoint_revision(cell):
    """The commit a checkpoint's cell pins, the longest hexadecimal code span, or None."""
    spans = [span for span in re.findall(r"`([^`]+)`", cell) if HEX.match(span)]
    return max(spans, key=len) if spans else None


def pinned_date(cell):
    """The first ISO date a cell names (the day a pin without a revision was read), or None."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", cell)
    return match.group(0) if match else None


def makefile_commit(makefile):
    match = re.search(r"^UPSTREAM_OPENJEV_COMMIT := (\S+)$", makefile, re.M)
    return match.group(1) if match else None


def resolved_pins(resolved_json):
    """Maps a Package.resolved identity to (version or None, revision)."""
    data = json.loads(resolved_json)
    return {pin["identity"]: (pin["state"].get("version"), pin["state"].get("revision", ""))
            for pin in data.get("pins", [])}


def requirement_version(requirements, package):
    match = re.search(rf"^{re.escape(package)}==(\S+)$", requirements, re.M)
    return match.group(1) if match else None


def local_pins(name, pinned, files):
    """What this repository's other files pin for a project, and the disagreements with
    THIRD_PARTY.md. `files` maps "Makefile", "Package.resolved" and "oracle" to their text (None
    when the file is missing). A file or an entry that is missing counts as a disagreement.
    Returns (facts, warnings), two lists of sentences."""
    facts, warnings = [], []

    def text_of(key, label):
        text = files.get(key)
        if text is None:
            warnings.append(f"{label} is missing, so its pin was not compared.")
        return text

    if name == "razorback16/openjev":
        text = text_of("Makefile", "The Makefile")
        commit = makefile_commit(text) if text is not None else None
        if text is not None and commit is None:
            warnings.append("The Makefile sets no `UPSTREAM_OPENJEV_COMMIT`.")
        elif commit is not None:
            facts.append(f"The Makefile's `UPSTREAM_OPENJEV_COMMIT` is `{commit}`.")
            if commit != pinned:
                warnings.append(f"The Makefile pins `{commit}`, THIRD_PARTY.md `{pinned}`.")
    elif name in ("ml-explore/mlx-swift", "ml-explore/mlx-swift-lm",
                  "huggingface/swift-transformers"):
        identity = name.split("/")[1]
        text = text_of("Package.resolved", "Package.resolved")
        pin = resolved_pins(text).get(identity) if text is not None else None
        if text is not None and pin is None:
            warnings.append(f"Package.resolved has no pin for {identity}.")
        elif pin is not None:
            version, revision = pin
            facts.append(f"Package.resolved resolves {version or 'revision'} `{revision[:7]}`.")
            if name == "ml-explore/mlx-swift" and version != pinned:
                warnings.append(f"Package.resolved resolves {version}, THIRD_PARTY.md pins "
                                f"{pinned}.")
            if name == "ml-explore/mlx-swift-lm" and not revision.startswith(pinned or "-"):
                warnings.append(f"Package.resolved resolves `{revision[:7]}`, THIRD_PARTY.md "
                                f"pins `{pinned}`.")
    elif name == "Blaizzy/mlx-vlm":
        text = text_of("oracle", "Tools/oracle/requirements.txt")
        version = requirement_version(text, "mlx-vlm") if text is not None else None
        if text is not None and version is None:
            warnings.append("Tools/oracle/requirements.txt locks no mlx-vlm.")
        elif version is not None:
            facts.append(f"Tools/oracle/requirements.txt locks mlx-vlm {version}.")
            if version != (pinned or "").lstrip("v"):
                warnings.append(f"Tools/oracle/requirements.txt locks mlx-vlm {version}, "
                                f"THIRD_PARTY.md pins {pinned}.")
    return facts, warnings


def area_of(path, areas=OPENJEV_AREAS):
    for area in areas:
        if any(path == p or (p.endswith("/") and path.startswith(p)) for p in area.prefixes):
            return area.name
    return OPENJEV_NEW_MODULE if path.startswith("openjev/") else OPENJEV_OTHER


def touches_read_policy(patch):
    """Whether a unified diff adds or removes a line that names the re-read policy."""
    return any(READ_POLICY.search(line) for line in patch.splitlines()
               if line[:1] in "+-" and not line.startswith(("+++", "---")))


def fact_value(text, fact):
    """A Fact's value in one revision of its file, or None."""
    if text is None:
        return None
    if fact.pattern == "":
        return pyproject_extras(text)
    match = re.search(fact.pattern, text)
    return match.group(0).strip() if match else None


def pyproject_extras(text):
    """The optional-dependency groups of a pyproject.toml other than `test`, comma-separated."""
    section = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not section:
        return None
    names = re.findall(r"^([A-Za-z0-9_.-]+)\s*=", section.group(1), re.M)
    return ", ".join(n for n in names if n != "test") or None


def utc(stamp):
    """An ISO 8601 time as `YYYY-MM-DDTHH:MM:SSZ` in UTC."""
    if not stamp:
        return None
    value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def first_line(message):
    return (message or "").strip().splitlines()[0].strip() if (message or "").strip() else ""


def api_commit(item):
    """A commit from the REST API as the review records it."""
    commit = item["commit"]
    author = (item.get("author") or {}).get("login") or commit["author"]["name"]
    return {"sha": item["sha"], "date": utc(commit["committer"]["date"]),
            "subject": first_line(commit["message"]), "author": author}


def parse_log(output):
    """Commits from `git log --format=%H%x1f%cI%x1f%an%x1f%s` output."""
    commits = []
    for line in output.splitlines():
        if not line.strip():
            continue
        sha, date, author, subject = line.split("\x1f", 3)
        commits.append({"sha": sha, "date": utc(date), "subject": subject.strip(),
                        "author": author})
    return commits


def parse_name_status(output):
    """Files from `git diff --name-status -M` output: path, status and the old path of a rename."""
    statuses = {"A": "added", "M": "modified", "D": "removed", "R": "renamed", "C": "copied",
                "T": "changed"}
    files = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = statuses.get(parts[0][:1], "modified")
        entry = {"path": parts[-1], "status": status}
        if status in ("renamed", "copied"):
            entry["previous"] = parts[1]
        files.append(entry)
    return files


def parse_numstat(output):
    """Maps a path to (additions, deletions) from `git diff --numstat -M` output."""
    counts = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if " => " in path:  # a rename: dir/{old => new}/file or old => new
            path = re.sub(r"\{[^{}]* => ([^{}]*)\}", r"\1", path)
            path = path.split(" => ")[-1]
            path = path.replace("//", "/")
        counts[path] = (int(added) if added.isdigit() else 0,
                        int(deleted) if deleted.isdigit() else 0)
    return counts


def parse_ls_remote(output):
    """(default branch, head commit) from `git ls-remote --symref URL HEAD` output."""
    branch = sha = None
    for line in output.splitlines():
        if line.startswith("ref: ") and line.endswith("\tHEAD"):
            branch = line[len("ref: "):-len("\tHEAD")].removeprefix("refs/heads/")
        elif line.endswith("\tHEAD"):
            sha = line.split("\t")[0]
    return branch, sha


# ---------------------------------------------------------------------------------------------
# Reading upstream


def retry(attempt, describe):
    """Runs `attempt()` up to ATTEMPTS times while it raises a transient error."""
    for number in range(1, ATTEMPTS + 1):
        try:
            return attempt()
        except NotFound:
            raise
        except ReviewError as error:
            if number == ATTEMPTS:
                raise ReviewError(f"{describe}: {error}") from None
            time.sleep(2 * number)


class GitHub:
    """GET requests to the REST API through a GitHub CLI (`gh api` or a wrapper with its
    interface). Every call passes --method GET, since gh turns a request with fields into a POST."""

    def __init__(self, command):
        self.command = command
        self.calls = 0

    def get(self, path):
        def attempt():
            self.calls += 1
            try:
                result = subprocess.run([self.command, "api", "--method", "GET", path],
                                        capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
            except subprocess.TimeoutExpired:
                raise ReviewError(f"no answer within {COMMAND_TIMEOUT} s") from None
            if result.returncode == 0:
                return json.loads(result.stdout or "null")
            message = (result.stderr or result.stdout).strip().splitlines()
            message = message[-1] if message else f"exit status {result.returncode}"
            if "HTTP 404" in message:
                raise NotFound(message)
            if re.search(r"HTTP (401|403|422)", message) and "rate limit" not in message:
                raise NotFound(message)  # not transient either
            raise ReviewError(message)

        return retry(attempt, f"{self.command} api {path}")

    def pages(self, path, key=None, limit=1000):
        """Every item of a paginated list endpoint (or of `key` in each page), up to `limit`."""
        items, page = [], 1
        separator = "&" if "?" in path else "?"
        while len(items) < limit:
            data = self.get(f"{path}{separator}per_page=100&page={page}")
            batch = data.get(key, []) if key else data
            items += batch
            if len(batch) < 100:
                break
            page += 1
        return items


def http_json(url):
    """GET a JSON document and its Link header's next URL, with retries."""

    def attempt():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                link = response.headers.get("Link") or ""
                match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                return json.loads(response.read().decode("utf-8")), match and match.group(1)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                raise NotFound(f"HTTP {error.code} for {url}") from None
            raise ReviewError(f"HTTP {error.code} for {url}") from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise ReviewError(f"{url}: {error}") from None

    return retry(attempt, url)


class Clone:
    """A local clone that already holds the commits a review needs. Read only: it is never
    fetched, checked out or written."""

    def __init__(self, path):
        self.path = Path(path)

    def git(self, *args):
        result = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise ReviewError(f"git {' '.join(args)}: {result.stderr.strip()}")
        return result.stdout

    def origin(self):
        try:
            return self.git("config", "--get", "remote.origin.url").strip()
        except ReviewError:
            return ""

    def has(self, revision):
        result = subprocess.run(["git", "-C", str(self.path), "cat-file", "-e",
                                 f"{revision}^{{commit}}"], capture_output=True)
        return result.returncode == 0

    def commit(self, revision):
        commits = parse_log(self.git("log", "-1", "--format=%H%x1f%cI%x1f%an%x1f%s", revision,
                                     "--"))
        return commits[0]

    def log(self, base, head, paths=()):
        return parse_log(self.git("log", "--format=%H%x1f%cI%x1f%an%x1f%s", f"{base}..{head}",
                                  "--", *paths))

    def files(self, base, head):
        files = parse_name_status(self.git("diff", "--name-status", "-M", base, head))
        counts = parse_numstat(self.git("diff", "--numstat", "-M", base, head))
        for entry in files:
            entry["additions"], entry["deletions"] = counts.get(entry["path"], (0, 0))
        return files

    def file_commits(self, base, head):
        """Maps each path changed in base..head to the commits that changed it, newest first."""
        output = self.git("log", "--format=%x1e%H", "--name-only", f"{base}..{head}")
        mapping = {}
        for block in output.split("\x1e"):
            lines = [line for line in block.splitlines() if line.strip()]
            for path in lines[1:]:
                mapping.setdefault(path, []).append(lines[0])
        return mapping

    def patch(self, base, head, path):
        return self.git("diff", base, head, "--", path)

    def show(self, revision, path):
        try:
            return self.git("show", f"{revision}:{path}")
        except ReviewError:
            return None


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ls_remote(url):
    """(default branch, head commit) of a public repository, with retries. git never prompts:
    a repository that is gone or private answers with an error instead."""

    def attempt():
        try:
            result = subprocess.run(["git", "ls-remote", "--symref", url, "HEAD"],
                                    capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
                                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        except subprocess.TimeoutExpired:
            raise ReviewError(f"no answer within {COMMAND_TIMEOUT} s") from None
        if result.returncode != 0:
            raise ReviewError(result.stderr.strip() or f"exit status {result.returncode}")
        branch, sha = parse_ls_remote(result.stdout)
        if not branch or not sha:
            raise NotFound("no HEAD in its answer")
        return branch, sha

    return retry(attempt, f"git ls-remote {url}")


# ---------------------------------------------------------------------------------------------
# Reviewing one project


def releases_since(releases, pinned, pinned_date):
    """The published releases newer than the pin: after the pinned release when the pin is a
    release tag, else published after the pinned commit's date. Newest first."""
    published = [r for r in releases if not r.get("draft")]
    tags = [r["tag_name"] for r in published]
    if pinned in tags:
        return published[:tags.index(pinned)]
    return [r for r in published if (r.get("published_at") or "") > (pinned_date or "")]


def release_record(release):
    return {"tag": release["tag_name"], "name": release.get("name") or release["tag_name"],
            "date": utc(release.get("published_at")), "prerelease": bool(release.get("prerelease")),
            "url": release.get("html_url")}


def watched_commits(watch, commits, files, files_complete, query):
    """The commits of base..head that touch a Watch, newest first.

    `commits` are base..head's commits, `files` its changed paths, and `query(path)` the commits
    of the head's history that touch `path` since the oldest commit of base..head (the commits API,
    or git log). A path the comparison's complete file list does not name costs no query."""
    in_range = {c["sha"] for c in commits}
    if not in_range:
        return []
    if watch.pattern:
        if not files_complete:
            raise ReviewError("the comparison lists too many files to match a pattern")
        paths = sorted({f["path"] for f in files if re.search(watch.pattern, f["path"])})
    elif files_complete and not any(f["path"] == watch.path
                                    or f["path"].startswith(watch.path + "/") for f in files):
        paths = []
    else:
        paths = [watch.path]
    found = {}
    for path in paths:
        for commit in query(path):
            if commit["sha"] in in_range:
                found[commit["sha"]] = commit
    return sorted(found.values(), key=lambda c: c["date"] or "", reverse=True)


def review_github(project, cell, github, clone=None, local=None, max_commits=MAX_COMMITS):
    """The review record of one GitHub project. A failure becomes the record's `error`."""
    url = f"https://github.com/{project.repo}"
    record = {"kind": "github", "name": project.repo, "url": url, "pinned_text": cell,
              "pinned": {"rev": pinned_revision(cell)}, "facts": [], "warnings": [],
              "watched": [], "files": [], "commits": [], "commit_count": 0, "releases": [],
              "latest_release": None, "error": None}
    if local is not None:
        record["facts"], record["warnings"] = local_pins(project.repo, record["pinned"]["rev"],
                                                         local)
    try:
        _review_github(project, record, github, clone, max_commits)
    except ReviewError as error:
        record["error"] = str(error)
    return record


def _review_github(project, record, github, clone, max_commits):
    repo, pinned = project.repo, record["pinned"]["rev"]
    if pinned is None:
        raise ReviewError("THIRD_PARTY.md names no revision for it")
    branch, head_sha = ls_remote(record["url"] + ".git")
    if clone is not None and not (clone.has(head_sha) and clone.has(pinned)):
        missing = (f"`{head_sha[:7]}`, the head of {branch}" if not clone.has(head_sha)
                   else f"the pin, `{pinned}`")
        record["facts"].append(f"The clone in {display_path(clone.path)} lacks {missing}, so the "
                               "GitHub API was read instead; `make upstream` fetches it.")
        clone = None
    record["source"] = f"git, in {display_path(clone.path)}" if clone else "the GitHub API"
    record["branch"] = branch

    def commit_of(revision):
        if clone:
            return clone.commit(revision)
        return api_commit(github.get(f"repos/{repo}/commits/{revision}"))

    pin = commit_of(pinned)
    record["pinned"].update(pin)
    head = commit_of(head_sha)
    record["head"] = head

    if clone:
        commits = clone.log(pin["sha"], head["sha"])
        ahead = len(commits)
        behind = len(clone.log(head["sha"], pin["sha"]))
        files = clone.files(pin["sha"], head["sha"]) if commits else []
        complete = True
        by_file = clone.file_commits(pin["sha"], head["sha"]) if project.areas else {}

        def query(path):
            return clone.log(pin["sha"], head["sha"], (path,))
    else:
        compare = github.get(f"repos/{repo}/compare/{pin['sha']}...{head['sha']}?per_page=100")
        ahead, behind = compare["ahead_by"], compare["behind_by"]
        items = list(compare.get("commits", []))
        page = 2
        while len(items) < compare.get("total_commits", 0):
            more = github.get(f"repos/{repo}/compare/{pin['sha']}...{head['sha']}"
                              f"?per_page=100&page={page}").get("commits", [])
            if not more:
                break
            items += more
            page += 1
        commits = [api_commit(item) for item in reversed(items)]
        files = [{"path": f["filename"], "status": f["status"], "additions": f["additions"],
                  "deletions": f["deletions"], "patch": f.get("patch"),
                  **({"previous": f["previous_filename"]} if f.get("previous_filename") else {})}
                 for f in compare.get("files", [])]
        complete = len(files) < COMPARE_FILE_CAP
        by_file = {}
        if project.areas and 0 < len(commits) <= 60:
            for commit in commits:
                detail = github.get(f"repos/{repo}/commits/{commit['sha']}")
                for f in detail.get("files", []):
                    by_file.setdefault(f["filename"], []).append(commit["sha"])
        since = min((c["date"] for c in commits), default=None)

        def query(path):
            items = github.pages(f"repos/{repo}/commits?sha={head['sha']}&path={path}"
                                 f"&since={since}", limit=MAX_WATCHED * 10)
            return [api_commit(item) for item in items]

    compare_url = f"{record['url']}/compare/{pin['sha'][:12]}...{head['sha'][:12]}"
    record["comparison"] = {"ahead": ahead, "behind": behind, "url": compare_url}
    record["commit_count"] = len(commits)
    record["commits"] = commits[:max_commits]
    if behind:
        record["warnings"].append(f"The pin is not an ancestor of {branch}: {branch} lacks "
                                  f"{behind} of its commits (a force-push or another branch).")

    for watch in project.watches:
        entry = {"what": watch.path or watch.pattern, "why": watch.why, "commits": [],
                 "pattern": bool(watch.pattern)}
        try:
            found = watched_commits(watch, commits, files, complete, query)
            entry["count"] = len(found)
            entry["commits"] = found[:MAX_WATCHED]
        except ReviewError as error:
            entry["error"] = str(error)
        record["watched"].append(entry)

    if project.areas:
        for entry in files:
            path = entry["path"]
            area = area_of(path, project.areas)
            tags = [area]
            if path in READ_POLICY_FILES and entry["status"] != "removed":
                patch = clone.patch(pin["sha"], head["sha"], path) if clone else entry.get("patch")
                if patch is None:
                    tags.append("read policy unknown: no patch")
                elif touches_read_policy(patch):
                    tags.append("read policy")
            entry["areas"] = tags
            entry["commits"] = by_file.get(path, [])
        record["files"] = [{k: v for k, v in f.items() if k != "patch"} for f in files]
        record["files_complete"] = complete
    record["files_count"] = len(files)

    def text_at(revision, path):
        if clone:
            return clone.show(revision, path)
        try:
            data = github.get(f"repos/{repo}/contents/{path}?ref={revision}")
        except NotFound:
            return None
        return base64.b64decode(data.get("content", "")).decode("utf-8", "replace")

    for fact in project.facts:
        at_pin = fact_value(text_at(pin["sha"], fact.path), fact)
        at_head = at_pin if head["sha"] == pin["sha"] else fact_value(
            text_at(head["sha"], fact.path), fact)
        record.setdefault("file_facts", []).append(
            {"label": fact.label, "path": fact.path, "pinned": at_pin, "head": at_head})

    releases = github.pages(f"repos/{repo}/releases", limit=100)
    stable = [r for r in releases if not r.get("draft") and not r.get("prerelease")]
    record["latest_release"] = release_record(stable[0]) if stable else None
    newer = releases_since(releases, pinned, pin["date"])
    for release in newer:
        entry = release_record(release)
        if HEX.match(pinned):  # does the release contain the pinned commit?
            try:
                status = github.get(f"repos/{repo}/compare/{pin['sha']}...{release['tag_name']}"
                                    "?per_page=1")
                entry["contains_pin"] = status["behind_by"] == 0
                entry["after_pin"] = status["ahead_by"]
            except ReviewError:
                pass
        record["releases"].append(entry)

    if project.pulls:
        pulls = github.pages(f"repos/{repo}/pulls?state=open&sort=updated&direction=desc",
                             limit=300)
        record["pulls"] = [{"number": pull["number"], "title": pull["title"],
                            "created": utc(pull["created_at"]), "updated": utc(pull["updated_at"]),
                            "author": (pull.get("user") or {}).get("login"),
                            "draft": bool(pull.get("draft")), "url": pull["html_url"]}
                           for pull in pulls if re.search(project.pulls, pull["title"])]
        record["pulls_filter"] = project.pulls


def review_checkpoint(checkpoint, cell):
    """The review record of one Hugging Face checkpoint."""
    url = f"{HUB}/{checkpoint.repo}"
    record = {"kind": "hub", "name": checkpoint.repo, "url": url, "pinned_text": cell,
              "pinned": {"rev": pinned_checkpoint_revision(cell), "date": pinned_date(cell)},
              "commits": [], "commit_count": 0, "error": None, "facts": [], "warnings": []}
    try:
        info, _ = http_json(f"{HUB}/api/models/{checkpoint.repo}")
        record["head"] = {"sha": info.get("sha"), "date": utc(info.get("lastModified"))}
        pinned = record["pinned"]["rev"]
        moved = (not info.get("sha", "").startswith(pinned) if pinned
                 else (record["head"]["date"] or "") > (record["pinned"]["date"] or "9999"))
        if moved:
            commits, found = [], False
            next_url = f"{HUB}/api/models/{checkpoint.repo}/commits/main"
            while next_url and not found and len(commits) < HUB_COMMIT_LIMIT:
                page, next_url = http_json(next_url)
                for item in page:
                    if pinned and item["id"].startswith(pinned):
                        found = True
                        break
                    if not pinned and utc(item["date"]) <= record["pinned"]["date"] + "T23:59:59Z":
                        found = True
                        break
                    commits.append({"sha": item["id"], "date": utc(item["date"]),
                                    "subject": first_line(item.get("title")),
                                    "author": ", ".join(a.get("user", "") for a in
                                                        item.get("authors", []))})
            # Not found with older history left to read: the search stopped at the limit.
            capped = not found and bool(next_url)
            record["commits"] = commits
            record["commit_count"] = len(commits)
            record["commits_capped"] = capped
            if capped:
                record["warnings"].append(f"The search stopped after main's newest {len(commits)} "
                                          "commits without reaching the pin, so there are more.")
            elif not found and pinned:
                record["warnings"].append(f"`{pinned[:8]}` is not in main's history: the pinned "
                                          "revision may be on another branch, or main was "
                                          "rewritten.")
    except ReviewError as error:
        record["error"] = str(error)
    return record


# ---------------------------------------------------------------------------------------------
# The review


def moved(record):
    """Whether a project has anything new since its pin."""
    if record["kind"] == "hub":
        head = record.get("head") or {}
        pinned = record["pinned"]
        if pinned.get("rev"):
            return bool(head.get("sha")) and not head["sha"].startswith(pinned["rev"])
        return (head.get("date") or "") > (pinned.get("date") or "9999")
    comparison = record.get("comparison") or {}
    return bool(comparison.get("ahead") or comparison.get("behind") or record.get("releases"))


def needs_attention(record):
    return moved(record) or bool(record.get("error")) or bool(record.get("warnings"))


def fingerprint(review):
    """What the review saw, for telling whether a later run saw something else: each pin, head,
    release, error and warning. The time of the run is left out."""
    state = state_of(review)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def state_of(review):
    """Maps each project to a short description of what it is at, for tracking_issue.py."""
    state = {}
    for record in review["projects"]:
        if record.get("error"):
            state[record["name"]] = "error: " + record["error"][:200]
            continue
        head = (record.get("head") or {}).get("sha", "")
        # The newest release since the pin, a prerelease included, else the latest release.
        newest = (record.get("releases") or [None])[0] or record.get("latest_release") or {}
        release = newest.get("tag", "")
        parts = [f"pin {record['pinned'].get('rev')}", f"head {head[:12]}"]
        if release:
            parts.append(f"release {release}")
        parts += record.get("warnings", [])
        state[record["name"]] = "; ".join(parts)
    return state


def collect(github_command, clones, max_commits=MAX_COMMITS, only=None, progress=None):
    """Reviews the projects the script follows, or those of them `only` names. `clones` maps
    owner/name to a local clone's path."""
    markdown = THIRD_PARTY.read_text(encoding="utf-8")
    pins = third_party_pins(markdown)
    local = {}
    for key, path in (("Makefile", MAKEFILE), ("Package.resolved", PACKAGE_RESOLVED),
                      ("oracle", ORACLE_REQUIREMENTS)):
        local[key] = path.read_text(encoding="utf-8") if path.exists() else None
    if shutil.which(github_command) is None:
        raise ReviewError(f"{github_command} is not on PATH; pass --gh with a GitHub CLI")
    github = GitHub(github_command)
    projects = []
    for project in GITHUB_PROJECTS:
        if only and project.repo not in only:
            continue
        if progress:
            progress(project.repo)
        cell = pins.get(project.repo)
        if cell is None:
            projects.append({"kind": "github", "name": project.repo,
                             "url": f"https://github.com/{project.repo}", "pinned": {"rev": None},
                             "pinned_text": None, "facts": [], "warnings": [],
                             "error": "THIRD_PARTY.md has no row that links it"})
            continue
        clone = None
        path = clones.get(project.repo)
        if path is not None:
            candidate = Clone(path if Path(path).is_absolute() else ROOT / path)
            origin = candidate.origin().removesuffix(".git").lower()
            if origin.endswith(project.repo.lower()):
                clone = candidate
        projects.append(review_github(project, cell, github, clone, local, max_commits))
    for checkpoint in HUB_CHECKPOINTS:
        if only and checkpoint.repo not in only:
            continue
        if progress:
            progress(checkpoint.repo)
        cell = pins.get("hf:" + checkpoint.repo)
        if cell is None:
            projects.append({"kind": "hub", "name": checkpoint.repo,
                             "url": f"{HUB}/{checkpoint.repo}", "pinned": {"rev": None},
                             "pinned_text": None, "facts": [], "warnings": [],
                             "error": "THIRD_PARTY.md has no row that links it"})
            continue
        projects.append(review_checkpoint(checkpoint, cell))
    review = {"format": FORMAT_VERSION,
              "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "projects": projects, "github_calls": github.calls}
    review["attention"] = [r["name"] for r in projects if needs_attention(r)]
    review["fingerprint"] = fingerprint(review)
    return review


# ---------------------------------------------------------------------------------------------
# Rendering


def short(sha):
    return f"`{sha[:7]}`" if sha else "?"


def day(stamp):
    return (stamp or "")[:10] or "?"


def plural(count, word, words=None):
    return f"{count} {word if count == 1 else (words or word + 's')}"


def commit_line(commit):
    subject = commit.get("subject") or "(no message)"
    return f"- {short(commit['sha'])} {day(commit.get('date'))} {subject}"


def summary_now(record):
    if record.get("error"):
        return "not read"
    head = record.get("head") or {}
    if record["kind"] == "hub":
        return f"{short(head.get('sha'))} on main, modified {day(head.get('date'))}"
    text = f"{short(head.get('sha'))} on {record.get('branch', 'the default branch')}"
    release = record.get("latest_release")
    if release:
        text = f"release {release['tag']} ({day(release['date'])}); " + text
    return text


def summary_since(record):
    if record.get("error"):
        return "error: " + record["error"]
    if not moved(record):
        return "nothing new" + ("; " + " ".join(record["warnings"]) if record.get("warnings")
                                 else "")
    parts = []
    if record["kind"] == "github":
        if record.get("releases"):
            parts.append(plural(len(record["releases"]), "release"))
        if record.get("commit_count"):
            parts.append(plural(record["commit_count"], "commit"))
        behind = (record.get("comparison") or {}).get("behind")
        if behind:
            parts.append(f"{behind} pinned commits missing from the branch")
    else:
        count = plural(record.get("commit_count", 0), "commit")
        parts.append(("at least " + count if record.get("commits_capped") else count)
                     if record.get("commits") else "main moved")
    return ", ".join(parts) + ("; " + " ".join(record["warnings"]) if record.get("warnings")
                               else "")


def summary_pinned(record):
    pinned = record.get("pinned") or {}
    rev = pinned.get("rev")
    if not rev:
        return f"as of {pinned['date']}" if pinned.get("date") else "none"
    text = f"`{rev[:7]}`" if HEX.match(rev) else rev
    if record["kind"] == "github" and pinned.get("sha") and not HEX.match(rev):
        text += f" {short(pinned['sha'])}"
    if pinned.get("date") and record["kind"] == "github":
        text += f" ({day(pinned['date'])})"
    return text


def render(review):
    """The review as Markdown."""
    projects = review["projects"]
    attention = [r for r in projects if needs_attention(r)]
    lines = [f"# Upstream review of {day(review['reviewed_at'])}", ""]
    lines.append(f"{plural(len(projects), 'pin')} of THIRD_PARTY.md, the ones "
                 "`Tools/upstream/review.py` follows, against their projects, read at "
                 f"{review['reviewed_at'].replace('T', ' ').replace('Z', ' UTC')}.")
    if attention:
        lines.append(f"{plural(len(attention), 'project')} of {len(projects)} moved or need a "
                     "look: " + ", ".join(r["name"] for r in attention) + ".")
    elif len(projects) == 1:
        lines.append("It is at its pin.")
    else:
        lines.append(f"All {len(projects)} are at their pins.")
    lines += ["", "| Project | Pinned | Now | Since the pin |", "|---|---|---|---|"]
    for record in projects:
        lines.append(f"| [{record['name']}]({record['url']}) | {summary_pinned(record)} | "
                     f"{summary_now(record)} | {summary_since(record)} |")
    for record in projects:
        lines += [""] + (render_github(record) if record["kind"] == "github"
                         else render_checkpoint(record))
    lines += ["", f"<!-- fingerprint {review['fingerprint']} -->"]
    return "\n".join(lines) + "\n"


def render_common(record):
    lines = [f"## {record['name']}", ""]
    if record.get("pinned_text"):
        lines.append(f"THIRD_PARTY.md pins: {record['pinned_text']}.")
    if record.get("error"):
        lines += ["", f"Not read: {record['error']}"]
        return lines, False
    return lines, True


def render_github(record):
    lines, ok = render_common(record)
    if not ok:
        for fact in record.get("facts", []):
            lines.append(fact)
        return lines
    pin, head = record["pinned"], record["head"]
    lines.append(f"The pin is {short(pin['sha'])} ({day(pin['date'])}, {pin['subject']}); "
                 f"{record['branch']} is at {short(head['sha'])} ({day(head['date'])}, "
                 f"{head['subject']}). Read with {record['source']}.")
    release = record.get("latest_release")
    lines.append(f"Latest release: {release['tag']} ({day(release['date'])})."
                 if release else "No release.")
    for fact in record.get("facts", []):
        lines.append(fact)
    for warning in record.get("warnings", []):
        lines.append(f"**Check:** {warning}")
    for fact in record.get("file_facts", []):
        pinned, now = (f"`{v}`" if v else f"none in `{fact['path']}`"
                       for v in (fact["pinned"], fact["head"]))
        if fact["pinned"] == fact["head"]:
            lines.append(f"{fact['label']}: {pinned}, the same at the pin and at "
                         f"{record['branch']}.")
        else:
            lines.append(f"{fact['label']}: {pinned} at the pin, {now} at {record['branch']}.")
    if record.get("releases"):
        lines += ["", "Releases since the pin, newest first:", ""]
        for release in record["releases"]:
            extra = " (prerelease)" if release.get("prerelease") else ""
            if "contains_pin" in release:
                extra += (f", {plural(release['after_pin'], 'commit')} after the pin"
                          if release["contains_pin"] else ", does not contain the pin")
            lines.append(f"- [{release['tag']}]({release['url']}) {day(release['date'])}{extra}")
    count = record.get("commit_count", 0)
    if not count:
        lines += ["", "No commits since the pin."]
        return lines + render_pulls(record)
    shown = record["commits"]
    lines += ["", f"{plural(count, 'commit')} since the pin"
              + (f", the newest {len(shown)} below" if len(shown) < count else "")
              + f" ([compare]({record['comparison']['url']})):", ""]
    lines += [commit_line(c) for c in shown]
    if record.get("files"):
        lines += ["", "Files changed, by area:", "", "| Area | File | Change | Lines | Commits |",
                  "|---|---|---|---|---|"]
        order = {a.name: i for i, a in enumerate(OPENJEV_AREAS)}
        for entry in sorted(record["files"],
                            key=lambda f: (order.get(f["areas"][0], len(order)), f["path"])):
            change = entry["status"] + (f" from `{entry['previous']}`" if entry.get("previous")
                                        else "")
            commits = " ".join(short(sha) for sha in entry.get("commits", [])[:8])
            lines.append(f"| {', '.join(entry['areas'])} | `{entry['path']}` | {change} | "
                         f"+{entry['additions']} -{entry['deletions']} | {commits} |")
        if not record.get("files_complete", True):
            lines.append(f"\nThe comparison lists only the first {COMPARE_FILE_CAP} files.")
    elif record.get("files_count"):
        lines.append(f"\n{plural(record['files_count'], 'file')} changed"
                     + (f" (the comparison lists at most {COMPARE_FILE_CAP})"
                        if record["files_count"] >= COMPARE_FILE_CAP else "") + ".")
    if record.get("watched"):
        lines += ["", "Commits that touch what this repository uses:", ""]
        for entry in record["watched"]:
            label = (f"Paths matching `{entry['what']}`, {entry['why']}" if entry.get("pattern")
                     else f"`{entry['what']}`, {entry['why']}")
            if entry.get("error"):
                lines.append(f"- {label}: not read ({entry['error']}).")
            elif not entry.get("commits"):
                lines.append(f"- {label}: none.")
            else:
                more = (f" ({entry['count']}, the newest {len(entry['commits'])} below)"
                        if entry["count"] > len(entry["commits"]) else "")
                lines.append(f"- {label}{more}:")
                lines += ["  " + commit_line(c) for c in entry["commits"]]
    lines += render_pulls(record)
    return lines


def render_pulls(record):
    if "pulls" not in record:
        return []
    which = ("Open pull requests" if record["pulls_filter"] == "."
             else f"Open pull requests whose title matches `{record['pulls_filter']}`")
    if not record["pulls"]:
        return ["", f"{which}: none."]
    lines = ["", f"{which}, not merged and so not in the comparison:", ""]
    for pull in record["pulls"]:
        draft = ", draft" if pull.get("draft") else ""
        lines.append(f"- [#{pull['number']}]({pull['url']}) {pull['title']} (by {pull['author']}, "
                     f"opened {day(pull['created'])}, updated {day(pull['updated'])}{draft})")
    return lines


def render_checkpoint(record):
    lines, ok = render_common(record)
    if not ok:
        return lines
    head = record["head"]
    if not moved(record):
        lines.append(f"`main` is still at {short(head['sha'])} (last modified "
                     f"{day(head['date'])}).")
    else:
        lines.append(f"`main` is now at {short(head['sha'])} (last modified {day(head['date'])}), "
                     "so a download that names no revision gets other files than the pin.")
        if record.get("commits"):
            count = plural(record["commit_count"], "commit")
            lead = ("At least " + count if record.get("commits_capped")
                    else count[0].upper() + count[1:])
            lines += ["", f"{lead} since the pin:", ""]
            lines += [commit_line(c) for c in record["commits"]]
    for warning in record.get("warnings", []):
        lines.append(f"**Check:** {warning}")
    return lines


# ---------------------------------------------------------------------------------------------


def parse_clones(values):
    clones = {project.repo: project.clone for project in GITHUB_PROJECTS if project.clone}
    for value in values or []:
        repo, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--clone takes OWNER/NAME=PATH, not {value!r}")
        clones[repo] = path
    return clones


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gh", default="ghp",
                        help="the GitHub CLI to read the REST API with (default ghp; gh in CI)")
    parser.add_argument("--json", action="store_true", help="print the review as JSON")
    parser.add_argument("--from", dest="source", type=Path,
                        help="render a review that --json saved, without reading anything else")
    parser.add_argument("--clone", action="append", metavar="OWNER/NAME=PATH",
                        help="a local clone to read a project's history from "
                             "(default razorback16/openjev=Upstream/openjev)")
    parser.add_argument("--project", action="append", metavar="NAME",
                        help="review only this project (repeatable)")
    parser.add_argument("--max-commits", type=int, default=MAX_COMMITS,
                        help=f"commits listed per project, newest first (default {MAX_COMMITS})")
    args = parser.parse_args(argv)
    if args.source:
        review = json.loads(args.source.read_text(encoding="utf-8"))
    else:
        try:
            review = collect(args.gh, parse_clones(args.clone), args.max_commits, args.project,
                             progress=lambda name: print(f"reading {name}", file=sys.stderr))
        except ReviewError as error:
            print(f"review.py: {error}", file=sys.stderr)
            return 1
    if args.json:
        json.dump(review, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render(review))
    return 0


if __name__ == "__main__":
    sys.exit(main())
