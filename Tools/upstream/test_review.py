#!/usr/bin/env python3
"""Tests of the upstream review: no network and no GitHub account.

    python3 Tools/upstream/test_review.py

They cover review.py's parsing (THIRD_PARTY.md's table and the pins of the projects it follows,
git's and the APIs' output), its review of a project from a local clone (a repository the test
builds with git) and from the GitHub API (a stand-in that answers the requests the review makes), a
checkpoint on the Hub, the Markdown it renders, and tracking_issue.py's choice of what to do with
the tracking issue. Standard library and git only; CI runs it in the macOS job beside the JevBench
smoke test, and the Upstream review workflow runs it before each review.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import review  # noqa: E402
import tracking_issue  # noqa: E402

EM_DASH = chr(0x2014)  # spelled as a code point, so this file holds none
PIN_DATE = "2026-09-29T12:00:00+00:00"


def api_item(sha, date, message, login="someone"):
    return {"sha": sha, "author": {"login": login},
            "commit": {"message": message, "author": {"name": login},
                       "committer": {"date": date}}}


def hexsha(seed):
    return (seed * 40)[:40]


class FakeGitHub:
    """Answers the review's REST requests from a table of path prefixes, recording each one."""

    def __init__(self, answers, pages=None):
        self.answers, self.page_answers, self.calls = answers, pages or {}, []

    def get(self, path):
        self.calls.append(path)
        for prefix, answer in self.answers.items():
            if path.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise review.NotFound(f"no answer for {path}")

    def pages(self, path, key=None, limit=1000):
        self.calls.append(path)
        for prefix, answer in self.page_answers.items():
            if path.startswith(prefix):
                return answer
        return []


class ParsingTests(unittest.TestCase):
    def test_table_rows_keep_code_spans_whole(self):
        rows = review.table_rows(
            "| a | b |\n|---|:---:|\n| `x | y` | [z](https://e) |\nnot a row\n")
        self.assertEqual(rows, [["a", "b"], ["`x | y`", "[z](https://e)"]])

    def test_pins_of_a_table(self):
        markdown = (
            "| Project | Role | Pinned revision | License |\n|---|---|---|---|\n"
            "| [a/up](https://github.com/a/up) | x | `dcd2094` (v0.5.0, 2026-09-29) | MIT |\n"
            "| [b/lib](https://github.com/b/lib) and [b/m](https://huggingface.co/b/m) | x | "
            "package 0.3.6; checkpoint `1a793eb` | MIT |\n"
            "| [c/kit](https://github.com/c/kit) | x | 0.32.2 | MIT |\n")
        pins = review.third_party_pins(markdown)
        self.assertEqual(set(pins), {"a/up", "b/lib", "hf:b/m", "c/kit"})
        self.assertEqual(review.pinned_revision(pins["a/up"]), "dcd2094")
        self.assertEqual(review.pinned_revision(pins["c/kit"]), "0.32.2")
        self.assertEqual(review.pinned_checkpoint_revision(pins["hf:b/m"]), "1a793eb")
        with self.assertRaises(review.ReviewError):
            review.third_party_pins("| Name | Version |\n|---|---|\n| a | 1 |\n")

    def test_revisions_of_a_cell(self):
        cases = {
            "`v0.6.15` (upstream's pin)": "v0.6.15",
            "`c043fb3` (2026-09-28)": "c043fb3",
            "0.32.2": "0.32.2",
            "2.23+": "2.23",
            "Model card as of 2026-09-29": None,
            "`main` only": None,
        }
        for cell, expected in cases.items():
            self.assertEqual(review.pinned_revision(cell), expected, cell)
        full = "a7a81407613811e8ba63af92ac0d852b809e191f"
        self.assertEqual(review.pinned_checkpoint_revision(f"`a7a81407` (`{full}`)"), full)
        self.assertEqual(review.pinned_checkpoint_revision(
            "`7b95e3887078ba56283c24f2578d6e5a06b9d7e8` (`main` on 2026-10-01)"),
            "7b95e3887078ba56283c24f2578d6e5a06b9d7e8")
        self.assertIsNone(review.pinned_checkpoint_revision("Model card as of 2026-09-29"))
        self.assertEqual(review.pinned_date("Model card as of 2026-09-29"), "2026-09-29")

    def test_every_reviewed_project_has_a_pin_in_third_party(self):
        pins = review.third_party_pins(review.THIRD_PARTY.read_text(encoding="utf-8"))
        for project in review.GITHUB_PROJECTS:
            self.assertIn(project.repo, pins)
            revision = review.pinned_revision(pins[project.repo])
            self.assertTrue(revision and (review.HEX.match(revision)
                                          or revision.lstrip("v")[0].isdigit()), project.repo)
        for checkpoint in review.HUB_CHECKPOINTS:
            cell = pins["hf:" + checkpoint.repo]
            self.assertTrue(review.pinned_checkpoint_revision(cell) or review.pinned_date(cell),
                            checkpoint.repo)

    def test_the_repository_pins_agree_with_third_party(self):
        # What a review reports under "Check": the Makefile, Package.resolved and the oracle's
        # lock against THIRD_PARTY.md. Moving one pin without the others fails here.
        pins = review.third_party_pins(review.THIRD_PARTY.read_text(encoding="utf-8"))
        local = {"Makefile": review.MAKEFILE.read_text(encoding="utf-8"),
                 "Package.resolved": review.PACKAGE_RESOLVED.read_text(encoding="utf-8"),
                 "oracle": review.ORACLE_REQUIREMENTS.read_text(encoding="utf-8")}
        for project in review.GITHUB_PROJECTS:
            facts, warnings = review.local_pins(
                project.repo, review.pinned_revision(pins[project.repo]), local)
            self.assertEqual(warnings, [], project.repo)
        sources = (review.ROOT / "Sources" / "OpenJevDiffusionGemma" / "Download"
                   / "ModelSource.swift").read_text(encoding="utf-8")
        for name in ("4bit", "8bit", "bf16"):
            cell = pins[f"hf:mlx-community/diffusiongemma-26B-A4B-it-{name}"]
            self.assertIn(review.pinned_checkpoint_revision(cell), sources, name)

    def test_local_pins_report_a_disagreement(self):
        facts, warnings = review.local_pins(
            "razorback16/openjev", "dcd2094", {"Makefile": "UPSTREAM_OPENJEV_COMMIT := 1234567\n"})
        self.assertEqual(facts, ["The Makefile's `UPSTREAM_OPENJEV_COMMIT` is `1234567`."])
        self.assertEqual(warnings, ["The Makefile pins `1234567`, THIRD_PARTY.md `dcd2094`."])
        resolved = json.dumps({"pins": [
            {"identity": "mlx-swift", "state": {"version": "0.32.3", "revision": "1960120abc"}},
            {"identity": "mlx-swift-lm", "state": {"revision": "3b339ad0123"}}]})
        _, warnings = review.local_pins("ml-explore/mlx-swift", "0.32.2",
                                        {"Package.resolved": resolved})
        self.assertEqual(warnings, ["Package.resolved resolves 0.32.3, THIRD_PARTY.md pins "
                                    "0.32.2."])
        _, warnings = review.local_pins("ml-explore/mlx-swift-lm", "c043fb3",
                                        {"Package.resolved": resolved})
        self.assertEqual(len(warnings), 1)
        _, warnings = review.local_pins("Blaizzy/mlx-vlm", "v0.6.15",
                                        {"oracle": "mlx==0.32.2\nmlx-vlm==0.6.15\n"})
        self.assertEqual(warnings, [])

    def test_a_missing_local_pin_is_a_disagreement(self):
        cases = [
            ("razorback16/openjev", "dcd2094", {"Makefile": None},
             "The Makefile is missing, so its pin was not compared."),
            ("razorback16/openjev", "dcd2094", {"Makefile": "PYTHON ?= python3.14\n"},
             "The Makefile sets no `UPSTREAM_OPENJEV_COMMIT`."),
            ("ml-explore/mlx-swift", "0.32.2", {"Package.resolved": None},
             "Package.resolved is missing, so its pin was not compared."),
            ("huggingface/swift-transformers", "af520cf",
             {"Package.resolved": json.dumps({"pins": []})},
             "Package.resolved has no pin for swift-transformers."),
            ("Blaizzy/mlx-vlm", "v0.6.15", {"oracle": None},
             "Tools/oracle/requirements.txt is missing, so its pin was not compared."),
            ("Blaizzy/mlx-vlm", "v0.6.15", {"oracle": "mlx==0.32.2\n"},
             "Tools/oracle/requirements.txt locks no mlx-vlm."),
        ]
        for name, pinned, files, warning in cases:
            facts, warnings = review.local_pins(name, pinned, files)
            self.assertEqual((facts, warnings), ([], [warning]), warning)

    def test_git_output(self):
        log = ("a" * 40 + "\x1f2026-09-30T08:00:00-04:00\x1fAda\x1fFix the read loop\n"
               + "b" * 40 + "\x1f2026-09-29T12:00:00+00:00\x1fBo\x1fAdd | a pipe\n")
        self.assertEqual(review.parse_log(log), [
            {"sha": "a" * 40, "date": "2026-09-30T12:00:00Z", "subject": "Fix the read loop",
             "author": "Ada"},
            {"sha": "b" * 40, "date": "2026-09-29T12:00:00Z", "subject": "Add | a pipe",
             "author": "Bo"}])
        self.assertEqual(review.parse_name_status(
            "M\topenjev/api.py\nA\topenjev/new.py\nR087\tdocs/a.md\tdocs/b.md\nD\told.py\n"),
            [{"path": "openjev/api.py", "status": "modified"},
             {"path": "openjev/new.py", "status": "added"},
             {"path": "docs/b.md", "status": "renamed", "previous": "docs/a.md"},
             {"path": "old.py", "status": "removed"}])
        self.assertEqual(review.parse_numstat(
            "3\t1\topenjev/api.py\n-\t-\tdata.bin\n0\t0\tdocs/{a.md => b.md}\n2\t2\tx => y\n"),
            {"openjev/api.py": (3, 1), "data.bin": (0, 0), "docs/b.md": (0, 0), "y": (2, 2)})
        self.assertEqual(review.parse_ls_remote("ref: refs/heads/main\tHEAD\n" + "c" * 40
                                                + "\tHEAD\n"), ("main", "c" * 40))

    def test_times_become_utc(self):
        self.assertEqual(review.utc("2026-07-15T12:31:54.000Z"), "2026-07-15T12:31:54Z")
        self.assertEqual(review.utc("2026-09-29T09:23:32-07:00"), "2026-09-29T16:23:32Z")
        self.assertEqual(review.utc("2026-09-29T16:23:32Z"), "2026-09-29T16:23:32Z")
        self.assertIsNone(review.utc(None))

    def test_areas_and_the_read_policy(self):
        self.assertEqual(review.area_of("openjev/api.py"), "wire")
        self.assertEqual(review.area_of("openjev/engine.py"), "engine")
        self.assertEqual(review.area_of("openjev/forjev.py"), review.OPENJEV_NEW_MODULE)
        self.assertEqual(review.area_of("tests/test_api.py"), "tests")
        self.assertEqual(review.area_of("docker/Dockerfile.clm"), "vLLM image")
        self.assertEqual(review.area_of("README.md"), review.OPENJEV_OTHER)
        self.assertTrue(review.touches_read_policy(
            "--- a/openjev/config.py\n+++ b/openjev/config.py\n@@\n"
            "-    auto_threshold: float = 0.1\n+    auto_threshold: float = 0.2\n"))
        # Context lines and file headers do not count, only added and removed lines.
        self.assertFalse(review.touches_read_policy(
            "--- a/openjev/engine.py\n+++ b/openjev/engine.py\n@@\n     seed + k * 7919\n"
            "-VOCAB = 262144\n+VOCAB = 262145\n"))

    def test_facts_of_a_file(self):
        pyproject = ("[project]\nname = 'openjev'\n\n[project.optional-dependencies]\n"
                     "test = ['pytest']\nmlx = [\"mlx-vlm==0.6.15; sys_platform == 'darwin'\"]\n"
                     "laya = ['laya==0.3.6']\n\n[project.urls]\nSource = 'x'\n")
        upstream = review.GITHUB_PROJECTS[0]
        requirement, extras = upstream.facts
        self.assertEqual(review.fact_value(pyproject, requirement), "mlx-vlm==0.6.15")
        self.assertEqual(review.fact_value(pyproject, extras), "mlx, laya")
        self.assertIsNone(review.fact_value(None, requirement))
        package = ('dependencies: [\n    .package(url: "https://github.com/ml-explore/mlx-swift", '
                   '.upToNextMinor(from: "0.32.3")),\n]\n')
        lm = next(p for p in review.GITHUB_PROJECTS if p.repo == "ml-explore/mlx-swift-lm")
        self.assertEqual(review.fact_value(package, lm.facts[0]),
                         '.package(url: "https://github.com/ml-explore/mlx-swift", '
                         '.upToNextMinor(from: "0.32.3")),')

    def test_releases_since_a_pin(self):
        releases = [{"tag_name": "v0.7.0", "published_at": "2026-09-07T00:00:00Z"},
                    {"tag_name": "v0.7.0rc0", "published_at": "2026-08-31T00:00:00Z",
                     "prerelease": True},
                    {"tag_name": "draft", "published_at": None, "draft": True},
                    {"tag_name": "v0.6.15", "published_at": "2026-08-18T00:00:00Z"},
                    {"tag_name": "v0.6.14", "published_at": "2026-08-17T00:00:00Z"}]
        self.assertEqual([r["tag_name"] for r in review.releases_since(releases, "v0.6.15", None)],
                         ["v0.7.0", "v0.7.0rc0"])
        self.assertEqual([r["tag_name"] for r in review.releases_since(
            releases, "c043fb3", "2026-09-01T00:00:00Z")], ["v0.7.0"])

    def test_watched_paths(self):
        commits = [{"sha": "a" * 40, "date": "2026-09-30T00:00:00Z", "subject": "x"},
                   {"sha": "b" * 40, "date": "2026-09-29T00:00:00Z", "subject": "y"}]
        files = [{"path": "Sources/A.swift"}, {"path": "Libraries/DiffusionGemma.swift"}]
        queried = []

        def query(path):
            queried.append(path)
            # the history before the range is filtered out
            return [commits[1], {"sha": "c" * 40, "date": "2026-09-01T00:00:00Z", "subject": "z"}]

        watch = review.Watch("a", path="Sources/A.swift")
        self.assertEqual(review.watched_commits(watch, commits, files, True, query), [commits[1]])
        # a path the complete file list does not name costs no query
        self.assertEqual(review.watched_commits(review.Watch("b", path="Package.swift"), commits,
                                                files, True, query), [])
        self.assertEqual(queried, ["Sources/A.swift"])
        # a directory watch matches the files under it
        self.assertEqual(review.watched_commits(review.Watch("c", path="Sources"), commits, files,
                                                True, query), [commits[1]])
        pattern = review.Watch("d", pattern=r"(?i)diffusion")
        self.assertEqual(review.watched_commits(pattern, commits, files, True, query),
                         [commits[1]])
        self.assertEqual(queried[-1], "Libraries/DiffusionGemma.swift")
        with self.assertRaises(review.ReviewError):  # 300 files: a pattern cannot be resolved
            review.watched_commits(pattern, commits, files, False, query)
        self.assertEqual(review.watched_commits(watch, [], files, True, query), [])


class GitHubCLITests(unittest.TestCase):
    """GitHub.get against a stand-in for gh, a script that answers like it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_sleep, review.time.sleep = review.time.sleep, lambda seconds: None

    def tearDown(self):
        review.time.sleep = self.saved_sleep

    def fake_gh(self, body):
        script = self.tmp / "gh"
        script.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                          "count = Path(sys.argv[0] + '.count')\n"
                          "n = int(count.read_text()) + 1 if count.exists() else 1\n"
                          "count.write_text(str(n))\n"
                          "assert sys.argv[1:4] == ['api', '--method', 'GET'], sys.argv\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return review.GitHub(str(script)), Path(str(script) + ".count")

    def test_a_read_is_a_get_and_returns_json(self):
        github, _ = self.fake_gh("print('{\"path\": \"' + sys.argv[4] + '\"}')\n")
        self.assertEqual(github.get("repos/a/b"), {"path": "repos/a/b"})

    def test_a_transient_failure_is_retried(self):
        github, count = self.fake_gh(
            "if n < 3:\n    sys.stderr.write('read: connection reset by peer\\n'); sys.exit(1)\n"
            "print('[]')\n")
        self.assertEqual(github.get("repos/a/b/releases"), [])
        self.assertEqual(count.read_text(), "3")

    def test_not_found_is_not_retried(self):
        github, count = self.fake_gh("sys.stderr.write('gh: Not Found (HTTP 404)\\n'); "
                                     "sys.exit(1)\n")
        with self.assertRaises(review.NotFound):
            github.get("repos/a/missing")
        self.assertEqual(count.read_text(), "1")

    def test_ls_remote_retries_and_never_prompts(self):
        git = self.tmp / "git"
        git.write_text(f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n"
                       "count = Path(sys.argv[0] + '.count')\n"
                       "n = int(count.read_text()) + 1 if count.exists() else 1\n"
                       "count.write_text(str(n))\n"
                       "assert os.environ.get('GIT_TERMINAL_PROMPT') == '0'\n"
                       "assert sys.argv[1:3] == ['ls-remote', '--symref'], sys.argv\n"
                       "if n == 1:\n    sys.stderr.write('unable to access\\n'); sys.exit(128)\n"
                       "print('ref: refs/heads/main\\tHEAD'); print('d' * 40 + '\\tHEAD')\n")
        git.chmod(git.stat().st_mode | stat.S_IEXEC)
        saved = os.environ["PATH"]
        os.environ["PATH"] = f"{self.tmp}{os.pathsep}{saved}"
        try:
            self.assertEqual(review.ls_remote("https://github.com/a/b.git"), ("main", "d" * 40))
        finally:
            os.environ["PATH"] = saved
        self.assertEqual((self.tmp / "git.count").read_text(), "2")

    def test_a_lasting_failure_names_the_request(self):
        github, count = self.fake_gh("sys.stderr.write('HTTP 502\\n'); sys.exit(1)\n")
        with self.assertRaises(review.ReviewError) as raised:
            github.get("repos/a/b")
        self.assertIn("repos/a/b", str(raised.exception))
        self.assertEqual(count.read_text(), str(review.ATTEMPTS))


class LocalCloneTests(unittest.TestCase):
    """Reviews upstream OpenJev's layout from a clone the test builds: the pin, then two commits."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "openjev"
        self.repo.mkdir()
        self.env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
                    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com"}
        self.git("init", "-q", "-b", "main")
        self.write("pyproject.toml", "[project.optional-dependencies]\ntest = ['pytest']\n"
                   "mlx = [\"mlx-vlm==0.6.15; sys_platform == 'darwin'\"]\nlaya = ['laya']\n")
        self.write("openjev/api.py", "ROUTES = ['/v1/systemone']\n")
        self.write("openjev/engine.py", "VOCAB = 262144\n\ndef read(seed, k):\n"
                   "    return seed + k * 7919\n")
        self.write("openjev/config.py", "auto_threshold = 0.1\n")
        self.write("tests/test_api.py", "def test_it():\n    pass\n")
        self.write("docs/a.md", "a long enough line of documentation\n" * 5)
        self.pin = self.commit("Merge pull request #9", PIN_DATE)
        self.write("openjev/api.py", "ROUTES = ['/v1/systemone', '/v1/models']\n")
        self.write("openjev/engine.py", "VOCAB = 262144\n\ndef read(seed, k):\n"
                   "    return seed + k * 7919 + 1\n")
        self.write("openjev/forjev.py", "class ForJev:\n    pass\n")
        self.write("pyproject.toml", "[project.optional-dependencies]\ntest = ['pytest']\n"
                   "mlx = [\"mlx-vlm==0.7.4; sys_platform == 'darwin'\"]\nlaya = ['laya']\n"
                   "forjev = ['httpx']\n")
        self.second = self.commit("Add the ForJev backend", "2026-09-30T08:00:00-04:00")
        self.write("tests/test_api.py", "def test_it():\n    assert True\n")
        self.git("mv", "docs/a.md", "docs/b.md")
        self.head = self.commit("Test the models route\n\nWith a body.", "2026-10-01T00:00:00Z")
        self.saved = review.ls_remote
        review.ls_remote = lambda url: ("main", self.head)

    def tearDown(self):
        review.ls_remote = self.saved

    def git(self, *args, date=None):
        env = dict(self.env)
        if date:
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
        return subprocess.run(["git", "-C", str(self.repo), "-c", "commit.gpgsign=false", *args],
                              check=True, capture_output=True, text=True, env=env).stdout

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message, date):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message, date=date)
        return self.git("rev-parse", "HEAD").strip()

    def test_the_review_of_a_clone(self):
        upstream = review.GITHUB_PROJECTS[0]
        pulls = [{"number": 10, "title": "Add ForJev backend", "created_at": "2026-09-29T14:15:23Z",
                  "updated_at": "2026-09-29T14:15:23Z", "user": {"login": "MhaWay"},
                  "draft": False, "html_url": "https://github.com/razorback16/openjev/pull/10"}]
        github = FakeGitHub({}, {"repos/razorback16/openjev/releases": [],
                                 "repos/razorback16/openjev/pulls": pulls})
        record = review.review_github(
            upstream, f"`{self.pin[:7]}` (v0.5.0, 2026-09-29)", github, review.Clone(self.repo),
            {"Makefile": f"UPSTREAM_OPENJEV_COMMIT := {self.pin[:7]}\n"})
        self.assertIsNone(record["error"])
        self.assertTrue(record["source"].startswith("git, in "))
        self.assertEqual(record["warnings"], [])
        self.assertEqual(record["pinned"]["sha"], self.pin)
        self.assertEqual(record["head"]["subject"], "Test the models route")
        self.assertEqual([c["sha"] for c in record["commits"]], [self.head, self.second])
        self.assertEqual(record["commits"][1]["date"], "2026-09-30T12:00:00Z")
        self.assertEqual(record["comparison"]["ahead"], 2)
        self.assertTrue(review.moved(record))
        areas = {f["path"]: f["areas"] for f in record["files"]}
        self.assertEqual(areas, {"openjev/api.py": ["wire"],
                                 "openjev/engine.py": ["engine", "read policy"],
                                 "openjev/forjev.py": [review.OPENJEV_NEW_MODULE],
                                 "pyproject.toml": ["dependencies"],
                                 "tests/test_api.py": ["tests"],
                                 "docs/b.md": [review.OPENJEV_OTHER]})
        files = {f["path"]: f for f in record["files"]}
        self.assertEqual(files["docs/b.md"]["status"], "renamed")
        self.assertEqual(files["docs/b.md"]["previous"], "docs/a.md")
        self.assertEqual(files["openjev/api.py"]["commits"], [self.second])
        self.assertEqual((files["openjev/api.py"]["additions"],
                          files["openjev/api.py"]["deletions"]), (1, 1))
        watched = {w["what"]: [c["sha"] for c in w["commits"]] for w in record["watched"]}
        self.assertEqual(watched, {"openjev": [self.second], "tests": [self.head],
                                   "pyproject.toml": [self.second]})
        facts = {f["label"]: (f["pinned"], f["head"]) for f in record["file_facts"]}
        self.assertEqual(facts["Upstream's mlx-vlm requirement"],
                         ("mlx-vlm==0.6.15", "mlx-vlm==0.7.4"))
        self.assertEqual(facts["Upstream's backends (pyproject.toml extras)"],
                         ("mlx, laya", "mlx, laya, forjev"))
        self.assertEqual([p["number"] for p in record["pulls"]], [10])
        # Only releases and pull requests come from the API.
        self.assertEqual(github.calls, ["repos/razorback16/openjev/releases",
                                        "repos/razorback16/openjev/pulls?state=open&sort=updated"
                                        "&direction=desc"])
        text = "\n".join(review.render_github(record))
        self.assertIn("| engine, read policy | `openjev/engine.py` | modified | +1 -1 |", text)
        self.assertIn(f"| {review.OPENJEV_NEW_MODULE} | `openjev/forjev.py` | added |", text)
        self.assertIn("Upstream's mlx-vlm requirement: `mlx-vlm==0.6.15` at the pin, "
                      "`mlx-vlm==0.7.4` at main.", text)
        self.assertIn("[#10](https://github.com/razorback16/openjev/pull/10) Add ForJev backend",
                      text)

    def test_a_clone_without_the_head_falls_back_to_the_api(self):
        review.ls_remote = lambda url: ("main", "f" * 40)
        github = FakeGitHub({"repos/razorback16/openjev/commits/": review.NotFound("gone")})
        record = review.review_github(review.GITHUB_PROJECTS[0], f"`{self.pin[:7]}`", github,
                                      review.Clone(self.repo))
        self.assertIn("lacks `fffffff`, the head of main", " ".join(record["facts"]))
        self.assertEqual(record["source"], "the GitHub API")
        self.assertIn("gone", record["error"])


class GitHubAPITests(unittest.TestCase):
    """Reviews a library from the REST API: the compare, its pages, watched paths and releases."""

    def setUp(self):
        self.pin, self.head = "c043fb3" + "0" * 33, "9afc3b5" + "0" * 33
        self.first, self.second = "1" * 40, "2" * 40
        self.saved = review.ls_remote
        review.ls_remote = lambda url: ("main", self.head)

    def tearDown(self):
        review.ls_remote = self.saved

    def test_the_review_of_a_library(self):
        project = review.GitHubProject(
            "owner/lib",
            watches=(review.Watch("loadWeights", path="Sources/Load.swift"),
                     review.Watch("requirements", path="Package.swift"),
                     review.Watch("a model upstream", pattern=r"(?i)diffusion")),
            facts=(review.Fact("Its mlx-swift requirement", "Package.swift", r"mlx-swift[^\n]*"),),
            pulls=r"(?i)diffusion")

        def contents(text):
            return {"content": base64.b64encode(text.encode()).decode()}

        compare = {"ahead_by": 2, "behind_by": 0, "total_commits": 2,
                   "commits": [api_item(self.first, "2026-09-29T23:36:05Z", "Named attachments"),
                               api_item(self.second, "2026-09-30T19:46:55Z",
                                        "require mlx-swift 0.32.3\n\nbody")],
                   "files": [{"filename": "Sources/Load.swift", "status": "modified",
                              "additions": 2, "deletions": 1, "patch": "@@"},
                             {"filename": "README.md", "status": "modified", "additions": 1,
                              "deletions": 0}]}
        github = FakeGitHub(
            {f"repos/owner/lib/commits/{self.pin[:7]}": api_item(self.pin, PIN_DATE, "pin"),
             f"repos/owner/lib/commits/{self.head}": api_item(self.head, "2026-10-02T20:02:02Z",
                                                             "head"),
             f"repos/owner/lib/compare/{self.pin}...{self.head}": compare,
             f"repos/owner/lib/contents/Package.swift?ref={self.pin}":
                 contents('.package(url: "x/mlx-swift", from: "0.32.2")\n'),
             f"repos/owner/lib/contents/Package.swift?ref={self.head}":
                 contents('.package(url: "x/mlx-swift", from: "0.32.3")\n'),
             f"repos/owner/lib/compare/{self.pin}...3.32.3": {"ahead_by": 1, "behind_by": 0}},
            {"repos/owner/lib/releases": [
                 {"tag_name": "3.32.3", "name": "3.32.3", "published_at": "2026-09-30T20:38:34Z",
                  "html_url": "https://github.com/owner/lib/releases/tag/3.32.3"},
                 {"tag_name": "3.31.4", "published_at": "2026-06-30T16:31:18Z"}],
             "repos/owner/lib/commits?sha=": [
                api_item(self.second, "2026-09-30T19:46:55Z", "require mlx-swift 0.32.3"),
                api_item("3" * 40, "2026-09-01T00:00:00Z", "before the pin")],
             "repos/owner/lib/pulls": [
                 {"number": 352, "title": "Base Gemma diffusion implementation",
                  "created_at": "2026-06-15T12:43:05Z", "updated_at": "2026-10-02T23:10:17Z",
                  "user": {"login": "aleroot"}, "draft": False,
                  "html_url": "https://github.com/owner/lib/pull/352"},
                 {"number": 1, "title": "Something else", "created_at": "2026-06-15T12:43:05Z",
                  "updated_at": "2026-10-02T23:10:17Z", "user": {"login": "x"},
                  "html_url": "https://github.com/owner/lib/pull/1"}]})
        record = review.review_github(project, "`c043fb3` (2026-09-28)", github)
        self.assertIsNone(record["error"])
        self.assertEqual(record["source"], "the GitHub API")
        self.assertEqual([c["sha"] for c in record["commits"]], [self.second, self.first])
        self.assertEqual(record["commits"][0]["subject"], "require mlx-swift 0.32.3")
        watched = {w["what"]: [c["sha"] for c in w["commits"]] for w in record["watched"]}
        self.assertEqual(watched, {"Sources/Load.swift": [self.second], "Package.swift": [],
                                   "(?i)diffusion": []})
        # One commits query, for the one watched path the compare names, from the oldest commit.
        queries = [c for c in github.calls if c.startswith("repos/owner/lib/commits?")]
        self.assertEqual(queries, [f"repos/owner/lib/commits?sha={self.head}"
                                   "&path=Sources/Load.swift&since=2026-09-29T23:36:05Z"])
        self.assertEqual(record["latest_release"]["tag"], "3.32.3")
        self.assertEqual(record["releases"][0]["tag"], "3.32.3")
        self.assertTrue(record["releases"][0]["contains_pin"])
        self.assertEqual(record["releases"][0]["after_pin"], 1)
        self.assertEqual(len(record["releases"]), 1)
        self.assertEqual([p["number"] for p in record["pulls"]], [352])
        self.assertEqual(record["file_facts"][0]["pinned"], 'mlx-swift", from: "0.32.2")')
        self.assertEqual(record["file_facts"][0]["head"], 'mlx-swift", from: "0.32.3")')
        text = "\n".join(review.render_github(record))
        self.assertIn("- [3.32.3](https://github.com/owner/lib/releases/tag/3.32.3) 2026-09-30, "
                      "1 commit after the pin", text)
        self.assertIn("- `Package.swift`, requirements: none.", text)
        self.assertIn("- Paths matching `(?i)diffusion`, a model upstream: none.", text)
        self.assertIn("  - `2222222` 2026-09-30 require mlx-swift 0.32.3", text)
        self.assertIn("[#352](https://github.com/owner/lib/pull/352) Base Gemma diffusion "
                      "implementation (by aleroot, opened 2026-06-15, updated 2026-10-02)", text)

    def test_a_missing_pin_is_an_error_of_that_project(self):
        github = FakeGitHub({"repos/owner/lib/commits/": review.NotFound("No commit found")})
        record = review.review_github(review.GitHubProject("owner/lib"), "`c043fb3`", github)
        self.assertIn("No commit found", record["error"])
        self.assertTrue(review.needs_attention(record))
        self.assertIn("Not read: No commit found", "\n".join(review.render_github(record)))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.saved = review.http_json

    def tearDown(self):
        review.http_json = self.saved

    def answer(self, model, commits):
        def http_json(url):
            if url.endswith("/commits/main"):
                return commits, None
            return model, None

        review.http_json = http_json

    def test_a_moved_main(self):
        pinned = "a7a81407613811e8ba63af92ac0d852b809e191f"
        self.answer({"sha": "b" * 40, "lastModified": "2026-10-01T00:00:00.000Z"},
                    [{"id": "b" * 40, "title": "Update chat_template.jinja", "date":
                      "2026-10-01T00:00:00.000Z", "authors": [{"user": "someone"}]},
                     {"id": pinned, "title": "the pin", "date": "2026-07-15T12:31:54.000Z"}])
        record = review.review_checkpoint(review.HubCheckpoint("org/model"),
                                          f"`a7a81407` (`{pinned}`)")
        self.assertTrue(review.moved(record))
        self.assertEqual(record["commit_count"], 1)
        self.assertEqual(record["warnings"], [])
        text = "\n".join(review.render_checkpoint(record))
        self.assertIn("`main` is now at `bbbbbbb`", text)
        self.assertIn("- `bbbbbbb` 2026-10-01 Update chat_template.jinja", text)

    def test_a_search_that_stops_at_the_limit(self):
        pages = {"page1": ([{"id": f"{i:040x}", "title": f"commit {i}",
                             "date": "2026-10-01T00:00:00.000Z"} for i in range(3)], "page2"),
                 "page2": ([{"id": f"{i:040x}", "title": f"commit {i}",
                             "date": "2026-09-30T00:00:00.000Z"} for i in range(3, 6)], "page3")}

        def http_json(url):
            if url.endswith("/commits/main"):
                return pages["page1"]
            if url in pages:
                return pages[url]
            return {"sha": "f" * 40, "lastModified": "2026-10-01T00:00:00.000Z"}, None

        review.http_json = http_json
        saved, review.HUB_COMMIT_LIMIT = review.HUB_COMMIT_LIMIT, 4
        try:
            record = review.review_checkpoint(review.HubCheckpoint("org/model"), "`1a793eb`")
        finally:
            review.HUB_COMMIT_LIMIT = saved
        self.assertTrue(record["commits_capped"])
        self.assertEqual(record["commit_count"], 6)
        self.assertEqual(record["warnings"], ["The search stopped after main's newest 6 commits "
                                              "without reaching the pin, so there are more."])
        self.assertIn("At least 6 commits since the pin:", "\n".join(review.render_checkpoint(
            record)))
        self.assertEqual(review.summary_since(record),
                         "at least 6 commits; " + record["warnings"][0])

    def test_a_pin_that_left_main(self):
        self.answer({"sha": "c" * 40, "lastModified": "2026-10-01T00:00:00.000Z"},
                    [{"id": "c" * 40, "title": "rewrite", "date": "2026-10-01T00:00:00.000Z"}])
        record = review.review_checkpoint(review.HubCheckpoint("org/model"), "`1a793eb`")
        self.assertIn("is not in main's history", record["warnings"][0])

    def test_a_pin_by_date(self):
        self.answer({"sha": "f7f5b7f5" + "0" * 32, "lastModified": "2026-07-15T16:35:45.000Z"}, [])
        record = review.review_checkpoint(review.HubCheckpoint("google/model"),
                                          "Model card as of 2026-09-29")
        self.assertFalse(review.moved(record))
        self.assertIn("`main` is still at `f7f5b7f`", "\n".join(review.render_checkpoint(record)))

    def test_an_unreachable_hub(self):
        def http_json(url):
            raise review.ReviewError("timed out")

        review.http_json = http_json
        record = review.review_checkpoint(review.HubCheckpoint("org/model"), "`1a793eb`")
        self.assertEqual(record["error"], "timed out")


def sample_review():
    """A review with a moved project, an unchanged one, a moved checkpoint and a failure."""
    moved = {"kind": "github", "name": "ml-explore/mlx-swift", "url": "https://github.com/x",
             "pinned_text": "0.32.2", "pinned": {"rev": "0.32.2", "sha": "2b5e877" + "0" * 33,
                                                 "date": "2026-09-28T18:21:07Z", "subject": "pin"},
             "branch": "main", "head": {"sha": "1960120" + "0" * 33, "date": "2026-09-30T18:18:47Z",
                                        "subject": "fix MLXLogger (#493)"},
             "source": "the GitHub API", "facts": ["Package.resolved resolves 0.32.2 `2b5e877`."],
             "warnings": [], "comparison": {"ahead": 1, "behind": 0, "url": "https://compare"},
             "commit_count": 1, "commits": [{"sha": "1960120" + "0" * 33,
                                             "date": "2026-09-30T18:18:47Z",
                                             "subject": "fix MLXLogger (#493)"}],
             "files_count": 7, "files": [], "releases": [
                 {"tag": "0.32.3", "name": "0.32.3", "date": "2026-09-30T18:20:13Z",
                  "prerelease": False, "url": "https://release"}],
             "latest_release": {"tag": "0.32.3", "date": "2026-09-30T18:20:13Z"},
             "watched": [{"what": "Source/MLX", "why": "the Swift API", "pattern": False,
                          "count": 1, "commits": [{"sha": "1960120" + "0" * 33,
                                                   "date": "2026-09-30T18:18:47Z",
                                                   "subject": "fix MLXLogger (#493)"}]},
                         {"what": "Source/Cmlx/mlx", "why": "the core", "pattern": False,
                          "count": 0, "commits": []}],
             "error": None}
    still = {"kind": "github", "name": "razorback16/openjev", "url": "https://github.com/y",
             "pinned_text": "`dcd2094`", "pinned": {"rev": "dcd2094", "sha": "dcd2094" + "0" * 33,
                                                    "date": "2026-09-29T16:23:32Z",
                                                    "subject": "Merge"},
             "branch": "main", "head": {"sha": "dcd2094" + "0" * 33, "date": "2026-09-29T16:23:32Z",
                                        "subject": "Merge"},
             "source": "git, in Upstream/openjev", "facts": [], "warnings": [],
             "comparison": {"ahead": 0, "behind": 0, "url": "https://compare"}, "commit_count": 0,
             "commits": [], "files": [], "releases": [], "latest_release": None, "watched": [],
             "pulls": [], "pulls_filter": ".", "error": None}
    checkpoint = {"kind": "hub", "name": "org/model", "url": "https://huggingface.co/org/model",
                  "pinned_text": "`1a793eb`", "pinned": {"rev": "1a793eb", "date": None},
                  "head": {"sha": "2" * 40, "date": "2026-10-01T00:00:00Z"}, "commits": [],
                  "commit_count": 0, "facts": [], "warnings": [], "error": None}
    failed = {"kind": "hub", "name": "org/gone", "url": "https://huggingface.co/org/gone",
              "pinned_text": "`8af2496`", "pinned": {"rev": "8af2496", "date": None},
              "commits": [], "commit_count": 0, "facts": [], "warnings": [],
              "error": "HTTP 404 for https://huggingface.co/api/models/org/gone"}
    result = {"format": 1, "reviewed_at": "2026-10-02T23:57:27Z",
              "projects": [still, moved, checkpoint, failed]}
    result["attention"] = [r["name"] for r in result["projects"] if review.needs_attention(r)]
    result["fingerprint"] = review.fingerprint(result)
    return result


class RenderTests(unittest.TestCase):
    def test_the_report(self):
        data = sample_review()
        self.assertEqual(data["attention"], ["ml-explore/mlx-swift", "org/model", "org/gone"])
        text = review.render(data)
        self.assertTrue(text.startswith("# Upstream review of 2026-10-02\n"))
        self.assertIn("3 projects of 4 moved or need a look: ml-explore/mlx-swift, org/model, "
                      "org/gone.", text)
        self.assertIn("| [razorback16/openjev](https://github.com/y) | `dcd2094` (2026-09-29) | "
                      "`dcd2094` on main | nothing new |", text)
        self.assertIn("| [ml-explore/mlx-swift](https://github.com/x) | 0.32.2 `2b5e877` "
                      "(2026-09-28) | release 0.32.3 (2026-09-30); `1960120` on main | 1 release, "
                      "1 commit |", text)
        self.assertIn("| main moved |", text)
        self.assertIn("| not read | error: HTTP 404", text)
        self.assertIn("- [0.32.3](https://release) 2026-09-30", text)
        self.assertIn("1 commit since the pin ([compare](https://compare)):", text)
        self.assertIn("- `Source/Cmlx/mlx`, the core: none.", text)
        self.assertIn("Open pull requests: none.", text)
        self.assertIn("`main` is now at `2222222`", text)
        self.assertTrue(text.endswith(f"<!-- fingerprint {data['fingerprint']} -->\n"))
        self.assertNotIn(EM_DASH, text)

    def test_a_fact_a_revision_lacks(self):
        record = sample_review()["projects"][0]
        record["file_facts"] = [{"label": "Upstream's mlx-vlm requirement",
                                 "path": "pyproject.toml", "pinned": "mlx-vlm==0.6.15",
                                 "head": None}]
        self.assertIn("Upstream's mlx-vlm requirement: `mlx-vlm==0.6.15` at the pin, none in "
                      "`pyproject.toml` at main.", "\n".join(review.render_github(record)))

    def test_json_round_trip_renders_the_same(self):
        data = sample_review()
        self.assertEqual(review.render(json.loads(json.dumps(data))), review.render(data))

    def test_the_fingerprint_ignores_the_time_and_follows_the_heads(self):
        data = sample_review()
        later = json.loads(json.dumps(data))
        later["reviewed_at"] = "2026-11-01T06:37:00Z"
        self.assertEqual(review.fingerprint(later), data["fingerprint"])
        later["projects"][1]["head"]["sha"] = "3" * 40
        self.assertNotEqual(review.fingerprint(later), data["fingerprint"])

    def test_a_prerelease_since_the_pin_changes_the_state(self):
        data = sample_review()
        before = review.state_of(data)["ml-explore/mlx-swift"]
        data["projects"][1]["releases"].insert(0, {"tag": "0.33.0rc1", "prerelease": True})
        after = review.state_of(data)["ml-explore/mlx-swift"]
        self.assertTrue(before.endswith("release 0.32.3"))
        self.assertTrue(after.endswith("release 0.33.0rc1"))

    def test_the_header_counts_the_pins_it_read(self):
        data = sample_review()
        data["projects"] = data["projects"][:1]
        data["attention"] = []
        text = review.render(data)
        self.assertIn("1 pin of THIRD_PARTY.md, the ones `Tools/upstream/review.py` follows, "
                      "against their projects, read at 2026-10-02 23:57:27 UTC.\nIt is at its "
                      "pin.", text)

    def test_the_scripts_hold_no_em_dash(self):
        for name in ("review.py", "tracking_issue.py", "test_review.py"):
            self.assertNotIn(EM_DASH, (HERE / name).read_text(encoding="utf-8"), name)


class TrackingIssueTests(unittest.TestCase):
    repo = "Algorythm-Canada/OpenJevSwift"

    def issue(self, number, state, data):
        return {"number": number, "state": state, "url": f"https://issue/{number}",
                "body": tracking_issue.body_of(data, self.repo)}

    def test_nothing_new_changes_nothing(self):
        data = sample_review()
        data["attention"] = []
        self.assertEqual(tracking_issue.plan(data, [], self.repo), [])

    def test_the_first_review_opens_the_issue(self):
        data = sample_review()
        [(action, number, body)] = tracking_issue.plan(data, [], self.repo, "https://run")
        self.assertEqual((action, number), ("create", None))
        self.assertTrue(body.startswith(f"<!-- upstream-review fingerprint={data['fingerprint']}"))
        self.assertIn("# Upstream review of 2026-10-02", body)
        self.assertIn("([this run](https://run))", body)
        self.assertEqual(tracking_issue.header_of(body),
                         (data["fingerprint"], review.state_of(data)))
        self.assertEqual(tracking_issue.command_for("gh", self.repo, action, number),
                         ["gh", "issue", "create", "--repo", self.repo, "--title",
                          tracking_issue.TITLE, "--label", "area/ci", "--body-file", "-"])

    def test_the_same_state_changes_nothing_open_or_closed(self):
        data = sample_review()
        for state in ("OPEN", "CLOSED"):
            self.assertEqual(tracking_issue.plan(data, [self.issue(70, state, data)], self.repo),
                             [], state)

    def test_a_new_state_updates_the_open_issue_and_says_what_moved(self):
        before = sample_review()
        after = json.loads(json.dumps(before))
        after["reviewed_at"] = "2026-11-01T06:37:00Z"
        after["projects"][1]["head"]["sha"] = "3" * 40
        after["fingerprint"] = review.fingerprint(after)
        issues = [{"number": 5, "state": "OPEN", "body": "an unrelated issue"},
                  self.issue(70, "OPEN", before)]
        actions = tracking_issue.plan(after, issues, self.repo)
        self.assertEqual([(a, n) for a, n, _ in actions], [("edit", 70), ("comment", 70)])
        comment = actions[1][2]
        self.assertTrue(comment.startswith("Updated by the review of 2026-11-01."))
        self.assertIn("- ml-explore/mlx-swift: was `pin 0.32.2; head 196012000000; release "
                      "0.32.3`, now `pin 0.32.2; head 333333333333; release 0.32.3`.", comment)
        self.assertNotIn("razorback16/openjev", comment)

    def test_a_closed_issue_is_reopened_when_upstream_moves_again(self):
        before = sample_review()
        after = json.loads(json.dumps(before))
        after["projects"][2]["head"]["sha"] = "4" * 40
        after["fingerprint"] = review.fingerprint(after)
        actions = tracking_issue.plan(after, [self.issue(70, "CLOSED", before)], self.repo)
        self.assertEqual([(a, n) for a, n, _ in actions],
                         [("reopen", 70), ("edit", 70), ("comment", 70)])
        self.assertTrue(actions[2][2].startswith("Reopened by the review of 2026-10-02, which "
                                                 "differs from the one this issue was closed "
                                                 "with."))

    def test_the_state_survives_an_error_that_could_end_the_comment(self):
        data = sample_review()
        data["projects"][3]["error"] = "a message with --> inside"
        data["fingerprint"] = review.fingerprint(data)
        body = tracking_issue.body_of(data, self.repo)
        self.assertEqual(body.split("\n")[1].count("-->"), 1)
        self.assertEqual(tracking_issue.header_of(body)[1]["org/gone"],
                         "error: a message with --> inside")

    def test_the_open_issue_wins_over_closed_ones(self):
        data = sample_review()
        issues = [self.issue(90, "CLOSED", data), self.issue(80, "OPEN", data),
                  self.issue(85, "OPEN", data)]
        self.assertEqual(tracking_issue.tracking_issue(issues)["number"], 80)
        self.assertEqual(tracking_issue.tracking_issue(issues[:1])["number"], 90)
        self.assertIsNone(tracking_issue.tracking_issue([{"number": 1, "state": "OPEN",
                                                          "body": None}]))

    def test_a_broken_run_leaves_the_issue_alone(self):
        data = sample_review()
        for record in data["projects"]:
            record["error"] = "connection reset"
        with self.assertRaises(tracking_issue.TrackingError):
            tracking_issue.plan(data, [], self.repo)

    def test_a_long_review_is_cut_to_fit(self):
        data = sample_review()
        data["projects"][1]["commits"] = [
            {"sha": f"{i:040x}", "date": "2026-09-30T00:00:00Z", "subject": "x" * 90}
            for i in range(2000)]
        body = tracking_issue.body_of(data, self.repo)
        self.assertLessEqual(len(body), tracking_issue.BODY_LIMIT)
        self.assertIn("The review was cut to fit an issue body.", body)
        self.assertEqual(tracking_issue.header_of(body)[0], data["fingerprint"])

    def test_dry_run_prints_and_runs_nothing(self):
        printed = []

        def run(*args, **kwargs):
            raise AssertionError("a dry run ran a command")

        tracking_issue.apply([("edit", 70, "line one\nline two")], "gh", self.repo, True, run,
                             printed.append)
        self.assertEqual(printed[0], f"would run: gh issue edit 70 --repo {self.repo} "
                                     "--body-file -")
        self.assertIn("  | line two", printed)
        self.assertEqual(tracking_issue.command_for("gh", self.repo, "create", None)[5:7],
                         ["--title", tracking_issue.TITLE])
        printed.clear()
        tracking_issue.apply([("create", None, "body")], "gh", self.repo, True, run, printed.append)
        self.assertIn("--title 'Upstream changes since the pinned revisions'", printed[0])
        self.assertEqual(tracking_issue.describe([("create", None, "")], dry_run=True),
                         "Dry run: would open the tracking issue.")
        self.assertEqual(tracking_issue.describe([("reopen", 70, ""), ("edit", 70, "")]),
                         "Done: reopened and updated #70.")
        self.assertEqual(tracking_issue.describe([]), "Nothing to change on the tracking issue.")

    def test_apply_sends_each_body_on_standard_input(self):
        sent = []

        def run(command, input=None, **kwargs):
            sent.append((command, input))
            return subprocess.CompletedProcess(command, 0, stdout="https://issue/71\n", stderr="")

        tracking_issue.apply([("reopen", 70, ""), ("comment", 70, "hello")], "gh", self.repo,
                             False, run, lambda line: None)
        self.assertEqual(sent, [(["gh", "issue", "reopen", "70", "--repo", self.repo], ""),
                                (["gh", "issue", "comment", "70", "--repo", self.repo,
                                  "--body-file", "-"], "hello")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
