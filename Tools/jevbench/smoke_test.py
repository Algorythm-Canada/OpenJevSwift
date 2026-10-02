#!/usr/bin/env python3
"""Smoke test of the JevBench harness: no model, no server to start and no download.

    python3 Tools/jevbench/smoke_test.py

It runs testdata/items.jsonl, seven authored items in JevBench's format, against a fake
/v1/systemone server inside the process, and checks the request each item becomes, the skip rule,
JevBench's accuracy, Brier score and ECE on answers whose values are worked out by hand below, the
compare command, SemIf's TypeSafe metrics on synthetic rows, the published-row comparison and the
vendored files' pins. Standard library only; CI runs it in the macOS job.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402

ITEMS = HERE / "testdata" / "items.jsonl"
MODEL = "verdict-1.4"

# The fake servers' answers, keyed by the question's instructions. Every probability is a dyadic
# fraction, so the expected scores below are exact in binary floating point.
ANSWERS_A = {
    "Which intent does the message express?":
        {"type": "choice", "choice": "beta", "confidence": 0.5,
         "probabilities": {"alpha": 0.25, "beta": 0.625, "gamma": 0.125}},
    "The invoice has been paid.": {"type": "noul", "noul": 0.75},
    "The invoice has been settled.": {"type": "noul", "noul": 0.625},
    "How severe is the incident?":
        {"type": "score", "score": 1.5, "confidence": 0.3,
         "legend": {"0": "Cosmetic", "1": "Some users blocked", "2": "Data loss"},
         "probabilities": {"0": 0.125, "1": 0.25, "2": 0.625}},
    "Which key comes first?":
        {"type": "choice", "choice": "x", "confidence": 0.0,
         "probabilities": {"x": 0.5, "y": 0.5}},
    "The server refuses this question.": (400, {"detail": "This question cannot be asked."}),
}
ANSWERS_B = {
    **ANSWERS_A,
    "Which intent does the message express?":
        {"type": "choice", "choice": "alpha", "confidence": 0.2,
         "probabilities": {"alpha": 0.5, "beta": 0.375, "gamma": 0.125}},
    "The invoice has been settled.": {"type": "noul", "noul": 0.375},
    "How severe is the incident?":
        {"type": "score", "score": 1.375, "confidence": 0.2,
         "legend": {"0": "Cosmetic", "1": "Some users blocked", "2": "Data loss"},
         "probabilities": {"0": 0.125, "1": 0.375, "2": 0.5}},
    "The server refuses this question.": {"type": "noul", "noul": 0.5},
}


def synthetic_typesafe() -> tuple:
    """Three rows in the shape SemIf's builder writes, in two cases, and their questions."""
    noul = {"id": "r1", "family": "typesafe_x", "group_id": "case-a", "primitive": "noul",
            "options": [{"id": "true", "description": "true: yes"},
                        {"id": "false", "description": "false: no"}],
            "label": 0, "target_distribution": [0.75, 0.25],
            "published_models": {"typesafe": {"model": "typesafe:x",
                                              "distribution": [0.75, 0.25]}}}
    choice = {"id": "r2", "family": "typesafe_x", "group_id": "case-a", "primitive": "choice",
              "options": [{"id": key, "description": f"{key}: {key}"} for key in "abc"],
              "label": 1, "target_distribution": [0.25, 0.5, 0.25],
              "published_models": {"typesafe": {"model": "typesafe:x",
                                                "distribution": [0.25, 0.5, 0.25]}}}
    other = {**noul, "id": "r3", "group_id": "case-b", "label": 1,
             "target_distribution": [0.25, 0.75],
             "published_models": {"typesafe": {"model": "typesafe:x",
                                               "distribution": [0.25, 0.75]}}}
    noul_question = {"type": "noul", "instructions": "It holds.", "criteria": None}
    choice_question = {"type": "choice", "instructions": "Which?",
                       "criteria": {"a": "A", "b": "B", "c": "C"}}
    return [noul, choice, other], {"r1": noul_question, "r2": choice_question,
                                   "r3": noul_question}


class FakeServer:
    """A /v1/systemone server on an ephemeral port that answers from a table and records every
    request body it receives, byte for byte."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.bodies = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.send_header("x-request-id", "req_" + hashlib.sha256(data).hexdigest()[:32])
                self.send_header("server-timing", "model;dur=1.5, server;dur=0.25, total;dur=1.75")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply(200, {"models": [{"name": MODEL, "description": "fake",
                                                 "release_date": "2026-10-01"}]})
                else:
                    self.reply(200, {"status": "ok"})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["content-length"]))
                server.bodies.append(raw)
                question = json.loads(raw)["questions"]["decision"]
                answer = server.answers[question["instructions"]]
                if isinstance(answer, tuple):
                    self.reply(*answer)
                else:
                    self.reply(200, {"model": MODEL, "answers": {"decision": answer},
                                     "usage": {"input_tokens": 10, "output_tokens": 0}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def run(answers: dict, server: str, cache: Path) -> tuple:
    dataset = harness.load_item_file(ITEMS)
    with FakeServer(answers) as fake:
        doc = harness.run_dataset(dataset, fake.url, MODEL, server, cache, progress=lambda _: None)
    return doc, fake.bodies


class HarnessSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="jevbench-smoke-"))
        cls.doc_a, cls.bodies_a = run(ANSWERS_A, "fake-a", cls.tmp)
        cls.doc_b, _ = run(ANSWERS_B, "fake-b", cls.tmp)
        cls.items = {item["id"]: item for item in cls.doc_a["items"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_item_becomes_jevbench_typesafe_request(self):
        dataset = harness.load_item_file(ITEMS)
        sent = [task for task in dataset.tasks if task.id != "t-choice-wide"]
        self.assertEqual(len(self.bodies_a), len(sent))
        for task, raw in zip(sent, self.bodies_a):
            question = {"type": task.question["type"],
                        "instructions": task.question["instructions"]}
            if task.question.get("criteria") is not None:
                question["criteria"] = task.question["criteria"]
            expected = {"state": task.state, "model": MODEL, "questions": {"decision": question}}
            # the bytes JevBench's own http_post_json sends: json.dumps with its defaults
            self.assertEqual(raw, json.dumps(expected).encode("utf-8"), task.id)
            body = json.loads(raw)
            self.assertEqual(list(body), ["state", "model", "questions"])
            if isinstance(task.question.get("criteria"), dict):
                self.assertEqual(list(body["questions"]["decision"]["criteria"]),
                                 list(task.question["criteria"]), "option order is kept")
            record = self.items[task.id]["request"]
            self.assertEqual(record["body_sha256"], hashlib.sha256(raw).hexdigest())
            state = task.state if isinstance(task.state, str) else json.dumps(
                task.state, ensure_ascii=False)
            self.assertEqual(record["state"]["sha256"],
                             hashlib.sha256(state.encode("utf-8")).hexdigest())
        settled = json.loads(self.bodies_a[2])["questions"]["decision"]
        self.assertNotIn("criteria", settled, "a noul without criteria sends none")
        state = json.loads(self.bodies_a[4])["state"]
        self.assertEqual(list(state), ["b", "a", "note"], "an object state keeps its key order")

    def test_noul_answer_maps_to_yes_and_no(self):
        self.assertEqual(self.items["t-noul-1"]["probabilities"], {"yes": 0.75, "no": 0.25})
        self.assertEqual(self.items["t-noul-1"]["predicted"], "yes")

    def test_shape_no_backend_accepts_is_skipped_and_counted(self):
        wide = self.items["t-choice-wide"]
        self.assertEqual(wide["status"], "skipped")
        self.assertIn("25 options", wide["skip_reason"])
        self.assertNotIn("request", wide, "a skipped item is never sent")
        task = harness.load_item_file(ITEMS).by_id["t-choice-wide"]
        self.assertIsNone(harness.shape_skip_reason(task, "laya-1.0"), "Laya takes 255 options")
        counts = self.doc_a["summary"]["counts"]
        self.assertEqual((counts["items"], counts["answered"], counts["skipped"],
                          counts["refused"]), (7, 5, 1, 1))
        overall = self.doc_a["summary"]["overall"]
        self.assertEqual((overall["n_planned"], overall["n_attempted"]), (7, 6))
        self.assertAlmostEqual(overall["coverage"], 6 / 7)

    def test_refusal_is_recorded_and_counts_as_wrong(self):
        refused = self.items["t-refused"]
        self.assertEqual((refused["status"], refused["http_status"]), ("refused", 400))
        self.assertFalse(refused["correct"])
        self.assertEqual(refused["response"], {"detail": "This question cannot be asked."})

    def test_scoring_arithmetic(self):
        overall = self.doc_a["summary"]["overall"]
        # correct: t-choice-1, t-noul-1, t-score-1, t-dict-state (a tie: JevBench takes the
        # smallest label); wrong: t-noul-2 and the refusal; t-choice-wide is not attempted
        self.assertEqual(overall["n_correct"], 4)
        self.assertAlmostEqual(overall["accuracy"], 4 / 6)
        self.assertEqual(self.items["t-dict-state"]["predicted"], "x")
        # Brier, the multi-class sum over the five answered items (the refusal has no distribution):
        # 0.21875 + 0.125 + 0.78125 + 0.21875 + 0.5
        self.assertEqual(overall["calibration_n"], 5)
        self.assertAlmostEqual(overall["brier_mean"], 1.84375 / 5)
        # ECE over 10 equal-width bins of top-label confidence: bin 5 (0.5, right), bin 6 (three
        # at 0.625, two right), bin 7 (0.75, right)
        self.assertAlmostEqual(overall["ece"]["ece"],
                               0.5 / 5 + 3 / 5 * abs(2 / 3 - 0.625) + 0.25 / 5)
        self.assertAlmostEqual(overall["ordinal_mae"], 0.5)  # expected level 1.5 against 2
        pairs = overall["paraphrase_consistency"]
        self.assertEqual((pairs["pairs"], pairs["both_valid"], pairs["agree"]), (1, 1, 1))
        tiers = self.doc_a["summary"]["per_tier"]
        self.assertAlmostEqual(tiers["easy"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(tiers["hard"]["accuracy"], 2 / 3)
        self.assertEqual(self.doc_a["summary"]["model_time"], {"n": 6, "mean_ms": 1.5,
                                                               "p50_ms": 1.5})

    def test_renormalization_band(self):
        task = harness.load_item_file(ITEMS).by_id["t-noul-1"]
        rounded = harness.jb_scoring.score_task({"no": 0.25, "yes": 0.74}, task)
        self.assertTrue(rounded["valid"] and rounded["renormalized"])
        self.assertAlmostEqual(rounded["probs"]["yes"], 0.74 / 0.99)
        broken = harness.jb_scoring.score_task({"no": 0.25, "yes": 0.7}, task)
        self.assertFalse(broken["valid"])

    def test_result_file_round_trip_and_rescoring(self):
        path = self.tmp / "out" / "result.json"
        harness.write_result(path, self.doc_a)
        self.assertTrue(path.read_bytes().isascii(), "a result file is ASCII")
        self.assertNotIn(str(Path.home()), path.read_text(), "nor names the home folder")
        doc = harness.read_result(path)
        doc.pop("_path")
        self.assertEqual(doc, json.loads(json.dumps(self.doc_a)))
        self.assertEqual(harness.summarize_doc(doc), json.loads(json.dumps(self.doc_a["summary"])))
        self.assertEqual(self.items["t-noul-1"]["timing"]["server"],
                         {"model": 1.5, "server": 0.25, "total": 1.75})
        self.assertTrue(self.items["t-noul-1"]["request_id"].startswith("req_"))

    def test_compare(self):
        result = harness.compare_docs(self.doc_a, self.doc_b)
        overall = result["overall"]
        self.assertEqual((overall["items"], overall["agree"], overall["identical"]), (5, 3, 2))
        self.assertAlmostEqual(overall["mean_abs_diff"], 1.25 / 12)
        self.assertEqual(overall["max_abs_diff"], 0.25)
        choice, noul, score = (result["per_type"][kind] for kind in ("choice", "noul", "score"))
        self.assertEqual((choice["items"], choice["agree"], choice["max_item"], choice["max_label"]),
                         (2, 1, "t-choice-1", "beta"))
        self.assertAlmostEqual(choice["mean_abs_diff"], 0.5 / 5)
        self.assertEqual((noul["items"], noul["agree"], noul["max_item"]), (2, 1, "t-noul-2"))
        self.assertAlmostEqual(noul["mean_abs_diff"], 0.5 / 4)
        self.assertEqual((score["max_abs_diff"], score["max_label"]), (0.125, "2"))
        self.assertAlmostEqual(score["mean_abs_diff"], 0.25 / 3)
        self.assertEqual([(d["id"], d["a"], d["b"], d["b_margin"]) for d in result["disagreements"]],
                         [("t-choice-1", "beta", "alpha", 0.125), ("t-noul-2", "yes", "no", 0.25)])
        self.assertEqual(result["correctness"], {"both_correct": 3, "both_wrong": 0, "a_only": 1,
                                                 "b_only": 1, "mcnemar_p": 1.0})
        self.assertEqual(result["answered_by_one_only"],
                         [{"id": "t-refused", "a": "refused", "b": "answered"}])
        self.assertEqual(result["largest"][0]["diff"], 0.25)
        # t-dict-state's reference answer is an exact tie, inside the 0.01 margin of the bound
        self.assertEqual([d["id"] for d in result["near_ties"]], ["t-dict-state"])
        with self.assertRaises(ValueError):
            harness.compare_docs(self.doc_a, {**self.doc_b, "model": "laya-1.0"})
        # an item only the reference run holds is listed too
        extra = {**self.items["t-noul-1"], "id": "t-extra"}
        wider = {**self.doc_b, "items": self.doc_b["items"] + [extra]}
        self.assertIn({"id": "t-extra", "a": "absent", "b": "answered"},
                      harness.compare_docs(self.doc_a, wider)["answered_by_one_only"])

    def test_command_line(self):
        a, b = self.tmp / "cli" / "a.json", self.tmp / "cli" / "b.json"
        with FakeServer(ANSWERS_A) as fake, contextlib.redirect_stdout(io.StringIO()):
            status = harness.main(["--cache", str(self.tmp), "run", "--items", str(ITEMS),
                                   "--base-url", fake.url, "--model", MODEL, "--server", "fake-a",
                                   "--output", str(a)])
        self.assertEqual(status, 0)
        harness.write_result(b, self.doc_b)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            harness.main(["compare", str(a), str(b), "--json"])
        self.assertEqual(json.loads(printed.getvalue())["overall"]["agree"], 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(harness.main(["--cache", str(self.tmp), "summary", str(a)]), 0)

    def test_published_row_comparison(self):
        doc = json.loads(json.dumps(self.doc_a))
        doc["dataset"]["name"] = "jevbench"
        doc["published_row"] = {"key": "row", "display": "Row", "setup": "elsewhere"}
        for item, code in zip(doc["items"], "ccwcccw"):
            item["published"] = {"outcome": code, "latency_s": 0.1}
        result = harness.against_published(doc, self.tmp)
        # skipped t-choice-wide is left out; ours: c c w c . c w(refused), published: c c w c . c w
        self.assertEqual(result["items"], 6)
        self.assertEqual(result["outcomes"]["both_correct"], 4)
        self.assertEqual(result["outcomes"]["both_wrong"], 2)
        self.assertEqual(result["outcomes"]["agreement"], 1.0)
        self.assertIsNone(result["board"])

    def test_typesafe_run_keeps_no_typesafe_text_or_reference(self):
        rows, questions = synthetic_typesafe()
        tasks = [harness.typesafe_task(row, questions[row["id"]], {"doc": row["id"]})
                 for row in rows]
        dataset = harness.Dataset("typesafe102", tasks, {t.id: "typesafe102" for t in tasks},
                                  {"name": "typesafe102"}, {row["id"]: row for row in rows})
        echo = {"detail": [{"type": "string_type", "loc": ["body", "state"], "msg": "bad",
                            "input": "TYPESAFE DOCUMENT TEXT"}]}
        # a 422 that echoes the request, and a choice rounded to four places that sums to 1.0005,
        # which JevBench takes as it is and SemIf's evaluator would refuse
        answers = {"It holds.": (422, echo),
                   "Which?": {"type": "choice", "choice": "a", "confidence": 0.0,
                              "probabilities": {"a": 0.3335, "b": 0.3335, "c": 0.3335}}}
        with FakeServer(answers) as fake:
            doc = harness.run_dataset(dataset, fake.url, MODEL, "fake", self.tmp,
                                      progress=lambda _: None)
        refused = [item for item in doc["items"] if item["status"] == "refused"]
        self.assertEqual([item["error"] for item in refused], ["HTTP 422", "HTTP 422"])
        self.assertTrue(all("correct" not in item for item in doc["items"]),
                        "correctness would give the reference answer away")
        path = self.tmp / "typesafe" / "result.json"
        harness.write_result(path, doc)
        self.assertNotIn("TYPESAFE DOCUMENT TEXT", path.read_text())
        semif = doc["summary"]["semif"]
        self.assertEqual((semif["rows"], semif["equal_case_modal_agreement"]), (1, 0.0))
        scored = harness.with_correctness(harness.read_result(path), dataset)
        self.assertEqual([item.get("correct") for item in scored["items"]], [False, False, False])

    def test_typesafe_rows_and_semif_metrics(self):
        rows, questions = synthetic_typesafe()
        noul, choice, other = rows
        noul_question, choice_question = questions["r1"], questions["r2"]
        tasks = [harness.typesafe_task(noul, noul_question, {"doc": 1}),
                 harness.typesafe_task(choice, choice_question, {"doc": 2}),
                 harness.typesafe_task(other, noul_question, {"doc": 3})]
        self.assertEqual([(t.labels, t.expected) for t in tasks],
                         [(["no", "yes"], "yes"), (["a", "b", "c"], "b"), (["no", "yes"], "no")])
        self.assertNotIn("criteria", tasks[0].question)
        with self.assertRaises(ValueError):
            harness.typesafe_task(choice, {**choice_question, "criteria": {"b": "B", "a": "A",
                                                                          "c": "C"}}, {})
        dataset = harness.Dataset("typesafe102", tasks, {t.id: "typesafe102" for t in tasks},
                                  {"name": "typesafe102"},
                                  {row["id"]: row for row in (noul, choice, other)})
        answers = {"r1": {"yes": 0.75, "no": 0.25}, "r2": {"a": 0.5, "b": 0.25, "c": 0.25},
                   "r3": {"yes": 0.5, "no": 0.5}}
        doc = {"dataset": {"name": "typesafe102"}, "model": MODEL, "items": []}
        for task in tasks:
            scored = harness.jb_scoring.score_task(answers[task.id], task)
            doc["items"].append({"id": task.id, "tier": "typesafe102", "family": task.family,
                                 "type": task.question["type"], "labels": task.labels,
                                 "status": "answered", "probabilities": answers[task.id],
                                 "valid": True, "predicted": scored["predicted"],
                                 "correct": scored["correct"], "timing": {"wall_ms": 1.0}})
        summary = harness.summarize_doc(doc, dataset)
        semif = summary["semif"]
        # case a: r1 agrees (TV 0), r2 does not (TV 0.25); case b: r3 ties and SemIf takes the
        # first option, true, against the reference false (TV 0.25)
        self.assertEqual((semif["rows"], semif["cases"]), (3, 2))
        self.assertAlmostEqual(semif["equal_case_modal_agreement"], (0.5 + 0.0) / 2)
        self.assertAlmostEqual(semif["equal_case_total_variation"], (0.125 + 0.25) / 2)
        self.assertEqual(semif["published"]["typesafe"]["equal_case_modal_agreement"], 1.0)
        # JevBench breaks r3's tie towards the smaller label, no, which is right
        self.assertAlmostEqual(summary["overall"]["accuracy"], 2 / 3)

    def test_vendored_files_match_their_pins(self):
        found = harness.check_vendored()
        self.assertEqual(set(found), set(harness.VENDORED))
        copy = self.tmp / "vendor"
        shutil.copytree(harness.VENDOR, copy)
        with open(copy / "jevbench" / "jevbench" / "scoring.py", "a") as handle:
            handle.write("\n")
        saved, harness.VENDOR = harness.VENDOR, copy
        try:
            with self.assertRaises(harness.PinError):
                harness.check_vendored()
            # a run checks before its first request, so a changed copy sends nothing
            with FakeServer(ANSWERS_A) as fake, self.assertRaises(harness.PinError):
                harness.run_dataset(harness.load_item_file(ITEMS), fake.url, MODEL, "fake",
                                    self.tmp, progress=lambda _: None)
            self.assertEqual(fake.bodies, [])
        finally:
            harness.VENDOR = saved

    def test_pins_match_the_rest_of_the_repository(self):
        import servers

        common = (harness.ROOT / "Tools" / "encoders" / "common.py").read_text()
        for backend, prefix in (("verdict", "VERDICT"), ("laya", "LAYA")):
            repo, revision = servers.CHECKPOINTS[backend][:2]
            self.assertIn(f'\n{prefix}_REPO = "{repo}"\n', common)
            self.assertIn(f'\n{prefix}_REVISION = "{revision}"\n', common)
        self.assertIn(f'\nUPSTREAM_COMMIT = "{servers.pinned_upstream_commit()}"\n', common)
        third_party = (harness.ROOT / "THIRD_PARTY.md").read_text()
        for commit in (harness.JEVBENCH_COMMIT, harness.SEMIF_COMMIT):
            self.assertIn(f"`{commit[:7]}`", third_party)

    def test_cached_files_must_be_the_pinned_ones(self):
        cached = self.tmp / "cached.jsonl"
        cached.write_bytes(b"{}\n")
        digest = hashlib.sha256(b"{}\n").hexdigest()
        self.assertEqual(harness.require_pinned(cached, 3, digest), cached)
        with self.assertRaises(harness.PinError):
            harness.require_pinned(cached, 3, "0" * 64)
        with self.assertRaises(harness.PinError):  # an empty cache is refused, not read
            harness.load_typesafe(self.tmp / "empty-cache", fetch=False)

    def test_a_package_that_fails_its_check_stops_the_swift_server(self):
        import subprocess
        import servers

        def fake_run(command, *args, **kwargs):
            failing = any("manifest.py" in str(part) for part in command)
            return subprocess.CompletedProcess(command, 1 if failing else 0,
                                               stdout="verdict-m18-fp16 differs\n", stderr="")

        saved, servers.subprocess.run = servers.subprocess.run, fake_run
        try:
            with self.assertRaises(SystemExit) as stopped:
                servers.swift_server("verdict", Path(sys.executable), str(self.tmp))
        finally:
            servers.subprocess.run = saved
        self.assertIn("is not the published package", str(stopped.exception))

    def test_function_loads_follow_the_capacity_each_run_records(self):
        import report
        import servers

        def run(dataset, capacity=None):
            server = {"name": "swift"}
            if capacity is not None:
                server["function_capacity"] = capacity
            # Four lengths in turn, twice: 128, 1,024, 512 and 256 tokens, one shape each.
            items = [{"status": "answered", "usage": {"input_tokens": tokens},
                      "timing": {"server": {"model": 10.0}}}
                     for tokens in [100, 1000, 300, 200] * 2]
            return {"model": "laya-1.0", "dataset": {"name": dataset}, "server": server,
                    "items": items}

        # Recorded without a capacity, as before D-042: two functions, so after the four first
        # loads every request loads its function again. Every function: none again.
        two = report.function_load_rows([run("jevbench"), run("typesafe102")])
        self.assertEqual(two[0][:4], ["laya-1.0", "16", "4", "12"])
        every = report.function_load_rows([run("jevbench", "all"), run("typesafe102", "all")])
        self.assertEqual(every[0][:4], ["laya-1.0", "16", "4", "0"])
        self.assertEqual(every[0][5], "128: 10, 256: 10, 512: 10, 1024: 10")
        # A JevBench run from a newer server with an older TypeSafe file cannot share one cache.
        with self.assertRaises(SystemExit) as stopped:
            report.function_load_rows([run("jevbench", "all"), run("typesafe102")])
        self.assertIn("different function capacities", str(stopped.exception))

        # servers.py reads the capacity from the Swift server's settings line.
        for line, kept in [("encoder_batch=16 encoder_functions=all encoder_models=x", "all"),
                           ("encoder_batch=16 encoder_functions=3 encoder_models=x", 3),
                           ("encoder_batch=16 encoder_models=x", 2)]:
            log = self.tmp / "server.log"
            log.write_text(f"info openjev: [openjev] settings: backend=laya {line}\n")
            self.assertEqual(servers.swift_function_capacity(log), kept)

    def test_compare_reports_d014_subsets(self):
        result = harness.compare_docs(self.doc_a, self.doc_b)
        # the reference's top two are at least 0.5 apart only on t-noul-1 (0.75 against 0.25),
        # where both runs answer yes; every fake prompt is 10 tokens, so none is long
        self.assertEqual(result["bounds"], {"confident_items": 1, "confident_agree": 1,
                                            "long_items": 0, "long_mean_abs_diff": None})
        long_a = json.loads(json.dumps(self.doc_a))
        long_b = json.loads(json.dumps(self.doc_b))
        for doc in (long_a, long_b):
            for item in doc["items"]:
                if item["id"] == "t-noul-2":
                    item["usage"] = {"input_tokens": 2000, "output_tokens": 0}
        bounds = harness.compare_docs(long_a, long_b)["bounds"]
        # t-noul-2: yes 0.625 against 0.375 and no 0.375 against 0.625, over two labels
        self.assertEqual((bounds["long_items"], bounds["long_mean_abs_diff"]), (1, 0.25))

    def test_report_handles_diffusiongemma_runs(self):
        import report

        def run_doc(server, backend, outcomes, base=None):
            doc = json.loads(json.dumps(base or self.doc_a))
            doc["model"] = "openjev-0.1"
            doc["dataset"]["name"] = "jevbench"
            doc["server"] = {"name": server, "backend": backend,
                             "implementation": ("OpenJevSwift" if server == "swift"
                                                else "razorback16/openjev"),
                             "version": "0.5.0", "commit": "dcd20947", "python": "3.12.2",
                             "packages": {"mlx": "0.32.2", "mlx-vlm": "0.6.15"},
                             "device": "mlx, the GPU", "dtype": "the checkpoint's",
                             "runtime": "MLX on the GPU (D-039)"}
            doc["published_row"] = {"key": "row", "display": "Row", "setup": "elsewhere"}
            for item, code in zip(doc["items"], outcomes):
                item["published"] = {"outcome": code, "latency_s": 0.1}
            doc["summary"] = harness.summarize_doc(doc)
            return doc

        # the published outcomes differ from the Swift run's on t-noul-1 and from upstream's on
        # t-choice-1 (both runs here answer as doc_a: c c w c . c w)
        swift = run_doc("swift", "mlx", "cwwcccw")
        upstream = run_doc("upstream", "mlx", "wcwcccw")
        # the runs table leaves out the timings of a model that shared the GPU
        rows = report.runs_rows([swift])
        self.assertEqual(rows[0][-3:], ["not reported"] * 3)
        self.assertNotEqual(report.runs_rows([self.doc_a])[0][-1], "not reported")
        # upstream's MLX server is not described as PyTorch
        environment = report.environment_rows([upstream])
        self.assertEqual(environment[0][4], "MLX 0.32.2 and mlx-vlm 0.6.15 on the GPU, the "
                                            "checkpoint's weights")
        # a setting beyond the defaults is shown with the runtime
        capped = json.loads(json.dumps(swift))
        capped["server"]["settings"] = {"OPENJEV_BACKEND": "mlx",
                                        "OPENJEV_MLX_CACHE_LIMIT_GB": "4"}
        self.assertEqual(report.environment_rows([capped])[0][4],
                         "MLX on the GPU (D-039), OPENJEV_MLX_CACHE_LIMIT_GB=4")
        # a Swift list of differing outcomes that is not upstream's is shown, and said to differ
        tables = report.published_tables([swift, upstream], self.tmp)
        self.assertIn("openjev-0.1 differs", tables[2])
        self.assertIn("| openjev-0.1 | swift | noul | 1 | t-noul-1 |", tables[3])
        self.assertIn("| openjev-0.1 | upstream | choice | 1 | t-choice-1 |", tables[3])
        same = report.published_tables([swift, run_doc("upstream", "mlx", "cwwcccw")], self.tmp)
        self.assertIn("the Swift run's list is the same", same[2])
        self.assertNotIn("| swift |", same[3])
        # with one server's run only, nothing is claimed about the other
        alone = report.published_tables([upstream], self.tmp)
        self.assertIn("openjev-0.1 only upstream's run", alone[2])
        self.assertNotIn("is the same", alone[2])
        alone = report.published_tables([swift], self.tmp)
        self.assertIn("openjev-0.1 only swift's run", alone[2])
        self.assertIn("| openjev-0.1 | swift | noul | 1 | t-noul-1 |", alone[3])
        # DiffusionGemma gets D-014's table instead of the encoders' near-tie rule: the same
        # answers are within every bound, doc_b's against doc_a's are not
        compared = harness.compare_docs(swift, upstream)
        self.assertTrue(report.within_d014(compared["overall"], compared["bounds"]))
        upstream = run_doc("upstream", "mlx", "wcwcccw", base=self.doc_b)
        compared = harness.compare_docs(swift, upstream)
        self.assertFalse(report.within_d014(compared["overall"], compared["bounds"]))
        tables = report.agreement_tables([swift, upstream])
        self.assertIn("D-014", tables[-2])
        self.assertIn("| openjev-0.1 | jevbench | 5 | 0.1042 | n/a over 0 | 3 of 5 (60.0%) | "
                      "1 of 1 (100.0%) | no |", tables[-1])
        self.assertNotIn("openjev-0.1", tables[-3], "no near-tie row for DiffusionGemma")
        # a pair with no item both runs answered renders, and meets no bound
        for doc in (swift, upstream):
            for item in doc["items"]:
                item["status"] = "failed"
        compared = harness.compare_docs(swift, upstream)
        self.assertFalse(report.within_d014(compared["overall"], compared["bounds"]))
        self.assertIn("| openjev-0.1 | jevbench | 0 | n/a | n/a over 0 | 0 of 0 (n/a) |",
                      report.agreement_tables([swift, upstream])[-1])

    def test_upstream_confidence(self):
        import calibration

        self.assertEqual(calibration.upstream_confidence([0.5, 0.5]), 0.0)
        self.assertEqual(calibration.upstream_confidence([1.0, 0.0]), 1.0)
        self.assertEqual(calibration.upstream_confidence([1.0]), 1.0)
        entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        self.assertAlmostEqual(calibration.upstream_confidence([0.75, 0.25]),
                               1 - entropy / math.log(2))
        self.assertAlmostEqual(calibration.upstream_confidence([1 / 3] * 3), 0.0)
        # Every choice and score answer the committed DiffusionGemma runs hold carries the value
        # the formula gives for its probabilities, on both servers, so a noul's computed value is
        # the one upstream would have sent.
        for server in ("swift", "upstream"):
            path = harness.RESULTS / f"openjev-0.1-{server}.json"
            if not path.exists():
                continue
            answers = [item["answer"] for item in harness.read_result(path)["items"]
                       if item["status"] == "answered" and item["type"] in ("choice", "score")]
            self.assertGreater(len(answers), 100)
            for answer in answers:
                self.assertAlmostEqual(
                    calibration.upstream_confidence(list(answer["probabilities"].values())),
                    answer["confidence"], places=12)

    def test_calibration_rows_and_metrics(self):
        import calibration

        rows = calibration.rows_of(self.doc_a)
        # the refusal has no distribution and the wide choice was skipped
        self.assertEqual([row["id"] for row in rows],
                         ["t-choice-1", "t-noul-1", "t-noul-2", "t-score-1", "t-dict-state"])
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["t-noul-1"]["group"], "g1")
        self.assertEqual(by_id["t-choice-1"]["group"], "t-choice-1")
        self.assertEqual((by_id["t-choice-1"]["confidence"],
                          by_id["t-choice-1"]["confidence_source"]), (0.5, "answer"))
        self.assertEqual(by_id["t-noul-1"]["confidence_source"], "computed")
        self.assertTrue(by_id["t-dict-state"]["correct"], "a tie goes to the smallest label")
        found = calibration.metrics(rows)
        overall = self.doc_a["summary"]["overall"]
        # JevBench's own ECE and Brier score over the same five distributions
        self.assertAlmostEqual(found["ece"], overall["ece"]["ece"])
        self.assertAlmostEqual(found["brier"], overall["brier_mean"])
        self.assertAlmostEqual(found["accuracy"], 4 / 5)
        self.assertAlmostEqual(found["nll"], -(2 * math.log(0.625) + math.log(0.75)
                                               + math.log(0.375) + math.log(0.5)) / 5)
        # right answers' tops 0.625, 0.75, 0.625, 0.5 against the wrong one's 0.625
        self.assertAlmostEqual(found["auroc_top"], (0.5 + 1 + 0.5 + 0) / 4)
        # confidence: 0.5, 0.19, 0.3 and 0.0 right against 0.05 wrong (the computed noul's)
        self.assertAlmostEqual(found["auroc_confidence"], 3 / 4)
        bins = calibration.ece(rows)["bins"]
        self.assertEqual([entry["n"] for entry in bins], [0, 0, 0, 0, 0, 1, 3, 1, 0, 0])
        self.assertAlmostEqual(bins[6]["accuracy"], 2 / 3)

    def test_temperature_scaling(self):
        import calibration

        probs = {"no": 0.2, "yes": 0.8}
        self.assertEqual(calibration.tempered(probs, 1.0), probs)
        halved = calibration.tempered(probs, 2.0)  # p^(1/2), renormalised
        self.assertAlmostEqual(halved["yes"], 2 / 3)
        self.assertEqual(calibration.tempered({"a": 0.0, "b": 1.0}, 0.5), {"a": 0.0, "b": 1.0})
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calibration.tempered(probs, value)
        # Three answers of yes at 0.8, two right: the NLL is least where q(yes) = 2/3, so at
        # T = logit(0.8) / logit(2/3) = ln 4 / ln 2 = 2.
        rows = [{"id": f"r{i}", "group": f"g{i}", "type": "noul", "probs": probs,
                 "expected": expected, "predicted": "yes", "correct": expected == "yes",
                 "top": 0.8, "confidence": 0.0} for i, expected in enumerate(["yes", "yes", "no"])]
        self.assertAlmostEqual(calibration.fit_temperature(rows), 2.0, places=6)
        scaled = calibration.at_temperature(rows[0], 2.0)
        self.assertAlmostEqual(scaled["top"], 2 / 3)
        self.assertEqual(harness.jb_scoring.argmax_label(scaled["probs"]), "yes")
        # every row is held out once, by the T fitted without its group
        held, temperatures = calibration.out_of_fold(rows)
        self.assertEqual([row["id"] for row in held], ["r0", "r1", "r2"])
        self.assertEqual(len(temperatures), 3, "three groups fill three of the five folds")
        # r2 is the only wrong answer: without it the fit sharpens to the lower bound (within
        # 1e-5: so close to it the NLLs differ by less than a float's resolution)
        self.assertAlmostEqual(held[2]["temperature"], calibration.BOUNDS[0], places=5)

    def test_folds_and_bootstrap(self):
        import calibration

        rows = [{"id": f"r{i}", "group": f"g{i // 2}", "correct": i % 3 == 0, "top": 0.5 + i / 40}
                for i in range(20)]
        folds = calibration.fold_map(rows)
        self.assertEqual(folds, calibration.fold_map(rows), "the assignment is deterministic")
        self.assertEqual(sorted(folds), [f"g{i}" for i in range(10)])
        self.assertEqual(sorted(folds.values()), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        low, high = calibration.bootstrap(rows, calibration.ece_value)
        self.assertLessEqual(low, high)
        self.assertEqual(calibration.bootstrap(rows, calibration.ece_value), [low, high])
        self.assertEqual(calibration.bootstrap(rows, lambda draw: 1.0), [1.0, 1.0])
        with self.assertRaises(ValueError):
            calibration.out_of_fold([{**rows[0], "probs": {"a": 1.0}, "expected": "a"}])

    def test_calibration_tables_from_result_files(self):
        import calibration

        results = self.tmp / "calibration-results"
        for doc, server in ((self.doc_a, "swift"), (self.doc_b, "upstream")):
            copy = json.loads(json.dumps(doc))
            copy["dataset"]["name"] = "jevbench"
            copy["server"]["name"] = server
            harness.write_result(results / f"verdict-1.4-{server}.json", copy)
        text = calibration.render(results, self.tmp, model="verdict-1.4")
        self.assertIn("| jevbench | swift | 5 | 80.0% |", text)
        # the fake items have two tiers: easy (three, two right) and hard (two, both right)
        self.assertIn("| jevbench | easy | swift | 3 | 66.7% |", text)
        self.assertIn("| jevbench | hard | swift | 2 | 100.0% |", text)
        self.assertIn("none: every answer is right |", text)
        self.assertIn("| 0.6 to 0.7 | 3 | 0.625 | 66.7% |", text)
        self.assertIn("### Temperature scaling fitted offline", text)
        self.assertEqual(calibration.render(results, self.tmp, model="laya-1.0"),
                         f"no result files for laya-1.0 under {results}\n")
        # a deployment's own items, where `run --items` writes them by default, are fitted too
        own = json.loads(json.dumps(self.doc_a))
        own["model"], own["server"]["name"] = "laya-1.0", "mine"
        path = results / harness.default_output("items", "laya-1.0", "mine").relative_to(
            harness.RESULTS)
        self.assertEqual(path.parent.name, "items")
        harness.write_result(path, own)
        custom = calibration.render(results, self.tmp, model="laya-1.0")
        self.assertIn("| items | mine | 5 | 80.0% |", custom)
        self.assertIn("| items | mine | 5 | 4 |", custom, "a T is fitted on the items")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            harness.main(["--cache", str(self.tmp), "calibration", "--results", str(results),
                          "--model", "verdict-1.4"])
        self.assertEqual(printed.getvalue(), text + "\n")

    def test_extra_settings_are_checked(self):
        import servers

        self.assertEqual(servers.extra_settings(["OPENJEV_MLX_CACHE_LIMIT_GB=4",
                                                 "OPENJEV_AUTO_MAX=1"]),
                         {"OPENJEV_MLX_CACHE_LIMIT_GB": "4", "OPENJEV_AUTO_MAX": "1"})
        self.assertEqual(servers.extra_settings([]), {})
        # what servers.py sets itself, a name outside OPENJEV_ and a pair without a value
        for pair in ("OPENJEV_PORT=1", "OPENJEV_BACKEND=laya", "OPENJEV_MLX_MODEL=/tmp/x",
                     "OPENJEV_HOST=0.0.0.0", "HF_TOKEN=x", "OPENJEV_MLX_CACHE_LIMIT_GB"):
            with self.assertRaises(SystemExit, msg=pair):
                servers.extra_settings([pair])
        # a setting that can hold a credential is refused, and its value is never printed
        for pair in ("OPENJEV_API_KEY=sk-secret", "OPENJEV_ORIGIN_SECRET=sk-secret",
                     "OPENJEV_MODEL_ROUTES=m=https://user:sk-secret@host/v1",
                     "OPENJEV_UPSTREAM=https://user:sk-secret@host", "OPENJEV_HUB_TOKEN=sk-secret",
                     "OPENJEV_PROXY_PASSWORD=sk-secret"):
            with self.assertRaises(SystemExit, msg=pair) as refused:
                servers.extra_settings([pair])
            self.assertNotIn("sk-secret", str(refused.exception))

    def test_helpers(self):
        self.assertEqual(harness.parse_server_timing("model;dur=41.2, server;dur=2.8, total;dur=44"),
                         {"model": 41.2, "server": 2.8, "total": 44.0})
        self.assertIsNone(harness.parse_server_timing(None))
        self.assertEqual(harness.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(harness.mcnemar_exact(5, 0), 2 / 32)
        self.assertEqual(harness.mcnemar_exact(6, 1), 2 * 8 / 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
