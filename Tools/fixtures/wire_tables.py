#!/usr/bin/env python3
"""Record upstream OpenJev's wire contract as fixtures for OpenJevCore's wire types.

The script builds upstream's FastAPI app exactly as upstream's tests do, with
``openjev.api.create_app(Settings(...), tokenizer=FakeTokenizer())`` inside
``fastapi.testclient.TestClient``, sends it requests and records the responses byte for byte.
No model and no network are involved: the engine points at a local port where nothing listens,
so every request that passes validation ends in upstream's own 503, which is recorded too.

Requirements: Python 3.10 or later with fastapi and httpx, and the pinned upstream checkout at
Upstream/openjev (``make upstream``). Set up once, from the repository root:

    python3 -m venv Tools/fixtures/.venv
    Tools/fixtures/.venv/bin/python -m pip install "fastapi>=0.115" httpx

Then regenerate:

    Tools/fixtures/.venv/bin/python Tools/fixtures/wire_tables.py

The output goes to Fixtures/wire/. Each file records the upstream commit and the Python, FastAPI,
Starlette, pydantic and pydantic-core versions that wrote it. Running the script twice with the
same versions gives the same files.
"""

import base64
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
OUTPUT = ROOT / "Fixtures" / "wire"
UPSTREAM_COMMIT = "dcd2094"

# Settings read the environment; clear it so the defaults are upstream's own.
for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
# Upstream logs every rejected request; the fixtures record the responses instead.
logging.getLogger("openjev").setLevel(logging.CRITICAL)

sys.path.insert(0, str(UPSTREAM))

import fastapi  # noqa: E402
import httpx  # noqa: E402
import pydantic  # noqa: E402
import pydantic_core  # noqa: E402
import starlette  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openjev import config as oj_config  # noqa: E402
from openjev.api import create_app  # noqa: E402
from openjev.config import ENCODER_MODELS, Settings, served_models  # noqa: E402
from openjev.engine import Engine, to_answer  # noqa: E402

# Nothing listens on the discard port, so the engine's first read fails to connect and upstream
# answers 503 "inference backend unavailable: ConnectError" without any model.
NO_BACKEND = "http://127.0.0.1:9"
REQUEST_ID = re.compile(r"^req_[0-9a-f]{32}$")
REQUEST_ID_FORMAT = "req_ followed by 32 lowercase hex characters"


class FakeTokenizer:
    """One integer id per token, where a token is a run of ASCII letters, one digit, one
    whitespace character or one other character. "AA" is one token, so upstream's label
    discovery finds exactly 255 single-token labels. Ids are assigned in order of first use."""

    TOKEN = re.compile(r"[A-Za-z]+|\d|\s|[^\w\s]")

    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for token in self.TOKEN.findall(text):
            out.append(self.ids.setdefault(token, len(self.ids) + 1))
        return out

    def apply_chat_template(self, messages, **kwargs):
        """Only a valid request with think > 0 gets here, on its way to the 503; the ids are
        never recorded, so any stable rendering will do."""
        return self.encode("\n".join(f"{m['role']}: {m['content']}" for m in messages))


def upstream_head():
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generator():
    return {
        "script": "Tools/fixtures/wire_tables.py",
        "upstream": "razorback16/openjev",
        "upstream_commit": UPSTREAM_COMMIT,
        "python": sys.version.split()[0],
        "fastapi": fastapi.__version__,
        "starlette": starlette.__version__,
        "pydantic": pydantic.VERSION,
        "pydantic_core": pydantic_core.__version__,
        "httpx": httpx.__version__,
    }


def compact(value):
    """How FastAPI's JSONResponse renders a body."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"))


# Request bodies ------------------------------------------------------------------------------

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
QUICKSTART = {  # Jev's quickstart request, as upstream's tests hold it
    "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
    "model": "jev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this",
                       "criteria": {"billing": "Payment or subscription issues",
                                    "technical": "Bugs or integration problems",
                                    "sales": "Pricing or account questions"}},
        "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"},
    },
}
SMALL = {"state": "x", "model": "jev-latest", "questions": {"a": {"type": "noul"}}}
BIG_PLACEHOLDER = "@@BIG@@"
BIG_REPEAT, BIG_COUNT = "QUFB", 2 * 1024 * 1024  # 8 MB of base64, 6 MB decoded: over the 5 MB default


def with_(base, **fields):
    out = dict(base)
    out.update(fields)
    return out


def without(base, *names):
    return {k: v for k, v in base.items() if k not in names}


def question(q):
    return with_(QUICKSTART, questions={"q": q})


def nested(levels):
    deep = {"a": None}
    for _ in range(levels):
        deep = {"a": deep}
    return deep


def json_cases():
    """(name, body) pairs sent as application/json against the default settings."""
    cases = [
        ("quickstart", QUICKSTART),
        ("quickstart_explicit_defaults", with_(QUICKSTART, steps=1, think=0, sequential=False)),
        ("small_noul", SMALL),
        ("missing_state", without(QUICKSTART, "state")),
        ("missing_model", without(QUICKSTART, "model")),
        ("missing_questions", without(QUICKSTART, "questions")),
        ("missing_everything", {}),
        ("questions_empty", with_(QUICKSTART, questions={})),
        ("questions_list", with_(QUICKSTART, questions=[])),
        ("questions_string", with_(QUICKSTART, questions="x")),
        ("state_number", with_(QUICKSTART, state=5)),
        ("state_float", with_(QUICKSTART, state=1.5)),
        ("state_true", with_(QUICKSTART, state=True)),
        ("state_null", with_(QUICKSTART, state=None)),
        ("state_object", with_(QUICKSTART, state={"b": 1, "a": [2, {"z": None}]})),
        ("state_list", with_(QUICKSTART, state=[1, "two", {"3": 4}])),
        ("state_empty_string", with_(QUICKSTART, state="")),
        ("model_number", with_(QUICKSTART, model=5)),
        ("model_null", with_(QUICKSTART, model=None)),
        ("model_unknown", with_(QUICKSTART, model="gpt-4")),
        ("model_pinned_jev", with_(QUICKSTART, model="jev-1.13.0")),
        ("question_string", question("x")),
        ("question_null", question(None)),
        ("question_list", question([])),
        ("question_no_type", question({"instructions": "x"})),
        ("question_type_nope", question({"type": "nope", "instructions": "x"})),
        ("question_type_number", question({"type": 5})),
        ("question_type_null", question({"type": None})),
        ("question_type_nope_and_state_missing",
         without(question({"type": "nope"}), "state")),
        ("question_type_nope_and_no_type",
         with_(QUICKSTART, questions={"a": {"type": "nope"}, "b": {"instructions": "x"}})),
        ("instructions_number", question({"type": "noul", "instructions": 5})),
        ("instructions_object", question({"type": "noul", "instructions": {"k": [1, 2]}})),
        ("instructions_null", question({"type": "noul", "instructions": None})),
        ("noul_criteria_string", question({"type": "noul", "criteria": "x"})),
        ("noul_criteria_list", question({"type": "noul", "criteria": []})),
        ("noul_criteria_true_number", question({"type": "noul", "criteria": {"true": 5}})),
        ("noul_criteria_both_numbers", question({"type": "noul", "criteria": {"true": 5, "false": False}})),
        ("noul_criteria_unknown_key", question({"type": "noul", "criteria": {"yes": 5}})),
        ("noul_criteria_null", question({"type": "noul", "criteria": None})),
        ("choice_criteria_list", question({"type": "choice", "criteria": ["a", "b"]})),
        ("choice_criteria_missing", question({"type": "choice", "instructions": "x"})),
        ("choice_criteria_null", question({"type": "choice", "criteria": None})),
        ("choice_criteria_empty", question({"type": "choice", "instructions": "x", "criteria": {}})),
        ("choice_criteria_value_number", question({"type": "choice", "criteria": {"a": 5, "b": "ok", "c": True}})),
        ("choice_single_option", question({"type": "choice", "criteria": {"billing": "anything"}})),
        ("choice_255_options", question({"type": "choice", "criteria": {f"o{j}": None for j in range(255)}})),
        ("choice_256_options", question({"type": "choice", "criteria": {f"o{j}": None for j in range(256)}})),
        ("score_criteria_dict", question({"type": "score", "criteria": {"a": "b"}})),
        ("score_criteria_empty", question({"type": "score", "criteria": []})),
        ("score_criteria_missing", question({"type": "score"})),
        ("score_criteria_null_level", question({"type": "score", "criteria": ["ok", None, 5]})),
        ("score_single_level", question({"type": "score", "criteria": ["fine"]})),
        ("score_10_levels", question({"type": "score", "criteria": [f"l{i}" for i in range(10)]})),
        ("score_11_levels", question({"type": "score", "criteria": [f"l{i}" for i in range(11)]})),
        ("questions_many_errors", with_(QUICKSTART, questions={
            "a": {"type": "choice"}, "b": {"type": "score", "criteria": []}, "c": {"type": "noul", "criteria": "x"}})),
    ]
    for field, values in [
        ("steps", [0, 1, 8, 9, -1, "3", " 3 ", "\t3\n", "\u30003", "3.0", "3.", "3.00", "3.5", "+3", "-0", "0003",
                   "1_0", "_1", "1_", "1__0", "1_0.0", ".0", "1e3", "0x10", "+-3", "abc", "", "٣", "1" * 4300, " " + "1" * 4300, "1" * 4301,
                   True, False, 2.0, -0.0, 2.5, 100.0, 1e20, 10**30, None, [], {}]),
        ("samples", [0, 1, 32, 33]),
        ("think", [-1, 0, 4096, 4097]),
        ("sequential", ["yes", "YES", "no", "off", "true", "t", "F", "On", "y", "N", "0", "1", " yes", "maybe", "",
                        1, 0, 2, -1, 1.0, 0.0, -0.0, 2.0, 0.5, None, [], {}, True]),
    ]:
        for v in values:
            label = json.dumps(v, ensure_ascii=True)
            if len(label) > 40:
                label = f"string_of_{len(v)}_characters" + ("_padded" if v.startswith(" ") else "")
            cases.append((f"{field}_{label}", with_(SMALL, **{field: v})))
    cases += [
        ("extension_errors_together", with_(QUICKSTART, steps=9, samples=0, think="x", sequential="maybe")),
        ("extension_errors_with_body_errors", with_(without(QUICKSTART, "state"), questions={}, steps=0)),
        ("images_string", with_(QUICKSTART, images="abc")),
        ("images_number_item", with_(QUICKSTART, images=[5])),
        ("images_empty_object", with_(QUICKSTART, images=[{}])),
        ("images_object_number_fields", with_(QUICKSTART, images=[{"content_type": 5, "base64": None}])),
        ("images_null", with_(QUICKSTART, images=None)),
        ("images_empty_list", with_(QUICKSTART, images=[])),
        ("images_url", with_(QUICKSTART, images=["https://example.com/a.png"])),
        ("images_bmp", with_(QUICKSTART, images=[{"content_type": "image/bmp", "base64": PNG}])),
        ("images_invalid_base64", with_(QUICKSTART, images=[{"content_type": "image/png", "base64": "not base64!"}])),
        ("images_nine", with_(QUICKSTART, images=[f"data:image/png;base64,{PNG}"] * 9)),
        ("images_valid", with_(QUICKSTART, images=[f"data:image/png;base64,{PNG}",
                                                   {"content_type": "image/jpeg", "base64": PNG}])),
        ("images_with_think", with_(QUICKSTART, images=[f"data:image/png;base64,{PNG}"], think=64)),
        ("images_with_sequential", with_(QUICKSTART, images=[f"data:image/png;base64,{PNG}"], sequential=True)),
        ("questions_257", with_(QUICKSTART, questions={f"q{i}": {"type": "noul", "instructions": "x"} for i in range(257)})),
        ("question_nested_1000", with_(QUICKSTART, questions={"q": nested(1000)})),
        ("unknown_top_level_field", with_(QUICKSTART, extra={"anything": [1, 2]})),
        ("state_long_string_missing_model", without(with_(QUICKSTART, state="y" * 600), "model")),
        ("state_wide_object_invalid_model", with_(QUICKSTART, state={f"k{i}": i for i in range(25)}, model=[1])),
    ]
    return cases


# Recording ---------------------------------------------------------------------------------

def app_client(settings_kwargs):
    kwargs = dict(upstream=NO_BACKEND)
    kwargs.update(settings_kwargs)
    return kwargs, TestClient(create_app(Settings(**kwargs), tokenizer=FakeTokenizer()))


def record(name, settings, response, method, path, headers, body_text=None, body_base64=None,
           body_expand=None, streamed=False):
    rid = response.headers.get("x-request-id")
    if rid is None or not REQUEST_ID.match(rid) or response.headers.get("x-typesafe-request-id") != rid:
        raise SystemExit(f"{name}: request id headers are not as expected: {dict(response.headers)}")
    request = {"method": method, "path": path, "headers": headers}
    if body_text is not None:
        request["body_text"] = body_text
    if body_base64 is not None:
        request["body_base64"] = body_base64
    if body_expand is not None:
        request["body_expand"] = body_expand
    if streamed:
        request["streamed_without_content_length"] = True
    response_headers = {"content-type": response.headers.get("content-type")}
    if "retry-after" in response.headers:
        response_headers["retry-after"] = response.headers["retry-after"]
    return {
        "name": name,
        "settings": settings,
        "request": request,
        "response": {"status": response.status_code, "body_text": response.content.decode("utf-8"),
                     "headers": response_headers},
        "request_id_header_format": REQUEST_ID_FORMAT,
        "server_timing_present": "server-timing" in response.headers,
    }


def recorded_settings(kwargs):
    """The Settings fields that differ from upstream's defaults, as JSON."""
    defaults = Settings()
    return {k: v for k, v in kwargs.items() if getattr(defaults, k) != v}


def http_cases():
    out = []
    kwargs, client = app_client({})
    settings = recorded_settings(kwargs)
    json_headers = {"content-type": "application/json"}
    with client as c:
        for path in ("/health", "/v1/models"):
            out.append(record(f"get{path.replace('/', '_')}", settings, c.get(path), "GET", path, {}))
        for name, body in json_cases():
            text = compact(body)
            r = c.post("/v1/systemone", content=text.encode("utf-8"), headers=json_headers)
            out.append(record(name, settings, r, "POST", "/v1/systemone", json_headers, body_text=text))

        # The oversize image: the body is recorded with a placeholder that the reader expands.
        for form in ("object", "data_url"):
            if form == "object":
                image = {"content_type": "image/png", "base64": BIG_PLACEHOLDER}
            else:
                image = f"data:image/png;base64,{BIG_PLACEHOLDER}"
            template = compact(with_(QUICKSTART, images=[image]))
            real = template.replace(BIG_PLACEHOLDER, BIG_REPEAT * BIG_COUNT)
            r = c.post("/v1/systemone", content=real.encode("utf-8"), headers=json_headers)
            out.append(record(f"images_oversize_{form}", settings, r, "POST", "/v1/systemone", json_headers,
                              body_text=template,
                              body_expand={"placeholder": BIG_PLACEHOLDER, "repeat": BIG_REPEAT,
                                           "count": BIG_COUNT}))

        # Bodies that are not a JSON object, or not JSON at all.
        raw = [
            ("body_empty", b"", json_headers),
            ("body_json_null", b"null", json_headers),
            ("body_json_array", b"[]", json_headers),
            ("body_json_string", b'"x"', json_headers),
            ("body_json_number", b"5", json_headers),
            ("body_malformed_truncated", b'{"state": ', json_headers),
            ("body_malformed_single_quotes", b"{'state': 'x'}", json_headers),
            ("body_malformed_trailing_comma", b'{"state": "x",}', json_headers),
            ("body_malformed_missing_colon", b'{"state" "x"}', json_headers),
            ("body_malformed_missing_comma", b'{"state": "x" "model": "y"}', json_headers),
            ("body_malformed_extra_data", b"{} {}", json_headers),
            ("body_malformed_unterminated_string", b'{"state": "x', json_headers),
            ("body_malformed_control_character", b'{"state": "a\x01b"}', json_headers),
            ("body_malformed_invalid_escape", b'{"state": "\\q"}', json_headers),
            ("body_malformed_non_ascii_before_error", '{"state": "éé", }'.encode("utf-8"), json_headers),
            ("body_invalid_utf8", b'{"state": "\xff"}', json_headers),
            ("body_no_content_type", compact(SMALL).encode("utf-8"), {}),
            ("body_text_plain", compact(SMALL).encode("utf-8"), {"content-type": "text/plain"}),
            ("body_json_charset", compact(SMALL).encode("utf-8"), {"content-type": "application/json; charset=utf-8"}),
            ("body_vendor_json", compact(SMALL).encode("utf-8"), {"content-type": "application/vnd.api+json"}),
        ]
        for name, body, headers in raw:
            r = c.post("/v1/systemone", content=body, headers=headers)
            try:
                text = body.decode("utf-8")
                out.append(record(name, settings, r, "POST", "/v1/systemone", headers, body_text=text))
            except UnicodeDecodeError:
                out.append(record(name, settings, r, "POST", "/v1/systemone", headers,
                                  body_base64=base64.b64encode(body).decode("ascii")))

    # The body cap.
    kwargs, client = app_client({"max_body_bytes": 512})
    settings = recorded_settings(kwargs)
    with client as c:
        text = compact(QUICKSTART)
        assert len(text.encode("utf-8")) > 512
        r = c.post("/v1/systemone", content=text.encode("utf-8"), headers=json_headers)
        out.append(record("body_over_cap", settings, r, "POST", "/v1/systemone", json_headers, body_text=text))
        r = c.post("/v1/systemone", content=iter([text.encode("utf-8")]), headers=json_headers)
        assert "content-length" not in r.request.headers
        out.append(record("body_over_cap_streamed", settings, r, "POST", "/v1/systemone", json_headers,
                          body_text=text, streamed=True))
        small = compact(SMALL)
        r = c.post("/v1/systemone", content=small.encode("utf-8"), headers=json_headers)
        out.append(record("body_under_cap", settings, r, "POST", "/v1/systemone", json_headers, body_text=small))
        r = c.get("/v1/models")
        out.append(record("body_cap_get_models", settings, r, "GET", "/v1/models", {}))

    # Authentication. Header values are sent as UTF-8 bytes; Starlette decodes them as latin-1.
    auth_sets = [
        ({"api_key": "sk-test"}, [
            ("auth_key_missing", "GET", "/v1/models", {}),
            ("auth_key_wrong", "GET", "/v1/models", {"authorization": "Bearer nope"}),
            ("auth_key_right", "GET", "/v1/models", {"authorization": "Bearer sk-test"}),
            ("auth_key_without_bearer_prefix", "GET", "/v1/models", {"authorization": "sk-test"}),
            ("auth_key_padded", "GET", "/v1/models", {"authorization": "Bearer  sk-test "}),
            ("auth_key_lowercase_bearer", "GET", "/v1/models", {"authorization": "bearer sk-test"}),
            ("auth_key_health_open", "GET", "/health", {}),
            ("auth_key_missing_post", "POST", "/v1/systemone", {"content-type": "application/json"}),
            ("auth_key_right_post", "POST", "/v1/systemone",
             {"content-type": "application/json", "authorization": "Bearer sk-test"}),
        ]),
        ({"origin_secret": "s3"}, [
            ("auth_origin_missing", "GET", "/v1/models", {}),
            ("auth_origin_wrong", "GET", "/v1/models", {"x-origin-secret": "nope"}),
            ("auth_origin_right", "GET", "/v1/models", {"x-origin-secret": "s3"}),
            ("auth_origin_health_open", "GET", "/health", {}),
        ]),
        ({"api_key": "sk-test", "origin_secret": "s3"}, [
            ("auth_both_missing", "GET", "/v1/models", {}),
            ("auth_both_origin_only", "GET", "/v1/models", {"x-origin-secret": "s3"}),
            ("auth_both_right", "GET", "/v1/models", {"x-origin-secret": "s3", "authorization": "Bearer sk-test"}),
            ("auth_both_non_ascii_origin", "GET", "/v1/models",
             {"x-origin-secret": "s€", "authorization": "Bearer sk-test"}),
            ("auth_both_non_ascii_key", "GET", "/v1/models",
             {"x-origin-secret": "s3", "authorization": "Bearer sk-t€st"}),
        ]),
    ]
    for settings_kwargs, requests in auth_sets:
        kwargs, client = app_client(settings_kwargs)
        settings = recorded_settings(kwargs)
        with client as c:
            for name, method, path, headers in requests:
                sent = {k: v.encode("utf-8") for k, v in headers.items()}
                if method == "GET":
                    r = c.get(path, headers=sent)
                    out.append(record(name, settings, r, method, path, headers))
                else:
                    small = compact(SMALL)
                    r = c.post(path, content=small.encode("utf-8"), headers=sent)
                    out.append(record(name, settings, r, method, path, headers, body_text=small))
    return out


# Answers -----------------------------------------------------------------------------------

def schema_question(wire):
    """The internal question upstream's build_schema makes from a wire question, for to_answer."""
    engine = Engine(Settings(upstream=NO_BACKEND), FakeTokenizer())
    schema = engine.build_schema({"q": wire})
    return schema["questions"][0]


def answer_cases():
    choice3 = {"type": "choice", "instructions": "Which team should handle this",
               "criteria": {"billing": "Payment or subscription issues",
                            "technical": "Bugs or integration problems",
                            "sales": "Pricing or account questions"}}
    tie = {"type": "choice", "criteria": {"left": None, "right": None}}
    wide = {"type": "choice", "criteria": {f"o{j}": None for j in range(255)}}
    score3 = {"type": "score", "instructions": "How frustrated the customer appears",
              "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]}
    score_obj = {"type": "score", "criteria": ["a", {"k": 1}]}
    score10 = {"type": "score", "criteria": [f"level {i}" for i in range(10)]}
    noul = {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}
    layout = [1.0, 0.0, 1e-05, 0.30000000000000004, 0.123456789012345678]
    rows = [
        ("noul_70_30", noul, [0.7, 0.3]),
        ("noul_certain", noul, [1.0, 0.0]),
        ("choice_three", choice3, [0.08, 0.85, 0.07]),
        ("choice_tie", tie, [0.5, 0.5]),
        ("choice_255_uniform", wide, [1 / 255] * 255),
        ("score_three", score3, [0.15, 0.55, 0.30]),
        ("score_object_legend", score_obj, [0.25, 0.75]),
        ("score_ten", score10, [0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.2, 0.06, 0.04]),
    ]
    rows += [(f"noul_layout_{i}", noul, [v, 1.0 - v]) for i, v in enumerate(layout)]
    rows.append(("choice_layout", {"type": "choice", "criteria": {f"c{i}": None for i in range(len(layout))}}, layout))
    rows.append(("score_layout", {"type": "score", "criteria": [f"l{i}" for i in range(len(layout))]}, layout))
    out = []
    for name, wire, p in rows:
        answer = to_answer(schema_question(wire), p)
        out.append({"name": name, "question": wire, "probabilities": p, "body_text": compact(answer)})
    return out


def response_case():
    questions = dict(QUICKSTART["questions"])
    questions["only"] = {"type": "choice", "instructions": "Which team", "criteria": {"billing": "anything at all"}}
    questions["level"] = {"type": "score", "instructions": "How bad", "criteria": [{"text": "fine", "weight": 1}]}
    engine = Engine(Settings(upstream=NO_BACKEND), FakeTokenizer())
    schema = engine.build_schema(questions)
    reads = {"department": [0.08, 0.85, 0.07], "frustration": [0.15, 0.55, 0.30], "is_urgent": [0.7, 0.3]}
    answers = dict(schema["forced"])
    for q in schema["questions"]:
        answers[q["key"]] = to_answer(q, reads[q["key"]])
    answers = {k: answers[k] for k in questions}
    body = {"model": oj_config.MODEL_VERSION, "answers": answers, "usage": {"input_tokens": 123, "output_tokens": 0}}
    return {"name": "full_response", "questions": questions, "reads": reads, "body_text": compact(body)}


# Requests ----------------------------------------------------------------------------------

def request_bodies():
    return [
        ("quickstart", QUICKSTART),
        ("non_ascii_state", with_(QUICKSTART, state="Café ñ \U0001f600 日本語 \u0000\u007f")),
        ("object_state", with_(QUICKSTART, state={"zeta": 1, "alpha": {"b": 2.5, "a": [True, None, "x"]},
                                                  "mid": {"y": 1e-07, "x": 10**20}})),
        ("list_state", with_(QUICKSTART, state=[3, "two", {"one": 1.0}, [], {}])),
        ("described_objects_and_arrays", with_(QUICKSTART, questions={
            "pick": {"type": "choice", "instructions": {"task": "route", "steps": [1, 2]},
                     "criteria": {"zeta": {"why": "last letter"}, "alpha": ["first", "letter"], "mid": None,
                                  "plain": "text"}},
            "rate": {"type": "score", "instructions": ["how", "bad"],
                     "criteria": ["fine", {"level": 1}, ["very", "bad"]]},
            "flag": {"type": "noul", "instructions": None,
                     "criteria": {"true": {"means": "yes"}, "false": ["no"]}},
        })),
        ("noul_partial_criteria", with_(QUICKSTART, questions={
            "a": {"type": "noul", "criteria": {"false": "not urgent"}},
            "b": {"type": "noul", "criteria": {}},
        })),
        ("extension_fields", with_(QUICKSTART, steps=4, samples=16, think=256, sequential=True)),
        ("extension_bounds", with_(QUICKSTART, steps=8, samples=32, think=0, sequential=False)),
        ("images_data_url", with_(QUICKSTART, images=[f"data:image/png;base64,{PNG}"])),
        ("images_object_and_url", with_(QUICKSTART, images=[{"content_type": "image/jpeg", "base64": PNG},
                                                            f"data:image/gif;base64,{PNG}"])),
    ]


def request_cases():
    from openjev.api import SystemOneRequest

    out = []
    for name, body in request_bodies():
        compact_text = compact(body)
        dumped = compact(SystemOneRequest.model_validate(body).model_dump(mode="json", exclude_unset=True))
        if dumped != compact_text:
            raise SystemExit(f"{name}: pydantic's dump differs from the body; write the body in field order")
        out.append({"name": name, "default": json.dumps(body), "compact": compact_text})
    return out


# Models ------------------------------------------------------------------------------------

def model_cases():
    listings = []
    for backend in ["vllm", "mlx", *ENCODER_MODELS]:
        version, names, _ = served_models(backend)
        client = TestClient(create_app(Settings(backend=backend, upstream=NO_BACKEND), tokenizer=FakeTokenizer()))
        r = client.get("/v1/models")  # the listing needs no lifespan, so no encoder model loads
        listings.append({"backend": backend, "model_version": version, "accepted_names": sorted(names),
                         "status": r.status_code, "body_text": r.content.decode("utf-8")})
    routes = {"verdict-1.4": "http://verdict:8000", "custom-model": "http://custom:8000"}
    client = TestClient(create_app(Settings(model_routes=routes, upstream=NO_BACKEND), tokenizer=FakeTokenizer()))
    r = client.get("/v1/models")
    listings.append({"backend": "vllm", "model_routes": routes, "model_version": oj_config.MODEL_VERSION,
                     "accepted_names": sorted(served_models("vllm")[1]), "status": r.status_code,
                     "body_text": r.content.decode("utf-8")})
    return listings


# Output ------------------------------------------------------------------------------------

def write(name, payload):
    """One top-level key per line and one list entry per line: small files with readable diffs."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in {"generator": generator(), **payload}.items():
        if isinstance(value, list):
            items = ",\n".join(" " + json.dumps(v, ensure_ascii=False) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        else:
            lines.append(f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    (OUTPUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {name}: {len(text.encode('utf-8'))} bytes")


def main():
    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    write("cases.json", {"no_backend": NO_BACKEND, "cases": http_cases()})
    write("answers.json", {"answers": answer_cases(), "response": response_case()})
    write("requests.json", {"requests": request_cases()})
    write("models.json", {"listings": model_cases()})


if __name__ == "__main__":
    main()
