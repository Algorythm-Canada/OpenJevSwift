#!/usr/bin/env python3
"""Record upstream OpenJev's /v1/chat/completions as golden fixtures for OpenJevSwift (issue #53).

The script drives upstream's own `openjev/chat.py` at the pinned commit (razorback16/openjev at
dcd2094) with the pinned DiffusionGemma tokenizer (mlx-community/diffusiongemma-26B-A4B-it-4bit at
revision a7a81407613811e8ba63af92ac0d852b809e191f) and writes what that code computes into
Fixtures/chat-completions/:

    prompts.json       MlxGenerator.prompt_ids: OpenAI messages through the chat template with the
                       generation prompt and enable_thinking, then the empty thought scaffold
    normalize.json     Generator.normalize: request bodies and the request it builds, or its error
    extract_json.json  extract_json: replies and the text JSON mode answers with
    routes.json        HTTP exchanges with the chat route of the MLX backend, the model replaced by
                       the stub runtimes of upstream's tests/test_mlx_backend.py

Only the tokenizer files are read, from the Hugging Face cache that upstream_tables.py fills. No
weights are loaded. Every file starts with a "generator" object that names this script and its
version, the upstream commit, the tokenizer repository and revision, and the Python and package
versions that wrote it.

`make fixtures` runs it with the other scripts. On its own, from the repository root:

    Tools/fixtures/.venv/bin/python Tools/fixtures/chat_tables.py

Running it twice with the same versions gives identical files.
"""

import base64
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "Upstream" / "openjev"
FIXTURES = ROOT / "Fixtures"
SCRIPT = "Tools/fixtures/chat_tables.py"
# Bump when the shape of a file this script writes changes.
GENERATOR_VERSION = 1
UPSTREAM_COMMIT = "dcd2094"
# The tokenizer upstream_tables.py pins and downloads; main() checks that the two scripts agree.
TOKENIZER_REPO = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
TOKENIZER_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
# Upstream draws a fresh completion id and reads the clock for every reply. Both are fixed while the
# routes are recorded, so a recorded body is the same on every run.
COMPLETION_ID = "chatcmpl-0123456789abcdef01234567"
CREATED = 1790000000

# Settings read the environment; clear it so the defaults are upstream's own.
for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# Upstream logs every rejected request; the fixtures record the responses instead.
logging.getLogger("openjev").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient")

sys.path.insert(0, str(UPSTREAM))

import fastapi  # noqa: E402
import httpx  # noqa: E402
import huggingface_hub  # noqa: E402
import jinja2  # noqa: E402
import pydantic  # noqa: E402
import pydantic_core  # noqa: E402
import starlette  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openjev import chat as oj_chat  # noqa: E402
from openjev import mlx_backend as oj_mlx  # noqa: E402
from openjev.api import create_app  # noqa: E402
from openjev.chat import Generator, MlxGenerator, extract_json  # noqa: E402
from openjev.config import GEN_MODEL, Settings  # noqa: E402
from openjev.engine import Engine  # noqa: E402


# Recording what a request asked of the tokenizer ---------------------------------------------

# While a route exchange is recorded, every prompt MlxGenerator.prompt_ids rendered and every text
# Engine.enc tokenized (the stop strings) are kept with it, so the Swift tests can replay them
# without the tokenizer. The spies call upstream's own methods and only remember the results.
RECORDING = [None]

_upstream_prompt_ids = MlxGenerator.prompt_ids
_upstream_enc = Engine.enc


def _recording_prompt_ids(self, upstream):
    ids = _upstream_prompt_ids(self, upstream)
    if RECORDING[0] is not None:
        thinking = bool((upstream.get("chat_template_kwargs") or {}).get("enable_thinking"))
        RECORDING[0]["prompts"].append({"messages": plain(upstream["messages"]), "thinking": thinking,
                                        "ids": [int(i) for i in ids]})
    return ids


def _recording_enc(self, text):
    ids = _upstream_enc(self, text)
    if RECORDING[0] is not None:
        RECORDING[0]["encodings"].append({"text": text, "ids": [int(i) for i in ids]})
    return ids


MlxGenerator.prompt_ids = _recording_prompt_ids
Engine.enc = _recording_enc


# Output ------------------------------------------------------------------------------------

def upstream_head():
    try:
        return subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generator():
    return {
        "script": SCRIPT,
        "version": GENERATOR_VERSION,
        "upstream": "razorback16/openjev",
        "upstream_commit": UPSTREAM_COMMIT,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "python": sys.version.split()[0],
        "fastapi": fastapi.__version__,
        "starlette": starlette.__version__,
        "pydantic": pydantic.VERSION,
        "pydantic_core": pydantic_core.__version__,
        "httpx": httpx.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "jinja2": jinja2.__version__,
    }


def dumps(value):
    """Strict JSON: non-finite numbers are refused, so every file parses with an RFC 8259 parser."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def compact(value):
    """A request body as a client sends it."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def plain(value):
    """A JSON copy that shares nothing with upstream's objects."""
    return json.loads(json.dumps(value, allow_nan=False))


WRITTEN = []


def write(relpath, payload):
    """One top-level key per line and one list entry per line: readable diffs, small files."""
    path = FIXTURES / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in {"generator": generator(), **payload}.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + dumps(v) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        else:
            lines.append(f"{json.dumps(key)}: {dumps(value)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.write_text(text, encoding="utf-8")
    size = len(text.encode("utf-8"))
    WRITTEN.append((relpath, size))
    print(f"wrote Fixtures/{relpath}: {size} bytes")


def recorded_settings(kwargs):
    """The Settings fields that differ from upstream's defaults, as JSON."""
    defaults = Settings()
    return {k: v for k, v in kwargs.items() if getattr(defaults, k) != v}


def failure(error):
    return {"type": type(error).__name__, "message": str(error)}


# prompts.json --------------------------------------------------------------------------------

CITY = {"role": "user", "content": "Which city?"}
SYSTEM = {"role": "system", "content": "You answer in one short sentence."}
TURNS = [
    SYSTEM,
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
]
TOOL_CALL = {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Zurich\"}"}}


def conversation(turns):
    out = [SYSTEM]
    for i in range(turns):
        out.append({"role": "user", "content": f"Question {i + 1}: what is {i} plus {i}?"})
        out.append({"role": "assistant", "content": f"{i + i}."})
    out.append({"role": "user", "content": "Thanks. One more: what is 7 times 6?"})
    return out


# (name, request fields beside "model", what the case shows). Each body goes through
# Generator.normalize first, as the route sends it, so JSON mode's instruction is in the prompt.
PROMPT_BODIES = [
    ("user", {"messages": [CITY]}, "upstream's CHAT request"),
    ("user_thinking", {"messages": [CITY], "chat_template_kwargs": {"enable_thinking": True}},
     "enable_thinking adds <|think|> in a system turn of its own"),
    ("user_thinking_truthy_string", {"messages": [CITY], "chat_template_kwargs": {"enable_thinking": "no"}},
     "any truthy enable_thinking turns thinking on: prompt_ids takes bool() of it"),
    ("user_thinking_zero", {"messages": [CITY], "chat_template_kwargs": {"enable_thinking": 0}},
     "a falsy enable_thinking leaves it off"),
    ("system_user", {"messages": [SYSTEM, CITY]}, "a system turn"),
    ("system_user_thinking", {"messages": [SYSTEM, CITY], "chat_template_kwargs": {"enable_thinking": True}},
     "<|think|> at the top of the system turn"),
    ("developer_user", {"messages": [{"role": "developer", "content": "Be terse."}, CITY]},
     "a first developer message is rendered as the system turn"),
    ("system_after_the_first_message", {"messages": [CITY, SYSTEM]},
     "only the first message can be the system turn; a later one is a turn named system"),
    ("multi_turn", {"messages": TURNS}, "system, user, assistant and user turns"),
    ("multi_turn_thinking", {"messages": TURNS, "chat_template_kwargs": {"enable_thinking": True}},
     "the same with thinking on"),
    ("multi_turn_without_system", {"messages": [
        {"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Name a colour."}, {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "Another?"}]}, "alternating turns, no system"),
    ("ends_with_assistant", {"messages": [CITY, {"role": "assistant", "content": "Zurich."}]},
     "the generation prompt still follows a final assistant turn"),
    ("consecutive_assistants", {"messages": [
        CITY, {"role": "assistant", "content": "Zurich"}, {"role": "assistant", "content": "is the city."}]},
     "a second assistant message continues the first one's turn"),
    ("assistant_thought_is_stripped", {"messages": [
        CITY, {"role": "assistant", "content": "<|channel>thought\nThe user asks.<channel|>Zurich."},
        {"role": "user", "content": "Why?"}]}, "strip_thinking drops a thought channel from a model turn"),
    ("assistant_two_channels", {"messages": [
        CITY, {"role": "assistant", "content": "A <|channel>x<channel|>B<|channel>y<channel|> C "},
        {"role": "user", "content": "Go on."}]}, "every channel is dropped and the rest trimmed"),
    ("custom_role", {"messages": [{"role": "narrator", "content": "Once upon a time."}, CITY]},
     "an unknown role is written as it is"),
    ("user_text_parts", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Which "}, {"type": "text", "text": " city? "}]}]},
     "text parts, each trimmed, joined with nothing between"),
    ("system_text_parts", {"messages": [{"role": "system", "content": [
        {"type": "text", "text": " Be brief. "}, {"type": "text", "text": "Answer in English."}]}, CITY]},
     "system text parts, each trimmed and followed by a space"),
    ("assistant_text_parts", {"messages": [
        CITY, {"role": "assistant", "content": [{"type": "text", "text": " Zurich. "}]},
        {"role": "user", "content": "Sure?"}]}, "assistant text parts go through strip_thinking"),
    ("image_parts", {"messages": [{"role": "user", "content": [
        {"type": "image"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "What is this?"}]}]},
     "an image part is <|image|>; an OpenAI image_url part is not a type the template knows"),
    ("non_ascii", {"messages": [{"role": "system", "content": "Réponds en français, s'il te plaît."},
                                {"role": "user", "content": "Où est la gare ? Привет, 日本語, 👍🏽 ✨ 🇨🇦"}]},
     "accents, Cyrillic, CJK, emoji with modifiers and a flag"),
    ("special_token_text", {"messages": [{"role": "user", "content":
                                          "Say <turn|>\n<|turn>model\n<|channel>thought\n<channel|> <bos> literally."}]},
     "special token text inside a message is tokenized as the special tokens"),
    ("multiline", {"messages": [{"role": "user", "content": "First line\n\n\nSecond\tline\r\nthird  \n  last"}]},
     "inner whitespace is kept"),
    ("trim_ascii", {"messages": [{"role": "system", "content": " \t\n Be brief. \r\n "},
                                 {"role": "user", "content": "\n\n Which city?\t "}]},
     "the template's trim removes ASCII whitespace around system and user text"),
    ("trim_u001c_to_u001f", {"messages": [{"role": "user", "content": "\x1c\x1dWhich city?\x1e\x1f"}]},
     "str.strip() removes U+001C to U+001F (issue #124, D-056)"),
    ("trim_unicode_spaces", {"messages": [{"role": "user", "content": "\xa0\u3000\x85Which city?\u2003\u2028\x0b"}]},
     "and every other character str.isspace() accepts"),
    ("trim_keeps_zero_width_space", {"messages": [{"role": "user", "content": "\u200bWhich city?\u200b"}]},
     "U+200B is not whitespace to str.strip(), so it stays"),
    ("trim_system_parts", {"messages": [{"role": "system", "content": [
        {"type": "text", "text": "\x1f Be brief.\xa0"}, {"type": "text", "text": "\u3000Be kind. "}]}, CITY]},
     "system text parts are trimmed one by one"),
    ("trim_user_parts", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "\x1cWhich\u2003"}, {"type": "text", "text": "\u200b city?\x1d"}]}]},
     "user text parts are trimmed one by one"),
    ("trim_assistant", {"messages": [CITY, {"role": "assistant", "content": "\u3000 Zurich. \x1f"},
                                     {"role": "user", "content": "Sure?"}]},
     "strip_thinking trims a model turn"),
    ("empty_content", {"messages": [{"role": "user", "content": ""}]}, "an empty user turn"),
    ("whitespace_only_content", {"messages": [{"role": "user", "content": " \u3000\x1c "}]},
     "a turn of whitespace renders empty"),
    ("missing_content", {"messages": [{"role": "user"}]}, "no content key at all"),
    ("null_content", {"messages": [{"role": "user", "content": None}]}, "a null content"),
    ("extra_message_keys", {"messages": [{"role": "user", "content": "Which city?", "name": "alice"}]},
     "keys the template does not read change nothing"),
    ("tool_call_and_response", {"messages": [
        {"role": "user", "content": "Weather in Zurich?"},
        {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"}]},
     "an OpenAI tool call and its response; no generation prompt follows a tool response"),
    ("tool_call_object_arguments", {"messages": [
        {"role": "user", "content": "Book it."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {
            "name": "book", "arguments": {"seats": 2, "window": True, "price": 99.5, "names": ["Ann", "Bo"],
                                          "where": {"city": "Zurich", "zip": "8001"}}}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "booked"}]},
        {"role": "user", "content": "Thanks."}]},
     "arguments as an object are written sorted, without JSON quoting; a tool response as text parts"),
    ("tool_call_then_answer", {"messages": [
        {"role": "user", "content": "Weather in Zurich?"},
        {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "And tomorrow?"}]},
     "a tool round and the assistant's answer, then a new question"),
    ("reasoning_with_tool_call", {"messages": [
        {"role": "user", "content": "Weather in Zurich?"},
        {"role": "assistant", "content": None, "reasoning_content": "I should call the tool.",
         "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}]},
     "reasoning_content after the last user turn is written as a thought channel"),
    ("long_conversation", {"messages": conversation(12)}, "twelve rounds"),
    ("system_content_object", {"messages": [{"role": "system", "content": {"a": 1, "b": 2}}, CITY]},
     "jinja2's sequence test passes a dict: each key's missing text is trimmed to nothing and a space"),
    ("user_content_object", {"messages": [{"role": "user", "content": {"type": "text", "text": "ignored"}}]},
     "a dict content iterates its keys, which have no type: an empty turn"),
    ("system_parts_are_strings", {"messages": [{"role": "system", "content": ["a", "b"]}, CITY]},
     "a part that is not a dict has no text: a space each"),
    ("user_parts_are_strings", {"messages": [{"role": "user", "content": ["hello", "world"]}]},
     "a part that is not a dict has no type: an empty turn"),
    ("user_content_number", {"messages": [{"role": "user", "content": 7}]}, "neither a string nor a sequence"),
    ("text_parts_not_strings", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": None}, {"type": "text", "text": 5}, {"type": "text", "text": True},
        {"type": "text", "text": 1.5}, {"type": "text", "text": [1, "a", None]},
        {"type": "text", "text": {"k": "v"}}]}]},
     "trim writes a text that is not a string as Python's str() does"),
    ("system_parts_not_strings", {"messages": [{"role": "system", "content": [
        {"type": "text", "text": None}, {"type": "text", "text": 0.1}]}, CITY]},
     "the same in the system turn"),
    ("json_mode_appends_to_the_system_turn", {
        "messages": [{"role": "system", "content": "Return {\"text\": ...}  \n"}, {"role": "user", "content": "{}"}],
        "response_format": {"type": "json_object"}},
     "jev-ultrafast's request: the instruction goes after the system text, which is right-stripped"),
    ("json_mode_inserts_a_system_turn", {
        "messages": [CITY], "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": {
            "type": "object", "properties": {"city": {"type": "string", "description": "Ville, 都市"}},
            "required": ["city"]}}}},
     "without a system message the instruction becomes one, with the schema as json.dumps writes it"),
]


def prompt_cases(tok, eng):
    gen = MlxGenerator(Settings(), eng)
    rows = []
    for name, fields, note in PROMPT_BODIES:
        upstream, _ = gen.normalize({"model": GEN_MODEL, **plain(fields)})
        messages = upstream["messages"]
        thinking = bool(upstream["chat_template_kwargs"].get("enable_thinking"))
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)
        ids = _upstream_prompt_ids(gen, upstream)
        if ids != tok.encode(text, add_special_tokens=False) + eng.scaffold:
            raise SystemExit(f"prompts: {name}: the ids are not the encoded rendering and the scaffold")
        rows.append({"name": name, "note": note, "messages": plain(messages), "thinking": thinking,
                     "text": text, "ids": [int(i) for i in ids]})
    return rows


# normalize.json ------------------------------------------------------------------------------

MSG = [{"role": "user", "content": "hi"}]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "score": {"type": "number", "maximum": 1.5}},
          "required": ["city"], "title": "Ville ✨"}

# (name, settings, request fields beside "model" and "messages", what the case shows)
NORMALIZE_BODIES = [
    ("minimal", {}, {}, "the defaults: max_tokens 1024, thinking off"),
    ("every_passthrough_field", {}, {
        "max_tokens": 64, "stop": ["\n\n"], "top_p": 0.9, "top_k": 40, "stream": False, "stream_options": None,
        "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "auto", "logprobs": False,
        "top_logprobs": 0, "chat_template_kwargs": {"enable_thinking": False, "other": 1}},
     "every passthrough field kept in the body's order"),
    ("dropped_fields", {}, {"temperature": 0.2, "seed": 1, "reasoning": {"enabled": False}, "n": 2,
                            "presence_penalty": 0.5, "frequency_penalty": 0.5, "min_p": 0.1, "logit_bias": {"1": 2},
                            "user": "u", "max_tokens": 10},
     "everything else is dropped"),
    ("field_order_kept", {}, {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 5, "stop": "x"},
     "max_tokens and chat_template_kwargs keep their place when the body has them"),
    ("max_tokens_capped", {}, {"max_tokens": 99999}, "capped at gen_max_tokens"),
    ("max_tokens_at_cap", {}, {"max_tokens": 8192}, "exactly the cap"),
    ("max_tokens_huge_integer", {}, {"max_tokens": 10 ** 30}, "an integer of any size is capped"),
    ("max_tokens_one", {}, {"max_tokens": 1}, "the smallest"),
    ("max_tokens_custom_cap", {"gen_max_tokens": 100}, {"max_tokens": 5000}, "OPENJEV_GEN_MAX_TOKENS"),
    ("max_tokens_null", {}, {"max_tokens": None}, "null is the default"),
    ("max_completion_tokens", {}, {"max_completion_tokens": 50}, "the newer name"),
    ("max_completion_tokens_null", {}, {"max_completion_tokens": None}, "null is the default"),
    ("max_tokens_wins", {}, {"max_tokens": 3, "max_completion_tokens": "bad"},
     "max_tokens is read when present; max_completion_tokens is not checked"),
    ("max_tokens_null_hides_max_completion_tokens", {}, {"max_tokens": None, "max_completion_tokens": 5},
     "a present max_tokens wins even when null: the default applies"),
    ("max_tokens_string", {}, {"max_tokens": "abc"}, "upstream's test: a string"),
    ("max_tokens_true", {}, {"max_tokens": True}, "a bool is refused, not read as 1"),
    ("max_tokens_false", {}, {"max_tokens": False}, "False is refused"),
    ("max_tokens_float", {}, {"max_tokens": 3.9}, "a float is refused, not truncated"),
    ("max_tokens_integral_float", {}, {"max_tokens": 50.0}, "50.0 is a float"),
    ("max_tokens_zero", {}, {"max_tokens": 0}, "not positive"),
    ("max_tokens_negative", {}, {"max_tokens": -5}, "not positive"),
    ("max_tokens_huge_negative", {}, {"max_tokens": -(10 ** 30)}, "not positive"),
    ("max_tokens_list", {}, {"max_tokens": [1, "a", None, True, 2.5]}, "repr of a list"),
    ("max_tokens_object", {}, {"max_tokens": {"a": None, "b": "it's", "c": [False]}}, "repr of a dict"),
    ("max_tokens_quoted_string", {}, {"max_tokens": "it's \"50\"\n"}, "repr picks its quotes and escapes"),
    ("max_completion_tokens_string", {}, {"max_completion_tokens": "50"}, "upstream's test: the message names max_tokens"),
    ("thinking_on", {}, {"chat_template_kwargs": {"enable_thinking": True}}, "kept"),
    ("thinking_other_keys", {}, {"chat_template_kwargs": {"other": 1}}, "enable_thinking goes first"),
    ("thinking_null_kwargs", {}, {"chat_template_kwargs": None}, "null is no kwargs"),
    ("thinking_empty_list_kwargs", {}, {"chat_template_kwargs": []}, "a falsy value is no kwargs"),
    ("stream_forces_usage", {}, {"stream": True}, "stream_options.include_usage is forced on"),
    ("stream_usage_false_forced", {}, {"stream": True, "stream_options": {"include_usage": False, "x": 1}},
     "even when the client turned it off, keeping the other options in place"),
    ("stream_truthy", {}, {"stream": 1}, "any truthy stream streams"),
    ("stream_false_keeps_options", {}, {"stream": False, "stream_options": {"include_usage": False}},
     "without streaming the options pass through as they are"),
    ("stream_null_options", {}, {"stream": True, "stream_options": None}, "null options"),
    ("json_object_appends", {}, {"messages": [{"role": "system", "content": "Be brief.  \n "}, *MSG],
                                 "response_format": {"type": "json_object"}},
     "appended to a system string, right-stripped first"),
    ("json_object_inserts", {}, {"response_format": {"type": "json_object"}}, "a system message is inserted"),
    ("json_object_system_parts", {}, {"messages": [{"role": "system", "content": [{"type": "text", "text": "x"}]}, *MSG],
                                      "response_format": {"type": "json_object"}},
     "a system message whose content is not a string gets one inserted ahead of it"),
    ("json_object_developer", {}, {"messages": [{"role": "developer", "content": "Be brief."}, *MSG],
                                   "response_format": {"type": "json_object"}},
     "a developer message is not a system message"),
    ("json_object_system_unicode_space", {}, {"messages": [{"role": "system", "content": "Be brief.\u3000\x1c"}, *MSG],
                                              "response_format": {"type": "json_object"}},
     "rstrip() is str.rstrip(), the characters str.isspace() accepts"),
    ("json_schema", {}, {"response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": SCHEMA}}},
     "the schema written by json.dumps(ensure_ascii=False)"),
    ("json_schema_without_schema", {}, {"response_format": {"type": "json_schema", "json_schema": {"name": "c"}}},
     "no schema, no schema text"),
    ("json_schema_empty_schema", {}, {"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}},
     "an empty schema is falsy"),
    ("json_schema_null", {}, {"response_format": {"type": "json_schema", "json_schema": None}}, "null json_schema"),
    ("json_object_with_a_schema", {}, {"response_format": {"type": "json_object", "json_schema": {"schema": [1, "a"]}}},
     "json_object reads json_schema too, whatever the schema's type"),
    ("json_schema_string_schema", {}, {"response_format": {"type": "json_schema", "json_schema": {"schema": "any"}}},
     "a string schema is written as JSON"),
    ("response_format_text", {}, {"response_format": {"type": "text"}}, "not JSON mode"),
    ("response_format_null", {}, {"response_format": None}, "not JSON mode"),
    ("response_format_type_case", {}, {"response_format": {"type": "JSON_OBJECT"}}, "the type is matched exactly"),
    ("json_mode_keeps_other_messages", {}, {"messages": [{"role": "system", "content": "S", "name": "n"},
                                                         {"role": "user", "content": "U"}],
                                            "response_format": {"type": "json_object"}},
     "the system message keeps its other keys"),
    # Where upstream raises something other than ValueError, the route crashes with a 500; the port
    # refuses those requests with a 400 instead. Recorded so the departure is visible.
    ("crash_kwargs_list", {}, {"chat_template_kwargs": [1]}, "a non-mapping kwargs: TypeError"),
    ("crash_stream_options_string", {}, {"stream": True, "stream_options": "yes"}, "TypeError"),
    ("crash_response_format_string", {}, {"response_format": "json_object"}, "AttributeError"),
    ("crash_json_schema_string", {}, {"response_format": {"type": "json_schema", "json_schema": "s"}},
     "AttributeError"),
    ("json_mode_string_message", {}, {"messages": ["hi"], "response_format": {"type": "json_object"}},
     "dict('hi') raises ValueError, which upstream answers as a 400 with Python's message"),
    ("json_mode_number_message", {}, {"messages": [7], "response_format": {"type": "json_object"}},
     "dict(7) raises TypeError"),
]


def normalize_cases():
    rows = []
    for name, settings, fields, note in NORMALIZE_BODIES:
        body = {"model": GEN_MODEL, "messages": MSG, **fields}
        row = {"name": name, "note": note, "settings": recorded_settings(settings), "body": plain(body)}
        try:
            upstream, json_mode = Generator(Settings(**settings)).normalize(plain(body))
        except Exception as e:  # noqa: BLE001 - every outcome is recorded
            row["error"] = failure(e)
        else:
            row["upstream"] = plain(upstream)
            row["json_mode"] = json_mode
        rows.append(row)
    return rows


# extract_json.json ---------------------------------------------------------------------------

EXTRACT_TEXTS = [
    ("object", '{"city": "Zurich"}'),
    ("compact_object", '{"city":"Zurich","n":[1,2]}'),
    ("array", '[1, "two", 3.5]'),
    ("prose_around", 'Sure! Here it is: {"city": "Zurich"} Hope that helps.'),
    ("fenced_json", 'Sure!\n```json\n{"text": "Zurich"}\n```'),
    ("fenced_plain", '```\n{"text": "Zurich"}\n```'),
    ("fenced_uppercase_label", '```JSON\n{"text": "Zurich"}\n```'),
    ("fenced_without_newlines", '```json{"a": 1}```'),
    ("fence_inside_prose", 'Answer:\n```json\n{"a": 1}\n```\nDone.'),
    ("surrounding_whitespace", '  \n\t{"a": 1}\n  '),
    ("unicode_whitespace_and_fences", '\u3000```json\xa0{"a": 1}\u2003```\x85'),
    ("crlf_fences", '```json\r\n{"a": 1}\r\n```\r\n'),
    ("only_a_fence", '```'),
    ("empty_fenced_block", '```json\n```'),
    ("two_fences", '``````'),
    ("nothing_json", 'No JSON here, sorry.'),
    ("nothing_json_with_whitespace", '  no json  \n'),
    ("empty", ''),
    ("braces_in_prose", 'Use {name} as a placeholder, then {"a": 1}.'),
    ("invalid_then_valid", '{bad} [1, 2'),
    ("first_valid_wins", '[1, 2] {"a": 1}'),
    ("nested_first_bracket", 'x [{"a": [1, {"b": null}]}, true] y'),
    ("unterminated_then_inner", '{"a": {"b": 1}'),
    ("whitespace_inside", '{ "a" : [ 1 , 2 ] ,\n\t"b" : { } }'),
    ("empty_containers", '{} []'),
    ("empty_array_first", '[] {}'),
    ("duplicate_keys", '{"a": 1, "b": 2, "a": 3}'),
    ("escapes", '{"s": "quote \\" backslash \\\\ slash \\/ tab \\t nl \\n cr \\r bs \\b ff \\f nul \\u0000 del \\u007f"}'),
    ("unicode_escapes", '{"e": "\\u00e9\\u4e2d\\ud83d\\ude00", "k\\u00e9y": 1}'),
    ("raw_unicode", '{"city": "Zürich", "emoji": "👍🏽", "ls": "\u2028"}'),
    ("control_character_in_string", '{"a": "line\nbreak"} {"b": 2}'),
    ("numbers", '[0, -0, 1, -1, 1.0, -0.0, 0.1, 1e5, 1E5, 1e-5, 1.5e+300, 123456789012345678901234567890, 1e16, 2.5e-7]'),
    ("float_overflow", '[1e400, -1e400]'),
    ("nan_and_infinity", '{"a": NaN, "b": Infinity, "c": -Infinity}'),
    ("leading_zero_in_array", '[01] [1]'),
    ("long_integer", '[' + '9' * 5000 + '] {"ok": true}'),
    ("literals", '[true, false, null]'),
    ("bad_literal", '[True] [true]'),
    ("trailing_comma", '[1, 2,] {"a": [1]}'),
    ("single_quotes", "{'a': 1} [\"b\"]"),
    ("deep_nesting", '[' * 200 + ']' * 200),
    ("string_with_brackets", '{"text": "[not an array] {nor an object}"}'),
    ("closing_fence_only", '{"a": 1}\n```'),
    ("opening_fence_only", '```json\n{"a": 1}'),
]


def extract_cases():
    return [{"name": name, "text": text, "result": extract_json(text)} for name, text in EXTRACT_TEXTS]


# routes.json ---------------------------------------------------------------------------------

class DeferredFuture(concurrent.futures.Future):
    """A call that starts on its worker only once a done callback is registered."""

    def __init__(self, pool, fn, args):
        super().__init__()
        self._start = (pool, fn, args)

    def add_done_callback(self, fn):
        super().add_done_callback(fn)
        start, self._start = self._start, None
        if start is not None:
            pool, call, args = start
            pool.submit(self._run, call, args)

    def _run(self, call, args):
        if not self.set_running_or_notify_cancel():
            return
        try:
            result = call(*args)
        except BaseException as e:  # noqa: BLE001 - handed to the awaiting coroutine
            self.set_exception(e)
        else:
            self.set_result(result)


class OrderedExecutor:
    """The stub runtimes' one worker thread, with the completion order of upstream's CPython.

    MlxGenerator.stream relies on its end marker being queued after every chunk the runtime
    emitted with call_soon_threadsafe. On CPython 3.12, which upstream's container runs,
    run_in_executor's future always completes through call_soon_threadsafe too, so it does. On
    3.14, which writes these fixtures, a call that has already finished when wrap_future
    registers its callback completes the future at once (asyncio.futures._chain_future), so the
    end marker can overtake the last chunks: a stub generates far faster than a model, and upstream's
    own one-token stream lost its only chunk on a fresh app in about half of 30 tries. Starting each
    call only once its callback is registered restores 3.12's order, so a recording is the same on
    every run."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, call, *args):
        return DeferredFuture(self.pool, call, args)

    def shutdown(self, wait=True, cancel_futures=False):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class StubRuntime:
    """upstream's tests/test_mlx_backend.py StubRuntime, for generation: a fixed reply, one token a
    word, then a last segment with no token, which the real runtime's detokenizer emits when it
    stops and nobody is billed for."""

    REPLY = ['{"city"', ': "', "Zurich", '"']
    TAIL = "}"

    def __init__(self, model_path):
        self.pool = OrderedExecutor()
        self.prompt_cache_entries = None
        self.generations = []

    def set_cache_limit(self, gb):
        return None

    def record(self, prompt, max_tokens, stop_ids, skip_special):
        self.generations.append({"prompt": [int(i) for i in prompt], "max_tokens": max_tokens,
                                 "stop_ids": [int(i) for i in stop_ids],
                                 "skip_special": [int(i) for i in (skip_special or ())]})

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.record(prompt, max_tokens, stop_ids, skip_special)
        ids, finish = [], "stop"
        for i, text in enumerate(self.REPLY):
            if len(ids) >= max_tokens:
                finish = "length"
                break
            token = 1000 + i
            if token in stop_ids:
                break
            ids.append(token)
            if not emit(text, token):
                return ids, len(prompt), "cancelled"
        emit(self.TAIL, None)
        return ids, len(prompt), finish

    def close(self):
        self.pool.shutdown()


# The thought-channel markers reach the detokenizer fused into a later segment, as upstream's test
# recorded them from a leaking run against real weights.
LEAKY_EMITS = [(100, ""), (45518, ""), (107, ""), (101, ""), (34699, ""),
               (6819, "<|channel>thought\n<channel|>six"), (6589, " seven"),
               (10155, " eight"), (3595, " nine"), (None, " ten")]
CLEAN_EMITS = [(34699, ""), (6819, "six"), (6589, " seven"),
               (10155, " eight"), (3595, " nine"), (None, " ten")]
MARKER_IDS = [100, 45518, 107, 101]


class ReplayRuntime(StubRuntime):
    """upstream's ReplayRuntime: the recorded emits of a reply that opened a thought channel, clean
    when the runtime is asked to skip the marker ids, as the real generator is."""

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.record(prompt, max_tokens, stop_ids, skip_special)
        skipping = set(skip_special or ()) >= set(MARKER_IDS)
        ids = []
        for token, text in (CLEAN_EMITS if skipping else LEAKY_EMITS):
            if token is None:
                emit(text, None)
                break
            ids.append(token)
            if not emit(text, token):
                return ids, len(prompt), "cancelled"
        return ids, len(prompt), "stop"


class OneTokenRuntime(StubRuntime):
    """upstream's OneTokenRuntime: one token with no text of its own, flushed with the end."""

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        self.record(prompt, max_tokens, stop_ids, skip_special)
        emit("", 1000)
        emit("7", None)
        return [1000], len(prompt), "stop"


RUNTIMES = {"stub": StubRuntime, "replay": ReplayRuntime, "one_token": OneTokenRuntime}
RUNTIME_NOTES = {
    "stub": "StubRuntime: emits '{\"city\"', ': \"', 'Zurich', '\"' as tokens 1000 to 1003, stopping early at a "
            "token in stop_ids or at max_tokens (finish 'length'), then '}' with no token; finish 'stop'.",
    "replay": "ReplayRuntime: emits the recorded segments of a reply that opened a thought channel, "
              "clean when skip_special holds the four marker ids; finish 'stop'.",
    "one_token": "OneTokenRuntime: emits '' for token 1000, then '7' with no token; finish 'stop'.",
}

JSON_HEADERS = {"content-type": "application/json"}
CHAT = {"model": GEN_MODEL, "messages": [CITY]}
LONG = {"model": GEN_MODEL, "messages": [{"role": "user", "content": "word " * 200}]}


def single_token_stop(tok):
    """A stop string that is the stub's third token (1002) alone, so a stop shows in a reply."""
    text = tok.decode([1002])
    if tok.encode(text, add_special_tokens=False) != [1002]:
        raise SystemExit(f"routes: token 1002 ({text!r}) does not round-trip as one token")
    return text


def route_runs(tok):
    """(settings, runtime, [(name, body, headers, running, note)]) in recording order. A body is a
    dict sent as compact JSON, or the exact text or bytes to send."""
    stop = single_token_stop(tok)
    default = [
        ("completion", CHAT, None, None, "upstream's test_chat_completion_on_mlx"),
        ("completion_model_alias", dict(CHAT, model="diffusiongemma"), None, None, "the second accepted name"),
        ("completion_json_mode", dict(CHAT, response_format={"type": "json_object"}, temperature=0.7, seed=3,
                                      max_tokens=99999), None, None, "upstream's test_chat_json_mode_and_max_tokens"),
        ("completion_jev_ultrafast", {
            "model": GEN_MODEL, "max_tokens": 1024, "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False}, "temperature": 0.2, "seed": 1,
            "messages": [{"role": "system", "content": "Return {\"text\": ...}"}, {"role": "user", "content": "{}"}]},
         None, None, "upstream's test_chat_normalizes_jev_ultrafast_request, on the MLX backend"),
        ("completion_json_schema", dict(CHAT, response_format={"type": "json_schema", "json_schema": {
            "name": "c", "schema": {"type": "object", "properties": {"city": {"type": "string"}}}}},
            max_completion_tokens=50), None, None, "the schema in the instruction; max_completion_tokens"),
        ("completion_length", dict(CHAT, max_tokens=2), None, None, "the reply stops at max_tokens: finish length"),
        ("completion_stop", dict(CHAT, stop=[stop, "two tokens here", "\n\n"]), None, None,
         "a single-token stop string ends the reply; multi-token ones are dropped"),
        ("completion_stop_string", dict(CHAT, stop=stop), None, None, "stop as one string"),
        ("completion_thinking", dict(CHAT, chat_template_kwargs={"enable_thinking": True}), None, None,
         "thinking on in the prompt"),
        ("completion_max_tokens_null", dict(CHAT, max_tokens=None), None, None, "null is the default"),
        ("completion_max_completion_tokens", dict(CHAT, max_completion_tokens=50), None, None,
         "upstream's test_chat_max_tokens_must_be_an_integer: 50 is sent as max_tokens"),
        ("completion_without_content_type", CHAT, {}, None, "request.json() reads any body"),
        ("completion_text_plain", CHAT, {"content-type": "text/plain"}, None, "whatever its content type"),
        ("completion_byte_order_mark", b"\xef\xbb\xbf" + compact(CHAT).encode(), None, None,
         "json.loads drops a UTF-8 byte order mark"),
        ("completion_nan_in_body", '{"model":"diffusiongemma-26b","messages":[{"role":"user","content":"Which city?"}],'
                                   '"top_p":NaN}', None, None,
         "json.loads reads NaN; the port's parser refuses it (D-016)"),
        ("stream", dict(CHAT, stream=True, stream_options={"include_usage": True}), None, None,
         "upstream's test_chat_stream_on_mlx"),
        ("stream_usage_forced", dict(CHAT, stream=True), None, None, "include_usage is forced on"),
        ("stream_usage_false", dict(CHAT, stream=True, stream_options={"include_usage": False}), None, None,
         "even when the client turned it off"),
        ("stream_json_mode", dict(CHAT, stream=True, response_format={"type": "json_object"}), None, None,
         "a stream is not reduced to the JSON object"),
        ("stream_length", dict(CHAT, stream=True, max_tokens=2), None, None, "finish length in the finish chunk"),
        ("stream_truthy", dict(CHAT, stream=1), None, None, "any truthy stream streams"),
        ("invalid_json", "{", None, None, "request.json() raises"),
        ("empty_body", "", None, None, "an empty body is not JSON"),
        ("body_array", "[1]", None, None, "not an object"),
        ("body_string", '"hi"', None, None, "not an object"),
        ("long_integer_in_body", '{"model":"diffusiongemma-26b","messages":[],"n":' + "9" * 5000 + "}", None, None,
         "int() refuses more than 4,300 digits: a ValueError"),
        ("messages_missing", {"model": GEN_MODEL}, None, None, "no messages"),
        ("messages_empty", {"model": GEN_MODEL, "messages": []}, None, None, "empty messages"),
        ("messages_not_an_array", {"model": GEN_MODEL, "messages": "hi"}, None, None, "a string"),
        ("messages_object", {"model": GEN_MODEL, "messages": {"role": "user"}}, None, None, "an object"),
        ("messages_checked_before_model", {"model": 42, "messages": []}, None, None, "messages first"),
        ("model_missing", {"messages": CHAT["messages"]}, None, None, "upstream's test_chat_model_is_required"),
        ("model_null", dict(CHAT, model=None), None, None, "null"),
        ("model_number", dict(CHAT, model=42), None, None, "not a string"),
        ("model_unknown", dict(CHAT, model="gpt-4"), None, None, "upstream's test_chat_rejects_an_unknown_model"),
        ("model_unknown_quotes", dict(CHAT, model="it's \"x\""), None, None, "repr of the name"),
        ("model_unknown_non_ascii", dict(CHAT, model="modèle\n"), None, None, "repr of the name"),
        ("model_case_matters", dict(CHAT, model="DiffusionGemma-26B"), None, None, "names match exactly"),
        ("model_checked_before_max_tokens", dict(CHAT, model="gpt-4", max_tokens="abc"), None, None, "404 first"),
        ("max_tokens_string", dict(CHAT, max_tokens="abc"), None, None, "upstream's test"),
        ("max_tokens_true", dict(CHAT, max_tokens=True), None, None, "upstream's test"),
        ("max_tokens_float", dict(CHAT, max_tokens=3.9), None, None, "upstream's test"),
        ("max_tokens_zero", dict(CHAT, max_tokens=0), None, None, "upstream's test"),
        ("max_tokens_negative", dict(CHAT, max_tokens=-5), None, None, "upstream's test"),
        ("max_completion_tokens_string", dict(CHAT, max_completion_tokens="50"), None, None, "upstream's test"),
        ("capacity", CHAT, None, 40, "upstream's test_chat_capacity_is_refused: running at 8 + 32"),
        ("capacity_before_max_tokens", dict(CHAT, max_tokens="abc"), None, 40, "the 529 comes before max_tokens"),
        ("model_before_capacity", dict(CHAT, model="gpt-4"), None, 40, "the 404 comes before the 529"),
        ("capacity_streaming", dict(CHAT, stream=True), None, 40, "a stream is refused the same way"),
        ("below_capacity", CHAT, None, 39, "one place left"),
        # upstream failures: the route raises and Starlette answers a bare 500
        ("message_without_role", {"model": GEN_MODEL, "messages": [{"content": "hi"}]}, None, None,
         "upstream failure: the template raises UndefinedError"),
        ("message_not_an_object", {"model": GEN_MODEL, "messages": ["hi"]}, None, None,
         "upstream failure: the template raises UndefinedError"),
        ("message_role_not_a_string", {"model": GEN_MODEL, "messages": [{"role": 5, "content": "hi"}]}, None, None,
         "upstream failure: the template raises TypeError"),
        ("json_mode_string_message", dict(CHAT, messages=["hi"], response_format={"type": "json_object"}),
         None, None, "dict('hi') raises a ValueError, answered as a 400 with Python's message"),
        ("response_format_string", dict(CHAT, response_format="json_object"), None, None,
         "upstream failure: normalize raises AttributeError"),
        ("stop_number", dict(CHAT, stop=5), None, None, "upstream failure: stop_ids raises TypeError"),
        ("chat_template_kwargs_list", dict(CHAT, chat_template_kwargs=[1]), None, None,
         "upstream failure: normalize raises TypeError"),
    ]
    return [
        ({"backend": "mlx", "mlx_model": "/models/dg"}, "stub", default),
        ({"backend": "mlx", "mlx_model": "/models/dg", "gen_max_tokens": 100}, "stub", [
            ("max_tokens_custom_cap", dict(CHAT, max_tokens=5000), None, None, "OPENJEV_GEN_MAX_TOKENS caps it"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg", "mlx_max_prompt": 8}, "stub", [
            ("prompt_over_limit", LONG, None, None, "upstream's test_chat_refuses_an_over_long_prompt"),
            ("prompt_over_limit_stream", dict(LONG, stream=True), None, None, "refused before the stream starts"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg", "mlx_max_prompt": 15}, "stub", [
            ("prompt_one_over_limit", CHAT, None, None, "CHAT's prompt is 16 tokens with the scaffold"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg", "mlx_max_prompt": 16}, "stub", [
            ("prompt_at_limit", CHAT, None, None, "exactly the limit is served"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg", "api_key": "secret"}, "stub", [
            ("api_key_missing", CHAT, None, None, "the /v1/ middleware's 403, in Jev's error shape"),
            ("api_key_given", CHAT, {"content-type": "application/json", "authorization": "Bearer secret"}, None,
             "served"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg", "max_body_bytes": 100}, "stub", [
            ("body_over_the_cap", LONG, None, None, "the /v1/ middleware's 413, in Jev's error shape"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg"}, "replay", [
            ("thought_channel_completion", CHAT, None, None, "upstream's test_the_thought_channel_never_reaches_a_chat_client"),
            ("thought_channel_stream", dict(CHAT, stream=True), None, None,
             "upstream's test_the_thought_channel_never_reaches_a_streaming_client"),
        ]),
        ({"backend": "mlx", "mlx_model": "/models/dg"}, "one_token", [
            ("one_token_stream", dict(CHAT, stream=True), None, None, "upstream's test_a_one_token_reply_still_streams"),
            ("one_token_completion", CHAT, None, None, "the same reply whole"),
        ]),
    ]


def send(client, body, headers):
    if isinstance(body, bytes):
        content, text = body, None
    elif isinstance(body, str):
        content, text = body.encode("utf-8"), body
    else:
        text = compact(body)
        content = text.encode("utf-8")
    request = {"method": "POST", "path": "/v1/chat/completions", "headers": headers}
    if text is not None:
        request["body_text"] = text
    else:
        request["body_base64"] = base64.b64encode(content).decode("ascii")
    return request, client.post("/v1/chat/completions", content=content, headers=headers)


REQUEST_ID = re.compile(r"req_[0-9a-f]{32}")


def route_cases(tok):
    rows = []
    fixed_time = types.SimpleNamespace(time=lambda: CREATED + 0.75)
    saved = (oj_mlx.MlxRuntime, oj_chat.completion_id, oj_chat.time)
    try:
        oj_chat.completion_id = lambda: COMPLETION_ID
        oj_chat.time = fixed_time
        for settings, runtime, exchanges in route_runs(tok):
            oj_mlx.MlxRuntime = RUNTIMES[runtime]
            # An upstream failure is recorded as the bare 500 Starlette answers, not raised here.
            app = create_app(Settings(**settings), tokenizer=tok)
            with TestClient(app, raise_server_exceptions=False) as client:
                gen = client.app.state.generator
                generations = client.app.state.engine.runtime.generations
                for name, body, headers, running, note in exchanges:
                    headers = JSON_HEADERS if headers is None else headers
                    before = len(generations)
                    previous = gen.running
                    if running is not None:
                        gen.running = running
                    RECORDING[0] = {"prompts": [], "encodings": []}
                    try:
                        request, r = send(client, body, headers)
                    finally:
                        recorded = RECORDING[0]
                        RECORDING[0] = None
                        if running is not None:
                            gen.running = previous
                    rid = r.headers.get("x-request-id")
                    if rid is not None and not (REQUEST_ID.fullmatch(rid)
                                                and r.headers.get("x-typesafe-request-id") == rid):
                        raise SystemExit(f"routes: {name}: unexpected request id headers {dict(r.headers)}")
                    response_headers = {"content-type": r.headers.get("content-type")}
                    if "retry-after" in r.headers:
                        response_headers["retry-after"] = r.headers["retry-after"]
                    rows.append({
                        "name": name, "note": note, "settings": recorded_settings(settings), "runtime": runtime,
                        "running": running, "request": request,
                        "response": {"status": r.status_code, "headers": response_headers,
                                     "body_text": r.content.decode("utf-8")},
                        "request_id_headers": rid is not None,
                        "server_timing_present": "server-timing" in r.headers,
                        "prompts": recorded["prompts"], "encodings": recorded["encodings"],
                        "generations": generations[before:],
                    })
    finally:
        oj_mlx.MlxRuntime, oj_chat.completion_id, oj_chat.time = saved
    return rows


# -------------------------------------------------------------------------------------------

def check_pins():
    text = (ROOT / "Tools" / "fixtures" / "upstream_tables.py").read_text(encoding="utf-8")
    for name, value in (("TOKENIZER_REPO", TOKENIZER_REPO), ("TOKENIZER_REVISION", TOKENIZER_REVISION),
                        ("UPSTREAM_COMMIT", UPSTREAM_COMMIT)):
        if f'\n{name} = "{value}"\n' not in text:
            raise SystemExit(f"{name} differs from Tools/fixtures/upstream_tables.py's; keep the pins together")
    head = upstream_head()
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")


def main():
    check_pins()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    eng = Engine(Settings(), tok)
    markers = eng.thought_open + eng.thought_close
    if markers != MARKER_IDS:
        raise SystemExit(f"the thought-channel markers are {markers}, not {MARKER_IDS}")
    write("chat-completions/prompts.json", {"scaffold": eng.scaffold, "cases": prompt_cases(tok, eng)})
    write("chat-completions/normalize.json", {"cases": normalize_cases()})
    write("chat-completions/extract_json.json", {"cases": extract_cases()})
    write("chat-completions/routes.json", {
        "completion_id": COMPLETION_ID, "created": CREATED, "markers": markers, "scaffold": eng.scaffold,
        "runtimes": RUNTIME_NOTES, "cases": route_cases(tok)})
    total = sum(size for _, size in WRITTEN)
    print(f"{len(WRITTEN)} files, {total} bytes")


if __name__ == "__main__":
    main()
