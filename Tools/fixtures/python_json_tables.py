#!/usr/bin/env python3
"""Write the CPython reference tables for OpenJevCore's JSON writer, parser and decode errors.

The tables record what CPython's json module produces, so that the Swift tests compare against
Python itself rather than against expectations written by hand. The script uses only the standard
library and a fixed seed, so running it twice with the same Python version gives the same files.

Usage (from the repository root):

    python3 Tools/fixtures/python_json_tables.py

The output goes to Fixtures/python-json/. Each file records the Python version that wrote it.
"""

import base64
import json
import math
import random
import struct
import sys
from pathlib import Path

SEED = 20260929
FLOAT_COUNT = 5000
OUTPUT = Path(__file__).resolve().parents[2] / "Fixtures" / "python-json"


def generator():
    return {
        "script": "Tools/fixtures/python_json_tables.py",
        "python": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "seed": SEED,
    }


def bits_hex(x):
    return struct.pack(">d", x).hex()


def float_table(rng):
    edge = [
        0.0, -0.0, 0.1 + 0.2, 0.1, 0.5, 1.0, -1.0, 100.0, 1e15, 1e16, 1e-4, 1e-5, 1e17, 1e22, 1e23,
        9999999999999998.0, 999999999999999.9, 0.00011, 0.000099, 2.0**53, 2.0**53 + 2, 2.0**63,
        2.0**64, 5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308,
        1.7976931348623157e308, -1.7976931348623157e308, 123456789012345678.0, 1.5e300, 1e-7,
        math.pi, math.e, 1 / 3, 2 / 3, 1e100, 1e-100, 4.35, 0.3, 1234.5678, 1e21, 1e-310,
    ]
    values = list(edge)
    seen = {bits_hex(x) for x in values}
    while len(values) < FLOAT_COUNT:
        kind = rng.randrange(4)
        if kind == 0:
            # Any finite bit pattern: covers every exponent range, subnormals included.
            bits = rng.getrandbits(64)
            if (bits >> 52) & 0x7FF == 0x7FF:
                continue
            x = struct.unpack(">d", struct.pack(">Q", bits))[0]
        elif kind == 1:
            # Short decimals, the values people actually write, near every layout boundary.
            digits = rng.randrange(1, 8)
            mantissa = rng.randrange(10 ** (digits - 1), 10**digits)
            x = float(f"{mantissa}e{rng.randrange(-25, 25)}")
        elif kind == 2:
            # Integral values around 2**53 and the fixed to exponential switch at 1e16.
            x = float(rng.randrange(1, 10 ** rng.randrange(1, 20)))
        else:
            x = rng.uniform(-1000, 1000)
        if rng.randrange(2):
            x = -x
        key = bits_hex(x)
        if key in seen:
            continue
        seen.add(key)
        values.append(x)
    return [{"hex": bits_hex(x), "repr": repr(x), "dumps": json.dumps(x)} for x in values]


def string_table():
    samples = [
        "",
        "plain ascii",
        'double "quotes" inside',
        "single 'quotes' inside",
        "back\\slash and \\\\ double",
        "slash / is never escaped",
        "".join(chr(c) for c in range(0x20)),
        "tab\tnewline\ncarriage\rbackspace\bformfeed\f",
        "DEL \x7f and C1 \x80 \x9f",
        "line separator   paragraph separator  ",
        "café",
        "café",
        "中文字符 日本語 한국어",
        "emoji \U0001F600 \U0001F44D\U0001F3FD family \U0001F468‍\U0001F469‍\U0001F467",
        "astral \U00010000 \U0010FFFF \U0001D11E",
        "combining à́̂ and Hangul 각",
        "BOM ﻿ and noncharacter ￾ ￿",
        "NUL \x00 in the middle",
        "mixed: \"\\\né\U0001F600\x01",
        "right-to-left שלום مرحبا",
        "long " + "x" * 600,
    ]
    return [
        {
            "scalars": [ord(c) for c in s],
            "dumps": json.dumps(s),
            "dumps_unicode": json.dumps(s, ensure_ascii=False),
        }
        for s in samples
    ]


def document_texts(rng):
    texts = {
        "scalars": '[null, true, false, 0, -0, 1, -1, 12345678901234567890123456789, 1.0, -0.0, '
        '1e2, 1E-7, 2.5e+300, 0.1, 100, 1.5]',
        "nested": '{"state": {"name": "Ada", "tags": ["a", "b", {"deep": [1, [2, [3, {}]]]}], '
        '"empty": {}, "list": []}, "questions": {"q2": {"type": "yesno"}, '
        '"q1": {"type": "choice", "criteria": {"z": null, "a": "first"}}}}',
        "non_ascii_keys": '{"été": 1, "zebra": 2, "中": 3, "\U0001F600": 4, '
        '"A": 5, "a": 6, "É": 7}',
        "equivalent_keys": '{"caf\\u00e9": "precomposed", "cafe\\u0301": "decomposed", '
        '"é": 1, "é": 2}',
        "duplicate_keys": '{"a": 1, "b": 2, "a": 3, "c": {"x": 1, "x": [2]}, "b": 4}',
        "escapes": '{"s": "\\"\\\\\\/\\b\\f\\n\\r\\t\\u0000\\u001f\\u007f\\u2028\\ud83d\\ude00"}',
        "seed_key": '[{"text": "The cat sat."}, {"q1": {"type": "yesno", "description": '
        '"Is there a cat?"}, "q0": {"type": "score", "criteria": [1, 2.5, "high"]}}]',
        "string_state": '"just a string state"',
        "floats": '[0.30000000000000004, 1e15, 1e16, 0.0001, 0.00001, 123456789012345678, '
        '123456789012345678.0, 5e-324, 1.7976931348623157e308]',
    }
    for i in range(12):
        texts[f"random_{i:02d}"] = json.dumps(random_value(rng, 0), ensure_ascii=bool(i % 2))
    return texts


def random_string(rng):
    alphabet = ["a", "b", "Z", " ", '"', "\\", "\n", "\x01", "é", "é", "中",
                "\U0001F600", " ", "/", "\x7f"]
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 8)))


def random_value(rng, depth):
    kind = rng.randrange(8 if depth < 4 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.choice([True, False])
    if kind == 2:
        return rng.randrange(-(10**20), 10**20)
    if kind == 3:
        bits = rng.getrandbits(64)
        x = struct.unpack(">d", struct.pack(">Q", bits))[0]
        return x if math.isfinite(x) else rng.random()
    if kind in (4, 5):
        return random_string(rng)
    if kind == 6:
        return [random_value(rng, depth + 1) for _ in range(rng.randrange(0, 5))]
    return {random_string(rng): random_value(rng, depth + 1) for _ in range(rng.randrange(0, 5))}


def document_table(rng):
    rows = []
    for name, text in document_texts(rng).items():
        value = json.loads(text)
        rows.append({
            "name": name,
            "text": text,
            "dumps": json.dumps(value),
            "dumps_sorted": json.dumps(value, sort_keys=True),
            "dumps_compact": json.dumps(value, separators=(",", ":")),
            "dumps_unicode": json.dumps(value, ensure_ascii=False),
        })
    return rows


# Documents for decode_errors.json: what json.loads does with the bytes of a request body. The
# named ones cover every message CPython's scanner (Modules/_json.c) raises and the places where
# it is more lenient than RFC 8259; the rest are seeded mutations of request-like documents.
NAMED_DOCUMENTS = [
    ("empty", b""),
    ("whitespace_only", b" \t\r\n "),
    ("open_object", b"{"),
    ("open_array", b"["),
    ("open_string", b'"'),
    ("key_without_colon", b'{"a"'),
    ("colon_without_value", b'{"a":'),
    ("value_without_close", b'{"a":1'),
    ("comma_without_key", b'{"a":1,'),
    ("object_trailing_comma", b'{"a":1,}'),
    ("object_trailing_comma_spaced", b'{"a":1 , }'),
    ("array_trailing_comma", b"[1,]"),
    ("array_trailing_comma_newline", b"[1,\n]"),
    ("array_comma_without_value", b"[1,"),
    ("array_without_close", b"[1"),
    ("array_missing_comma", b"[1 2]"),
    ("array_double_comma", b"[1,,2]"),
    ("array_leading_comma", b"[,1]"),
    ("object_leading_comma", b"{,}"),
    ("object_double_comma", b'{"a":1,,"b":2}'),
    ("object_double_colon", b'{"a"::1}'),
    ("object_number_key", b"{1:2}"),
    ("object_single_quotes", b"{'a':1}"),
    ("object_missing_colon", b'{"a" 1}'),
    ("object_missing_comma", b'{"a":1 "b":2}'),
    ("mismatched_close_array", b'{"a":[1}'),
    ("mismatched_close_object", b'{"a":{"b":1]}'),
    ("extra_data", b"{} {}"),
    ("extra_close", b"[1]]"),
    ("extra_word", b"true false"),
    ("trailing_whitespace", b"{}  \n"),
    ("trailing_nul", b'{"a":1}\x00'),
    ("lone_close_bracket", b"]"),
    ("lone_close_brace", b"}"),
    ("lone_comma", b","),
    ("lone_colon", b":"),
    ("truncated_null", b"nul"),
    ("truncated_true", b"tru"),
    ("truncated_false", b"fals"),
    ("misspelled_null", b"nulx"),
    ("null_in_array", b"[null]"),
    ("minus_alone", b"-"),
    ("minus_letter", b"-x"),
    ("double_minus", b"--1"),
    ("leading_zero", b"01"),
    ("leading_zero_in_array", b"[01]"),
    ("fraction_without_digits", b"1."),
    ("fraction_before_exponent", b"1.e5"),
    ("exponent_without_digits", b"1e"),
    ("exponent_sign_without_digits", b"1e+"),
    ("exponent_without_digits_in_array", b"[1e+]"),
    ("exponent_without_digits_in_object", b'{"a":1.5e}'),
    ("number_then_letter", b"[1e5x]"),
    ("leading_dot", b".5"),
    ("leading_plus", b"+1"),
    ("minus_in_object", b'{"a":-}'),
    ("negative_zero", b'{"a":-0}'),
    ("nan", b"NaN"),
    ("infinity", b"Infinity"),
    ("negative_infinity", b"-Infinity"),
    ("nan_in_array", b"[NaN, Infinity, -Infinity]"),
    ("truncated_infinity", b"[Infinit"),
    ("lowercase_nan", b"nan"),
    ("lowercase_inf", b"inf"),
    ("float_overflow", b"1e400"),
    ("float_overflow_negative", b"[-1e400]"),
    ("integer_4300_digits", b"1" * 4300),
    ("integer_4301_digits", b"1" * 4301),
    ("negative_integer_4300_digits", b"-" + b"1" * 4300),
    ("negative_integer_4301_digits", b"-" + b"1" * 4301),
    ("float_4301_digits", b"1" * 4301 + b".5"),
    ("exponent_4301_digits", b"1" * 4301 + b"e5"),
    ("long_integer_then_error", b"[" + b"1" * 4301 + b",}"),
    ("error_then_long_integer", b"[1,}" + b"1" * 4301),
    ("lone_high_surrogate", b'"\\ud83d"'),
    ("lone_low_surrogate", b'"\\ude00"'),
    ("high_surrogate_then_letter", b'"\\ud83dx"'),
    ("high_surrogate_then_escape", b'"\\ud83d\\u0041"'),
    ("high_surrogate_then_short_escape", b'"\\ud83d\\u12"'),
    ("high_surrogate_then_bad_escape", b'"\\ud83d\\uZZZZ"'),
    ("surrogate_pair", b'"\\ud83d\\ude00"'),
    ("short_unicode_escape", b'"\\u12"'),
    ("bad_unicode_escape", b'"\\u12G4"'),
    ("empty_unicode_escape", b'"\\u"'),
    ("unicode_escape_then_end", b'"\\u0041'),
    ("unicode_escape_cut", b'"\\u004'),
    ("unicode_escape_with_non_ascii", '"\\u00é"'.encode()),
    ("unicode_escape_cut_with_non_ascii", '"\\u0é'.encode()),
    ("invalid_escape", b'"\\x"'),
    ("backslash_at_end", b'"\\'),
    ("backslash_at_end_after_text", b'"a\\'),
    ("control_character", b'"\x01"'),
    ("tab_in_string", b'"\t"'),
    ("newline_in_string", b'"\n"'),
    ("escaped_nul", b'{"a":"b\\u0000"}'),
    ("non_ascii_then_control", '"é\x02"'.encode()),
    ("non_ascii_then_invalid_escape", '"é\\q"'.encode()),
    ("non_ascii_key_then_bad_value", '{"é": x}'.encode()),
    ("non_ascii_trailing_comma", '["é", ]'.encode()),
    ("non_ascii_value", "é".encode()),
    ("non_ascii_in_object", '{"a":é}'.encode()),
    ("line_separator_in_array", "[ ]".encode()),
    ("astral_then_error", '{"😀": "中文" x}'.encode()),
    ("vertical_tab", b"[\x0b1]"),
    ("form_feed", b"[1\x0c]"),
    ("no_break_space_after", '{"a":1} '.encode()),
    ("no_break_space_before", " {}".encode()),
    ("byte_order_mark", b"\xef\xbb\xbf{}"),
    ("byte_order_mark_then_error", b"\xef\xbb\xbf{,}"),
    ("byte_order_mark_alone", b"\xef\xbb\xbf"),
    ("two_byte_order_marks", b"\xef\xbb\xbf\xef\xbb\xbf{}"),
    ("invalid_utf8", b'{"state": "\xff"}'),
    ("invalid_utf8_after_error", b'{,"\xff"}'),
    ("truncated_utf8", b'{"state": "\xc3'),
    ("overlong_utf8", b'"\xc0\xaf"'),
    ("wire_truncated", b'{"state": '),
    ("wire_single_quotes", b"{'state': 'x'}"),
    ("wire_trailing_comma", b'{"state": "x",}'),
    ("wire_missing_colon", b'{"state" "x"}'),
    ("wire_missing_comma", b'{"state": "x" "model": "y"}'),
    ("wire_extra_data", b"{} {}"),
    ("wire_unterminated_string", b'{"state": "x'),
    ("wire_control_character", b'{"state": "a\x01b"}'),
    ("wire_invalid_escape", b'{"state": "\\q"}'),
    ("wire_non_ascii_before_error", '{"state": "éé", }'.encode()),
    ("deep_arrays_truncated", b"[" * 3000 + b"]" * 2999),
    ("deep_objects_then_error", b'{"a":' * 2000 + b"1" + b"}" * 1999 + b","),
    ("deep_arrays", b"[" * 3000 + b"]" * 3000),
]

MUTATION_SEEDS = [
    b'{"state":"x","model":"jev-latest","questions":{"a":{"type":"noul"}}}',
    b'{"state": {"name": "Ada", "tags": ["a", "b", {"deep": [1, [2, [3, {}]]]}]}, '
    b'"n": -12.5e+3, "t": true, "f": false, "z": null}',
    '{"é": "中文 😀", "k": ["\\u00e9\\ud83d\\ude00", "\\n\\t\\"\\\\\\/"]}'.encode(),
    b'[1, 2.5, -0, 0e1, 1E-2, "x", [], {}, [[]], {"a": {}}]',
    b'  {"a" : 1 ,\n "b":[ 1 , 2 ] }  ',
    b'"just a string"',
    b"123",
    b"-1.5e10",
    b"[true, null, NaN, Infinity, -Infinity]",
]

MUTATION_PIECES = [
    b'"', b"{", b"}", b"[", b"]", b":", b",", b" ", b"\n", b"\\", b"u", b"0", b"1", b"9", b"-",
    b"+", b".", b"e", b"E", b"n", b"t", b"f", b"a", b"l", b"s", b"N", b"I", b"x", b"\x01",
    b"\x1f", b"\x7f", "é".encode(), "😀".encode(), b"\xff", b"\t", b"\r", b"/", b"b", b"r",
]

MUTATION_COUNT = 600


def mutated(rng):
    """A request-like document with one to three random edits."""
    s = bytearray(rng.choice(MUTATION_SEEDS))
    for _ in range(rng.randrange(1, 4)):
        op = rng.randrange(5)
        p = rng.randrange(len(s) + 1)
        if op == 0 and s:
            del s[min(p, len(s) - 1)]
        elif op == 1:
            s[p:p] = rng.choice(MUTATION_PIECES)
        elif op == 2 and s:
            i = min(p, len(s) - 1)
            s[i:i + 1] = rng.choice(MUTATION_PIECES)
        elif op == 3:
            s = s[:p]
        else:
            q = rng.randrange(len(s) + 1)
            s[p:p] = s[min(p, q):max(p, q)][:10]
    return bytes(s)


def reads_as_utf8(raw):
    """Whether json.loads reads the bytes as UTF-8 with strict decoding: it also detects UTF-16
    and UTF-32, and its surrogatepass decoding lets encoded surrogates through, which the Swift
    port does not reproduce (decision D-031)."""
    if json.detect_encoding(raw) not in ("utf-8", "utf-8-sig"):
        return False
    doc = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    try:
        doc.decode("utf-8")
    except UnicodeDecodeError:
        try:
            doc.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            return True
        return False
    return True


def loads_outcome(raw):
    """What json.loads(raw) does, or None when it runs out of stack (RecursionError)."""
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        return {"outcome": "JSONDecodeError", "msg": e.msg, "pos": e.pos}
    except UnicodeDecodeError:
        return {"outcome": "UnicodeDecodeError"}
    except RecursionError:
        return None
    except ValueError:
        # int() past sys.int_max_str_digits
        return {"outcome": "ValueError"}
    return {"outcome": "accepted"}


def decode_error_table(rng):
    documents = list(NAMED_DOCUMENTS)
    seen = {raw for _, raw in documents}
    while len(documents) < len(NAMED_DOCUMENTS) + MUTATION_COUNT:
        raw = mutated(rng)
        if raw in seen or not reads_as_utf8(raw):
            continue
        seen.add(raw)
        documents.append((f"mutation_{len(documents) - len(NAMED_DOCUMENTS):03d}", raw))
    rows = []
    for name, raw in documents:
        if not reads_as_utf8(raw):
            raise SystemExit(f"decode_errors: {name} is not read as UTF-8 by json.loads")
        outcome = loads_outcome(raw)
        if outcome is None:
            raise SystemExit(f"decode_errors: {name} ran out of stack")
        row = {"name": name, "bytes": base64.b64encode(raw).decode("ascii")}
        # A readable copy of short documents; the bytes are what the tests use.
        if len(raw) <= 200:
            try:
                row["text"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                pass
        row.update(outcome)
        rows.append(row)
    return rows


def write(name, rows):
    path = OUTPUT / name
    body = {"generator": generator(), "rows": rows}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(OUTPUT.parents[1])} ({path.stat().st_size} bytes)")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write("float_repr.json", float_table(random.Random(SEED)))
    write("strings.json", string_table())
    write("documents.json", document_table(random.Random(SEED + 1)))
    write("decode_errors.json", decode_error_table(random.Random(SEED + 2)))


if __name__ == "__main__":
    main()
