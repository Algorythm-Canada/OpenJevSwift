#!/usr/bin/env python3
"""Write the CPython reference tables for OpenJevCore's JSON writer and parser.

The tables record what CPython's json module produces, so that the Swift tests compare against
Python itself rather than against expectations written by hand. The script uses only the standard
library and a fixed seed, so running it twice with the same Python version gives the same files.

Usage (from the repository root):

    python3 Tools/fixtures/python_json_tables.py

The output goes to Fixtures/python-json/. Each file records the Python version that wrote it.
"""

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


if __name__ == "__main__":
    main()
