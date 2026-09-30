# CPython JSON reference tables

These tables record what CPython's `json` module produces, so that the tests for
`Sources/OpenJevCore/JSON` compare against Python itself. They do not depend on upstream OpenJev;
they pin down the `json.dumps` behaviour that upstream relies on (`ensure_ascii`, `sort_keys`,
separators and float `repr`), and the `json.loads` errors that upstream's FastAPI app turns into
its `json_invalid` 422 and its "There was an error parsing the body" 400.

| File | Contents |
|---|---|
| `float_repr.json` | 5,000 doubles as IEEE-754 bit patterns in hex, with `repr(x)` and `json.dumps(x)` |
| `strings.json` | Strings as lists of code points, with `json.dumps(s)` and `json.dumps(s, ensure_ascii=False)` |
| `documents.json` | JSON texts (including duplicate and canonically equivalent keys) with `json.dumps` of the parsed value under the default, `sort_keys=True`, compact and `ensure_ascii=False` options |
| `decode_errors.json` | 728 documents as base64 bytes (and as text when short and UTF-8), each with how `json.loads(bytes)` ends: `accepted`, `JSONDecodeError` with its `msg` and `pos` (characters from the start, after any byte order mark), `UnicodeDecodeError`, or `ValueError` (an integer over `int()`'s 4,300 digits). 128 are chosen to reach every message of CPython's scanner and the places where it accepts more than RFC 8259; 600 are seeded mutations of request-like documents. Only documents `json.loads` reads as UTF-8 are kept: no UTF-16 or UTF-32, and no encoded surrogates. |

Each file records the Python version, the script and the seed under `generator`.

Regenerate from the repository root with any Python 3 (standard library only):

```bash
python3 Tools/fixtures/python_json_tables.py
```

The output is deterministic for a given Python version. `make fixtures` regenerates these tables
together with every other fixture (see [../README.md](../README.md)). Tests that need a table skip
with a message when it is missing.
