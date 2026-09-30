# CPython JSON reference tables

These tables record what CPython's `json` module produces, so that the tests for
`Sources/OpenJevCore/JSON` compare against Python itself. They do not depend on upstream OpenJev;
they pin down the `json.dumps` behaviour that upstream relies on (`ensure_ascii`, `sort_keys`,
separators and float `repr`).

| File | Contents |
|---|---|
| `float_repr.json` | 5,000 doubles as IEEE-754 bit patterns in hex, with `repr(x)` and `json.dumps(x)` |
| `strings.json` | Strings as lists of code points, with `json.dumps(s)` and `json.dumps(s, ensure_ascii=False)` |
| `documents.json` | JSON texts (including duplicate and canonically equivalent keys) with `json.dumps` of the parsed value under the default, `sort_keys=True`, compact and `ensure_ascii=False` options |

Each file records the Python version, the script and the seed under `generator`.

Regenerate from the repository root with any Python 3 (standard library only):

```bash
python3 Tools/fixtures/python_json_tables.py
```

The output is deterministic for a given Python version. Tests that need a table skip with a
message when it is missing.
