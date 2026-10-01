# SDK compatibility suite (issue #39)

"TypeSafe's SDKs work unchanged" is upstream's headline claim. This suite runs those SDKs, as an
application would use them, against the Swift server, and fails when one of them cannot decode an
answer, maps an error to the wrong class, or retries differently. The CI job `SDK compatibility`
runs it on Linux on every pull request (decision D-040,
[docs/development.md](../../docs/development.md)).

## What runs

`run.py` starts `openjev-stub-server`, the real application over the test support's stub backends,
three times on free ports of 127.0.0.1, each with `OPENJEV_API_KEY=sk-test`:

| Server | Settings | Answers |
|---|---|---|
| main | `OPENJEV_MODEL_ROUTES=laya-1.0=http://127.0.0.1:{laya}` | The DiffusionGemma stub: each question's first label 0.7, 123 input tokens a read; `laya-1.0` forwarded to the Laya server |
| laya | `OPENJEV_BACKEND=laya` | The Laya stub: the second option 0.7, 99 input tokens a batch |
| overloaded | `OPENJEV_MAX_QUEUE=0` | A 529 with `retry-after: 1` for every request (D-027) |

Each SDK reaches main and overloaded through a recording proxy and runs every scenario in a
process of its own, configured from `TYPESAFE_BASE_URL` and `TYPESAFE_API_KEY`:

| Scenario | The SDK must |
|---|---|
| `quickstart` | send its default model, `jev-latest`, and decode Jev's quickstart answers: `SystemOneResponse` with `.choices`, `.scores` and `.nouls` in Python, the typed answers in TypeScript, and the response's request id |
| `models` | list the DiffusionGemma models, then `laya-1.0`, as Fixtures/wire/models.json records them |
| `wrong_key` | raise its authentication error (`TypeSafeAuthenticationError`, `AuthenticationError`) carrying the request id, without retrying |
| `overloaded` | retry the 529 twice, each time after the one second `retry-after` asks for and with `X-TypeSafe-Retry-Count`, then raise its API error (`TypeSafeInternalServerError`, a `TypeSafeAPIError`; `InternalServerError`, an `APIError`) |
| `samples_33` | send `samples: 33` (`extra_body` in Python, an extra request property in TypeScript) and raise its 422 error with the message `samples: Input should be less than or equal to 32` |
| `routed` | decode the Laya server's answers to a `laya-1.0` request, forwarded by main (issue #38) |

A scenario prints what the SDK observed; `run.py` checks it, and the exchanges its proxy recorded,
against what the server must answer. A failed check prints every HTTP exchange it made, and the
servers' logs. Every exchange and log is written to `exchanges/` (or `--exchanges`). The servers
must also exit 0 on SIGTERM. The exit status is 1 when anything failed.

`--swift-sdk` also builds `swift/`, a package of its own, and runs NSStudent's JevSwiftSDK, which
takes a base URL: `models`, `wrong_key`, `overloaded` and `routed`. It sends only state, model
and questions, so `samples_33` does not apply. It writes the questions and the choice criteria from
Swift dictionaries, whose order changes from run to run, and the DiffusionGemma stub answers from
tokenizations recorded in upstream's order, so its `quickstart` answers are read through the Laya
stub, in `routed`. CI runs it.

## Pins

| SDK | Version | Where |
|---|---|---|
| [typesafe-sdk](https://pypi.org/project/typesafe-sdk/) | 0.7.2 (2026-09-26), the newest on PyPI on 2026-10-01 | `requirements.txt`, a complete lock for CPython 3.12 |
| [@typesafe-ai/sdk](https://www.npmjs.com/package/@typesafe-ai/sdk) | 0.6.0 (2026-09-15), the newest on npm on 2026-10-01; Node.js 20 or later | `typescript/package.json` and `typescript/package-lock.json` |
| [NSStudent/JevSwiftSDK](https://github.com/NSStudent/JevSwiftSDK) | 0.1.0, commit `ce35d20` | `swift/Package.swift` |

## Running it

From the repository root, once, with CPython 3.12 and Node.js 20 or later:

```bash
make sdk-compat-venv
```

Then:

```bash
make sdk-compat
```

`make sdk-compat` builds `openjev-stub-server` and runs `run.py` with the virtual environment's
Python. `make sdk-compat SDK_COMPAT_ARGS=--swift-sdk` also runs JevSwiftSDK. `run.py --help` lists
the options; `--server` takes a stub server built elsewhere, as CI's Linux build.

| Path | What it is |
|---|---|
| `run.py` | The runner: the servers, the proxies, the expectations. Standard library only |
| `python_checks.py` | The Python SDK's scenarios |
| `typescript/checks.mjs` | The TypeScript SDK's scenarios, under Node.js |
| `swift/` | JevSwiftSDK's scenarios, a Swift package the main package never builds |
| `requirements.txt` | The lock for `.venv` |
