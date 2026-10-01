# Deployment: a Mac mini as a decision server

`openjev serve` runs the Jev-compatible server on a Mac as one process, with no Docker and no
Python. It serves DiffusionGemma (`openjev-0.1`, `OPENJEV_BACKEND=mlx`) on MLX and Verdict
(`verdict-1.4`) and Laya (`laya-1.0`) on Core ML. This document follows upstream's README, "Run
your own", where it applies to a Mac.

## What the Mac needs

| Backend | Model | In this build | Mac | Memory |
|---|---|---|---|---|
| `verdict` | `verdict-1.4`, 151M parameters, Core ML | yes | Apple silicon, macOS 15 or later | about 1 GB at peak on the GPU (spike #56) |
| `mlx` | `openjev-0.1`, DiffusionGemma 26B-A4B, 4-bit, MLX | yes | Apple silicon | about 16 GB to load (MLX holds 14.35 GiB) and more in service (R4); 32 GB or more recommended |
| `laya` | `laya-1.0`, 421M parameters, Core ML | yes | Apple silicon, macOS 15 or later | about 2.9 GB at peak on the GPU (spike #56) |

On an M3 Max, Verdict reads one question in 7.5 to 20.3 ms depending on its length, and a batch
of 16 in 4.3 to 19.3 ms per question ([spikes/encoder-runtime.md](spikes/encoder-runtime.md)). Like
upstream, one server serves one backend; run one process per backend, each on its own port.
Forwarding between them (`OPENJEV_MODEL_ROUTES`) comes with issue #38.

## Build

With Xcode 27 or later and the Metal Toolchain ([development.md](development.md)):

```bash
git clone https://github.com/Algorythm-Canada/OpenJevSwift.git
cd OpenJevSwift
swift build -c release --product openjev
```

The binary is in the folder `swift build -c release --show-bin-path` prints. Copy it to a fixed
place, for example:

```bash
sudo install -m 755 "$(swift build -c release --show-bin-path)/openjev" /usr/local/bin/openjev
```

`openjev --version` prints the package version. On Linux the same command builds the CLI
without the encoder backends, so `verdict` exits 3 there.

## The models' files

Verdict needs its converted Core ML package (`verdict-m18-fp16`, 306 MB) and the checkpoint's
tokenizer and calibrator (D-033):

- **Downloaded on first start.** With `OPENJEV_ENCODER_MODELS` unset, the first start downloads
  the package from the `verdict-m18-fp16-v1` release of `Algorythm-Canada/openjev-models`, and
  the tokenizer and calibrator from `heman10x/rlcd-modernbert-151m` at `8af2496`, into
  `~/Library/Application Support/OpenJevSwift/encoders/verdict-m18-fp16/` of the user the server
  runs as. Every file's size and SHA-256 are checked against the manifest the binary embeds, and a
  later start downloads only what is missing or changed. The server binds its port once the files
  are there and the model has loaded.
- **A local folder.** For a Mac without network access, or a package converted with
  `Tools/encoders/convert_verdict.py` ([Tools/encoders/README.md](../Tools/encoders/README.md)),
  set `OPENJEV_ENCODER_MODELS` to the folder that holds `verdict-m18-fp16.mlpackage`; nothing is
  downloaded then. The tokenizer and calibrator are read from
  `$OPENJEV_ENCODER_MODELS/verdict-m18-fp16/tokenizer/` when that folder holds them, else from
  the Hugging Face cache snapshot of the checkpoint at its pinned revision: `HF_HUB_CACHE`, else
  `HF_HOME/hub`, else `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.

The first start compiles the package and keeps the result beside it
(`verdict-m18-fp16.mlmodelc`); later starts reuse the compiled copy until the package changes.

Laya on a Mac runs its multifunction package, `laya-m18-fp16` (810 MB), with its tokenizer and
`rl_agent_config.json` (D-037). Its release `laya-m18-fp16-v1` is published, so without
`OPENJEV_ENCODER_MODELS` the first start downloads about 850 MB into Application Support and
checks every file's SHA-256; a local package converted with `Tools/encoders/convert_laya.py` works
through the variable, as for Verdict. On an M3 Max, the first `decide` took 12 seconds, most of it
compiling the package, and later ones 2.5 seconds.

## Settings

The server reads upstream's `OPENJEV_*` variables, with upstream's defaults and startup checks
(D-013). The ones a Mac deployment sets:

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_BACKEND` | `mlx` | `mlx`, `verdict` or `laya`. Upstream's default, `vllm`, does not exist in this port (D-030). |
| `OPENJEV_HOST` | `127.0.0.1` | The address to bind. `0.0.0.0` serves the network. |
| `OPENJEV_PORT` | `8080` | The port to bind. `0` picks a free one, which the `serving on` line names. |
| `OPENJEV_API_KEY` | unset | Require `Authorization: Bearer <key>` on `/v1/` routes. |
| `OPENJEV_ORIGIN_SECRET` | unset | Require `X-Origin-Secret` (for a server behind a proxy). |
| `OPENJEV_ENCODER_MODELS` | unset | A folder of converted Core ML packages, used instead of downloading (this port's, D-033). |
| `OPENJEV_ENCODER_BATCH` | `16` | Questions per backend call. On a Mac, Verdict splits a call into Core ML calls of at most 16 questions. |
| `OPENJEV_MAX_QUEUE` | `512` | Decisions inside the server before a 529. `0` refuses every request, as upstream's does. |
| `OPENJEV_MAX_QUESTIONS` | `256` | Questions per request before a 400. |
| `OPENJEV_MAX_BODY_BYTES` | `67108864` | Request body limit before a 413. |
| `OPENJEV_WARMUP` | `1` | `0` skips the warm-up read before the server opens. |
| `OPENJEV_LOG_LEVEL` | `info` | `trace`, `debug`, `info`, `notice`, `warning`, `error` or `critical`. |
| `HF_HOME`, `HF_HUB_CACHE` | unset | Where the Hugging Face cache is: DiffusionGemma's checkpoint, and the tokenizer of a local folder. |
| `HF_TOKEN` | unset | A Hugging Face token for a gated repository; an empty value counts as unset. |

The DiffusionGemma settings are upstream's: `OPENJEV_MLX_MODEL` (a directory, or a Hub
repository, `repo@revision` for a revision; the default repository loads its pinned revision),
`OPENJEV_MLX_MAX_PROMPT` (32768), `OPENJEV_MLX_PROMPT_CACHE` (12 prefills), `OPENJEV_MLX_CACHE_LIMIT_GB`
(unset leaves MLX's buffer pool alone, `0` disables it), `OPENJEV_CANVAS` and the others in
upstream's table.

`serve` takes six flags. The first five override their variable:

| Flag | Variable |
|---|---|
| `--backend <name>` | `OPENJEV_BACKEND` |
| `--host <address>` | `OPENJEV_HOST` |
| `--port <port>` | `OPENJEV_PORT` |
| `--log-level <level>` | `OPENJEV_LOG_LEVEL` |
| `--no-warmup` | `OPENJEV_WARMUP=0` |
| `--shutdown-timeout <seconds>` | none; the time requests in flight get after SIGTERM, 0 to 86400, 30 by default |

A flag's value is checked as its variable's is, so `--port abc` fails with
`OPENJEV_PORT='abc' is not a int`, the wording of upstream's `_env_num`, which this port uses for
every variable (D-030).

## Run it by hand

```bash
OPENJEV_BACKEND=verdict openjev serve
```

From another terminal:

```bash
curl -s localhost:8080/v1/models
```

Jev's quickstart request, the `quickstart` case of `Fixtures/wire/cases.json`, gets three
answers:

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' --data-binary @- <<'JSON'
{"state":"Hi, I've been trying to connect my Stripe account but keep getting a 403 error.","model":"jev-latest","questions":{"department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"Payment or subscription issues","technical":"Bugs or integration problems","sales":"Pricing or account questions"}},"frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["Calm, just stating facts","Frustrated but civil","Very angry, strong language"]},"is_urgent":{"type":"noul","instructions":"The message conveys urgency or time-sensitivity"}}}
JSON
```

Ctrl-C stops the server gracefully.

## launchd

A LaunchDaemon starts the server at boot, before anyone logs in, and restarts it when it fails.
Run it as an ordinary user, here `openjev`, whose Application Support folder receives the model.
Save this as `/Library/LaunchDaemons/local.openjev.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>local.openjev.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/openjev</string>
        <string>serve</string>
        <string>--shutdown-timeout</string>
        <string>30</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OPENJEV_BACKEND</key>
        <string>verdict</string>
        <key>OPENJEV_HOST</key>
        <string>0.0.0.0</string>
        <key>OPENJEV_PORT</key>
        <string>8080</string>
        <key>OPENJEV_API_KEY</key>
        <string>replace-with-a-long-random-key</string>
    </dict>
    <key>UserName</key>
    <string>openjev</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>ExitTimeOut</key>
    <integer>40</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>/Users/openjev/Library/Logs/openjev.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/openjev/Library/Logs/openjev.log</string>
</dict>
</plist>
```

- **`KeepAlive`** with `SuccessfulExit` false restarts the server whenever it exits with a status
  other than 0: a crash, a failed load, or a shutdown that had to cancel requests. A stop that
  ends with status 0 stays stopped. `<key>KeepAlive</key><true/>` restarts it in every case.
  A restart waits at least `ThrottleInterval` seconds, so invalid settings (status 2) repeat their
  message every 10 seconds in the log until they are fixed.
- **`ExitTimeOut`** is how long launchd waits after SIGTERM before it sends SIGKILL. Its default
  is system-defined (launchd.plist(5)), so set it, above `--shutdown-timeout`, or launchd may
  kill the server before the requests in flight have finished.
- **`ProcessType`** `Interactive` lifts the light limits on CPU and I/O that launchd applies to a
  job without a `ProcessType`.
- **The API key** is in the file, so keep it readable by root alone.

Load, stop, restart and inspect it:

```bash
sudo chown root:wheel /Library/LaunchDaemons/local.openjev.serve.plist
sudo chmod 600 /Library/LaunchDaemons/local.openjev.serve.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/local.openjev.serve.plist
sudo launchctl kickstart -k system/local.openjev.serve
sudo launchctl print system/local.openjev.serve
sudo launchctl bootout system/local.openjev.serve
```

`bootstrap` loads the job, which starts at once because of `RunAtLoad`; `kickstart -k` restarts
it; `print` shows its state and its last exit status; and `bootout` stops it with SIGTERM and
unloads it.

## Health check

`GET /health` answers `{"status":"ok"}` without authentication. The server binds its port only
after the model has loaded and warmed up, as upstream's lifespan does, so the first successful
answer means it is ready:

```bash
curl -fsS http://127.0.0.1:8080/health
```

`GET /v1/models` names the model a server serves, and needs the API key when one is set.

## Logs

Everything goes to standard error, one line per event, as swift-log writes it:
`{time} {level} openjev: [{module}] {message}`. A start and one request look like this:

```text
2026-10-01T11:51:53-0400 info openjev: [openjev] settings: host=0.0.0.0 port=8080 backend=verdict log_level=info warmup=on max_queue=512 max_questions=256 max_body_bytes=67108864 encoder_batch=16 encoder_models=downloads api_key=set origin_secret=unset model_routes=none
2026-10-01T11:51:53-0400 info openjev: [openjev] loading verdict-1.4 (OPENJEV_BACKEND=verdict)
2026-10-01T11:51:55-0400 info openjev: [openjev] warming up
2026-10-01T11:51:56-0400 info openjev: [HummingbirdCore] Server started and listening on 0.0.0.0:8080
2026-10-01T11:51:56-0400 info openjev: [openjev] serving on 0.0.0.0:8080
2026-10-01T11:51:56-0400 info openjev: [OpenJevServer] POST /v1/systemone 200 54.5ms req_176a5c837d4f88a26775a92c4cb7bbc1
```

- **Requests.** Each request gets one info line: the method, the path without its query string,
  the status, the milliseconds and the request id, which is also the response's `x-request-id`.
  A client that went away before its answer shows 499 (nginx's code); its decision was
  cancelled.
- **Refusals.** As upstream, a 422 and a plain-detail 400 also get a warning that says where the
  request was wrong and why, for example
  `400 req_... body.questions.q.criteria: Too many score levels. Must have at most 10 levels.`
  A backend that fails during a decision gets an error line with its 503 message.
- **What is never written.** A body, a state, instructions, a header value or a query string.
  The API key and the origin secret appear only as `set` or `unset`, and the model routes as
  names, without their URLs.
- **Level.** `OPENJEV_LOG_LEVEL=warning` keeps refusals and failures and drops the phase and
  request lines.
- **Rotation.** launchd opens the log file once. After rotating it, for example with
  newsyslog(8), restart the job (`launchctl kickstart -k`) so the server writes to the new file.

## Graceful shutdown

On SIGTERM, which `launchctl bootout` sends, or SIGINT (Ctrl-C), the server:

1. Stops accepting connections: a new connection is refused, and an idle keep-alive connection
   is closed.
2. Lets each request in flight finish and send its answer, then closes its connection.
3. Releases the model, logs `released verdict-1.4` and `stopped`, and exits with status 0.

Requests still running after `--shutdown-timeout` seconds (30 by default) are cancelled, the
model is released, and the exit status is 1. With none running, the exit status is 0 whatever the
timeout, 0 included. When the timeout cancels the server, Hummingbird also logs
`Waiting on child channel: CancellationError()` at error level. A second signal does not cut the
wait short.

## Exit statuses

| Status | Meaning | Example |
|---|---|---|
| 0 | Success; for `serve`, a clean shutdown | SIGTERM with no request left |
| 1 | Any other failure | the address is in use; a shutdown that cancelled requests; `decide` whose backend failed during the read (it prints the 503 body) |
| 2 | Invalid settings or command line; the message names the variable | `openjev: OPENJEV_PORT='eighty' is not a int`; `openjev: unknown backend 'vllm'; use one of mlx, laya, verdict (OPENJEV_BACKEND)` |
| 3 | The backend cannot run: not in this build, or it failed to load | `openjev: openjev-0.1 failed to load (OPENJEV_BACKEND=mlx): /models/dg is not a DiffusionGemma checkpoint: it lacks config.json, ...`; a download that fails its checksum |
| 4 | `decide` only: the request was refused (a 4xx or the 529) | an unknown model, a malformed body |

A message goes to standard error, prefixed with `openjev:`. A command line the parser refuses
exits 2, as Python's argparse does, with the parser's message and usage.

## openjev decide

`openjev decide` answers one request without a server, through the same engine and the same
code as `POST /v1/systemone`, and prints the response body to standard output exactly as the
server sends it, without a trailing newline. The body is read from `--request <file>`, or from
standard input without it:

```bash
openjev decide --backend verdict --request case.json > answer.json
```

It is for scripts that need one decision, for checking a model on a machine before serving it,
and for regression files: record a set of answers, upgrade, record them again and compare the
bytes. A refused request prints the server's error body to standard error and exits 4, so a
script can tell a refusal (fix the request) from a failure (status 1 or 3, fix the deployment).
`decide` reads the same settings as `serve`, skips the warm-up read, and loads the model for each
call, which for Verdict takes about two seconds.

## openjev models

`openjev models` prints the `GET /v1/models` body of the selected backend without loading a
model, for example to check what a server will list before starting it:

```bash
openjev models --backend verdict
```

## DiffusionGemma

Upstream's Apple silicon section applies as it is: `OPENJEV_BACKEND=mlx openjev serve` runs
DiffusionGemma inside the process on MLX, with the `mlx-community/diffusiongemma-26B-A4B-it-4bit`
weights (`OPENJEV_MLX_MODEL`) at their pinned revision. On first start they are downloaded into the
Hugging Face cache in huggingface_hub's layout (13 files, 16.58 GB), resumed after an interruption
and checked file by file, so a cache upstream or mlx-vlm filled is used as is. Loading the 4-bit
weights takes about 16 GB of memory; `OPENJEV_MLX_CACHE_LIMIT_GB` bounds MLX's buffer pool
(upstream's README, "MLX memory"; the figures are in docs/07 R4). Reads run one at a time on the
GPU, so a Mac serves a few requests per second, not a fleet.
