#!/usr/bin/env python3
"""The SDK compatibility suite (issue #39): TypeSafe's official SDKs against the Swift server.

Starts openjev-stub-server, the real application over the test support's stub backends, three
times on free ports of 127.0.0.1, each with OPENJEV_API_KEY=sk-test:

- main: the DiffusionGemma stub, with laya-1.0 routed to the Laya server (OPENJEV_MODEL_ROUTES);
- laya: the Laya stub, which main forwards to;
- overloaded: the DiffusionGemma stub with OPENJEV_MAX_QUEUE=0, which refuses every request with
  a 529 (D-027).

Each SDK reaches main and overloaded through a recording proxy, and runs every scenario in its
own process: the Python SDK (python_checks.py, with the interpreter this script runs under), the
TypeScript SDK under Node (typescript/checks.mjs) and, with --swift-sdk, NSStudent's JevSwiftSDK
(swift/). A scenario prints what the SDK observed; this script checks it, and the exchanges the
proxy saw, against what the server must answer. A failed check prints every HTTP exchange it
made. Every exchange and the servers' logs are also written under --exchanges. The exit status is
1 when any check failed.

Standard library only: run it with Tools/sdk-compat/.venv/bin/python (make sdk-compat-venv),
which also holds typesafe-sdk. docs/09-conformance-and-testing.md describes the suite.
"""

import argparse
import http.client
import http.server
import json
import math
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
API_KEY = "sk-test"
SCENARIOS = ["quickstart", "models", "wrong_key", "overloaded", "samples_33", "routed"]
# JevSwiftSDK leaves out two scenarios. It sends state, model and questions only, so an extension
# field cannot reach the server (samples_33). And it writes the questions and the choice criteria
# from Swift dictionaries, whose order changes from run to run, while the DiffusionGemma stub
# replays the tokenizations upstream's tests recorded, in their order only (quickstart); its
# answers are read through the Laya stub instead, which tokenizes nothing (routed).
SWIFT_SCENARIOS = [name for name in SCENARIOS if name not in ("samples_33", "quickstart")]
# The headers a proxy does not pass on: they describe one connection, not the exchange.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade",
    "content-length",
}


class CheckFailure(AssertionError):
    """What a check found wrong."""


def expect(condition, message):
    if not condition:
        raise CheckFailure(message)


def header(headers, name):
    """The first value of the header `name` in a list of (name, value) pairs, or None."""
    return next((value for key, value in headers if key.lower() == name), None)


# MARK: Exchanges and the recording proxy


@dataclass
class Exchange:
    """One HTTP request a proxy passed on, and the answer it passed back."""

    check: str
    started: float
    method: str
    path: str
    request_headers: list
    request_body: bytes
    finished: float = 0.0
    status: int = 0
    reason: str = ""
    response_headers: list = field(default_factory=list)
    response_body: bytes = b""
    error: str = ""

    def request_json(self):
        return json.loads(self.request_body)

    def describe(self):
        """The exchange as text, as near to the wire as the proxy saw it."""
        lines = [f"> {self.method} {self.path} HTTP/1.1"]
        lines += [f"> {name}: {value}" for name, value in self.request_headers]
        lines += [">", *(f"> {line}" for line in self.request_body.decode("utf-8", "replace").splitlines())]
        if self.error:
            lines.append(f"< (no answer: {self.error})")
        else:
            lines.append(f"< HTTP/1.1 {self.status} {self.reason}")
            lines += [f"< {name}: {value}" for name, value in self.response_headers]
            lines += ["<", *(f"< {line}" for line in self.response_body.decode("utf-8", "replace").splitlines())]
        lines.append(f"  ({(self.finished - self.started) * 1000:.1f} ms)")
        return "\n".join(lines)


class Recorder:
    """The exchanges of every proxy, each under the name of the check that was running."""

    def __init__(self):
        self.lock = threading.Lock()
        self.check = ""
        self.exchanges = []

    def add(self, exchange):
        with self.lock:
            self.exchanges.append(exchange)

    def of(self, check):
        with self.lock:
            return [exchange for exchange in self.exchanges if exchange.check == check]


class RecordingProxy:
    """An HTTP/1.1 proxy on a free port of 127.0.0.1 in front of one server: it passes every
    request on unchanged, but for the headers that describe one connection, and records it."""

    def __init__(self, name, target_port, recorder):
        self.name = name

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def read_body(self):
                """The request's body, sent with a length or in chunks."""
                if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
                    body = b""
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                                pass
                            return body
                        body += self.rfile.read(size)
                        self.rfile.readline()
                length = int(self.headers.get("content-length") or 0)
                return self.rfile.read(length) if length else b""

            def handle_one(self):
                body = self.read_body()
                headers = [(key, value) for key, value in self.headers.items()]
                exchange = Exchange(recorder.check, time.monotonic(), self.command, self.path, headers, body)
                connection = http.client.HTTPConnection("127.0.0.1", target_port, timeout=120)
                try:
                    connection.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
                    for key, value in headers:
                        if key.lower() not in HOP_BY_HOP:
                            connection.putheader(key, value)
                    if body or self.command in ("POST", "PUT", "PATCH"):
                        connection.putheader("Content-Length", str(len(body)))
                    connection.endheaders(body or None)
                    answer = connection.getresponse()
                    exchange.status, exchange.reason = answer.status, answer.reason
                    exchange.response_headers = answer.getheaders()
                    exchange.response_body = answer.read()
                except OSError as error:
                    exchange.error = f"{type(error).__name__}: {error}"
                finally:
                    connection.close()
                    exchange.finished = time.monotonic()
                    recorder.add(exchange)
                if exchange.error:
                    self.send_error(502, exchange.error)
                    return
                self.send_response_only(exchange.status, exchange.reason)
                for key, value in exchange.response_headers:
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(exchange.response_body)))
                self.end_headers()
                self.wfile.write(exchange.response_body)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = handle_one

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name=f"proxy-{name}", daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


# MARK: The stub servers


class StubServer:
    """openjev-stub-server in a child process, which prints the port it bound."""

    def __init__(self, name, binary, settings, log_directory):
        self.name = name
        self.log_path = log_directory / f"server-{name}.log"
        environment = {key: value for key, value in os.environ.items() if not key.startswith("OPENJEV_")}
        environment.update({"OPENJEV_HOST": "127.0.0.1", "OPENJEV_PORT": "0", "OPENJEV_API_KEY": API_KEY})
        environment.update(settings)
        self.log = open(self.log_path, "wb")
        self.process = subprocess.Popen(
            [str(binary)], env=environment, stdout=subprocess.PIPE, stderr=self.log, stdin=subprocess.DEVNULL
        )
        try:
            self.port = self.read_port(timeout=60)
        except Exception:
            self.process.kill()
            self.process.wait()
            self.log.close()
            raise

    def read_port(self, timeout):
        deadline = time.monotonic() + timeout
        line = b""
        while not line.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.process.poll() is not None:
                raise RuntimeError(f"{self.name}: the server did not print its port\n{self.log_text()}")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 64)
                if not chunk:
                    continue
                line += chunk
        return int(line)

    def log_text(self):
        self.log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self):
        """Stops the server with SIGTERM, as launchd does, and returns its exit status."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        return self.process.returncode


# MARK: What the server must answer


def load_expected_listing():
    """The DiffusionGemma listing, then laya-1.0's entry, as Fixtures/wire/models.json records
    them: main lists its own models, then the model routed to the Laya server."""
    listings = json.loads((ROOT / "Fixtures/wire/models.json").read_text())["listings"]

    def body(backend):
        listing = next(item for item in listings if item["backend"] == backend and "model_routes" not in item)
        return json.loads(listing["body_text"])["models"]

    return body("mlx") + body("laya")


def expect_one(exchanges, path, status):
    expect(len(exchanges) == 1, f"expected one exchange, got {len(exchanges)}")
    exchange = exchanges[0]
    expect(exchange.path == path, f"the SDK asked {exchange.path}, not {path}")
    expect(exchange.status == status, f"the server answered {exchange.status}, not {status}")
    return exchange


def expect_server_headers(exchange):
    """Upstream's headers, on every answer: equal request ids of the recorded form, and the
    server's timing."""
    request_id = header(exchange.response_headers, "x-typesafe-request-id") or ""
    expect(
        request_id.startswith("req_") and len(request_id) == 36,
        f"x-typesafe-request-id is {request_id!r}, not req_ and 32 hex characters",
    )
    expect(header(exchange.response_headers, "x-request-id") == request_id, "x-request-id differs")
    expect(header(exchange.response_headers, "server-timing") is not None, "no server-timing")
    return request_id


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def expect_answers(observed, exchange, *, model, first, choice_index, score, noul, tokens):
    """The answers of the stub that served the request. The choice depends on the criteria's
    order in the body the SDK sent, which a dictionary-based SDK does not keep; `first` says
    whether the SDK keeps the order of the answer's probabilities."""
    criteria = list(exchange.request_json()["questions"]["department"]["criteria"])
    chosen = criteria[choice_index]
    expect(observed["model"] == model, f"model {observed['model']!r}, not {model!r}")
    department = observed["department"]
    expect(department["choice"] == chosen, f"department chose {department['choice']!r}, not {chosen!r}")
    expect(set(department["probabilities"]) == set(criteria), "probabilities are not over the criteria")
    if first:
        # An SDK that keeps the order the server wrote keeps the criteria's order.
        expect(list(department["probabilities"]) == criteria, "probabilities out of the criteria's order")
    for label, probability in department["probabilities"].items():
        expected = 0.7 if label == chosen else 0.15
        expect(close(probability, expected), f"P({label}) is {probability}, not {expected}")
    frustration = observed["frustration"]
    expect(close(frustration["score"], score), f"frustration scored {frustration['score']}, not {score}")
    expect(frustration["legend"].get("0") == "Calm, just stating facts", f"legend {frustration['legend']}")
    expect(close(observed["is_urgent"]["noul"], noul), f"is_urgent is {observed['is_urgent']['noul']}, not {noul}")
    expect(observed["usage"] == {"input_tokens": tokens, "output_tokens": 0}, f"usage {observed['usage']}")


def verify_quickstart(sdk, observed, exchanges):
    exchange = expect_one(exchanges, "/v1/systemone", 200)
    request_id = expect_server_headers(exchange)
    expect(header(exchange.request_headers, "authorization") == f"Bearer {API_KEY}", "not the key")
    expect(exchange.request_json()["model"] == "jev-latest", "the SDK did not send its default model")
    # The DiffusionGemma stub gives each question's first label 0.7 and reads once for 123 tokens.
    expect_answers(observed, exchange, model="openjev-0.1", first=sdk != "swift", choice_index=0,
                   score=0.15 * 1 + 0.15 * 2, noul=0.7, tokens=123)
    if observed.get("request_id") is not None:
        expect(observed["request_id"] == request_id, f"request id {observed['request_id']}, not {request_id}")


def verify_models(sdk, observed, exchanges):
    exchange = expect_one(exchanges, "/v1/models", 200)
    expect_server_headers(exchange)
    expected = load_expected_listing()
    expect(observed["models"] == expected, f"listing {json.dumps(observed['models'])}")


ERRORS = {
    # scenario: the error each SDK raises, as (python, typescript, swift)
    "wrong_key": ("TypeSafeAuthenticationError", "AuthenticationError", "http"),
    "overloaded": ("TypeSafeInternalServerError", "InternalServerError", "http"),
    "samples_33": ("TypeSafeUnprocessableEntityError", "UnprocessableEntityError", None),
}


def expect_error(sdk, scenario, observed, exchange, status, message):
    expected = ERRORS[scenario][("python", "typescript", "swift").index(sdk)]
    expect(observed["error"] == expected, f"raised {observed['error']}, not {expected}")
    expect(observed["api_error"], "the error is not the SDK's API error")
    expect(observed["status"] == status, f"status {observed['status']}, not {status}")
    expect(message in observed["message"], f"the message {observed['message']!r} lacks {message!r}")
    request_id = expect_server_headers(exchange)
    expect(observed["request_id"] == request_id, f"request id {observed['request_id']}, not {request_id}")


def verify_wrong_key(sdk, observed, exchanges):
    # A 401 is not retried.
    exchange = expect_one(exchanges, "/v1/systemone", 401)
    expect(header(exchange.request_headers, "authorization") == "Bearer sk-wrong", "not the wrong key")
    expect_error(sdk, "wrong_key", observed, exchange, 401,
                 "Cannot authenticate with the server. Please check your API key and try again.")


def verify_overloaded(sdk, observed, exchanges):
    expect(len(exchanges) == 3, f"expected the request and two retries, got {len(exchanges)} exchanges")
    for exchange in exchanges:
        expect(exchange.status == 529, f"the server answered {exchange.status}, not 529")
        expect(header(exchange.response_headers, "retry-after") == "1", "no retry-after: 1")
    if sdk != "swift":
        counts = [header(exchange.request_headers, "x-typesafe-retry-count") for exchange in exchanges]
        expect(counts == [None, "1", "2"], f"X-TypeSafe-Retry-Count {counts}")
    # retry-after: 1 honoured: the SDKs' own backoff would wait at most 0.5 s before the first retry.
    for before, after in zip(exchanges, exchanges[1:]):
        gap = after.started - before.finished
        expect(gap >= 0.95, f"retried after {gap:.3f} s, before the second retry-after asked for")
    expect_error(sdk, "overloaded", observed, exchanges[-1], 529, "OpenJev is at capacity. Retry shortly.")


def verify_samples_33(sdk, observed, exchanges):
    exchange = expect_one(exchanges, "/v1/systemone", 422)
    expect(exchange.request_json().get("samples") == 33, "samples did not reach the server")
    # The SDKs join a 422's items as "loc: msg", without "body".
    expect_error(sdk, "samples_33", observed, exchange, 422,
                 "samples: Input should be less than or equal to 32")


def verify_routed(sdk, observed, exchanges):
    exchange = expect_one(exchanges, "/v1/systemone", 200)
    request_id = expect_server_headers(exchange)
    expect(exchange.request_json()["model"] == "laya-1.0", "the SDK did not ask laya-1.0")
    # Answered by the Laya server through main: its stub gives the second option 0.7, so P(yes) is
    # 0.3, and reads one batch for 99 tokens.
    expect_answers(observed, exchange, model="laya-1.0", first=sdk != "swift", choice_index=1,
                   score=0.15 * 0 + 0.7 * 1 + 0.15 * 2, noul=0.3, tokens=99)
    if observed.get("request_id") is not None:
        expect(observed["request_id"] == request_id, "the request id is not main's")


VERIFY = {
    "quickstart": verify_quickstart,
    "models": verify_models,
    "wrong_key": verify_wrong_key,
    "overloaded": verify_overloaded,
    "samples_33": verify_samples_33,
    "routed": verify_routed,
}


# MARK: Running the checks


def drive(command, environment, timeout=120):
    """Runs one scenario's driver and returns what it printed, parsed."""
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise CheckFailure(f"{' '.join(map(str, command))} exited {result.returncode}:\n{result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise CheckFailure(f"the driver printed no JSON:\n{result.stdout}\n{result.stderr}") from None


def swift_checks_binary():
    """Builds the JevSwiftSDK driver and returns its path."""
    package = HERE / "swift"
    subprocess.run(["swift", "build", "--package-path", str(package)], check=True)
    bin_path = subprocess.run(
        ["swift", "build", "--package-path", str(package), "--show-bin-path"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(bin_path) / "jev-swift-sdk-checks"


def stub_server_binary():
    """openjev-stub-server, built by `swift build` at the repository root."""
    subprocess.run(["swift", "build", "--package-path", str(ROOT), "--product", "openjev-stub-server"], check=True)
    bin_path = subprocess.run(
        ["swift", "build", "--package-path", str(ROOT), "--show-bin-path"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(bin_path) / "openjev-stub-server"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", type=Path, help="openjev-stub-server; built with swift build when left out")
    parser.add_argument("--node", default="node", help="the Node.js executable (default: node)")
    parser.add_argument("--swift-sdk", action="store_true", help="also run NSStudent's JevSwiftSDK")
    parser.add_argument("--exchanges", type=Path, default=HERE / "exchanges",
                        help="where to write every exchange and the servers' logs (default: Tools/sdk-compat/exchanges)")
    args = parser.parse_args()

    server_binary = args.server or stub_server_binary()
    drivers = {
        "python": lambda scenario: [sys.executable, str(HERE / "python_checks.py"), scenario],
        "typescript": lambda scenario: [args.node, str(HERE / "typescript/checks.mjs"), scenario],
    }
    if args.swift_sdk:
        swift_binary = swift_checks_binary()
        drivers["swift"] = lambda scenario: [str(swift_binary), scenario]
    if shutil.which(args.node) is None:
        sys.exit(f"{args.node} is not on PATH; install Node.js 20 or later")
    if not (HERE / "typescript/node_modules/@typesafe-ai/sdk").is_dir():
        sys.exit("typescript/node_modules is missing; run make sdk-compat-venv")

    if args.exchanges.exists():
        shutil.rmtree(args.exchanges)
    args.exchanges.mkdir(parents=True)
    recorder = Recorder()
    servers, proxies, results = [], [], []
    try:
        laya = StubServer("laya", server_binary, {"OPENJEV_BACKEND": "laya"}, args.exchanges)
        servers.append(laya)
        main_server = StubServer(
            "main", server_binary, {"OPENJEV_MODEL_ROUTES": f"laya-1.0=http://127.0.0.1:{laya.port}"},
            args.exchanges)
        servers.append(main_server)
        overloaded = StubServer("overloaded", server_binary, {"OPENJEV_MAX_QUEUE": "0"}, args.exchanges)
        servers.append(overloaded)
        main_proxy = RecordingProxy("main", main_server.port, recorder)
        overloaded_proxy = RecordingProxy("overloaded", overloaded.port, recorder)
        proxies += [main_proxy, overloaded_proxy]

        environment = {key: value for key, value in os.environ.items() if not key.startswith("TYPESAFE_")}
        environment.update({
            "TYPESAFE_BASE_URL": main_proxy.url,
            "TYPESAFE_API_KEY": API_KEY,
            "SDK_COMPAT_OVERLOADED_URL": overloaded_proxy.url,
        })
        for sdk, command in drivers.items():
            for scenario in SWIFT_SCENARIOS if sdk == "swift" else SCENARIOS:
                check = f"{sdk} {scenario}"
                recorder.check = check
                started = time.monotonic()
                try:
                    observed = drive(command(scenario), environment)
                    VERIFY[scenario](sdk, observed, recorder.of(check))
                    results.append((check, None, time.monotonic() - started))
                except Exception as error:
                    detail = str(error) if isinstance(error, CheckFailure) else traceback.format_exc()
                    results.append((check, detail, time.monotonic() - started))
    finally:
        for proxy in proxies:
            proxy.stop()
        statuses = {server.name: server.stop() for server in servers}

    for server, status in statuses.items():
        # A clean shutdown on SIGTERM exits 0.
        if status != 0:
            results.append((f"server {server} exits 0 on SIGTERM", f"it exited {status}", 0.0))

    with open(args.exchanges / "exchanges.txt", "w") as log:
        for exchange in recorder.exchanges:
            log.write(f"## {exchange.check}\n{exchange.describe()}\n\n")

    failed = [result for result in results if result[1] is not None]
    for check, detail, seconds in results:
        print(f"{'ok  ' if detail is None else 'FAIL'} {check} ({seconds:.1f} s)")
    for check, detail, _ in failed:
        print(f"\n=== {check} failed\n{detail}")
        exchanges = recorder.of(check)
        print(f"--- {len(exchanges)} HTTP exchange(s)")
        for exchange in exchanges:
            print(exchange.describe())
    if failed:
        for server in servers:
            print(f"\n--- the {server.name} server's log ({server.log_path})")
            print(server.log_path.read_text(errors="replace")[-4000:])
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed; exchanges in {args.exchanges}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
