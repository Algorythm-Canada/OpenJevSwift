# ``OpenJevServer``

The Jev-compatible HTTP server: upstream OpenJev's routes, settings, errors and headers over any
engine of OpenJevCore, on Hummingbird.

## Overview

The server answers `POST /v1/systemone`, `GET /v1/models` and `GET /health` as upstream's FastAPI
app does, so TypeSafe's SDKs and anything written against upstream work against it unchanged. It
reads upstream's `OPENJEV_*` settings with upstream's defaults and startup checks
(``ServerSettings``), sends upstream's error bodies and the `x-typesafe-request-id`,
`x-request-id` and `server-timing` headers on every response, authenticates `/v1/` routes with
`OPENJEV_API_KEY` and `OPENJEV_ORIGIN_SECRET`, and forwards the models `OPENJEV_MODEL_ROUTES`
names to the OpenJev servers that serve them. A client that goes away cancels its decision, and a
graceful shutdown lets the requests in flight finish before it releases the model.

The `openjev` command line tool runs this server: <doc:RunningTheServer> covers it, and
<doc:Configuration> lists every setting. The module builds on macOS and Linux. It is not a library
product, so packages that depend on OpenJevSwift never link Hummingbird; the `openjev` tool, the
SDK suite's stub server and the tests link it. Its API is documented for them and for anyone who
embeds the server in a fork.

A server over a loaded service, in a service group that stops it on SIGTERM:

```swift
let settings = try ServerSettings(environment: ProcessInfo.processInfo.environment)
let service = try await provider.makeService(settings: settings)
var configuration = ServiceGroupConfiguration(
    services: [DecisionServer(settings: settings, service: service, logger: logger)],
    gracefulShutdownSignals: [.sigterm, .sigint], logger: logger)
configuration.maximumGracefulShutdownDuration = .seconds(30)
try await ServiceGroup(configuration: configuration).run()
```

`provider` is a ``BackendProvider``: ``DecisionBackendProvider`` builds a
``/OpenJevCore/DecisionEngine`` over a ``/OpenJevCore/DecisionBackend``, and
``QuestionReadBackendProvider`` an ``/OpenJevCore/EncoderDecisionEngine`` over a
``/OpenJevCore/QuestionReadBackend``, each configured from the settings.

## Topics

### Essentials

- <doc:RunningTheServer>
- <doc:Configuration>

### Settings

- ``ServerSettings``
- ``ServerSettingsError``

### Loading the model

- ``BackendProvider``
- ``DecisionBackendProvider``
- ``QuestionReadBackendProvider``

### Serving

- ``DecisionServer``
- ``ShutdownInterrupted``
- ``OpenJevApplication``
- ``OpenJevRequestContext``
- ``SystemOneHandler``

### Version

- ``openJevServerVersion``
