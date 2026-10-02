# ``OpenJevCore``

The decision engine of OpenJevSwift: Jev's request and answer types, the prompts and canvases
upstream OpenJev builds, and the two engines that answer a request through a model backend.

## Overview

OpenJevSwift is a native Swift implementation of
[OpenJev](https://github.com/razorback16/openjev), the open, Jev-compatible "System One" decision
server, at upstream commit `dcd2094`. A request is a state and typed questions: a noul (yes or
no), a choice among named options or a score over ordered levels. The answer to each is a
probability distribution and a confidence, read from a model's probabilities rather than
generated as text, so an answer cannot go off-schema.

This module is everything that does not depend on a model. It parses and validates requests as
upstream's FastAPI app does, builds the question schema, the prompts, the answer templates and
the seeded canvases byte for byte, runs upstream's read policies, and writes the answers with
Jev's exact key sets. It depends on Foundation alone and builds for macOS 14, iOS 17 and Linux.
It never reads environment variables: settings are typed values that the caller passes in.

Two engines answer a request, each over its own kind of backend:

- ``DecisionEngine`` reads canvases through a ``DecisionBackend``. The DiffusionGemma runtime of
  the OpenJevDiffusionGemma module is one.
- ``EncoderDecisionEngine`` reads whole questions through a ``QuestionReadBackend``. Verdict and
  Laya, in the OpenJevEncoders module, are two.

Both engines conform to ``SystemOneService``, which the OpenJevServer module serves over HTTP.
<doc:GettingStarted> loads a backend and answers a request in an app.

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:RequestsAndAnswers>
- <doc:ImplementingABackend>

### Requests

- ``SystemOneRequest``
- ``Question``
- ``NoulCriteria``
- ``Described``
- ``ImageInput``
- ``RequestValidator``
- ``ValidationProblem``
- ``ReadOptions``

### Answers

- ``Decision``
- ``Answer``
- ``SystemOneResponse``
- ``Usage``
- ``ModelInfo``
- ``ModelsResponse``
- ``HealthResponse``

### The wire format

- ``WireEncodable``
- ``WireEncoder``
- ``WireDecodingError``
- ``WireError``
- ``TypedErrorBody``
- ``PlainDetailBody``
- ``ValidationErrorItem``
- ``LocComponent``

### Engines

- ``DecisionEngine``
- ``EngineConfiguration``
- ``EncoderDecisionEngine``
- ``EncoderEngineConfiguration``
- ``SystemOneService``
- ``ServedModels``
- ``KnownEncoderModels``
- ``ModelTimeRecorder``
- ``ModelReleasing``

### Backends

- ``DecisionBackend``
- ``CanvasRead``
- ``ReadPrompt``
- ``ReadResult``
- ``SlotRead``
- ``ThoughtGeneration``
- ``BackendCapabilities``
- ``DecisionTokenizer``
- ``TokenizerError``
- ``QuestionReadBackend``
- ``BatchReadResult``
- ``EncoderQuestion``

### Errors

- ``SchemaError``
- ``OverloadedError``
- ``BackendRefusal``
- ``BackendContractError``

### JSON

- ``JSONValue``
- ``JSONObject``
- ``OrderedMap``
- ``JSONParser``
- ``JSONParseError``
- ``PythonJSONWriter``
- ``JSONWriteError``
- ``PythonJSONLoads``

### Schema and prompts

- ``QuestionKind``
- ``QuestionSchemaBuilder``
- ``QuestionSchema``
- ``ReadQuestion``
- ``EncoderQuestionSchemaBuilder``
- ``EncoderQuestionSchema``
- ``AnswerFormat``
- ``TextOf``
- ``SystemText``
- ``StateText``
- ``AnswerText``
- ``EngineTokens``
- ``LabelDiscovery``
- ``LabelSet``

### Canvases and seeds

- ``CanvasGeometry``
- ``CanvasGeometryError``
- ``ReadGrouping``
- ``TemplateResolver``
- ``ResolvedTemplate``
- ``TemplateCache``
- ``CanvasBuilder``
- ``SeededCanvas``
- ``SeedDerivation``
- ``MT19937``
- ``PythonRandom``
- ``SHA256``

### Distributions

- ``SlotDistribution``
- ``Confidence``
- ``ReadAveraging``
- ``pythonSum(_:)``

### Images

- ``ImageValidation``
- ``ImagePart``
- ``ImageLimits``

### Version

- ``openJevCoreVersion``
