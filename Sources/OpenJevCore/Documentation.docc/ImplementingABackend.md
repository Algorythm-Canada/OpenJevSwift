# Implementing a backend

Put a model behind one of the two backend protocols and let an engine do everything else.

## Overview

The engines own everything upstream's engines own apart from the model: the option checks, the
question schema, the prompts, the batching or the canvases, the seeds, the read policies, the
queue bound and the answers. A backend owns the model, and which protocol it adopts depends on how
the model answers a question:

- ``DecisionBackend``: the model fills a canvas. Given a prompt and a canvas of answer slots, it
  returns the log-probabilities of the labels at each slot. DiffusionGemma reads this way, as do
  upstream's MLX and vLLM backends. ``DecisionEngine`` drives it.
- ``QuestionReadBackend``: the model scores a question's options directly, from the state and the
  question alone. Verdict and Laya read this way, as do upstream's JevK5 and CLM.
  ``EncoderDecisionEngine`` drives it.

Both protocols require `Sendable`: an engine calls its backend from many tasks.

## Canvas backends

### What a canvas backend provides

- ``DecisionBackend/tokenizer``, a ``DecisionTokenizer``.
  ``DecisionEngine/init(backend:configuration:)`` encodes the markers with it, discovers the
  single-token choice labels (``LabelDiscovery``) and tokenizes every answer template through it,
  so its ids must be the model's own.
- ``DecisionBackend/maxPromptTokens``, the longest prompt one read or thought may carry;
  `Int.max` for no limit.
- ``DecisionBackend/capabilities``, the ``BackendCapabilities`` the backend honours: `steps`,
  `samples`, `think`, `sequential` and `images`.
- ``DecisionBackend/modelName``, the name the engine's refusals use.
- ``DecisionBackend/read(_:)``: one denoise over ``CanvasRead/canvas``, conditioned on
  ``CanvasRead/prompt``, for ``CanvasRead/steps`` passes. Between passes the slots keep their
  argmax and the rest of the canvas stays as it is. It returns a ``ReadResult`` with one
  ``SlotRead`` per slot of ``CanvasRead/slots``, in slot order, and the prompt tokens processed.
  Build it with ``ReadResult/init(tops:labelIDs:promptTokens:)`` from each slot's raw map of
  token id to log-probability (the top 20 tokens and every label, as upstream's MLX runtime
  returns them), so the engine's distributions and entropies are upstream's.
- ``DecisionBackend/think(prompt:budget:stopIDs:)``: up to `budget` tokens generated after
  `prompt`, stopping at the first of `stopIDs`, as a ``ThoughtGeneration``. A backend that does not
  generate sets `think` off in its capabilities and throws here.

### What the engine guarantees a canvas backend

- Options the capabilities turn off are refused before any read, with upstream's message
  (`"{model} does not support {field}"`), as are images sent with `think` or `sequential`.
- A token prompt longer than ``DecisionBackend/maxPromptTokens`` is refused with upstream's
  ``SchemaError`` before the read, and a read never asks for more than
  ``DecisionEngine/maxLabelIDs`` distinct label ids.
- Canvases, slot positions, label ids and seeds are upstream's, so a recorded request replays the
  same ``CanvasRead`` values; ``CanvasRead`` is `Hashable` for that reason.
- Reads run concurrently, up to ``EngineConfiguration/maxInflight`` at once, and the results are
  placed by group and read, so the answers never depend on the order calls finish. A backend that
  can run only one call at a time serialises them itself, as the DiffusionGemma runtime does by
  being an actor.
- Each call's time, the wait for a permit included, is added to ``ModelTimeRecorder/current`` when
  the call ends, whether it returned, threw or was cancelled. The server reports the sum as
  `server-timing`'s `model`.
- A cancelled decision cancels its task, so a read waiting for a permit leaves without one. A
  backend that checks `Task.isCancelled` between steps can stop a read early.
- The engine cuts a thought at the first thought-close id and appends the close marker, as
  upstream's `think` does.

### What the engine expects of a canvas backend

- One ``SlotRead`` per slot. Another count is a programming error and stops the process, as the
  engine's preconditions treat any broken backend contract.
- Errors that say what happened. ``SchemaError`` is a request the model cannot take, which the
  server answers with a 400 and the error's message. ``BackendRefusal`` is the model refusing the
  request, a 400 `the model rejected this request`. Any other error is a failure, which the server
  answers with the 503 that names the error's type.

## Question backends

### What a question backend provides

- ``QuestionReadBackend/modelInfo``, a ``ModelInfo``: the served name, which every refusal and
  every response names, with the description and release date `GET /v1/models` lists.
  ``KnownEncoderModels`` holds upstream's entries.
- ``QuestionReadBackend/maxChoices``, the most options one choice may have: 24 for Verdict, 255
  for the others. The schema builder refuses more with upstream's message.
- ``QuestionReadBackend/maxPromptTokens``, the longest sequence a backend that refuses long
  inputs accepts, or `nil` for a backend that truncates, as Verdict and Laya do. The engine does
  not count tokens; the backend applies its own limit.
- ``QuestionReadBackend/readBatch(state:stateText:questions:)``: one distribution per
  ``EncoderQuestion`` of the batch, in the batch's order, each in the question's option order
  (``EncoderQuestion/choices``), and the input tokens processed. A noul's distribution is
  `[P(true), 1 - P(true)]`.

### What the engine guarantees a question backend

- `images`, `steps` or `samples` above 1, a `think` other than 0 and `sequential: true` are refused
  before any read, as upstream's encoder engines refuse them.
- Forced answers, a choice with one option or a score with one level, never reach the backend; a
  request made only of them makes no call at all.
- Questions arrive in request order, in batches of ``EncoderEngineConfiguration/batchSize``, one
  backend call at a time under ``EncoderEngineConfiguration/maxInflight``, which is upstream's one
  model thread.
- `stateText` is ``StateText/render(_:)`` of `state`; a backend takes whichever form its prompt
  needs.
- ``EncoderDecisionEngine/warmUp()`` reads ``EncoderDecisionEngine/warmUpQuestions`` against
  ``EncoderDecisionEngine/warmUpState`` once, outside the queue bound, when the caller asks.

### What the engine checks

Every distribution a question backend returns: one per question, one value per option, each in
`[0, 1]`, which also rules out NaN and the infinities, and a sum within 1e-6 of 1. A distribution
that breaks the contract is a ``BackendContractError``, which the server answers as a backend
failure, rather than an answer built from bad numbers.

## Releasing the model

A backend that holds something worth releasing adopts ``ModelReleasing``. Both engines adopt it and
pass ``ModelReleasing/close()`` on to their backend, and the server calls it once its requests have
finished when it stops. Verdict and Laya release their loaded Core ML functions; a later read loads
them again.

## Serving a new backend

The OpenJevServer module wraps a canvas backend in a `DecisionBackendProvider` and a question
backend in a `QuestionReadBackendProvider`, each of which builds the engine from the server's
settings. The `openjev` command line tool lists its backends in a registry, one entry each, under
the name `OPENJEV_BACKEND` selects. The repository's `OpenJevTestSupport` target, which is not a
product, holds the stub backends the tests use: `StubBackend` answers as the stubs of upstream's
`tests/test_api.py` do, which is how `Fixtures/policies` was recorded, and
`StubQuestionReadBackend` as the fake engine of upstream's `tests/test_encoders.py`.
