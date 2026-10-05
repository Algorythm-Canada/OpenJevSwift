# Spike #21: chat template rendering parity

Question. Does swift-jinja render Gemma 4's `chat_template.jinja` as Python's `transformers`
does for the `[system, user]` prompts upstream OpenJev builds, with the generation prompt and
`enable_thinking` off and on, or does the port need a hand-rolled prompt builder?

Answer. The engine matches. Every row of `Fixtures/chat-prompts/prompts.json` renders to the
recorded text and the recorded ids, thinking off and on. No builder is needed. The decision is
recorded in D-008 of [../06-decisions.md](../06-decisions.md) and risk R2 in
[../07-risks-and-unknowns.md](../07-risks-and-unknowns.md).

Later (issue #124, 2026-10-03): one difference no row held. swift-jinja's `trim` strips
Foundation's `whitespacesAndNewlines`, and jinja2's strips what Python's `str.strip()` strips, so a
state that starts or ends with U+001C to U+001F or U+200B rendered differently. The port renders
the template with jinja2's `trim` since D-054, and `prompts.json` holds those states.

Versions the result holds for: swift-transformers 1.3.4, swift-jinja 2.5.1, swift-huggingface
0.11.0, mlx-swift-lm `c043fb3`; tokenizer `mlx-community/diffusiongemma-26B-A4B-it-4bit` at
`a7a81407613811e8ba63af92ac0d852b809e191f`; fixtures from transformers 5.17.0, tokenizers
0.23.2, Jinja2 3.1.6. Run on 2026-09-30, macOS 27, Apple silicon.

## How the template is found

`tokenizer_config.json` has no `chat_template` key; the template is the separate 17 KB
`chat_template.jinja`. swift-transformers 1.3.4 reads that file on its own when a tokenizer is
loaded from a folder: `LanguageModelConfigurationFromHub.loadConfig(modelFolder:)`
(Sources/Hub/Hub.swift, lines 275 to 296) looks for `chat_template.jinja`, then
`chat_template.json`, and merges the text into the tokenizer configuration as `chat_template`.
`PreTrainedTokenizer.hasChatTemplate` is therefore true, and `applyChatTemplate` uses the
file's text. Nothing has to be read or passed explicitly. The `SwiftTransformersTokenizer` test
suite checks `hasChatTemplate`.

The same code path serves loading through the Hub (`filesToDownload` includes
`chat_template.jinja`, Hub.swift line 224), so a future download-on-first-use loader keeps the
template.

## Two paths, one result

swift-transformers' `applyChatTemplate` renders the template and tokenizes the text in one
call; it never returns the text. Python's fixture records both, so the port renders twice:

- Ids: `SwiftTransformersTokenizer.chatPromptIDs(system:user:thinking:)` calls
  `applyChatTemplate(messages:chatTemplate:addGenerationPrompt:truncation:maxLength:tools:additionalContext:)`
  with `addGenerationPrompt: true` and `additionalContext: ["enable_thinking": thinking]`.
  That is upstream's `apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
  enable_thinking=thinking)`. The rendered text is tokenized with `addSpecialTokens: false`,
  which is right because the template writes `<bos>` itself.
- Text: `chatPromptText(system:user:thinking:)` compiles `chat_template.jinja` with swift-jinja
  directly, `Template(source, with: .init(lstripBlocks: true, trimBlocks: true))`, the options
  swift-transformers and Python's `transformers` both use, and renders it with the context
  swift-transformers builds: `messages`, `add_generation_prompt`, `enable_thinking` and the
  special token attributes of `tokenizer_config.json` (`bos_token`, `eos_token`, `unk_token`,
  `pad_token`, `mask_token`; the template reads `bos_token`).

The parity test checks, for every row and both thinking values, that the text equals the
recorded text, the ids equal the recorded ids, and `encode(text, addSpecialTokens: false)`
equals the ids, so the two paths are shown to agree with each other as well as with Python.

## Results

| Check | Rows | Matched | Mismatched |
|---|---|---|---|
| Rendered text equals `thinking_off.text` | 24 | 24 | 0 |
| Rendered text equals `thinking_on.text` | 24 | 24 | 0 |
| Ids equal `thinking_off.ids` | 24 | 24 | 0 |
| Ids equal `thinking_on.ids` | 24 | 24 | 0 |

The rows cover the quickstart system text chunked and unchunked, the 24-question and
255-option system texts, non-ASCII text, object, list and float states, empty and
whitespace-only states, surrounding whitespace, escapes and control characters, special token
text inside a state, emoji, a 400-word state, and every prompt the policy and error fixtures
rendered.

Template features exercised and rendered correctly by swift-jinja 2.5.1: `namespace()` with
attribute assignment, `{% set %}` blocks with `{% endset %}` (the `captured_content` capture),
`is string`, `is sequence`, `is mapping`, `is defined`, `message.get(...)`, `| trim`, `| length`,
`| default`, slicing `messages[1:]`, `range(...)` with negative step, nested macros (declared but
not reached without tools), `{{- -}}` whitespace control, and `enable_thinking is defined and
enable_thinking`.

## Byte layout of a read prompt

From the fixtures and the template, for `system` text S and `user` (state) text U, both
trimmed by the template's `| trim`:

```text
thinking off:  <bos><|turn>system\n{S}<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n
thinking on:   <bos><|turn>system\n<|think|>\n{S}<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n
```

As ids: `2, 105, 9731, 107` opens the system turn (`<bos>`, `<|turn>`, `system`, `\n`);
thinking on inserts `98, 107` (`<|think|>`, `\n`); `106, 107` closes each turn; `105, 2364, 107`
opens the user turn; the prompt ends with `105, 4368, 107` (`<|turn>`, `model`, `\n`). The
empty thought block `<|channel>thought\n<channel|>` is not part of the prompt: the engine puts
it at the head of the canvas (`EngineTokens.scaffold`).

## Image prompts (for #46)

mlx-vlm's processor path (`Upstream/openjev/openjev/mlx_backend.py`, lines 104 to 108) sends
the user content as a list of parts, images first: `[{"type": "image"}] * n +
[{"type": "text", "text": state}]`, with `add_generation_prompt=True` and `tokenize=False`, and
then expands the placeholders. Rendered through `SwiftTransformersTokenizer.renderChatTemplate`
with the same message shape, without a fixture (there is none yet; this is swift-jinja's
output, checked against the template's text by hand):

```text
1 image:  <bos><|turn>system\nAnswer the questions.<turn|>\n<|turn>user\n<|image|>What is in the picture?<turn|>\n<|turn>model\n
2 images: <bos><|turn>system\nAnswer the questions.<turn|>\n<|turn>user\n<|image|><|image|>What is in the picture?<turn|>\n<|turn>model\n
```

Ids for one image: `2, 105, 9731, 107, 7925, 506, 4137, 236761, 106, 107, 105, 2364, 107,
258880, 3689, 563, 528, 506, 6083, 236881, 106, 107, 105, 4368, 107`. Each `{"type": "image"}`
part becomes one `<|image|>` (id 258880) directly before the text, with no separator, and the
`applyChatTemplate` ids equal `encode` of the rendered text. So the same template path serves
image prompts; what #46 adds is the processor's expansion of each `<|image|>` into
`<|image>` (255999), 280 soft tokens (258880) and `<image|>` (258882), plus the pixel values.
The template also accepts `audio` and `video` parts (`<|audio|>`, `<|video|>`), which OpenJev
does not use.

## Hand-rolled builder

Not needed. Had the engine failed, the builder would have been the two layouts above with the
trimmed texts substituted, which the fixtures pin exactly. It is kept out of the code so there
is one rendering path to keep in parity.
