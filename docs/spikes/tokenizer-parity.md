# Spike #20: Gemma 4 tokenizer parity and load time with swift-transformers

Question. Does swift-transformers' `Tokenizers` tokenize the DiffusionGemma checkpoint as
Python's `tokenizers` does, on every text the engine tokenizes, and what does loading the 32 MB
`tokenizer.json` cost? If not, does OpenJevSwift need a custom tokenizer loader?

Answer. Encoding parity is complete: every fixture text encodes to the recorded ids, with and
without special tokens, and label discovery, the engine tokens and the special token ids all
match. Decoding has two departures, both in swift-transformers' post-processing rather than in
the BPE, and both handled generally in `SwiftTransformersTokenizer` without any text being
special-cased. No custom loader is needed. The decision is recorded in D-008 of
[../06-decisions.md](../06-decisions.md) and risk R1 in
[../07-risks-and-unknowns.md](../07-risks-and-unknowns.md).

Versions the result holds for: swift-transformers 1.3.4 (`c21fdcd`), swift-jinja 2.5.1,
swift-huggingface 0.11.0, mlx-swift-lm `c043fb3`; tokenizer
`mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407613811e8ba63af92ac0d852b809e191f`,
whose files were checked against the SHA-256 digests in
`Fixtures/tokenizer/special_tokens.json` before loading; fixtures from transformers 5.17.0 and
tokenizers 0.23.2. Run on 2026-09-30, macOS 27, Apple silicon, Xcode's Test action.

## Parity results

| Fixture | Check | Rows | Matched | Mismatched |
|---|---|---|---|---|
| `tokenizer/corpus.json` | `ids` (encode, no special tokens) | 917 | 917 | 0 |
| `tokenizer/corpus.json` | `ids_with_special_tokens` | 917 | 917 | 0 |
| `tokenizer/corpus.json` | `decoded` | 917 | 917 | 0 (3 before the trailing-bytes fix below) |
| `tokenizer/corpus.json` | `decoded_skip_special_tokens` | 917 | 917 | 0 (3 before the fix) |
| `tokenizer/engine_encodings.json` | `[text, ids]` pairs | 2,634 | 2,634 | 0 |
| `labels.json` | `labels`, `label_ids`, `base_ids`, 6 `rejected` | 255 + 6 | 255 + 6 | 0 |
| `tokenizer/special_tokens.json` | 23 `named` ids, `<\|video\|>`, `end_of_turn_text`, `engine` table | 26 | 26 | 0 |
| Replay agreement | every text either tokenizer knows: 3,252 texts, 917 decodes, 24 prompts | 4,241 | 4,241 | 0 |

The corpus categories (label_candidate 728, system_text 46, answer_text 34, marker 34,
whitespace 23, misc 12, byte_fallback 10, combining 9, special_token_text 7, emoji 6, state 6,
cjk 5, json_state 3) all encode identically. The three rows that decoded wrongly before the
fix are all `byte_fallback`: `"\0"` (`<0x00>`), `"\u{2028}"` (`<0xE2><0x80><0xA8>`) and
`"\u{10FFFF}"` (`<0xF4><0x8F><0xBF><0xBF>`), each a text made only of byte tokens.

Encoding details the corpus confirms: `add_special_tokens=True` adds nothing (the post-processor
has no special tokens); special tokens written as text inside a state are single ids
(`<turn|>` in a state is 106); `<end_of_turn>` is seven ordinary tokens; characters outside the
vocabulary fall back to `<0xNN>` byte tokens; NFC and NFD spellings stay distinct.

## Load time and memory

Measured by `SwiftTransformersTokenizer.load(from:)` with `ContinuousClock` around the
swift-transformers load, `task_info` resident size before and after, and `getrusage`
`ru_maxrss`, in a test process that had loaded nothing else (`TokenizerParityTests/loads()`
run alone):

| Figure | Value |
|---|---|
| Load wall time | 3.6 s (3.6 to 3.8 s over four runs) |
| Resident memory added by the load | 204 MB (175 to 224 MB over runs sharing the process with other tests) |
| Peak resident of the process after loading | 386 MB |
| mlx-swift-lm `#huggingFaceTokenizerLoader()` path, cold, run alone in a fresh process (`MLXTokenizerLoaderTests/sameResults()`): wall time | 3.8 s |
| The same: resident memory added, peak resident | 236 MB, 386 MB |
| The same path as a second load in a process that already holds the direct tokenizer | 3.6 s |

The two paths cost the same, as expected of the same code. The time is the whole of `LanguageModelConfigurationFromHub` plus `PreTrainedTokenizer.init`:
parsing the 32 MB JSON with yyjson into `Config` values, then building the vocabulary, merges
and added-token structures for 262,144 entries. It is paid once per process. Where the time
goes inside was not profiled; if start-up matters later, that is the place to look.

## The entry point

`SwiftTransformersTokenizer.load(from:)` does what `Tokenizers.AutoTokenizer.from(modelFolder:)`
does, in the same two steps, `LanguageModelConfigurationFromHub(modelFolder:)` then
`PreTrainedTokenizer(tokenizerConfig:tokenizerData:)`, with one line between them that sets
`clean_up_tokenization_spaces` to false when `tokenizer_config.json` does not set it (departure
2 below). `AutoTokenizer.from(modelFolder:)` itself was used first and gave the same encode
results; the direct construction is kept because it is the only way to fix the default.

mlx-swift-lm's path, `#huggingFaceTokenizerLoader()` from `MLXHuggingFace`, was loaded once in
the tests (`MLXTokenizerLoaderTests`). The macro expands to a `TokenizerLoader` whose
`load(from:)` calls `Tokenizers.AutoTokenizer.from(modelFolder:)` and wraps the result in a
private `TokenizerBridge` struct that forwards `encode`, `decode`, `convertTokenToId` and
`applyChatTemplate`. It gives the same ids as the direct loader for every corpus text and
every chat prompt, the same decodes except the three trailing-byte rows, and the type name
`TokenizerBridge`. Loaded cold in a fresh process it takes 3.8 s and 236 MB, the same as the
direct path within run-to-run noise. OpenJevSwift will not use it, because:

- it hides the upstream `Tokenizers.Tokenizer`, so the configuration cannot be adjusted and
  the object cannot be shared with code that needs the `Tokenizers` API;
- it pulls `MLXHuggingFace`, the `MLXHuggingFaceMacros` compiler plugin (swift-syntax) and, by
  the default trait, `MLXFoundationModels` into the model target (risk R18);
- the mlx-swift-lm revision pinned here does not depend on swift-transformers itself; the
  macro only works because this package already does. There is nothing to gain from the
  indirection.

If a later mlx-swift-lm loader wants a `TokenizerLoader`, a five-line conformance over
`SwiftTransformersTokenizer` provides one.

## Decode departures

Both were found by the fixture run and the tests that followed it; both are general, not tied
to particular texts, and both are pinned by a test over the unmodified swift-transformers path
(`MLXTokenizerLoaderTests/upstreamDecodeDepartures`) so that a swift-transformers update that
fixes them fails that test and the adapter's code can be removed.

### 1. Byte tokens at the end of a sequence are dropped (decoder)

Classification: decoder, `ByteFallbackDecoder`. Sources/Tokenizers/Decoder.swift lines 173 to
204 in swift-transformers 1.3.4 gather consecutive `<0xNN>` tokens into `byteTokens` and decode
them as UTF-8 when a non-byte token follows, but the loop never flushes `byteTokens` after the
last token. Python's `tokenizers` (decoders/byte_fallback.rs) flushes at the end.

Minimal reproduction, with the pinned tokenizer loaded through `AutoTokenizer.from(modelFolder:)`:

```swift
tokenizer.decode(tokens: [238], skipSpecialTokens: false)                    // "" in 1.3.4; Python: "\0"
tokenizer.decode(tokens: [482, 381, 429, 429], skipSpecialTokens: false)     // "" in 1.3.4; Python: "\u{10FFFF}"
tokenizer.decode(tokens: [238] + tokenizer.encode(text: "zero", addSpecialTokens: false),
                 skipSpecialTokens: false)                                   // "\0zero" in both
```

Estimated fix upstream: three lines after the loop in `ByteFallbackDecoder.decode` (flush
`byteTokens` as the other branch does), plus a test. Impact on OpenJev: reads never decode;
`think` decodes the thought the model wrote and a chat completion decodes its answer, so a
thought or answer ending in a byte-fallback character (rare: a control character, U+2028, or a
code point outside the vocabulary) would lose its last character.

What the adapter does: `SwiftTransformersTokenizer.decode` drops the special tokens first when
asked (from `specialTokenIDs`, the ids of every special token `tokenizer_config.json` names,
24 for this checkpoint), then splits off the trailing run of `<0xNN>` tokens, decodes the rest
with swift-transformers and the run as UTF-8 with U+FFFD per maximal invalid subpart, which
is what `String::from_utf8_lossy` does in `tokenizers`. Bytes followed by an ordinary token are
still decoded by swift-transformers, so its behaviour is unchanged wherever it was right.

### 2. `clean_up_tokenization_spaces` defaults to true (post-processing)

Classification: decoder post-processing, `PreTrainedTokenizer.cleanUp(text:)`. Line 532 of
Sources/Tokenizers/Tokenizer.swift reads
`tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)`. The checkpoint's
`tokenizer_config.json` has no such key. Python's `transformers` has defaulted
`clean_up_tokenization_spaces` to `False` since 4.45 (the fixtures were generated with 5.17.0),
so Python returns the decoded text unchanged while swift-transformers rewrites `" ."` to `"."`,
`" ,"` to `","`, `" 's"` to `"'s"` and so on.

Minimal reproduction:

```swift
let ids = tokenizer.encode(text: "a , b .", addSpecialTokens: false)
tokenizer.decode(tokens: ids, skipSpecialTokens: false)   // "a, b." in 1.3.4; Python: "a , b ."
```

No corpus row contains one of the rewritten patterns, which is why the fixture run did not
show it: it was found by reading `cleanUp` while fixing departure 1. The Python side is
therefore inferred from transformers' default and from the corpus README's observation that
every row round-trips, not recorded; follow-up C below records it.

Estimated fix upstream: change the default to false, or make it depend on the tokenizer class
as `transformers` did before 4.45; one line and a test. What the adapter does: sets
`clean_up_tokenization_spaces` to false in the configuration when the file does not set it,
before constructing `PreTrainedTokenizer`. A checkpoint that sets it true keeps it.

## Where the tests find the files, and CI

The suite reads the tokenizer directory from `OPENJEV_TEST_TOKENIZER` (a directory holding just
the three files), else `OPENJEV_TEST_MODEL` (the checkpoint directory, which holds them beside
the weights), else the Hugging Face cache snapshot the fixture generator downloads. The files
are checked against the fixture digests before loading. Without them the suites skip with a
comment naming `OPENJEV_TEST_MODEL`: the files are part of the checkpoint, so these are model
opt-in tests in the sense of docs/09-conformance-and-testing.md, and that is the one skip
`.github/scripts/check-test-log.sh` accepts. Hosted CI therefore builds the suite but does not
run it; the results above come from a developer machine, as the model tests' will. Follow-up E
below would make CI run it.

## Does OpenJevSwift need a custom loader?

No. The BPE model, normalizer (`Replace` of a space with `▁`), pre-tokenizer (`Split " "
MergedWithPrevious`), added-token splitting and byte fallback in swift-transformers 1.3.4 all
agree with Python's `tokenizers` on every text in the fixtures. The two departures are in the
decode path, are three lines each to fix upstream, and are handled in the adapter until then.

## Follow-up issues to create

A. Report the trailing byte token drop to huggingface/swift-transformers. Scope: the minimal
reproduction above, a pull request adding the flush after the loop in
`ByteFallbackDecoder.decode` with a test over `[<0x00>]` and a four-byte sequence, and, once a
release with the fix is pinned here, removal of the trailing-run code in
`SwiftTransformersTokenizer.decode` and of the first half of `upstreamDecodeDepartures`.

B. Report the `clean_up_tokenization_spaces` default to huggingface/swift-transformers. Scope:
the reproduction above with a pointer to transformers' 4.45 change, a pull request changing
`boolean(or: true)` to `boolean(or: false)` with a test, and, once pinned, removal of the
configuration line in `SwiftTransformersTokenizer.load` and the second half of
`upstreamDecodeDepartures`.

C. Extend the tokenizer corpus so the two departures are recorded by Python rather than
inferred. Scope: add rows to `Tools/fixtures/upstream_tables.py` for texts with `" ."`, `" ,"`,
`" ?"`, `" !"`, `" 's"`, `" n't"` patterns, for byte tokens before, between and after special
tokens (`decode([2, 238, 1], skip_special_tokens=True)`), and for an incomplete byte sequence
(`decode([482, 381])`); bump the script's `version`, regenerate the fixtures and extend
`TokenizerParityTests/decodeDepartures` to read the new rows instead of literals.

D. Record the image message shape for #46. Scope: add to `upstream_tables.py` a
`chat-prompts/image_prompts.json` rendered with the processor's `apply_chat_template` over
`[{"type": "image"}] * n + [{"type": "text", "text": state}]` for n in 1 and 2, with
`tokenize=False` and the ids of the text, so `ChatTemplateParityTests/imageMessages` compares
with a recording instead of the template's own text.

E. Run the tokenizer parity suite in CI. Scope: a step in the macOS job of
`.github/workflows/ci.yml` that fetches the four tokenizer files of the pinned revision (32 MB;
`fixtures.yml` already caches the same download keyed on the revision) into a directory and
exports `OPENJEV_TEST_TOKENIZER` before `swift test`, so the suite runs on every pull request
instead of skipping as a model test. Kept out of this pull request because the CI workflow was
landing in parallel (#77).
