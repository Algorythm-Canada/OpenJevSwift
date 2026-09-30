# Tokenizer fixtures

What the pinned DiffusionGemma tokenizer (`mlx-community/diffusiongemma-26B-A4B-it-4bit` at
revision `a7a81407613811e8ba63af92ac0d852b809e191f`, loaded with
`transformers.AutoTokenizer.from_pretrained`) produces. `Tools/fixtures/upstream_tables.py` writes
these files; see [../README.md](../README.md) for regenerating them.

## Files

| File | Contents |
|---|---|
| `special_tokens.json` | The tokenizer's special tokens and ids, upstream's token constants, the checkpoint's own special ids, the tokenizer pipeline options and the digests of the files read |
| `corpus.json` | Token ids, pieces and decodes for 917 texts |
| `engine_encodings.json` | Every text upstream's `Engine.enc` tokenized while the fixtures were made, with its ids |

### special_tokens.json

| Key | Contents |
|---|---|
| `tokenizer_class`, `vocabulary_size` | `GemmaTokenizer`, 262,144 ids, which is upstream's `VOCAB` |
| `named` | Every special token the tokenizer names, with its id: `bos`, `eos`, `unk`, `pad`, `mask` and the Gemma 4 tokens for images, audio, the thought channel, turns, thinking and tools |
| `extra_special_tokens` | `<\|video\|>` |
| `added_tokens` | The 24 added tokens with their `special`, `lstrip`, `rstrip`, `normalized` and `single_word` flags |
| `engine` | Upstream's constants and what they are in this vocabulary: `TURN_CLOSE` 106 is `<turn\|>`, `PAD` 0 is `<pad>`; the scaffold `<\|channel>thought\n<channel\|>` is `[100, 45518, 107, 101]` |
| `end_of_turn_text` | `<end_of_turn>` is not a token of this vocabulary. It encodes as seven ordinary tokens. Upstream's docs name 106 after it; here 106 is `<turn\|>`. |
| `model_config` | From the checkpoint's `config.json` at the same revision: end ids 1, 106 and 50 (`<eos>`, `<turn\|>`, `<\|tool_response>`), image token 258880, image start 255999 and end 258882, 280 soft tokens per image |
| `pipeline` | `tokenizer.json`'s normalizer, pre-tokenizer, post-processor, decoder and BPE options, without the vocabulary and merges |
| `files` | SHA-256 and size of `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `config.json` |

### corpus.json

`cases` holds one row per text:
`{text, categories, ids, ids_with_special_tokens, pieces, decoded, decoded_skip_special_tokens, round_trip}`.

- `ids` is `tokenizer.encode(text, add_special_tokens=False)`, what `Engine.enc` calls.
  `ids_with_special_tokens` is the same call with `add_special_tokens=True`.
- `pieces` is `convert_ids_to_tokens(ids)`. `decoded` is `decode(ids)`, and
  `decoded_skip_special_tokens` is `decode(ids, skip_special_tokens=True)`. `round_trip` says
  whether `decoded` equals `text`.
- `categories`: `label_candidate` (all 728 candidates after `"q1: "`), `marker` (the scaffold,
  the thought markers, `<end_of_turn>`, turn markers and answer fragments), `system_text` (every
  text in `system-texts/`), `answer_text` (each group's first-label answer text in `templates/`,
  alone and after the sequential join), `state`, `json_state` (states rendered with
  `json.dumps(state, ensure_ascii=False)`), `cjk`, `emoji`, `combining`, `whitespace` (leading,
  trailing and repeated spaces, tabs, newlines), `byte_fallback`, `special_token_text` and
  `misc`. The alternative answer texts are in `engine_encodings.json`, with their ids.

What the rows show:

- `add_special_tokens=True` adds nothing: the post-processor has no special tokens, and the chat
  template writes `<bos>` itself.
- Every row decodes back to its text.
- Special tokens written as plain text are special tokens: a state containing `<turn|>` encodes
  it as id 106.
- Characters outside the vocabulary fall back to byte tokens such as `<0xF4>`.

### engine_encodings.json

`cases` holds 2,634 pairs `[text, ids]`, sorted by text in code point order: every text
upstream's `Engine.enc` tokenized while the fixtures were made. That is label discovery, the
answer template of every group and every alternative label, the trial texts of `groups()`, and
the answer lines sequential reads write into the prompt. Together with
`chat-prompts/prompts.json` this is enough for a replay tokenizer to drive the fixture tests of
`OpenJevCore` on Linux, where swift-transformers is not available. On macOS the same pairs check
the real tokenizer.
