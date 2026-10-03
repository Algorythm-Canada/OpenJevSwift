# JevK5 fixture

`reads.json` holds what upstream OpenJev's `JevK5Engine.read_question` (razorback16/openjev at
`dcd2094`) sends and returns for 204 questions, with the model's letters read from the 4-bit MLX
conversion of JevK5 v0.2 through mlx-lm instead of vLLM; `Tools/jevk5/reference.py` writes it
([Tools/jevk5/README.md](../../Tools/jevk5/README.md) has the commands). The prompt, the option
texts and the readout of more than 16 options are the `jevk5` package's own (allebee/jevk5 at
v0.2.2, `0571ef3`, Apache-2.0); the tokenizer is the checkpoint's (`alibiserikbay/JevK5` at its
`v0.2` tag, `ea4804e`) as transformers loads it.

| Key | Contents |
|---|---|
| `generator` | The script and its version, the upstream, package and checkpoint pins, the conversion's SHA-256, the Python and package versions, the prefill step and the CPU |
| `temperature` | The calibration temperature from the conversion's `jevk5_config.json`, 1.532 |
| `tokenizer` | The letters' ids (`A` to `P` are 32 to 47), vLLM's `max_chars_per_token` for the vocabulary (128) and its size |
| `requests` | Per request, the input tokens upstream bills and the forced questions it answers without a read |
| `corpus` | The 26 requests of [Fixtures/encoders/corpus.json](../encoders/README.md), with the choices it cuts to 24 options for Verdict restored whole (40, 55, 100 and 255 options), and two requests whose states (about 5,000 and 11,000 tokens) take several prefill chunks |
| `reads` | Per read question: its options as `decision_options` writes them; every pass, with its option texts, its prompt (the text when it has at most 6,000 characters, the SHA-256 always), its length in Unicode scalars, its token count, the SHA-256 of its ids written as decimal numbers joined by commas, and the letters' logits in float32; the distribution and the token count upstream returns |
| `spreads` | `spread` over generated logits in 16 cases: ties at every cut, finals of more than 16 options, the tree method; each pass's texts and logits and the result |

The model-free tests in `Tests/OpenJevLetterReadoutTests` replay it: the prompts byte for byte,
the readout and every answer bit for bit, the billing exactly. The opt-in live tests
(`OPENJEV_JEVK5_MODEL`) hold the Swift tokenizer to the recorded ids and the Swift model to the
recorded logits. The file is ASCII: a text's punctuation is a JSON escape.
