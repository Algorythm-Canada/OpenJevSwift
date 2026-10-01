# The other System One models upstream serves

Upstream puts four other people's models behind the same API. Each has a fixed input format and
calibration that a Swift port must reproduce exactly. Facts below come from upstream's
`encoders.py` and tests, the model repositories and the checkpoints' configuration files.

## Verdict (`verdict-1.4`)

| Item | Value |
|---|---|
| Author, repository | Heman10x, [Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev), v1.4 inference engine |
| Checkpoint | [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m): `GLiClassModel` uni-encoder over ModernBERT-base (base `knowledgator/gliclass-modern-base-v2.0`), 151.4M parameters; `model.safetensors` 605 MB, `model.onnx` 606 MB, `model_fp16.onnx` 304 MB, `calibrator.json`, `tokenizer.json` 3.6 MB; Apache-2.0 |
| Input format | `<<LABEL>>label1<<LABEL>>label2...<<SEP>>text`. Noul: labels `true: {ins}` and `false: not {ins}`, text `Context:\n{context}\n\nEvaluate proposition: {ins}`. Choice: labels `It is {desc or name}`, text `Question: {ins}\n\nContext:\n{context}` (or the context alone without instructions). Score: labels `{desc} (Value: {float(i)})`. Every question gets a final label `insufficient evidence`. |
| Limits | 512 tokens, truncated silently (v1.4 cut 1024 to 512 to avoid positional drift); 24 options (25 logits); `class_token_index` 50368 |
| Calibration | `calibrator.json`: global temperature 2.8039 and a `per_k` table keyed by label count (2: 5.0069, 3: 5.0069, 4: 4.0314, 5: 3.0560, 6: 2.3898, 7: 2.3898, 9: 1.6668, 11: 3.3919, 17: 1.7200, 25: 1.5144). Upstream divides the first k logits by `per_k[k]` (falling back to the global temperature), softmaxes, drops the abstention entry, renormalises. |
| Upstream deviations | Ignores noul `criteria`; text only; batches of 16 sequences; `usage.input_tokens` is the attention-mask sum. |
| Swift feasibility | Built (#57): `VerdictBackend` in `OpenJevEncoders` runs the float16 Core ML package `verdict-m18-fp16` (D-011), converted by `Tools/encoders/convert_verdict.py`, one function per shape (batch 1 and 16 by 128, 256 and 512 tokens). On a Mac it reads up to 16 questions per call on the GPU, on an iPhone one per call on the Neural Engine. The prompt, the swift-transformers tokenization and the calibration match upstream on all 200 reference questions, exactly and within 1e-6. Core ML on an M3 Max stays within 0.0014 of PyTorch float32 on the GPU and 0.0076 on the Neural Engine (spike #56 bound: 0.02). The package, tokenizer and calibrator are downloaded on first use and checked by SHA-256 (D-033). No MLX port (D-011). |

## Laya (`laya-1.0`)

| Item | Value |
|---|---|
| Author, repository | Nandakishor M / Convai Innovations, [laya](https://github.com/NandhaKishorM/laya), package `laya` (upstream pins 0.3.6; 0.3.22 current) |
| Checkpoint | [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions): ModernBERT-large encoder (421M), `model.safetensors` 843 MB, `encoder/config.json`, `rl_agent_config.json`, `tokenizer/`; Apache-2.0. Fine-tuned on 2,000 labelled decisions across four workflows; reported 0.766 accuracy on the typed-decisions benchmark |
| Input format (`build_sequence`) | `[CLS] <type> question: <instructions> [SEP] [MASK]<opt0> [MASK]<opt1> ... [SEP] <state> [SEP]`. Options render as `label: description` (or the bare label), `level i: description` for scores, and for noul `false: <desc or "no, the statement does not hold">` then `true: <desc or "yes, the statement holds">` (semantic order false, true). Each option is capped at 48 tokens; the head (question plus options) is capped at `head_max_len` = 256; if fewer than 16 tokens remain, options are cut to `max(4, (256 − 16) / n)` tokens each; the question text is cut to what remains (at least 8 tokens); the state fills the rest of `max_len` = 1024 (right-truncated by default). The `[MASK]` positions are the option markers. |
| Head | `DecisionModel`: encoder hidden states plus a question-type embedding, a 2-layer `TransformerEncoder` head (norm-first, `hidden/64` heads), a scorer MLP (`LayerNorm → Linear → GELU → Linear(1)`) read at the marker positions, softmax over markers. An action head (`act_head`) exists but its output is unused for decisions. |
| Calibration | `rl_agent_config.json`: `temperature` per question type `[1.0148, 1.0374, 1.0575]` and `temperature_by_options` buckets (`noul:2` 1.9834, `choice:2` 1.9064, `choice:3-5` 1.7602, `choice:6-10` 1.0000, `choice:11+` 0.1006, `score:3-5` 1.2514). The `laya` package clamps temperatures to [0.5, 5.0] because `choice:11+` sharpens instead of softening. Probabilities are rounded to 4 decimals; upstream renormalises. |
| Limits | 1,024 tokens including options; up to 255 options but the head budget makes about 20 the practical maximum. |
| Swift feasibility | Same encoder family as Verdict at 3x the size, plus a custom head with a Transformer. Core ML tracing of the whole `DecisionModel` is plausible; parity must include the rounding and clamping rules. |

## JevK5 (`jevk5-0.2`)

| Item | Value |
|---|---|
| Author, repository | Alibi Serikbay, [jevk5](https://github.com/allebee/jevk5) (package 0.2.2) |
| Checkpoint | [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5): Qwen3.5-4B with a merged LoRA distilled from Qwen3.6-27B; `model.safetensors` 8.4 GB bf16; `jevk5_config.json` with the calibration temperature (1.532 for v0.2); Apache-2.0 |
| Prompt | System: "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning." User: `json.dumps({"evidence": state, "criterion": instructions, "options": [{"letter": "A", "description": ...}, ...]}, ensure_ascii=False)`. Rendered through a pinned Qwen3.5 chat template string with thinking off: `<\|im_start\|>system\n{system}<\|im_end\|>\n<\|im_start\|>user\n{user}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n`. |
| Options | `decision_options`: noul → `("true", desc or "The proposition is true.")`, `("false", ...)`; choice → its criteria; score → level indices. Each option text is `"{id}: {description}"`. Letters `A`..`P` (16). |
| Readout | The next-token logits of the 16 letter tokens at the answer position (each letter must be a single token); `softmax(logits / 1.532)`. More than 16 options: `spread` splits into near-equal groups, reads each, runs a "knockout" final of 16 and combines (sharpened by temperature 0.77); a "tree" method also exists. Each question is a separate pass; `usage.input_tokens` counts every pass. |
| Limits | 16,384 tokens, 400 beyond, never truncated. |
| Upstream results | On JevBench's 231 public items, upstream's vLLM path and JevK5's own run agreed on all 231 top answers and token counts; probabilities differed by 0.0012 median and 0.055 maximum. |
| Swift feasibility | Highest of the four: `MLXLLM` already runs Qwen3.5 (`Qwen35.swift`), the readout is a single forward pass reading 16 logits, the prompt is a fixed string. Needs an MLX conversion of the checkpoint (4-bit about 2.5 GB, 8-bit about 4.5 GB) that OpenJevSwift can point at; conversion is a one-time `mlx_lm.convert` step whose output could be published under the organisation's Hugging Face account or produced locally. Runs on Macs and, at 4-bit, on recent iPhones and iPads. |

## CLM (`clm-v0.1`)

| Item | Value |
|---|---|
| Author, repository | Contrastive-LM, [CLM](https://github.com/Contrastive-LM/CLM), package `contrastive-lm` 0.1.0 |
| Checkpoint | [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B): `CLM_v0.1-8B.pt` (75.6 MB): a state head and an action head, MLPs to 512 dimensions over the last-token embeddings of a frozen Qwen3-8B; Apache-2.0 |
| Mechanism | Embed the state with the question appended, and each option, with Qwen3-8B (vLLM pooling runner, `pooling_type LAST`); project through the heads; answer is a softmax over scaled cosines. 2,048 tokens; upstream truncates from the left so the question survives. Caches embeddings and projections. `usage.input_tokens` counts only texts that had to be embedded. |
| Known issue | Score questions can ignore the state (upstream CLM issue #3). |
| Swift feasibility | Qwen3-8B exists in `MLXLLM`; last-token hidden states are available; the heads are tiny MLPs (convert `.pt` to safetensors once). Memory: about 5 GB at 4-bit. Deferred behind the others because of its quality caveat and the extra conversion step. |

## What a Swift "encoder backend" shares

Upstream's `EncoderEngine` contract, which the Swift `OpenJevCore` will mirror as a sibling of
`DecisionBackend`:

- `build_schema` with the same forced answers and limits (`max_choices` 24 for Verdict, 255
  otherwise).
- A 400 for `images`, `steps > 1`, `samples > 1`, `think` and `sequential`
  (`"{model} does not support {field}"`).
- Batched reads of at most `OPENJEV_ENCODER_BATCH` (16) questions per forward pass.
- A distribution over the caller's options in the caller's order; noul is `[P(true), 1 − P(true)]`.
- Deterministic: the seed is unused.
- Its own `/v1/models` entry with the upstream description text and release date, and acceptance
  of `jev-latest` and `jev-preview` when it is the only model in the process.
