# DiffusionGemma 26B-A4B: what the Swift port must reproduce

Sources: the [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it), the
[mlx-community 4-bit checkpoint](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)
(`config.json`, `generation_config.json`, `tokenizer_config.json`), and mlx-vlm 0.6.15's
`mlx_vlm/models/diffusion_gemma/` (2,013 lines of Python, MIT), which upstream's MLX backend calls
and which the mlx-vlm authors verified against the Transformers implementation to within
`1.6e-7` maximum absolute difference.

## 1. Architecture

| Item | Value |
|---|---|
| `model_type` / architecture | `diffusion_gemma` / `DiffusionGemmaForBlockDiffusion` |
| Parameters | 25.2B total, 3.8B active (MoE), plus a ~550M vision tower |
| Text layers | 30, hidden 2816, 16 query heads |
| Attention pattern | 5 sliding-window layers then 1 full-attention layer, repeated; the last layer is forced to full attention |
| Sliding layers | head_dim 256, 8 KV heads, window 1024, RoPE default with theta 10,000, separate `v_proj` |
| Full layers | head_dim 512, 2 KV heads, RoPE "proportional" with `partial_rotary_factor` 0.25 and theta 1,000,000, **no `v_proj`: values are the keys** (pre-norm) passed through a parameter-free RMSNorm |
| Norms | `q_norm`, `k_norm` (RMSNorm with scale), `v_norm` (RMSNorm without scale); attention scale 1.0; four layer norms around attention and the feed-forward pair; RMS eps 1e-6 |
| Feed-forward | Dense GeGLU MLP (intermediate 2112) **and** a sparse MoE branch (128 experts, top-8, expert intermediate 704) in every layer; outputs are normed separately, summed, normed again, added to the residual |
| Router | RMS-normalise the input (no scale), multiply by a learned `scale` and `hidden_size^-0.5`, project to 128 scores, top-8, softmax over the 8, multiply by `per_expert_scale[indices]` |
| Embeddings | Vocabulary 262,144; tied input/output projection (`embed_tokens.as_linear`); embeddings scaled by `sqrt(hidden)` |
| Final logits | `tanh(x / 30) * 30` soft-capping, computed in float32 |
| Encoder and decoder | **Shared text weights.** The encoder is an autoregressive pass over the prompt (causal, with sliding windows) that fills the KV caches; the only encoder-specific parameters are 30 per-layer scalars (`layer_scalar`). The decoder is a bidirectional pass over a canvas of noise tokens that attends to the encoder KV plus itself. |
| Decoder extras | `layer_scalar` per decoder layer; a `self_conditioning` module (pre-norm → GeGLU MLP → post-norm) that adds the previous step's soft embeddings to the canvas embeddings |
| Canvas | 256 tokens by default (`canvas_length`); OpenJev uses 64 (vLLM `--diffusion-config canvas_length 64`) and, on MLX, a width rounded to 16 up to 64 |
| Context | 262,144 positions |
| Vision | Gemma 4 vision tower + multimodal embedder; a budget of 280 soft tokens per image (`max_soft_tokens`; 70, 140, 560 and 1,120 are budgets only the video processor accepts): each image is resized, preserving its aspect ratio, to the largest sides that are multiples of 48 and fit 2,520 patches of 16 pixels, so it gets `(H / 16) * (W / 16) / 9` soft tokens, at most 280 (253 for upstream's hot dog photo; `size` 224 by 224 in `processor_config.json` is not used); Pillow bicubic, rescaled by 1/255, not normalised; `<|image>` (`boi`) 255999, `<image|>` (`eoi`) 258882, `<|image|>` (placeholder and soft token) 258880; vision tokens attend bidirectionally within their own image block (`use_bidirectional_attention: "vision"`). See [spikes/vision-preprocessing.md](spikes/vision-preprocessing.md) |

Special token ids that the read path depends on: `pad` 0, `eos` {1, 106, 50}, `bos` 2,
`<turn|>` 106 (upstream's `TURN_CLOSE`). `<end_of_turn>` is not a token of this vocabulary: it
encodes as seven ordinary tokens ([Fixtures/tokenizer/README.md](../Fixtures/tokenizer/README.md)).
Thought-channel markers `<|channel>` and `<channel|>`; `<|think|>` at the start of the system
prompt enables thinking. The tokenizer class
is `GemmaTokenizer` (a byte-level BPE in `tokenizer.json`, 32 MB); the chat template ships as a
separate `chat_template.jinja` (Gemma 4's template with tools and channels) rather than inside
`tokenizer_config.json`.

## 2. Inference passes

### Encoder (prefill)

`encoder(input_ids, attention_mask, cache, pixel_values, mm_token_type_ids)`:

1. Embed tokens (image placeholders replaced by `pad` before embedding, then overwritten by
   projected vision features), scale by `sqrt(hidden)`.
2. Per layer: causal mask, sliding layers additionally masked to the window; with images, an
   overlay lets tokens of the same image block see each other.
3. Each layer runs with `decoder=False`, updating its KV cache (`KVCache` for full layers,
   `RotatingKVCache(max_size=1024)` for sliding layers, which after a one-piece prefill keeps
   every prompt position), and multiplies its output by the encoder's `layer_scalar` for that
   layer.
4. Return the final norm (unused by reads) and the caches.

Chunked prefill exists for long text-only prompts. Prompts with images are prefilled in one
piece because chunking cannot yet split on image-block boundaries. Upstream's reads never chunk
(`MlxRuntime._prefill`), and the port follows: in bfloat16 a chunked prefill moves read
probabilities by up to 0.62 (spike #22, D-036).

### Decoder (one denoise step)

`decoder(canvas_ids, cache, self_conditioning_*, decoder_attention_mask)`:

1. Embed the canvas and scale, then run the `self_conditioning` module:
   `post_norm(embeddings + down(geglu(gate(pre_norm(signal)), up(pre_norm(signal)))))`, where
   `post_norm` is an RMSNorm without a weight (so the checkpoint has no tensor for it). The
   signal is zeros on the first step, and the module still runs on them; afterwards it is
   `softmax(previous logits) @ embed_tokens.weight * sqrt(hidden)`. With quantized embeddings the
   implementations pass the logits themselves ("prefers logits self-conditioning") and project
   inside with a quantized matmul against the packed embedding.
2. Build masks per layer type: full-attention layers see every valid encoder position and the
   whole canvas; sliding layers see only the last `window − 1 = 1023` encoder positions plus the
   whole canvas. Canvas positions attend to each other bidirectionally.
3. Per layer, with `decoder=True`: compute Q, K, V for the canvas at `offset = encoder length`
   (RoPE positions continue after the prompt), concatenate the encoder's cached K, V (sliding
   layers: only the last 1023) in front of the canvas K, V, attend, and continue through the
   dense and MoE branches. The encoder caches are not modified.
4. Final norm, tied output projection, float32 soft-cap. Result: logits `[1, canvas, 262144]`.

### A "read" in upstream's sense

Given a prompt (chat template with the system and user messages, generation prompt on, thinking
off) and a canvas built as in [01-upstream-openjev.md](01-upstream-openjev.md) section 5.5:

1. Prefill the prompt once (cached).
2. Run exactly one decoder step, its self-conditioning signal zeros.
3. For each slot position, take the logits row, log-softmax in float32 (temperature 1), and keep
   the entries for the top 20 tokens plus every label id.

With `steps > 1`, between steps only the slot positions are overwritten with their argmax, and
self-conditioning from the previous logits is applied; every other canvas position keeps the
template token (what vLLM calls pinning). The final step's logits are read.

Note for the port: a read needs logits only at the slot positions, and the full-row logits are
used only for self-conditioning between steps. Projecting only the slot rows through the
262,144-wide tied head is close but not bit-identical: the smaller quantized matmul rounds
differently, and under the oracle's kernels only 29 of 156 slots matched the full projection.
The port projects every row (D-036).

### Text generation (for `think` and `/v1/chat/completions`)

mlx-vlm's `stream_diffusion_generate` (1,252 lines) implements the checkpoint's published policy
(`generation_config.json`): block-autoregressive canvases (min/max canvas length, full canvas
option), the entropy-bound sampler (`entropy_bound` 0.1) with a linear temperature schedule from
`t_max` 0.8 to `t_min` 0.4 over `max_denoising_steps` 48, a confidence-threshold sampler as an
alternative, self-conditioning between steps, stable-and-confident stopping (`confidence_threshold`
0.005, `stability_threshold` 1), EOS handling for {1, 106, 50} plus caller stop ids, committing a
finished block into the encoder cache (`diffusion_update_cache`) and a streaming detokenizer with
`skip_special_token_ids`. Upstream calls it at temperature 0. The Layr-Labs fork has a native
Swift version of this loop (see [04-swift-inference-landscape.md](04-swift-inference-landscape.md)).

What upstream actually runs (`MlxRuntime.generate`, which passes only `max_tokens`, the skipped ids
and `temperature=0.0`), and what the port reproduces (D-059):

1. **One block.** A canvas of `min(256, max(remaining, 64))` random ids from MLX's generator. Then
   up to 48 decoder passes over the encoder cache. Each pass divides the logits by the step's
   temperature, `0.4 + 0.4 × step / 48` for the countdown `step` 48 to 1, and takes the argmax. The
   last pass stops there. Otherwise the default **confidence-threshold** sampler, not the
   checkpoint's entropy-bound one, accepts the unrevealed positions whose probability is at least
   0.9 (at least the most probable one). It keeps them and re-noises every other position with
   fresh random ids. The next pass is conditioned on these logits (the quantized embedding's soft
   embeddings). A block ends when every position is accepted, when the argmax canvas equals the
   previous step's and its mean entropy is below 0.005, or after the 48th pass. The block is the
   last pass's argmax. On the recorded replies every block ended with every position accepted, in
   2 to 21 passes.
2. **Between blocks.** The whole block goes through the encoder after the prompt
   (`diffusion_update_cache`), so the next block attends to it. A full layer grows its buffer by
   256 positions. A sliding layer keeps its last 1,023 positions and appends the block.
3. **The reply.** The block's tokens are taken in order. The first EOS (1, 106, 50) or caller stop
   id ends it (`stop`) and is not returned; the `max_tokens`-th token ends it (`length`). Each token
   goes through the SentencePiece streaming detokenizer (`SPMStreamingDetokenizer`,
   `trim_space=False`), skipped ids dropped first. A word's text comes out with the token after it,
   and the last word comes after the last token.
4. **Randomness.** Even greedy, the initial canvas and the re-noised positions are random, and they
   change which positions are accepted when. Upstream never seeds MLX's generator, so its replies
   differ from one process to the next. The port seeds a generator for every reply.

## 3. The checkpoint the port will load

`mlx-community/diffusiongemma-26B-A4B-it-4bit` (revision `a7a81407`, converted with mlx-vlm
0.6.3, 16.58 GB):

| File | Size |
|---|---|
| `model-0000{1..4}-of-00004.safetensors` | 5.22 + 5.36 + 5.37 + 0.60 GB |
| `model.safetensors.index.json` | weight map |
| `config.json` | includes `quantization` and `generation_config` |
| `tokenizer.json` | 32 MB |
| `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`, `generation_config.json` | small |

Quantization: affine, group size 64, 4 bits by default, with an explicit per-module override map
to 8 bits for `model.decoder.embed_tokens`, every `self_attn.{q,k,v,o}_proj`, every dense
`mlp.{gate,up,down}_proj` and every `router.proj`. Experts stay at 4 bits. Weight names follow
`model.decoder.layers.N.*`; the encoder contributes only `model.encoder.language_model.layers.N.layer_scalar`
and the vision tower (`model.encoder.vision_tower.*`, `model.encoder.embed_vision.*`). mlx-vlm's
`sanitize` renames `experts.gate_up_proj` and `experts.down_proj` to `.weight`, drops
`rotary_emb` buffers and `lm_head.weight` (tied), and drops vision clipping calibration tensors
when unused. Storage dtypes are U32 (packed weights) and BF16 (scales, biases, norms).

`8bit` and `bf16` conversions exist in the same collection; upstream's cache-limit note says one
read on those can need more than 4 GB of pool.

## 4. Numbers upstream reports for this checkpoint on MLX

- About 16 GB to load; 17 GB resident idle on an M4 Pro.
- 30 to 50 MB per cached prefill; 27 GB after 360 unique prompts; 36 GB in an audit run.
- `mx.set_cache_limit(4 GB)` gave 23.5 GB with byte-identical answers.
- One 3-question read: 0.2 to 0.4 s (M3 Ultra), 0.39 s (M4 Max). 16 concurrent requests: about
  4 req/s. Reads are serialised on one thread.
- Prefill throughput on the Layr-Labs Swift port, M3 Ultra: about 1,650 tok/s at 10k context;
  peak memory 25 GiB during generation.

## 5. Facts that shape the design

- Only one decoder pass per read, so the expensive part is prefill plus one forward over a canvas
  of at most 64 tokens through 30 hybrid dense+MoE layers. MoE expert gathers dominate.
- The decoder is bidirectional and re-run per step, so canvas KV cannot be cached across steps;
  encoder KV can and must be cached across re-reads and samples.
- Sliding layers' decoder mask uses the last 1023 encoder positions in temporal order. With a
  rotating cache the physical order differs from the temporal order; mlx-vlm reorders
  (`_temporal_order`). Prompts longer than 1024 tokens exercise this path and need a parity test.
- Noise tokens are random vocabulary ids, not a `[MASK]` token. The result depends slightly on
  the noise, which is why re-reads average over seeds and why reproducing upstream's seeds matters
  for conformance.
- The vocabulary is 262,144 wide; a full-canvas logits tensor in float32 is `64 × 262144 × 4 B`
  = 64 MB per read. Slot-only projection avoids most of it.
