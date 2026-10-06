# Generation oracle fixtures

What upstream's own MLX generation returns on the pinned checkpoint, for the generation and
`think` parity tests (issues #50 to #52, D-058; layer 2 in
[docs/09-conformance-and-testing.md](../../docs/09-conformance-and-testing.md)).

## generation.json

[Tools/fixtures/generation_oracle.py](../../Tools/fixtures/generation_oracle.py) runs upstream's
`openjev.mlx_backend.MlxRuntime.generate` and `MlxEngine.decide` from upstream at `dcd2094`, which
run mlx-vlm 0.6.15's `stream_diffusion_generate` on MLX 0.32.2 with
`mlx-community/diffusiongemma-26B-A4B-it-4bit` at revision
`a7a81407613811e8ba63af92ac0d852b809e191f`, greedily (`temperature=0.0`) as upstream calls it.
Every run is made twice, the second time in reverse order with emptied prefill caches, and the
file is written only when the two agree bit for bit.

Upstream never seeds MLX's generator, which draws each block's initial canvas and the positions
re-noised at every step, so its replies differ from one process to the next. The script seeds it
with `mx.random.seed(seed)` before each reply and each `decide`, and records the seed;
`MLXRandom.RandomState(seed:)` in mlx-swift draws the same values (`sampler.initialize_canvas`).

- `generator` records the pins: upstream commit, checkpoint repository and revision (it is also
  the tokenizer), Python and package versions, the GPU and the SHA-256 of the wheel's
  `mlx.metallib`.
- `settings` is what mlx-vlm used: the checkpoint's `generation_config` as mlx-vlm loaded it, the
  EOS ids its stopping criteria held (1, 106, 50), the canvas length (256) and minimum (64), the
  step cap (48), the sampler (`confidence-threshold`, the default, at 0.9), the temperature (0),
  the detokenizer class (`SPMStreamingDetokenizer`, `trim_space` false), and upstream's
  thought-open and thought-close ids and read scaffold.
- `sampler` holds test vectors for each sampling function of `generate/diffusion.py` lines 285 to
  505, computed on the CPU: two canvas draws after each of three seeds; the 48 step temperatures
  (as Python floats and as the float32 the logits are divided by); categorical and argmax samples
  of synthetic logits at four temperatures after a seed; token probabilities; token entropies of
  two logit sets; entropy-bound masks at bounds 0.1, 0.5 and 0, with ties; confidence masks at
  four thresholds with and without `force_all`; and the stable-and-confident result over a
  sequence of six steps for four stopping settings. Arrays are `{shape, values}` or, for floats,
  `{shape, float32_bits}`.
- `generations` lists seven chat replies, each prompt built by upstream's `MlxGenerator.prompt_ids`
  (thinking off, then the read scaffold) and generated with the thought-channel markers skipped,
  as `MlxGenerator.generate` skips them: `short_answer`, `list` and `json` (#51's three prompts),
  `story` (640 tokens over blocks of 256, 256 and 128), `stop_comma` (ended by the extra stop id of
  `","`), `story_cut` (cut by `max_tokens` 40) and `short_answer_seed_7` (another seed). Each has
  the messages, `max_tokens`, the stop strings and ids, the skipped ids, the seed, the prompt ids,
  then what upstream returned: `prompt_tokens`, `finish_reason`, `stop_token` (the id that ended a
  `stop` reply), `generated` (the ids, without the stop id), `text` and `pieces` (every
  `emit(text, token)` call in order, the last with a null token for the buffered tail), and
  `blocks`. A block has its `canvas_length`, `initial_canvas`, `steps` (decoder passes),
  `canvas_draws` (random canvases drawn: the initial one and one per pass that did not stop at its
  argmax), `ended` (`all_revealed`, `stable_and_confident` or `max_denoising_steps`),
  `final_canvas` (the last pass's argmax, the whole canvas) and `committed` (the ids the reply took
  from it).
- `think` lists `MlxEngine.decide` with `think` for upstream's README example (`think` 128) and for
  24 sequential nouls (`think` 64): the request body, its seed, MLX's seed, then per thought the
  system and user texts, the prompt (thinking on, then the open marker), the budget and stop ids,
  the ids `MlxRuntime.generate` returned with its prompt tokens and finish reason, the thought after
  the cut, the prefix the read continues, and the thought tokens and prompt tokens billed; then the
  request's answers, `input_tokens` and `output_tokens`.

Note what the `list` and `json` texts show: upstream's chat path skips `enc("<|channel>thought\n")`
and `enc("<channel|>")`, and the first is three ids, the open marker, `thought` (45518) and the
newline (107), so a chat reply loses every newline and every `thought` token (D-058).

The run's timings are in `Tools/oracle/results/generation_run.json`, never in this file.
