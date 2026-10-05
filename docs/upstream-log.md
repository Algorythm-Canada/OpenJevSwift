# Upstream log

How this repository follows the projects it ports and depends on (issue #66), and a note for each
review. [THIRD_PARTY.md](../THIRD_PARTY.md) holds the pins. This page holds what was checked
against them, when, and what it meant for compatibility. Newest review first.

## The process

### When a review runs

- **Monthly.** The Upstream review workflow
  ([upstream-review.yml](../.github/workflows/upstream-review.yml)) runs on the first day of each
  month at 06:37 UTC. It runs `Tools/upstream/review.py`. When any project moved past its pin,
  failed to be read, or disagrees with another pin, it opens or updates one tracking issue labelled
  `area/ci`. There is one issue, updated in place: its body is replaced with the new review and a
  comment names what moved since the last update. A closed issue is reopened when a later review
  differs from the one it was closed with, and an issue whose body already shows the current state
  is left alone.
- **When a pinned project releases,** or before a pull request that moves a pin. Run the review by
  hand ([development.md](development.md#the-upstream-review)) or start the workflow from the
  Actions tab.

GitHub disables the schedule of a public repository's workflow after 60 days without activity in
the repository. The workflow's page then says so, and enabling it again is one click.

### What a review checks

The script gathers the facts. The reviewer reads the diffs it points at and decides.

| Project | What to look for | Where the report shows it |
|---|---|---|
| razorback16/openjev | Wire changes: routes, validation and error bodies in `api.py` and `chat.py`. Engine and read-policy changes in `engine.py` and `config.py`: templates, canvases, seeds, slot distributions, `OPENJEV_AUTO_THRESHOLD`, `OPENJEV_AUTO_MAX` and the 7919 seed step. New models or backends: a new module under `openjev/`, a new extra in `pyproject.toml`, a new Dockerfile. The mlx-vlm pin in `pyproject.toml`. Changes to upstream's tests, which the fixtures and the live suite mirror. Open pull requests show what may land next. | Files changed, by area (changed lines that name the re-read policy add "read policy"); the mlx-vlm requirement and the backends at the pin and at `main`; open pull requests |
| Blaizzy/mlx-vlm | Numerics fixes in `mlx_vlm/models/diffusion_gemma`, and in the modules its read imports: `gemma4`, `cache.py`, `switch_layers.py`, `rope_utils.py` and `base.py`. A fix to the read path changes what the port must compute, and the exact tier (D-014) would fail against a regenerated oracle. `generate/diffusion.py` matters for generation. Upstream OpenJev's pin decides which mlx-vlm the oracle runs. | Commits that touch what this repository uses; open pull requests about diffusion |
| ml-explore/mlx-swift-lm | API changes in what the DiffusionGemma port uses: `SwitchLinear`, `QuantizedSwitchLinear`, `gatherSort` and `scatterUnsort`; `loadWeights`; `BaseConfiguration`'s quantization types; `BaseLanguageModel`. `Gemma4Processor`, which only the Vision tests use (the library ports its own processor, tower and configuration, D-051 and D-054). Changes to `MLXLLM`'s Qwen3.5 text model (`Qwen35TextModel`, `Qwen35TextConfiguration`), which JevK5 runs whole (D-052), and to what its load and forward pass reach: `Qwen3NextMLP` and `Qwen3NextRMSNormGated`, `gatedDeltaUpdate` and its Metal kernel, the fused input projection that `loadWeights` prepares, the caches and masks, attention and RoPE, the compiled traces, and the MTP state keys through which the model returns its hidden states. Such a change can move JevK5's letter logits, which its opt-in live tests hold to `Fixtures/jevk5/reads.json`, or change how its checkpoint loads. Its mlx-swift requirement. New tags, since release 0.1.0 needs one at or after the pin. A DiffusionGemma model or pull request upstream (D-050). | Commits that touch what this repository uses; the mlx-swift requirement at the pin and at `main`; open pull requests about diffusion |
| ml-explore/mlx-swift | Releases, since the pin is exact. Changes to the vendored MLX core (`Source/Cmlx/mlx`, which computes every kernel) and to the Swift API. Platform fixes. | Releases; commits that touch what this repository uses |
| Layr-Labs/mlx-swift-lm | Changes to the fork's DiffusionGemma, the second reference and one of spike #22's oracles. | Commits that touch what this repository uses |
| huggingface/swift-transformers | Tokenizer changes (BPE, byte fallback, normalizers), which spike #20's parity rests on. Releases newer than the 1.3.4 that `Package.resolved` resolves. | Commits that touch what this repository uses; `Package.resolved` |
| The checkpoints on the Hub | A moved `main`. A download that names no revision, as upstream's server makes when no `OPENJEV_*_MODEL` setting names one, then gets other files than the pin. The port loads pinned revisions, so its own answers do not move. | "`main` is now at" |

A review also reports when the Makefile, `Package.resolved` or `Tools/oracle/requirements.txt`
disagree with THIRD_PARTY.md.

### What a review produces

1. A dated note below: what moved in each project and what it means for compatibility.
2. An issue in this repository, labelled `area/ci`, for each change that affects compatibility,
   with the evidence. Nothing else gets an issue.
3. The tracking issue closed once the note is written, or left open for the pull request that
   moves the pins to close.

### What moving a pin takes

One pull request per move, with:

1. **THIRD_PARTY.md and every other copy of the pin.** For upstream OpenJev: the Makefile's
   `UPSTREAM_OPENJEV_COMMIT`, the fixture scripts' pins, `FixturePinTests` and
   `Tools/encoders/common.py`. For the Swift packages: `Package.swift` and `Package.resolved`, which
   `swift package resolve` writes from the old file (SwiftPM 6.4 leaves the file alone while its
   pins stay the same, so a hand edit keeps a stale `originHash`). For mlx-swift and mlx-swift-lm
   also `Tools/oracle/UpstreamProbe`'s manifest and resolved file, since that package depends on
   this one by path and cannot resolve against other versions, and for mlx-swift `MLX_SWIFT_VERSION`
   in [mlx-probe.yml](../.github/workflows/mlx-probe.yml). For mlx-vlm:
   `Tools/oracle/requirements.txt` and `Tools/jevbench/requirements-upstream.txt`. For a checkpoint:
   `ModelSource.swift` or the encoder manifests, and for the 4-bit one, which is also the fixtures'
   tokenizer, `TOKENIZER_REVISION` in `Tools/fixtures/upstream_tables.py` and `FixturePinTests`.
2. **`make upstream` and the fixtures.** Check out the new commit, then `make fixtures-venv` and
   `make fixtures`. The [Fixtures workflow](../.github/workflows/fixtures.yml) regenerates every
   fixture on the pull request and fails on any byte that differs from the committed files.
3. **The oracle rerun,** when a read's inputs or its arithmetic may change: upstream's engine or
   MLX backend, mlx-vlm, MLX or the checkpoint. `Tools/fixtures/mlx_vlm_oracle.py` writes
   `Fixtures/oracle/reads.json` on the reference Mac. Then `ReadOracleTests` runs in both tiers,
   and `Tools/oracle/tolerance_stats.py` recomputes D-048's bounds if the fixture changed.
4. **The regression file.** `RegressionTests` must reproduce the port's own answers. The file
   records the mlx-swift version and the checkpoint, so a move of either is followed by a new
   recording with `OPENJEV_RECORD_REGRESSION=1`, once the reads are confirmed.
5. **A decision entry** in [06-decisions.md](06-decisions.md): what moved, what changed in the
   fixtures and the reads, and what the port changed to match.
6. **A note below,** and the tracking issue closed by the pull request.

A move of a package whose code ends up in the binaries, mlx-swift above all, also checks that they
still launch on the oldest systems the package declares, macOS 14 and iOS 17: `nm -m` on the built
`openjev` lists what it imports, and a symbol imported strongly that those systems lack stops every
binary at launch there, before any of its code runs (#119, D-053).

## Reviews

### 2026-10-04: mlx-swift 0.32.3 and mlx-swift-lm 3.32.3

Not a review: the pin move the 2026-10-02 review asked for in
[#119](https://github.com/Algorythm-Canada/OpenJevSwift/issues/119), made by pull request
[#126](https://github.com/Algorythm-Canada/OpenJevSwift/pull/126), which closes the issue. D-053
records it.

- **mlx-swift 0.32.2 to 0.32.3.** The one commit replaces `Logger.isEnabled(type:)` with
  `os_log_type_enabled`. The vendored MLX core is untouched, and the compiled `default.metallib`
  keeps its SHA-256 (`282550b0…`). Built from main, `openjev`, `openjev-bench` and the MLX test
  bundles import the missing symbol strongly; built on 0.32.3, none does. In the iOS Simulator, test
  bundles that link `OpenJevDiffusionGemma` or `OpenJevLetterReadout` stop at dyld on iOS 26.2 and
  18.5 with 0.32.2 ("Symbol not found") and load with 0.32.3.
- **mlx-swift-lm `c043fb3` to 3.32.3.** Of its five commits, two change library code: named image
  attachments (MLXLMCommon's chat and user input, MLXVLM's processors and models) and tool
  parameter booleans (MLXFoundationModels). Nothing the port takes from it changed, and MLXLLM,
  whose Qwen3.5 JevK5 runs on, did not change at all.
- **What the move checked, on the reference Mac.** Both tiers of `ReadOracleTests` give 0.32.2's
  figures, the exact tier all 63 reads bit for bit; `RegressionTests` reproduced every recorded
  value before the file was recorded again for 0.32.3; JevK5's live tests on the 4-bit conversion
  give #122's figures; the Vision tests, full tensors included, pass as before. JevK5's 28 corpus
  requests get byte-identical answers from main's build and this one's, at 8 and at 4 bits.
- **Copies of the pin the move found:** `Tools/oracle/UpstreamProbe`, which depends on this package
  by path and no longer resolved, and `mlx-probe.yml`. The process above now lists both, how to
  write `Package.resolved`, and the launch check.

### 2026-10-02

Run at 20:12 EDT (00:12 UTC on 2026-10-03) on the reference Mac with
`python3 Tools/upstream/review.py --gh ghp`, upstream OpenJev read from `Upstream/openjev` after
`make upstream`, the other projects through the REST API. Fingerprint `9b44ec67581c96b5`. Four of
the twelve pins have moved. One change affects compatibility, and it has an issue:
[#119](https://github.com/Algorythm-Canada/OpenJevSwift/issues/119). The Makefile,
`Package.resolved` and `Tools/oracle/requirements.txt` agree with THIRD_PARTY.md.

| Project | Pinned | Now | Since the pin |
|---|---|---|---|
| razorback16/openjev | `dcd2094` (2026-09-29) | `dcd2094` on main | nothing new |
| ml-explore/mlx-swift | 0.32.2 `2b5e877` (2026-09-28) | release 0.32.3 (2026-09-30); `1960120` on main | 1 release, 1 commit |
| ml-explore/mlx-swift-lm | `c043fb3` (2026-09-28) | release 3.32.3 (2026-09-30); `9afc3b5` on main | 1 release, 12 commits |
| Blaizzy/mlx-vlm | v0.6.15 `d734bd2` (2026-08-18) | release v0.7.4 (2026-09-28); `6ecadd7` on main | 8 releases, 366 commits |
| Layr-Labs/mlx-swift-lm | `eeba2af` (2026-09-29) | `363609d` on main | 63 commits |
| huggingface/swift-transformers | `af520cf` (2026-09-23) | release 1.3.4 (2026-09-02); `af520cf` on main | nothing new |
| google/diffusiongemma-26B-A4B-it | as of 2026-09-29 | `f7f5b7f` on main, modified 2026-07-15 | nothing new |
| mlx-community/diffusiongemma-26B-A4B-it-4bit | `a7a8140` | `a7a8140` on main, modified 2026-07-15 | nothing new |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | `7b95e38` | `7b95e38` on main, modified 2026-07-15 | nothing new |
| mlx-community/diffusiongemma-26B-A4B-it-bf16 | `2cd36f9` | `2cd36f9` on main, modified 2026-07-15 | nothing new |
| convaiinnovations/laya-typed-decisions | `1a793eb` | `1a793eb` on main, modified 2026-09-24 | nothing new |
| heman10x/rlcd-modernbert-151m | `8af2496` | `8af2496` on main, modified 2026-09-20 | nothing new |

- **razorback16/openjev: nothing new.** `dcd2094` is still the head of `main`, as on 2026-10-01.
  Upstream has no release or tag. Its `pyproject.toml` still requires `mlx-vlm==0.6.15` and
  declares the backends `mlx`, `laya` and `verdict`. One pull request is open, #10, "Add ForJev
  backend for typed decisions on an existing Qwen/vLLM server" (opened 2026-09-29, not updated
  since). If it merges, a review will see a new module and a new backend, and the port gets an
  issue then.
- **ml-explore/mlx-swift: affects compatibility.** 0.32.3 (2026-09-30) is one commit,
  "fix MLXLogger (#493)". 0.32.2 calls `os.Logger.isEnabled(type:)`, which macOS and iOS lack
  before 26.4. This package declares macOS 14 and iOS 17, and its `openjev` and `openjev-bench`
  binaries import the symbol strongly, so they would stop at launch there. The fix changes
  `Source/MLX/Logging.swift` and adds a C target. The vendored MLX core, MLXNN and MLXFast are
  unchanged, so no kernel moved. Issue
  [#119](https://github.com/Algorythm-Canada/OpenJevSwift/issues/119) has the evidence and
  what moving to 0.32.3 takes. No Mac or iPhone below 26.4 was available to show the crash.
- **ml-explore/mlx-swift-lm: no API the port uses changed; a tag now holds the pin.** 3.32.3
  (`3b339ad`, 2026-09-30) contains `c043fb3` and five commits after it, and `main` is twelve
  commits after it. `SwitchLayers.swift`, `Load.swift`, `BaseConfiguration.swift` and
  `LanguageModel.swift` are unchanged. `Gemma4.swift` changed once, 0493daf, in the message
  generator and processor; the port takes only `Gemma4VisionConfiguration` from that file. Its
  `Package.swift` now requires `.upToNextMinor(from: "0.32.3")` (390c5fd). 3.32.3 is the first tag
  at or after the pin, which release 0.1.0 (#65) needs. #119 proposes moving to it together with
  mlx-swift 0.32.3. Nothing named diffusion has merged. Pull request ml-explore/mlx-swift-lm#352,
  "Base Gemma diffusion implementation", is open and was rewritten on 2026-10-02; D-050 decides
  what this repository offers it, and the draft is [below](#draft-for-ml-exploremlx-swift-lm-not-posted).
- **Blaizzy/mlx-vlm: no fix to the read's arithmetic, as far as the diffs show; upstream still
  pins v0.6.15.** Eight releases since v0.6.15, up to v0.7.4 (2026-09-28), and 366 commits. In
  `mlx_vlm/models/diffusion_gemma`, two commits: b099148 strips channel markers from decoded text,
  and d8e6195 passes the automatic prefix cache to generation. `language.py` and `config.py`, which
  hold the read, are unchanged. The four commits to `generate/diffusion.py` change generation:
  the prefix cache, sampling at a small top-p or a low temperature, and CLI options. Of the
  modules the read imports, `gemma4` (8 commits), `cache.py` (16), `switch_layers.py` (2),
  `rope_utils.py` (1) and `base.py` (5) changed, mostly for other models, quantized KV caches,
  audio, batching and the prefix cache. Two hunks were read in full. a8715d5 changes
  `RotatingKVCache.make_mask`, the sliding-window mask of R15, but only for a cache that holds
  per-sequence lengths (batched requests); a single read gets the mask it had. 66e68ce adds a
  `stop_gradient` to Gemma 4's own router, which the read does not use and which no forward pass
  notices. Only an oracle run on a newer mlx-vlm can confirm that reads are unchanged, and none is
  due while upstream OpenJev pins 0.6.15. Related work, not a compatibility change: mlx-vlm added a
  typed decision API on 2026-09-30, `mlx_vlm.decision.predict` with choice, score, bool and
  multi-label questions (42d5ef1, #2396), Laya on it (06ff63d, #2397), and a decision CLI (f409f7f,
  #2402).
- **Layr-Labs/mlx-swift-lm: nothing that changes the reference.** 63 commits from 2026-09-30 to
  2026-10-02: model fixes, MiMo, the server and tests. None touches the fork's DiffusionGemma
  sources. Two change its DiffusionGemma tests: a046367 runs the bit-exact reference tests only on
  reference hardware, and f80c08c adds kernel tests for generation, caches, switch layers and
  samplers.
- **huggingface/swift-transformers: nothing new.** `af520cf` is still the head of `main`, and
  1.3.4, which `Package.resolved` resolves, is still the newest release.
- **The checkpoints: nothing new.** Every `main` is at its pin. The base model's card was last
  modified on 2026-07-15, before the 2026-09-29 that THIRD_PARTY.md records.

The tracking issue does not exist yet: the workflow opens it on its first run after the pull
request for #66 merges.

## Draft for ml-explore/mlx-swift-lm (not posted)

**Not posted.** This is the text D-050 recommends offering upstream. Posting it, and where, is the
maintainers' decision. ml-explore/mlx-swift-lm's AGENTS.md asks agents not to post for a user,
and its CONTRIBUTING.md asks whoever posts AI-drafted prose to read every word and disclose how AI
was used. Read and edit it, and confirm the AI usage line, before posting.

Where: as a comment on ml-explore/mlx-swift-lm#352, which is where the implementation is reviewed.
If a comment on someone else's pull request seems out of place, post it as a discussion titled
"DiffusionGemma: reads compared with mlx-vlm on the 4-bit checkpoint", with the first sentence
naming #352.

> OpenJevSwift (https://github.com/Algorythm-Canada/OpenJevSwift, Apache-2.0) ports DiffusionGemma's
> read path to Swift for a decision server: the prompt prefill and decoder passes over a canvas, on
> mlx-swift 0.32.3 and mlx-swift-lm 3.32.3, with no generation loop. We are not opening a second
> DiffusionGemma pull request, since this one covers more (generation and vision) and is already in
> review. What we can offer it is a comparison with mlx-vlm on the real checkpoint, which the unit
> tests here cannot make.
>
> - **63 reads recorded from mlx-vlm.** mlx-vlm 0.6.15 on MLX 0.32.2 with
>   `mlx-community/diffusiongemma-26B-A4B-it-4bit` at `a7a81407`: canvases 16 to 64 tokens wide,
>   prompts of 78 to 3,643 tokens (13 past the 1,024-token window), 1 to 3 steps. For every slot a
>   read keeps the float32 log-probabilities of the top 20 tokens and of every label, and each
>   prompt keeps digests of its prefill cache (layers 0 and 29). The file is
>   [Fixtures/oracle/reads.json](https://github.com/Algorythm-Canada/OpenJevSwift/blob/main/Fixtures/oracle/reads.json).
> - **A bit-for-bit comparison.** With the `mlx.metallib` of the mlx-metal wheel and mlx-vlm's RoPE
>   frequency table, our port gives all 63 reads bit for bit, caches included. On mlx-swift's own
>   kernels it differs in the last bit of the JIT-compiled `pow` and of the RoPE kernel, which moves
>   label probabilities by 0.013 on average; mlx-vlm with a chunked prefill moves them as much.
> - **What changed the bits for us:**
>   - The router. `Gemma4TextRouter` folds the scale into the norm weight and uses a plain softmax.
>     That is exact in real numbers but rounds differently from mlx-vlm's router.
>   - `experts.gate_up_proj` is fused and quantized. We load it into one `SwitchLinear` with 1,408
>     outputs and split after the gathered matmul, as mlx-vlm does.
>   - Self-conditioning: the precise softmax of the previous float32 logits, then `quantizedMM`
>     against the packed embedding with `transpose: false`, then the embedding scale. This is the
>     `_embed_canvas` point raised in review on 2026-08-21.
>   - A chunked prefill differs from a prefill in one piece by up to 0.6 in a label's probability,
>     in mlx-vlm against itself, so the recorded reads use one piece.
>
> If it is useful, we can run this pull request's model on those 63 reads and post the comparison,
> read by read, under both sets of kernels. That needs only the public 4-bit checkpoint. The method
> and the measurements are in
> [docs/spikes/backend-validation.md](https://github.com/Algorythm-Canada/OpenJevSwift/blob/main/docs/spikes/backend-validation.md)
> and decisions D-014 and D-048 of
> [docs/06-decisions.md](https://github.com/Algorythm-Canada/OpenJevSwift/blob/main/docs/06-decisions.md).
>
> AI usage: drafted with Claude (Anthropic) from our repository's measurements; I read and edited
> every word. The port itself was written with AI assistance, and its parity tests are what we
> rely on.
