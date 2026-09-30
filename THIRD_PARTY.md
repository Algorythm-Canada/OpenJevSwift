# Third-party references

This project ports behaviour and, later, code from the projects below. Commits are pinned
because several of them change daily. Update this table when a pin moves.

| Project | Role for OpenJevSwift | Pinned revision | License |
|---|---|---|---|
| [razorback16/openjev](https://github.com/razorback16/openjev) | Upstream. The wire API, engine algorithm, MLX backend and tests are the compatibility target. | `dcd2094` (v0.5.0, 2026-09-29) | Apache-2.0 |
| [vllm-project/vllm PR #57250](https://github.com/vllm-project/vllm/pull/57250) | Origin of the structured-read mechanism (`structured_server.py`) that upstream's `engine.py` adapts. Merged 2026-09-22. | vLLM `1b3b88ec` (upstream's Docker pin) | Apache-2.0 |
| [TypeSafe API docs](https://docs.typesafe.ai/) and [typesafe-ai/typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) | The Jev contract and the official client whose behaviour the server must satisfy. | SDK `f078f1e` (v0.7.2, 2026-09-26) | Docs proprietary; SDK per its repository |
| [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | The model. | Model card as of 2026-09-29 | Apache-2.0 plus Gemma Terms of Use |
| [mlx-community/diffusiongemma-26B-A4B-it-4bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) | The MLX checkpoint upstream's MLX backend loads by default; the checkpoint this port will load. | `a7a81407` | Apache-2.0 plus Gemma Terms of Use |
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `mlx_vlm/models/diffusion_gemma` and `generate/diffusion.py` | Reference implementation of DiffusionGemma on MLX (Python). The Swift port follows it. | `v0.6.15` (upstream's pin) | MIT |
| [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) | Array framework. | 0.32.2 | MIT |
| [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | Model primitives (SwitchGLU, KV caches, RoPE, Gemma 4 text and vision, tokenizer protocols, loaders). | `c043fb3` (2026-09-28) | MIT |
| [Layr-Labs/mlx-swift-lm](https://github.com/Layr-Labs/mlx-swift-lm) | Fork with a native DiffusionGemma Swift implementation (PR #157, merged 2026-09-26). Used as a second reference and numeric oracle, not as a dependency. | `eeba2af` (2026-09-29) | MIT |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | Tokenizers (GemmaTokenizer as BPE) and Jinja chat templates. | `af520cf` (2026-09-23) | Apache-2.0 |
| [python/cpython](https://github.com/python/cpython) `Modules/_randommodule.c` and `Lib/random.py` | `random.Random(int)` seeding, `getrandbits` and `randrange`, ported in `Sources/OpenJevCore/Random/` so canvases match upstream's. The MT19937 reference code by Matsumoto and Nishimura that CPython builds on keeps its notice in the file header. | CPython 3.14.7 (the fixtures' interpreter) | PSF-2.0; MT19937 reference code BSD-3-Clause |
| [python/cpython](https://github.com/python/cpython) `Modules/_json.c` and `Lib/json/decoder.py` | How `json.loads` refuses a document: its messages, positions and checks, ported in `Sources/OpenJevCore/JSON/PythonJSONLoads.swift` so the server's `json_invalid` 422 matches upstream's. | CPython 3.14.7 | PSF-2.0 |
| [fastapi/fastapi](https://github.com/fastapi/fastapi) and [Kludex/starlette](https://github.com/Kludex/starlette) | Upstream's web framework: how a request body is read (content type, `json_invalid`, "There was an error parsing the body"), how headers are read, and the error bodies the server reproduces. Behaviour only; no code is copied. | FastAPI 0.142.1, Starlette 1.7.0 (the fixtures' versions) | MIT; BSD-3-Clause |
| [hummingbird-project/hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP server framework. | 2.23+ | Apache-2.0 |
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) and [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | The `laya-1.0` model and its input format. | package 0.3.6 (upstream's pin); 0.3.22 current | Apache-2.0 |
| [Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) and [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) | The `verdict-1.4` model, prompt contract and calibrator. | v1.4 inference | Apache-2.0 |
| [allebee/jevk5](https://github.com/allebee/jevk5) and [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5) | The `jevk5-0.2` model and its letter-readout prompt. | `f26426d` (2026-09-28) | Apache-2.0 |
| [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) and [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | The `clm-v0.1` heads over Qwen3-8B. | package 0.1.0 | Apache-2.0 |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | Formerly also called OpenJev. A different project (direct logits from autoregressive models, NLI framing). Its readout is what JevK5 uses. Related work, not upstream. | `master` as of 2026-09-23 | MIT |
| [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) | JevBench v1, the public benchmark upstream reports against. | current | see repository |
