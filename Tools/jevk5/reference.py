#!/usr/bin/env python3
"""Record Fixtures/jevk5/reads.json: JevK5's prompts, passes and answers through upstream's own code.

Upstream OpenJev's `JevK5Engine` (razorback16/openjev at dcd2094, `openjev/encoders.py`) asks a
vLLM server for its letters' logprobs. Here the same engine runs with `letter_logprobs` replaced
by the converted checkpoint on MLX through mlx-lm; everything else is upstream's and the `jevk5`
package's own code: `EncoderEngine.build_schema`, `JevK5Engine.read_question` and its letter
softmax, and `jevk5.prompt` (`decision_options`, `prompt_text`, `spread`). The file holds:

    corpus      the 26 requests of Fixtures/encoders/corpus.json, with the choices Verdict's
                24-option limit cut there restored whole (20, 40, 55, 100 and 255 options), so
                that `spread` reads them in several passes, and two requests whose states (about
                5,000 and 11,000 tokens) take several prefill chunks
    reads       per read question: its options as `decision_options` writes them, every pass
                (the option texts, the prompt, its token ids from the checkpoint's tokenizer as
                transformers loads it, and the letters' logits from the 4-bit conversion), and
                the distribution and token count upstream returns
    spreads     `spread` over generated logits: ties at every cut, more than 256 options (the
                recursion) and the tree method, each pass's texts and logits and the result
    tokenizer   the letters' ids and vLLM's `max_chars_per_token` for the tokenizer

Setup, once, from the repository root:

    make upstream
    PY=~/Library/Caches/OpenJevSwift/jevk5/venv/bin/python
    /usr/local/bin/python3.12 -m venv ~/Library/Caches/OpenJevSwift/jevk5/venv
    $PY -m pip install -r Tools/jevk5/requirements.txt
    $PY -m pip install --no-deps -r Tools/jevk5/requirements-jevk5.txt
    $PY Tools/jevk5/convert.py --bits 4

Then `$PY Tools/jevk5/reference.py`. A pass runs as Swift's `Qwen35LetterReadoutModel` runs it:
one call up to 2,048 tokens, chunks of 2,048 through the caches beyond, the head applied to the
last position only. MLX's buffer pool is capped at 4 GB.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = ROOT / "Upstream" / "openjev"
UPSTREAM_COMMIT = "dcd2094"
FIXTURE = ROOT / "Fixtures" / "jevk5" / "reads.json"
SCRIPT = "Tools/jevk5/reference.py"
# Bump when the shape of the file changes.
GENERATOR_VERSION = 1
JEVK5_COMMIT = "0571ef373722dd10cace82c81580d7b675fe6b53"
DEFAULT_MODEL = Path.home() / "Library" / "Caches" / "OpenJevSwift" / "jevk5" / "jevk5-0.2-mlx-4bit"
PREFILL_STEP = 2048
# A prompt's text is kept in the file when it is at most this long; every prompt keeps its digest.
KEEP_PROMPT_CHARS = 6000

for _name in [n for n in os.environ if n.startswith("OPENJEV_")]:
    del os.environ[_name]
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(HERE))

import convert  # noqa: E402  (Tools/jevk5/convert.py: the pins and the qwen3_5_text model)

# The corpus rules of Tools/encoders/reference.py, which builds Fixtures/encoders/corpus.json.
# Its TRUNCATE cut is what Verdict's 24 options need; JevK5 reads the choices whole.
TRUNCATE = {
    ("indexed_12_mixed", "c2"): 6,
    ("indexed_12_mixed", "c5"): 7,
    ("indexed_12_mixed", "c8"): 12,
    ("indexed_12_mixed", "c11"): 20,
    ("widest_schema", "a"): 24,
    ("many_choices", "first"): 16,
    ("many_choices", "second"): 24,
    ("many_choices", "third"): 10,
}
OWN_STATE = [
    "quickstart", "six_questions", "lines_10_mixed", "indexed_11_mixed", "indexed_12_mixed",
    "nouls_8", "widest_schema", "many_choices", "described_objects_and_arrays", "non_ascii",
    "whitespace", "noul_criteria_variants", "choice_description_variants", "choice_names_special",
    "score_described_levels", "long_instructions", "state_object", "state_list", "state_floats",
    "state_empty_string", "state_escapes",
]
MIXED = [
    ("quickstart", "department"), ("quickstart", "frustration"), ("quickstart", "is_urgent"),
    ("six_questions", "is_repeat"), ("six_questions", "product"), ("six_questions", "tone"),
    ("lines_10_mixed", "n0"), ("lines_10_mixed", "c1"), ("lines_10_mixed", "s2"),
    ("indexed_12_mixed", "n0"), ("indexed_12_mixed", "s1"), ("indexed_12_mixed", "c2"),
    ("widest_schema", "a"), ("widest_schema", "b"), ("widest_schema", "c"),
    ("many_choices", "first"), ("many_choices", "second"), ("many_choices", "third"),
    ("many_choices", "flag"),
    ("described_objects_and_arrays", "pick"), ("described_objects_and_arrays", "rate"),
    ("described_objects_and_arrays", "flag"),
    ("non_ascii", "équipe"), ("non_ascii", "niveau"), ("non_ascii", "urgent"),
    ("whitespace", "padded"), ("whitespace", "levels"), ("whitespace", "flag"),
    ("noul_criteria_variants", "only_true"), ("noul_criteria_variants", "only_false"),
    ("noul_criteria_variants", "empty"), ("noul_criteria_variants", "null"),
    ("noul_criteria_variants", "both"),
    ("choice_description_variants", "opts"), ("choice_names_special", "names"),
    ("score_described_levels", "lvl"), ("long_instructions", "a"), ("long_instructions", "b"),
]
TRANSCRIPT_TEXTS = ["quickstart", "non_ascii", "whitespace", "state_escapes"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_digest(ids: list[int]) -> str:
    """The SHA-256 of the ids written as decimal numbers joined by commas."""
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def build_corpus(cut: bool) -> list[dict]:
    """The requests of Tools/encoders/reference.py's corpus, with or without its cut."""
    from openjev.encoders import WARMUP_QUESTIONS

    data = json.loads((ROOT / "Fixtures" / "schemas" / "schemas.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in data["cases"] if "schema" in c}

    def question(name, qid):
        q = json.loads(json.dumps(cases[name]["questions"][qid]))
        n = TRUNCATE.get((name, qid)) if cut else None
        if n is not None:
            q["criteria"] = dict(list(q["criteria"].items())[:n])
        return q

    def transcript(count):
        texts = [cases[n]["request"]["state"].strip() for n in TRANSCRIPT_TEXTS]
        return "\n".join(f"Message {i + 1}: {texts[i % len(texts)]}" for i in range(count))

    def conversation(count):
        texts = [cases[n]["request"]["state"] for n in TRANSCRIPT_TEXTS]
        customer = cases["state_object"]["request"]["state"]["customer"]
        return {"customer": customer, "messages": [texts[i % len(texts)] for i in range(count)]}

    requests = []

    def add(name, source, state, pairs):
        qs = {}
        for case, qid in pairs:
            key = qid if qid not in qs else f"{case}.{qid}"
            qs[key] = question(case, qid)
        requests.append({"name": name, "source": source, "state": state, "questions": qs})

    for name in OWN_STATE:
        case = cases[name]
        pairs = [(name, qid) for qid, q in case["questions"].items()
                 if not (q["type"] in ("choice", "score") and len(q["criteria"]) == 1)]
        add(name, f"schemas.json {name}, own state", case["request"]["state"], pairs)
    add("state_object_six", "schemas.json six_questions, state of state_object",
        cases["state_object"]["request"]["state"],
        [("six_questions", q) for q in cases["six_questions"]["questions"]])
    requests.append({"name": "warmup", "source": "upstream encoders.WARMUP_QUESTIONS, state \"warmup\"",
                     "state": "warmup", "questions": json.loads(json.dumps(
                         {k: dict(v, criteria=v.get("criteria")) for k, v in WARMUP_QUESTIONS.items()}))})
    add("medium_transcript", "the mixed set, state: a 16-message transcript of the fixture's text states",
        transcript(16), MIXED)
    add("long_transcript", "the mixed set, state: a 30-message transcript of the fixture's text states",
        transcript(30), MIXED)
    add("long_conversation", "schemas.json nouls_24, state: an 80-message conversation as JSON",
        conversation(80), [("nouls_24", q) for q in cases["nouls_24"]["questions"]])
    return requests


def long_requests() -> list[dict]:
    """Two requests past one prefill chunk: the quickstart questions under a 220-message
    conversation, and one noul under a 450-message one, both JSON states built as the corpus
    builds its long conversation."""
    data = json.loads((ROOT / "Fixtures" / "schemas" / "schemas.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in data["cases"] if "schema" in c}
    texts = [cases[n]["request"]["state"] for n in TRANSCRIPT_TEXTS]
    customer = cases["state_object"]["request"]["state"]["customer"]

    def conversation(count):
        return {"customer": customer, "messages": [texts[i % len(texts)] for i in range(count)]}

    quickstart = json.loads(json.dumps(cases["quickstart"]["questions"]))
    first_noul = next(iter(cases["nouls_8"]["questions"]))
    return [
        {"name": "chunked_conversation", "source": "schemas.json quickstart, state: a 220-message "
         "conversation as JSON (several prefill chunks)", "state": conversation(220),
         "questions": quickstart},
        {"name": "chunked_conversation_long", "source": "schemas.json nouls_8's first noul, state: a "
         "450-message conversation as JSON", "state": conversation(450),
         "questions": {first_noul: json.loads(json.dumps(cases["nouls_8"]["questions"][first_noul]))}},
    ]


def check_corpus(full: list[dict]) -> None:
    """The cut corpus must be Fixtures/encoders/corpus.json's requests exactly, so the whole one
    holds the same questions."""
    recorded = json.loads((ROOT / "Fixtures" / "encoders" / "corpus.json").read_text(encoding="utf-8"))
    if build_corpus(cut=True) != recorded["requests"]:
        sys.exit("the corpus rules no longer give Fixtures/encoders/corpus.json; update this script")
    if [r["name"] for r in full[:len(recorded["requests"])]] != [r["name"] for r in recorded["requests"]]:
        sys.exit("the whole corpus does not start with the encoder corpus's requests")


class Model:
    """The converted checkpoint on MLX through mlx-lm, read as the Swift backend reads it."""

    def __init__(self, folder: Path):
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from transformers import AutoTokenizer

        mx.set_cache_limit(convert.CACHE_LIMIT_BYTES)
        convert.register_text_model()
        self.mx = mx
        self.model, self.config = load_model(folder)
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder))
        self.letter_ids = [self.tokenizer.encode(letter, add_special_tokens=False)
                           for letter in "ABCDEFGHIJKLMNOP"]
        if any(len(ids) != 1 for ids in self.letter_ids):
            sys.exit(f"every answer letter must be one token, got {self.letter_ids}")

    def letter_logits(self, ids: list[int], letter_ids: list[int]) -> list[float]:
        mx = self.mx
        inputs = mx.array([ids])
        text_model = self.model.model
        cache = self.model.make_cache() if len(ids) > PREFILL_STEP else None
        start = 0
        while len(ids) - start > PREFILL_STEP:
            text_model(inputs[:, start:start + PREFILL_STEP], cache)
            mx.eval([c.state for c in cache])
            start += PREFILL_STEP
        hidden = text_model(inputs[:, start:], cache)
        logits = text_model.embed_tokens.as_linear(hidden[:, -1:, :]).reshape(-1)
        letters = logits[mx.array(letter_ids)].astype(mx.float32)
        mx.eval(letters)
        return [float(v) for v in letters.tolist()]


def make_engine(model: Model, temperature: float):
    """Upstream's JevK5Engine, with the model in this process instead of behind vLLM."""
    from openjev.encoders import JevK5Engine

    class RecordingEngine(JevK5Engine):
        def __init__(self):  # noqa: D107 - no vLLM, no thread pools
            self.temperature = temperature
            self.passes = []

        def letter_logprobs(self, prompt, count):
            ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            logits = model.letter_logits(ids, [i[0] for i in model.letter_ids[:count]])
            self.passes.append({"prompt": prompt, "ids": ids, "logits": logits})
            return logits, len(ids)

    return RecordingEngine()


def pass_record(texts: list[str], recorded: dict) -> dict:
    prompt, ids = recorded["prompt"], recorded["ids"]
    entry = {"texts": texts, "prompt_sha256": sha256_text(prompt), "chars": len(prompt),
             "tokens": len(ids), "ids_sha256": ids_digest(ids), "logits": recorded["logits"]}
    if len(prompt) <= KEEP_PROMPT_CHARS:
        entry["prompt"] = prompt
    return entry


def corpus_reads(corpus: list[dict], model: Model, temperature: float) -> tuple[list, list]:
    from jevk5.prompt import decision_options

    engine = make_engine(model, temperature)
    reads, totals = [], []
    for request in corpus:
        qs, forced = engine.build_schema(request["questions"])
        total = 0
        for q in qs:
            engine.passes = []
            question = {"type": q["type"], "instructions": q["raw_instructions"], "criteria": q["criteria"]}
            options = decision_options(question)
            probabilities, tokens = engine.read_question(request["state"], q)
            # The texts of each pass, recovered from the prompt's own JSON.
            passes = []
            for recorded in engine.passes:
                user = recorded["prompt"].split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>", 1)[0]
                texts = [o["description"] for o in json.loads(user)["options"]]
                passes.append(pass_record(texts, recorded))
            reads.append({"request": request["name"], "key": q["key"], "type": q["type"],
                          "options": [list(o) for o in options], "passes": passes,
                          "probabilities": probabilities, "tokens": tokens})
            total += tokens
        totals.append({"request": request["name"], "input_tokens": total,
                       "forced": list(forced)})
        print(f"{request['name']}: {len(qs)} questions, {total} tokens", flush=True)
    return reads, totals


def generated_logit(seed: str, text: str, levels: int | None) -> float:
    """A logit in [-4, 4) from the text's digest: `levels` distinct values when given, so that
    ties are common, else a multiple of 2**-20."""
    value = int(hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()[:12], 16)
    if levels:
        return -4.0 + 8.0 * (value % levels) / levels
    return -4.0 + (value % (8 << 20)) / float(1 << 20)


def spread_cases(temperature: float) -> list[dict]:
    from jevk5.prompt import spread

    cases = []
    plans = [
        ("20 options, two groups of 10 (upstream's test)", 20, "knockout", None),
        ("17 options, groups of 9 and 8, ties", 17, "knockout", 3),
        ("33 options, three groups of 11, keep 5, one free place", 33, "knockout", None),
        ("33 options, ties at the cut", 33, "knockout", 2),
        ("40 options", 40, "knockout", None),
        ("40 options, every logit equal", 40, "knockout", 1),
        ("55 options", 55, "knockout", None),
        ("100 options", 100, "knockout", None),
        ("100 options, ties", 100, "knockout", 4),
        ("255 options", 255, "knockout", None),
        ("255 options, ties", 255, "knockout", 5),
        ("257 options, a final of 17 read in passes again", 257, "knockout", None),
        ("300 options, ties", 300, "knockout", 3),
        ("40 options, tree", 40, "tree", None),
        ("255 options, tree", 255, "tree", None),
        ("300 options, tree with groups read in passes again", 300, "tree", 4),
    ]
    for title, n, method, levels in plans:
        texts = [f"o{i}: option {i}" for i in range(n)]
        seed = f"{title}"
        passes = []

        def read(pass_texts, seed=seed, levels=levels, passes=passes):
            logits = [generated_logit(seed, t, levels) for t in pass_texts]
            passes.append({"texts": list(pass_texts), "logits": logits})
            top = max(logits)
            w = [math.exp((v - top) / temperature) for v in logits]
            return [v / sum(w) for v in w]

        probabilities = spread(read, texts, method)
        cases.append({"title": title, "method": method, "texts": texts, "passes": passes,
                      "probabilities": probabilities})
    return cases


def generator(model_folder: Path) -> dict:
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "--short=7", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        sys.exit(f"Upstream/openjev is at {head}, expected {UPSTREAM_COMMIT}; run make upstream")
    listing = convert.listing(model_folder)
    weights = listing.get("model.safetensors")
    if listing != convert.OUTPUTS[4]:
        sys.exit(f"{model_folder} is not the pinned 4-bit conversion; run convert.py --check")
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                         text=True).stdout.strip()
    return {
        "script": SCRIPT, "version": GENERATOR_VERSION,
        "upstream": "razorback16/openjev", "upstream_commit": UPSTREAM_COMMIT,
        "jevk5": "allebee/jevk5", "jevk5_commit": JEVK5_COMMIT,
        "checkpoint_repo": convert.SOURCE_REPO, "checkpoint_revision": convert.SOURCE_REVISION,
        "conversion": "jevk5-0.2-mlx-4bit", "conversion_sha256": weights[1],
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name)
           for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "jevk5")},
        "prefill_step": PREFILL_STEP, "cpu": cpu,
    }


def write(path: Path, payload: dict) -> None:
    """One top-level key per line and one list entry per line, as Tools/fixtures writes, in
    ASCII (a text's punctuation is a JSON escape, which reads back the same)."""
    lines = []
    for key, value in payload.items():
        if isinstance(value, list) and value:
            items = ",\n".join(" " + json.dumps(v, ensure_ascii=True, allow_nan=False) for v in value)
            lines.append(f"{json.dumps(key)}: [\n{items}\n]")
        else:
            lines.append(f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=True, allow_nan=False)}")
    text = "{\n" + ",\n".join(lines) + "\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(text)} bytes", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="the 4-bit conversion's folder (default %(default)s)")
    args = parser.parse_args(argv)
    folder = Path(args.model).expanduser()
    gen = generator(folder)
    temperature = float(json.loads((folder / "jevk5_config.json").read_text())["temperature"])
    corpus = build_corpus(cut=False) + long_requests()
    check_corpus(corpus)
    model = Model(folder)
    reads, totals = corpus_reads(corpus, model, temperature)
    vocabulary = model.tokenizer.get_vocab()
    write(FIXTURE, {
        "generator": gen,
        "about": ("The 26 requests of Fixtures/encoders/corpus.json with the choices it cuts for "
                  "Verdict restored whole, and two requests with long states, read through "
                  "upstream's JevK5Engine.read_question with the 4-bit conversion on MLX in place "
                  "of vLLM; spread over generated logits; and the tokenizer's letters and longest "
                  "entry. Tools/jevk5/reference.py wrote it."),
        "temperature": temperature,
        "tokenizer": {"letter_ids": [i[0] for i in model.letter_ids],
                      "max_chars_per_token": max(len(t) for t in vocabulary),
                      "vocabulary": len(vocabulary)},
        "requests": totals,
        "corpus": corpus,
        "reads": reads,
        "spreads": spread_cases(temperature),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
