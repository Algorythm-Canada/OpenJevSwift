#!/usr/bin/env python3
"""Record PyTorch reference outputs of Verdict and Laya for spike #56.

The script builds a corpus of 200 questions from the requests in Fixtures/schemas/schemas.json,
runs both encoder models on the CPU through upstream OpenJev's own code at the pinned commit
(razorback16/openjev at dcd2094: EncoderEngine.build_schema, VerdictEngine.read_batch and
LayaEngine.read_batch), and writes what they see and return into Fixtures/encoders/:

    corpus.json   the requests: a state and its questions, as the API hands them to the engine
    verdict.json  per question: the prompt, its token ids, the raw logits and the calibrated
                  probabilities in float32, and the same read in float16 and bfloat16
    laya.json     per question: build_sequence's token ids and marker positions, the logits at
                  the markers, the probabilities before laya rounds them, laya's own answer and
                  the distribution upstream publishes, and the same read in float16 and bfloat16

The models run exactly as upstream loads them, in float32 on the CPU, with questions read in
batches of 16 (OPENJEV_ENCODER_BATCH). Small wrappers around the tokenizer and the model record
the tensors that upstream's read_batch passes and gets back; nothing in the read path is
reimplemented. Only the float16 and bfloat16 passes change anything: they cast a copy of each
model's weights to find the precision floor that a Core ML conversion should be judged against.

Setup, once, from the repository root:

    make upstream
    /usr/local/bin/python3.12 -m venv Tools/encoders/.venv
    Tools/encoders/.venv/bin/python -m pip install -r Tools/encoders/requirements.txt

Then:

    Tools/encoders/.venv/bin/python Tools/encoders/reference.py

The first run downloads both checkpoints at their pinned revisions into the Hugging Face cache
(about 1.5 GB). A run takes about 37 minutes on an M3 Max, most of it in the float16 and bfloat16
passes. Running it twice on the same machine with the same packages gives identical files.
"""

import copy
import json
import time
from pathlib import Path

import common
from common import FIXTURES, TORCH_THREADS, Recorder, context_of, f32

import laya
import numpy as np
import torch
from laya.common import QTYPES, build_sequence, render_options, temp_bucket
from openjev.encoders import VERDICT_MAX_LEN, verdict_prompt

SCRIPT = "Tools/encoders/reference.py"
# Bump when the shape of a file this script writes changes.
GENERATOR_VERSION = 1


def generator():
    gen = common.generator(SCRIPT, GENERATOR_VERSION, "torch", "transformers", "tokenizers", "gliclass", "laya",
                           "huggingface_hub", "numpy")
    gen.update({"device": "cpu", "torch_threads": TORCH_THREADS})
    return gen


def write(name, payload):
    common.write(FIXTURES / name, generator(), payload)


# The corpus.
#
# Every question comes from a request in Fixtures/schemas/schemas.json, in the form the API hands
# the engine (pydantic's model_dump: type, instructions, criteria). Verdict refuses a choice with
# more than 24 options, so the fixture's 40-, 55-, 100- and 255-option choices are cut to their
# first N options below; that covers 6, 7, 10, 12, 16, 20 and 24 options, which with the
# fixture's own 2- to 8-option choices reaches every per_k entry in Verdict's calibrator that a
# question can reach (k is the option count plus the abstention label, so k = 2 would take a
# one-option choice, which is forced) and three counts that fall back to its global temperature.
# Left out: the requests with forced questions (one option or one level, which neither model
# reads), the images request, the requests the schema refuses, nouls_24 under its own state (it is
# read under the long conversation below) and single_noul and nouls_30, plain nouls like nouls_8's.
#
# The fixture's states are short. Three long states are made from its text states, so that some
# prompts truncate: a 16-message transcript (about 415 tokens: Verdict truncates it only with a
# long label prefix), a 30-message transcript (about 790 tokens: Verdict always truncates, Laya
# only with a long head) and an 80-message conversation as a JSON state (about 2,080 tokens: both
# truncate). Upstream's own warmup request (encoders.WARMUP_QUESTIONS) is included as well.

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

# Requests read with their own state, in fixture order.
OWN_STATE = [
    "quickstart", "six_questions", "lines_10_mixed", "indexed_11_mixed", "indexed_12_mixed",
    "nouls_8", "widest_schema", "many_choices", "described_objects_and_arrays", "non_ascii",
    "whitespace", "noul_criteria_variants", "choice_description_variants", "choice_names_special",
    "score_described_levels", "long_instructions", "state_object", "state_list", "state_floats",
    "state_empty_string", "state_escapes",
]

# A mixed set of 38 distinct questions, read again under the long states.
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


def load_schema_cases():
    data = json.loads((common.ROOT / "Fixtures" / "schemas" / "schemas.json").read_text(encoding="utf-8"))
    return {c["name"]: c for c in data["cases"] if "schema" in c}


def question(cases, name, qid):
    q = json.loads(json.dumps(cases[name]["questions"][qid]))
    n = TRUNCATE.get((name, qid))
    if n is not None:
        q["criteria"] = dict(list(q["criteria"].items())[:n])
    return q


def transcript(cases, count):
    texts = [cases[n]["request"]["state"].strip() for n in TRANSCRIPT_TEXTS]
    return "\n".join(f"Message {i + 1}: {texts[i % len(texts)]}" for i in range(count))


def conversation(cases, count):
    texts = [cases[n]["request"]["state"] for n in TRANSCRIPT_TEXTS]
    customer = cases["state_object"]["request"]["state"]["customer"]
    return {"customer": customer, "messages": [texts[i % len(texts)] for i in range(count)]}


def build_corpus():
    from openjev.encoders import WARMUP_QUESTIONS

    cases = load_schema_cases()
    requests = []

    def add(name, source, state, pairs):
        qs = {}
        for case, qid in pairs:
            key = qid if qid not in qs else f"{case}.{qid}"
            qs[key] = question(cases, case, qid)
        requests.append({"name": name, "source": source, "state": state, "questions": qs})

    for name in OWN_STATE:
        case = cases[name]
        pairs = [(name, qid) for qid, q in case["questions"].items()
                 if not (q["type"] in ("choice", "score") and len(q["criteria"]) == 1)]
        add(name, f"schemas.json {name}, own state", case["request"]["state"], pairs)
    add("state_object_six", "schemas.json six_questions, state of state_object",
        cases["state_object"]["request"]["state"], [("six_questions", q) for q in cases["six_questions"]["questions"]])
    requests.append({"name": "warmup", "source": "upstream encoders.WARMUP_QUESTIONS, state \"warmup\"",
                     "state": "warmup", "questions": json.loads(json.dumps(
                         {k: dict(v, criteria=v.get("criteria")) for k, v in WARMUP_QUESTIONS.items()}))})
    add("medium_transcript", "the mixed set, state: a 16-message transcript of the fixture's text states",
        transcript(cases, 16), MIXED)
    add("long_transcript", "the mixed set, state: a 30-message transcript of the fixture's text states",
        transcript(cases, 30), MIXED)
    add("long_conversation", "schemas.json nouls_24, state: an 80-message conversation as JSON",
        conversation(cases, 80), [("nouls_24", q) for q in cases["nouls_24"]["questions"]])
    total = sum(len(r["questions"]) for r in requests)
    assert total == 200, total
    return requests


def verdict_reads(eng, corpus, raw_model):
    """Every question read through upstream's VerdictEngine.read, with the tensors it used."""
    rows = []
    for req in corpus:
        qs, forced = eng.build_schema(req["questions"])
        assert not forced, req["name"]
        tok, model = Recorder(eng.tok.__call__), Recorder(raw_model)
        eng_tok, eng.tok = eng.tok, tok
        eng.model = model
        try:
            probs, tokens = eng.read(req["state"], qs)
        finally:
            eng.tok = eng_tok
        assert len(tok.calls) == len(model.calls) == -(-len(qs) // eng.s.encoder_batch)
        context = context_of(req["state"])
        row_batches = []
        for b, ((_, _, enc), (_, _, out)) in enumerate(zip(tok.calls, model.calls)):
            ids, mask = enc["input_ids"], enc["attention_mask"]
            logits = out.logits.float().cpu().numpy()
            for i in range(ids.shape[0]):
                n = int(mask[i].sum())
                assert int(mask[i, :n].sum()) == n, "padding is on the right"
                row_batches.append((b, ids.shape[1], ids[i, :n].tolist(), logits[i]))
        assert sum(int(c[2]["attention_mask"].sum()) for c in tok.calls) == tokens
        for q, p, (b, padded, ids, logits) in zip(qs, probs, row_batches):
            prompt, k = verdict_prompt(q, context)
            full = len(eng_tok(prompt)["input_ids"])
            temperature = eng.per_k.get(k, eng.temperature)
            rows.append({
                "request": req["name"], "key": q["key"], "type": q["type"], "options": k - 1, "k": k,
                "prompt": prompt, "tokens_untruncated": full, "truncated": full > VERDICT_MAX_LEN,
                "input_ids": ids, "batch": b, "padded_length": padded,
                "temperature": temperature, "temperature_source": "per_k" if k in eng.per_k else "global",
                "logits": f32(logits[:k]), "probabilities": [float(v) for v in p],
            })
        rows[-1]["request_input_tokens"] = tokens
    return rows


def verdict_reference(corpus):
    started = time.perf_counter()
    eng = common.verdict_engine()
    path = eng.s.verdict_model
    print(f"verdict: loaded in {time.perf_counter() - started:.1f} s", flush=True)
    raw = eng.model
    reads = {}
    for name, dtype in (("float32", None), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        # Module.to casts in place, so each reduced precision starts from a copy of the float32 weights.
        model = raw if dtype is None else copy.deepcopy(raw).to(dtype)
        started = time.perf_counter()
        reads[name] = verdict_reads(eng, corpus, model)
        print(f"verdict {name}: 200 questions in {time.perf_counter() - started:.1f} s", flush=True)
        del model

    base = reads["float32"]
    rows = []
    for i, row in enumerate(base):
        out = dict(row)
        for name in ("float16", "bfloat16"):
            other = reads[name][i]
            assert other["input_ids"] == row["input_ids"]
            out[name] = {"logits": other["logits"], "probabilities": other["probabilities"]}
        rows.append(out)

    def floor(name):
        logit = max(max(abs(a - b) for a, b in zip(r["logits"], r[name]["logits"])) for r in rows)
        prob = max(max(abs(a - b) for a, b in zip(r["probabilities"], r[name]["probabilities"])) for r in rows)
        flips = [f"{r['request']}/{r['key']}" for r in rows
                 if int(np.argmax(r["probabilities"])) != int(np.argmax(r[name]["probabilities"]))]
        return {"max_abs_logit_difference": logit, "max_abs_probability_difference": prob,
                "top_label_changes": len(flips), "changed": flips}

    calibrator = json.loads(Path(path, "calibrator.json").read_text())
    write("verdict.json", {
        "about": ("Verdict (verdict-1.4) read through upstream's VerdictEngine on the CPU. prompt and "
                  "k come from verdict_prompt; k counts the caller's options plus the trailing "
                  "'insufficient evidence' label. input_ids are the tokenizer's output with "
                  "truncation at 512 and the batch padding removed; batch and padded_length say "
                  "which batch of 16 the question was read in and its padded width. logits are the "
                  "model's first k logits in float32. probabilities are upstream's: the first k "
                  "logits divided by temperature (calibrator per_k[k], else the global "
                  "temperature), softmax, the abstention dropped, renormalised. float16 and "
                  "bfloat16 hold the same read with the weights cast to that type. "
                  "request_input_tokens, on a request's last question, is the usage.input_tokens "
                  "upstream reports for the request (the attention-mask sum)."),
        "max_length": VERDICT_MAX_LEN,
        "encoder_batch": eng.s.encoder_batch,
        "class_token_index": int(raw.config.class_token_index),
        "text_token_index": int(raw.config.text_token_index),
        "pad_token_id": int(eng.tok.pad_token_id),
        "calibrator": {"temperature": calibrator["temperature"], "per_k": calibrator["per_k"]},
        "precision_floor": {"float16": floor("float16"), "bfloat16": floor("bfloat16")},
        "reads": rows,
    })


def laya_reads(eng, corpus):
    agent = eng.agent
    model = Recorder(agent.model)
    system_one = Recorder(agent.system_one)
    agent.model, agent.system_one = model, system_one
    rows, state_texts = [], {}
    try:
        for req in corpus:
            qs, forced = eng.build_schema(req["questions"])
            assert not forced, req["name"]
            before = len(model.calls)
            probs, tokens = eng.read(req["state"], qs)
            calls = model.calls[before:]
            answers = system_one.calls[before:]
            assert len(calls) == len(answers) == -(-len(qs) // eng.s.encoder_batch)
            flat = []
            for (args, _, out), (sargs, _, sout) in zip(calls, answers):
                input_ids, attention_mask, marker_pos, marker_mask, qtype = args
                logits = out[0].float().cpu().numpy()
                for r, (qkey, qdef) in enumerate(sargs[1].items()):
                    n = int(attention_mask[r].sum())
                    k = int(marker_mask[r].sum())
                    flat.append({
                        "ids": input_ids[r, :n].tolist(), "markers": marker_pos[r, :k].tolist(),
                        "qtype": int(qtype[r]), "logits": logits[r, :k], "answer": sout["answers"][qkey],
                        "laya_question": qdef, "padded_length": int(input_ids.shape[1]),
                        "batch_input_tokens": int(sout["usage"]["input_tokens"]),
                    })
            batch_sizes = [len(c[0][0]) for c in calls]
            for i, (q, p, f) in enumerate(zip(qs, probs, flat)):
                internal = agent._to_internal(f["laya_question"])
                max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
                seq, markers = build_sequence(agent.tok, req["state"], internal, max_len, head_max_len)
                assert seq == f["ids"] and markers == f["markers"], (req["name"], q["key"])
                # The same sequence with an empty state is everything before the state plus the final
                # [SEP], so the difference in length is how many state tokens the model read.
                state_read = len(seq) - len(build_sequence(agent.tok, "", internal, max_len, head_max_len)[0])
                state_tokens = len(agent.tok(context_of(req["state"]).replace(agent.tok.mask_token, " "),
                                             add_special_tokens=False)["input_ids"])
                k = len(markers)
                assert k == len(render_options(internal))
                qt = QTYPES[internal["t"]]
                bucket = temp_bucket(qt, k)
                temperature = agent.temperature_by_options.get(bucket, agent.temperature[qt])
                z = f["logits"] / temperature
                pr = np.exp(z - z.max())
                pr = pr / pr.sum()
                a = f["answer"]
                if internal["t"] == "noul":
                    assert a["noul"] == round(float(pr[1]), 4)
                else:
                    assert list(a["probabilities"].values()) == [round(float(v), 4) for v in pr]
                mask_token = agent.tok.mask_token
                texts = {"head": "%s question: %s" % (internal["t"], str(internal["ins"]).replace(mask_token, " ")),
                         "options": [" " + o.replace(mask_token, " ") for o in render_options(internal)]}
                state_texts[req["name"]] = context_of(req["state"]).replace(mask_token, " ")
                rows.append({
                    "request": req["name"], "key": q["key"], "type": q["type"], "options": k,
                    "laya_question": f["laya_question"], "texts": texts, "ids": f["ids"], "markers": f["markers"],
                    "qtype": f["qtype"], "batch": sum(1 for s in _cumulative(batch_sizes) if s <= i),
                    "padded_length": f["padded_length"],
                    "state_tokens": state_tokens, "state_tokens_read": state_read,
                    "truncated": state_read < state_tokens,
                    "bucket": bucket, "temperature": temperature,
                    "logits": f32(f["logits"]), "probabilities_unrounded": f32(pr),
                    "answer": a, "probabilities": [float(v) for v in p],
                })
            rows[-1]["request_input_tokens"] = tokens
    finally:
        agent.model, agent.system_one = model.fn, system_one.fn
    return rows, state_texts


def _cumulative(sizes):
    total = 0
    for s in sizes:
        total += s
        yield total


def laya_floor(rows, name):
    """The reduced-precision read against float32: largest differences and changed top answers."""
    def largest(key):
        return max(max(abs(a - b) for a, b in zip(r[key], r[name][key])) for r in rows)

    flips = [f"{r['request']}/{r['key']}" for r in rows
             if int(np.argmax(r["probabilities"])) != int(np.argmax(r[name]["probabilities"]))]
    return {"max_abs_logit_difference": largest("logits"),
            "max_abs_probability_difference_unrounded": largest("probabilities_unrounded"),
            "max_abs_published_probability_difference": largest("probabilities"),
            "top_label_changes": len(flips), "changed": flips}


def laya_reference(corpus):
    started = time.perf_counter()
    eng = common.laya_engine()
    print(f"laya: loaded in {time.perf_counter() - started:.1f} s", flush=True)
    started = time.perf_counter()
    rows, state_texts = laya_reads(eng, corpus)
    print(f"laya float32: 200 questions in {time.perf_counter() - started:.1f} s", flush=True)
    agent = eng.agent
    raw = agent.model
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        # A copy of the float32 weights in the reduced type, with the action head kept in float32
        # as LayaEngine.load does on a GPU (it takes float32 features).
        model = copy.deepcopy(raw).to(dtype)
        model.act_head.float()
        agent.model = model
        started = time.perf_counter()
        reduced, _ = laya_reads(eng, corpus)
        print(f"laya {name}: 200 questions in {time.perf_counter() - started:.1f} s", flush=True)
        for row, other in zip(rows, reduced):
            assert other["ids"] == row["ids"] and other["markers"] == row["markers"]
            row[name] = {k: other[k] for k in ("logits", "probabilities_unrounded", "probabilities")}
        agent.model = raw
        del model
    write("laya.json", {
        "about": ("Laya (laya-1.0) read through upstream's LayaEngine on the CPU, which calls laya's "
                  "Agent.system_one with at most 16 questions. laya_question is what upstream passes "
                  "to laya (instructions as text, criteria raw). ids and markers are build_sequence's "
                  "output for it, the token ids the model read and the positions of its [MASK] "
                  "option markers; they are the tokenization oracle for issue #58. logits are the "
                  "scorer's outputs at the markers in float32. probabilities_unrounded apply the "
                  "temperature as system_one does (temperature_by_options[bucket] if present, else "
                  "temperature[qtype], both clamped to [0.5, 5]) and softmax, before laya rounds to "
                  "4 decimals. answer is laya's own answer. probabilities is the distribution "
                  "upstream publishes, in the caller's option order: [noul, 1 - noul] for a noul, "
                  "otherwise laya's rounded probabilities renormalised. float16 and bfloat16 hold the "
                  "same read with the weights cast to that type (the action head stays float32, as "
                  "upstream keeps it on a GPU, where it serves Laya in bfloat16). "
                  "request_input_tokens, on a request's last question, is the usage.input_tokens "
                  "upstream reports. texts are the "
                  "strings build_sequence tokenizes without special tokens: the head, and each option "
                  "after its [MASK] marker (each cut to 48 tokens); state_texts holds each request's "
                  "state text. build_sequence then applies the head budget and max_len."),
        "max_len": agent.cfg.get("max_len"),
        "head_max_len": agent.cfg.get("head_max_len"),
        "encoder_batch": eng.s.encoder_batch,
        "special_tokens": {"cls": agent.tok.cls_token_id, "sep": agent.tok.sep_token_id,
                           "mask": agent.tok.mask_token_id, "pad": agent.tok.pad_token_id},
        "calibration": {
            "temperature_raw": agent.temperature_raw, "temperature_by_options_raw": agent.temperature_by_options_raw,
            "temperature": agent.temperature, "temperature_by_options": agent.temperature_by_options,
            "clamp": [laya.common.TEMP_MIN, laya.common.TEMP_MAX],
        },
        "qtypes": QTYPES,
        "precision_floor": {name: laya_floor(rows, name) for name in ("float16", "bfloat16")},
        "state_texts": state_texts,
        "reads": rows,
    })


def main():
    common.require_upstream()
    torch.set_num_threads(TORCH_THREADS)
    torch.manual_seed(0)
    corpus = build_corpus()
    write("corpus.json", {
        "about": ("The 200 questions of spike #56: requests of Fixtures/schemas/schemas.json with "
                  "their own states, the fixture's six_questions under state_object's state, "
                  "upstream's warmup request, and 38 fixture questions under two long transcripts "
                  "and nouls_24 under a long JSON conversation. Choices over 24 options are cut to "
                  "their first N options (truncated_choices). Each request is read in batches of 16 "
                  "in question order, as upstream reads it."),
        "truncated_choices": [{"request": a, "key": b, "options": n} for (a, b), n in TRUNCATE.items()],
        "questions": sum(len(r["questions"]) for r in corpus),
        "requests": corpus,
    })
    verdict_reference(corpus)
    laya_reference(corpus)


if __name__ == "__main__":
    main()
