#!/usr/bin/env python3
"""Convert Laya to Core ML and check each package against the PyTorch reference (spike #56).

Laya is laya's DecisionModel: a ModernBERT-large encoder, a question-type embedding added to every
position, a two-layer norm-first TransformerEncoder head and a scorer MLP
(convaiinnovations/laya-typed-decisions at the revision common.py pins). The converted model maps

    tokens  int32 [batch, 3, sequence]: plane 0 the token ids (padding 50283), plane 1 the
            attention mask, plane 2 the question type (choice 0, score 1, noul 2) in every position
    scores  float32 [batch, sequence]: the scorer's output at every position

with batch 1 or 16 and sequence 128, 256, 512 or 1024. The model does not gather the [MASK]
marker positions or apply the softmax, so the graph stays static; the caller reads the scores at
build_sequence's markers, divides by the calibrated temperature and takes the softmax, as laya's
Agent.system_one does. The action head is left out: upstream does not use it.

Two packages are written to common.MODELS, outside the repository:

    laya-e17-fp16.mlpackage  one program with enumerated input shapes, iOS 17, float16
    laya-m18-fp16.mlpackage  one function per input shape (b1_s128 to b16_s1024) sharing one copy
                             of the weights, iOS 18, float16

Each package then answers all 200 questions of Fixtures/encoders/corpus.json through upstream's
LayaEngine.read, with a stand-in for DecisionModel that gathers the Core ML scores at the markers,
and is compared with Fixtures/encoders/laya.json: the logits at the markers, the probabilities
before laya rounds them, laya's rounded answers and the distribution upstream publishes. As for
Verdict, the Python checks use the GPU and all units; the Swift harness measures the CPU and
Neural Engine settings. The report goes to docs/spikes/encoder-runtime/laya-coreml.json.

    Tools/encoders/.venv/bin/python Tools/encoders/convert_laya.py
"""

import argparse
import sys
import time

import common
import coreml_common as cc
import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from laya.common import QTYPES, temp_bucket
from torch import nn

SCRIPT = "Tools/encoders/convert_laya.py"
VERSION = 1
LENGTHS = (128, 256, 512, 1024)
BATCHES = (1, 16)
SHAPES = [(b, 3, s) for b in BATCHES for s in LENGTHS]
VARIANTS = {
    "laya-e17-fp16": ("enumerated", "fp16", ct.target.iOS17, ["CPU_AND_GPU", "ALL"]),
    "laya-m18-fp16": ("multifunction", "fp16", ct.target.iOS18, ["CPU_AND_GPU", "ALL"]),
}
# Converted only when named with --only: int8 weights with float16 computation (half the size, and
# less exact than bfloat16), and one program for one fixed shape, for the Neural Engine, which no
# multifunction Laya package loads for (see docs/spikes/encoder-runtime.md). The fixed-shape packages
# are checked from Swift only: the Python check reads every question, and most fit only one of them.
EXTRA = {
    "laya-m18-w8": ("multifunction", "w8", ct.target.iOS18, ["CPU_AND_GPU"]),
    **{f"laya-f18-b1s{s}-fp16": ("fixed", "fp16", ct.target.iOS18, []) for s in LENGTHS},
}
FIXED = {f"laya-f18-b1s{s}-fp16": (1, 3, s) for s in LENGTHS}


def head_layer(layer, h, padding):
    """nn.TransformerEncoderLayer's norm-first forward in eval mode, written with rank-4 tensors.

    PyTorch's own path unpacks the fused QKV projection through a rank-5 tensor. This one splits
    it, attends with an additive float mask and adds the feed-forward block, with the layer's own
    weights (dropout is the identity in eval mode). It matches the layer within 6e-7 in float32.
    """
    attn = layer.self_attn
    b, s, d = h.shape
    heads = attn.num_heads
    q, k, v = F.linear(layer.norm1(h), attn.in_proj_weight, attn.in_proj_bias).split(d, dim=-1)
    q, k, v = (t.reshape(b, s, heads, d // heads).transpose(1, 2) for t in (q, k, v))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=padding[:, None, None, :])
    h = h + attn.out_proj(a.transpose(1, 2).reshape(b, s, d))
    return h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))


class LayaCoreML(nn.Module):
    """tokens [batch, 3, sequence] to DecisionModel's scorer output at every position.

    Two rewrites keep the whole program on the Neural Engine. The question type's embedding is a
    product with a one-hot row instead of a gather: with the gather feeding the head, Core ML's
    plan put every operation of the model on the CPU, although the encoder, the head and the scorer
    were each planned on the Neural Engine alone. The head layers use head_layer above.
    """

    def __init__(self, decision_model):
        super().__init__()
        self.model = decision_model
        self.core = cc.ModernBertCore(decision_model.encoder, max(LENGTHS))
        for layer in decision_model.head.layers:
            assert layer.norm_first and layer.activation is F.relu

    def forward(self, tokens):
        input_ids, attention_mask, qtype = tokens[:, 0, :], tokens[:, 1, :], tokens[:, 2, 0]
        h = self.core(input_ids, attention_mask)
        onehot = (qtype[:, None] == torch.arange(3, dtype=qtype.dtype)[None, :]).to(h.dtype)
        h = h + (onehot @ self.model.type_emb.weight)[:, None, :]
        # An additive float mask instead of DecisionModel's boolean one: -1e4 is finite in float16.
        padding = (1.0 - attention_mask.to(torch.float32)) * cc.NEG
        for layer in self.model.head.layers:
            h = head_layer(layer, h, padding)
        return self.model.scorer(h).squeeze(-1).float()


class CoreMLLaya:
    """Stands in for DecisionModel in laya's Agent.system_one: Core ML scores read at the markers."""

    def __init__(self, runner, pad_id):
        self.runner, self.pad_id, self.shapes = runner, pad_id, []

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        rows = []
        for i in range(input_ids.shape[0]):
            n = int(attention_mask[i].sum())
            rows.append([input_ids[i, :n].tolist(), [1] * n, [int(qtype[i])] * n])
        scores, shape = self.runner.run(rows, self.pad_id)
        self.shapes.append(shape)
        logits = np.take_along_axis(scores, marker_pos.clamp(min=0).numpy(), axis=1)
        logits = torch.from_numpy(logits).masked_fill(~marker_mask, -1e4)
        # system_one reads the action head's output into answer["action"]; upstream drops it.
        return logits, torch.zeros((input_ids.shape[0], 2))


def check_wrapper(wrapper, reference):
    """The wrapper's scores at the markers against the reference logits, in PyTorch float32."""
    worst = 0.0
    with torch.inference_mode():
        for row in reference:
            n = len(row["ids"])
            tokens = torch.tensor([[row["ids"], [1] * n, [row["qtype"]] * n]], dtype=torch.int32)
            scores = wrapper(tokens)[0].numpy()
            logits = scores[row["markers"]]
            worst = max(worst, float(np.abs(logits - np.array(row["logits"], dtype=np.float32)).max()))
    return worst


def unrounded(agent, logits, qtype, k):
    """Agent.system_one's probabilities before it rounds them."""
    bucket = temp_bucket(qtype, k)
    t = agent.temperature_by_options.get(bucket, agent.temperature[qtype])
    z = np.asarray(logits, dtype=np.float32)[:k] / t
    p = np.exp(z - z.max())
    return p / p.sum()


def read_all(eng, corpus, model, batch):
    eng.s = common.settings(backend="laya", laya_model=eng.s.laya_model, encoder_batch=batch)
    agent = eng.agent
    recorder, answers = common.Recorder(model), common.Recorder(agent.system_one)
    saved = agent.model, agent.system_one
    agent.model, agent.system_one = recorder, answers
    rows = []
    try:
        for req in corpus:
            qs, _ = eng.build_schema(req["questions"])
            before = len(recorder.calls)
            probs, _ = eng.read(req["state"], qs)
            flat = []
            for (args, _, out), (sargs, _, sout) in zip(recorder.calls[before:], answers.calls[before:]):
                marker_mask, qtype = args[3], args[4]
                for r, key in enumerate(sargs[1]):
                    k = int(marker_mask[r].sum())
                    flat.append((out[0][r, :k].numpy(), int(qtype[r]), k, sout["answers"][key]))
            for q, p, (logits, qt, k, answer) in zip(qs, probs, flat):
                rows.append({"key": f"{req['name']}/{q['key']}", "logits": [float(v) for v in logits],
                             "probabilities_unrounded": [float(v) for v in unrounded(agent, logits, qt, k)],
                             "answer": answer, "probabilities": [float(v) for v in p]})
    finally:
        agent.model, agent.system_one = saved
    return rows


def rounded_values(answer):
    return [answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())


def parity(reference, rows):
    logit_max, logit_mean = cc.compare([r["logits"] for r in reference], [r["logits"] for r in rows])
    raw_max, raw_mean = cc.compare([r["probabilities_unrounded"] for r in reference],
                                   [r["probabilities_unrounded"] for r in rows])
    pub_max, pub_mean = cc.compare([r["probabilities"] for r in reference], [r["probabilities"] for r in rows])
    changed = [r["key"] for ref, r in zip(reference, rows)
               if int(np.argmax(ref["probabilities"])) != int(np.argmax(r["probabilities"]))]
    same = sum(1 for ref, r in zip(reference, rows) if rounded_values(ref["answer"]) == rounded_values(r["answer"]))
    worst = sorted(((max(abs(a - b) for a, b in zip(ref["probabilities_unrounded"], r["probabilities_unrounded"])),
                     r["key"]) for ref, r in zip(reference, rows)), reverse=True)[:5]
    return {
        "max_abs_logit_difference": logit_max, "mean_abs_logit_difference": logit_mean,
        "max_abs_probability_difference_unrounded": raw_max, "mean_abs_probability_difference_unrounded": raw_mean,
        "max_abs_published_probability_difference": pub_max, "mean_abs_published_probability_difference": pub_mean,
        "rounded_answers_identical": f"{same}/{len(rows)}",
        "top_label_agreement": f"{len(rows) - len(changed)}/{len(rows)}", "top_label_changes": changed,
        "largest_probability_differences": [{"question": k, "difference": d} for d, k in worst],
    }


def save(report):
    """Written after every package, so a crash in a later one keeps what came before."""
    common.write(common.RESULTS / "laya-coreml.json", common.generator(SCRIPT, VERSION, "torch", "coremltools", "transformers", "laya", "numpy"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-convert", action="store_true", help="validate packages converted earlier")
    parser.add_argument("--only", help="comma-separated variants to convert and check; the report keeps the others")
    args = parser.parse_args()
    common.require_upstream()
    torch.set_num_threads(common.TORCH_THREADS)
    torch.backends.mha.set_fastpath_enabled(False)  # the fused encoder-layer kernel has no Core ML mapping
    corpus = common.read_json(common.FIXTURES / "corpus.json")["requests"]
    reference = common.read_json(common.FIXTURES / "laya.json")
    reads = reference["reads"]
    common.MODELS.mkdir(parents=True, exist_ok=True)

    eng = common.laya_engine()
    agent = eng.agent
    pad_id = int(agent.tok.pad_token_id)
    assert QTYPES == reference["qtypes"]
    wrapper = LayaCoreML(agent.model).eval()
    report = {
        "about": ("Laya converted to Core ML with coremltools. Parity answers all 200 corpus questions "
                  "through upstream's LayaEngine.read, with a stand-in for DecisionModel that reads "
                  "each package's scores at the [MASK] markers, and compares with "
                  "Fixtures/encoders/laya.json (PyTorch float32 on the CPU). batch 1 reads one question "
                  "per call, batch 16 reads each request 16 questions at a time as upstream does; rows "
                  "are padded to the next enumerated shape. Timings here are Python wall clock on the "
                  "reference Mac; the Swift harness measures latency."),
        "interface": {"input": "tokens int32 [batch, 3, sequence]: token ids, attention mask, question type",
                      "output": "scores float32 [batch, sequence]", "batches": list(BATCHES),
                      "lengths": list(LENGTHS)},
        "wrapper_max_abs_logit_difference_float32": check_wrapper(wrapper, reads),
        "packages": {},
    }
    print(f"wrapper vs reference: {report['wrapper_max_abs_logit_difference_float32']:.2e}", flush=True)
    traced = cc.trace(wrapper, 3, LENGTHS[0])

    earlier = common.RESULTS / "laya-coreml.json"
    previous = common.read_json(earlier).get("packages", {}) if earlier.exists() else {}
    variants = VARIANTS
    if args.only:
        wanted = args.only.split(",")
        variants = {k: v for k, v in {**VARIANTS, **EXTRA}.items() if k in wanted}
        report["packages"] = dict(previous)
    for name, (kind, precision, target, units) in variants.items():
        path = common.MODELS / f"{name}.mlpackage"
        entry = {"kind": kind, "precision": precision, "minimum_deployment_target": target.name}
        if args.skip_convert and name in previous:
            entry.update({k: previous[name][k] for k in ("conversion_seconds", "operations") if k in previous[name]})
        if not args.skip_convert:
            if kind == "enumerated":
                seconds, ops = cc.build_enumerated(traced, SHAPES, precision, target, "scores", path)
            elif kind == "fixed":
                seconds, ops = cc.build_enumerated(traced, [FIXED[name]], precision, target, "scores", path)
            else:
                seconds, ops = cc.build_multifunction(traced, SHAPES, precision, target, "scores", path,
                                                      common.MODELS)
            entry.update({"conversion_seconds": round(seconds, 1), "operations": ops})
            print(f"{name}: converted in {seconds:.1f} s", flush=True)
        entry["package_bytes"] = common.directory_size(path)
        entry["parity"] = {}
        for unit in units:
            runner = cc.CoreMLRunner(path, kind, getattr(ct.ComputeUnit, unit), BATCHES, LENGTHS, 3, "scores")
            for batch in BATCHES:
                model = CoreMLLaya(runner, pad_id)
                started = time.perf_counter()
                try:
                    rows = read_all(eng, corpus, model, batch)
                except RuntimeError as error:  # Core ML's own message is the finding
                    entry["parity"][f"{unit} batch {batch}"] = {"error": str(error).replace("\n", " ")}
                    print(f"{name} {unit} batch {batch}: {error}", flush=True)
                    continue
                result = parity(reads, rows)
                result["python_seconds"] = round(time.perf_counter() - started, 2)
                result["shapes_used"] = sorted({cc.function_name(b, s) for b, s in model.shapes})
                entry["parity"][f"{unit} batch {batch}"] = result
                print(f"{name} {unit} batch {batch}: max |dp| {result['max_abs_probability_difference_unrounded']:.2e}, "
                      f"top labels {result['top_label_agreement']}, rounded {result['rounded_answers_identical']}",
                      flush=True)
            # A multifunction package loads each function inside the reads, so only its compile is timed.
            key = "python_load_seconds" if kind == "enumerated" else "python_compile_seconds"
            entry.setdefault(key, {})[unit] = round(runner.load_seconds, 2)
            runner.close()
        report["packages"][name] = entry
        save(report)
    save(report)


if __name__ == "__main__":
    sys.exit(main())
