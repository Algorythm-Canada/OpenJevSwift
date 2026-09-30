#!/usr/bin/env python3
"""Convert Verdict to Core ML and check each package against the PyTorch reference (spike #56).

Verdict is GLiClassModel, a uni-encoder over ModernBERT-base (heman10x/rlcd-modernbert-151m at the
revision common.py pins). The converted model maps

    tokens  int32 [batch, 2, sequence]: plane 0 the token ids (padding 50283), plane 1 the
            attention mask (1 for a token, 0 for padding)
    logits  float32 [batch, 25]: the head's 25 logits

with batch 1 or 16 and sequence 128, 256 or 512. The temperatures of calibrator.json, the softmax
and the removal of the abstention label stay outside the model, as upstream applies them.

Three packages are written to common.MODELS, outside the repository:

    verdict-e17-fp16.mlpackage  one program with enumerated input shapes, iOS 17, float16
    verdict-e17-fp32.mlpackage  the same in float32
    verdict-m18-fp16.mlpackage  one function per input shape (b1_s128 to b16_s512) sharing one
                                copy of the weights, iOS 18, float16

Each package is then read through upstream's own VerdictEngine.read with the Core ML model in
place of the PyTorch one, for all 200 questions of Fixtures/encoders/corpus.json, one question per
read and in batches of 16, and compared with Fixtures/encoders/verdict.json. The compute units
checked from Python leave out the ones that crash Core ML's CPU backend on macOS 27.0.1, and the
multifunction package's CPU and Neural Engine settings, which crashed coremltools' Python binding;
the Swift harness measures those (docs/spikes/encoder-runtime.md has the evidence). The report goes to
docs/spikes/encoder-runtime/verdict-coreml.json. The ONNX files the checkpoint ships are checked
against coremltools as well.

    Tools/encoders/.venv/bin/python Tools/encoders/convert_verdict.py
"""

import argparse
import sys
import time
from types import SimpleNamespace

import common
import coreml_common as cc
import coremltools as ct
import numpy as np
import torch
from torch import nn

SCRIPT = "Tools/encoders/convert_verdict.py"
VERSION = 1
LENGTHS = (128, 256, 512)
BATCHES = (1, 16)
SHAPES = [(b, 2, s) for b in BATCHES for s in LENGTHS]
VARIANTS = {
    "verdict-e17-fp16": ("enumerated", "fp16", ct.target.iOS17, ["CPU_AND_GPU", "ALL"]),
    "verdict-e17-fp32": ("enumerated", "fp32", ct.target.iOS17, ["CPU_ONLY", "CPU_AND_GPU", "ALL"]),
    "verdict-m18-fp16": ("multifunction", "fp16", ct.target.iOS18, ["CPU_AND_GPU", "ALL"]),
}
# Converted only when named with --only: int8 weights, float16 computation.
EXTRA = {
    "verdict-m18-w8": ("multifunction", "w8", ct.target.iOS18, ["CPU_AND_GPU"]),
}


class VerdictCoreML(nn.Module):
    """tokens [batch, 2, sequence] to GLiClassModel's 25 logits."""

    def __init__(self, gliclass_model):
        super().__init__()
        self.uni = gliclass_model.model
        self.core = cc.ModernBertCore(self.uni.encoder_model, max(LENGTHS))

    def forward(self, tokens):
        input_ids, attention_mask = tokens[:, 0, :], tokens[:, 1, :]
        hidden = self.core(input_ids, attention_mask)
        return self.uni.process_encoder_output(input_ids, attention_mask, hidden, None, None)[0]


class CoreMLVerdict:
    """Stands in for GLiClassModel in upstream's VerdictEngine.read_batch."""

    def __init__(self, runner, pad_id):
        self.runner, self.pad_id, self.shapes = runner, pad_id, []

    def __call__(self, input_ids, attention_mask, **kw):
        rows = []
        for i in range(input_ids.shape[0]):
            n = int(attention_mask[i].sum())
            rows.append([input_ids[i, :n].tolist(), [1] * n])
        logits, shape = self.runner.run(rows, self.pad_id)
        self.shapes.append(shape)
        return SimpleNamespace(logits=torch.from_numpy(logits))


def check_wrapper(wrapper, reference):
    """The wrapper against the reference logits, in PyTorch float32, one unpadded prompt at a time."""
    worst = 0.0
    with torch.inference_mode():
        for row in reference:
            n = len(row["input_ids"])
            tokens = torch.tensor([[row["input_ids"], [1] * n]], dtype=torch.int32)
            logits = wrapper(tokens)[0, : row["k"]].numpy()
            worst = max(worst, float(np.abs(logits - np.array(row["logits"], dtype=np.float32)).max()))
    return worst


def read_all(eng, corpus, model, batch):
    eng.s = common.settings(backend="verdict", verdict_model=eng.s.verdict_model, encoder_batch=batch)
    recorder = common.Recorder(model)
    eng.model = recorder
    rows = []
    for req in corpus:
        qs, _ = eng.build_schema(req["questions"])
        before = len(recorder.calls)
        probs, _ = eng.read(req["state"], qs)
        logits = [row for (_, _, out) in recorder.calls[before:] for row in out.logits.numpy()]
        for q, p, lg in zip(qs, probs, logits):
            rows.append({"key": f"{req['name']}/{q['key']}", "probabilities": [float(v) for v in p], "logits": lg})
    return rows


def parity(reference, rows):
    logits = [[float(v) for v in r["logits"][: ref["k"]]] for ref, r in zip(reference, rows)]
    logit_max, logit_mean = cc.compare([ref["logits"] for ref in reference], logits)
    prob_max, prob_mean = cc.compare([ref["probabilities"] for ref in reference], [r["probabilities"] for r in rows])
    changed = [r["key"] for ref, r in zip(reference, rows)
               if int(np.argmax(ref["probabilities"])) != int(np.argmax(r["probabilities"]))]
    worst = sorted(((max(abs(a - b) for a, b in zip(ref["probabilities"], r["probabilities"])), r["key"])
                    for ref, r in zip(reference, rows)), reverse=True)[:5]
    return {
        "max_abs_logit_difference": logit_max, "mean_abs_logit_difference": logit_mean,
        "max_abs_probability_difference": prob_max, "mean_abs_probability_difference": prob_mean,
        "top_label_agreement": f"{len(rows) - len(changed)}/{len(rows)}", "top_label_changes": changed,
        "largest_probability_differences": [{"question": k, "difference": d} for d, k in worst],
    }


def onnx_route():
    """What coremltools does with the ONNX export the checkpoint ships."""
    from pathlib import Path

    import onnx

    path = Path(common.snapshot(common.VERDICT_REPO, common.VERDICT_REVISION, ["model_fp16.onnx"])) / "model_fp16.onnx"
    model = onnx.load(str(path), load_external_data=False)
    ops = sorted({node.op_type for node in model.graph.node})
    try:
        ct.convert(str(path), minimum_deployment_target=ct.target.iOS17)
        outcome = "converted"
    except Exception as error:  # the converter's own message is the finding
        outcome = f"{type(error).__name__}: {str(error).splitlines()[0]}"
    return {
        "file": "model_fp16.onnx", "bytes": path.stat().st_size,
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
        "inputs": [i.name for i in model.graph.input], "outputs": [o.name for o in model.graph.output],
        "op_types": ops, "coremltools_convert": outcome,
    }


def save(report):
    """Written after every package, so a crash in a later one keeps what came before."""
    common.write(common.RESULTS / "verdict-coreml.json", common.generator(SCRIPT, VERSION, "torch", "coremltools", "transformers", "gliclass", "numpy"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-convert", action="store_true", help="validate packages converted earlier")
    parser.add_argument("--only", help="comma-separated variants to convert and check; the report keeps the others")
    args = parser.parse_args()
    common.require_upstream()
    torch.set_num_threads(common.TORCH_THREADS)
    corpus = common.read_json(common.FIXTURES / "corpus.json")["requests"]
    reference = common.read_json(common.FIXTURES / "verdict.json")
    reads = reference["reads"]
    common.MODELS.mkdir(parents=True, exist_ok=True)

    eng = common.verdict_engine()
    gliclass_model = eng.model
    pad_id = int(eng.tok.pad_token_id)
    wrapper = VerdictCoreML(gliclass_model).eval()
    report = {
        "about": ("Verdict converted to Core ML with coremltools from the PyTorch model (the ONNX "
                  "route is below). Parity reads all 200 corpus questions through upstream's "
                  "VerdictEngine.read with each package in place of the PyTorch model and compares "
                  "with Fixtures/encoders/verdict.json (PyTorch float32 on the CPU). batch 1 reads "
                  "one question per call, batch 16 reads each request 16 questions at a time as "
                  "upstream does; rows are padded to the next enumerated shape. Timings here are "
                  "Python wall clock on the reference Mac and include coremltools overhead; the Swift "
                  "harness measures latency."),
        "interface": {"input": "tokens int32 [batch, 2, sequence]: token ids, attention mask",
                      "output": "logits float32 [batch, 25]", "batches": list(BATCHES), "lengths": list(LENGTHS)},
        "wrapper_max_abs_logit_difference_float32": check_wrapper(wrapper, reads),
        "precision_floor_pytorch_cpu": reference["precision_floor"],
        "packages": {},
    }
    print(f"wrapper vs reference: {report['wrapper_max_abs_logit_difference_float32']:.2e}", flush=True)
    traced = cc.trace(wrapper, 2, LENGTHS[0])

    earlier = common.RESULTS / "verdict-coreml.json"
    previous = common.read_json(earlier).get("packages", {}) if earlier.exists() else {}
    variants = VARIANTS
    if args.only:
        wanted = args.only.split(",")
        variants = {k: v for k, v in {**VARIANTS, **EXTRA}.items() if k in wanted}
        report["packages"] = dict(previous)
    for name, (kind, precision, target, units) in variants.items():
        path = common.MODELS / f"{name}.mlpackage"
        entry = {"kind": kind, "precision": precision, "minimum_deployment_target": str(target).split(".")[-1]}
        if args.skip_convert and name in previous:
            entry.update({k: previous[name][k] for k in ("conversion_seconds", "operations") if k in previous[name]})
        if not args.skip_convert:
            if kind == "enumerated":
                seconds, ops = cc.build_enumerated(traced, SHAPES, precision, target, "logits", path)
            else:
                seconds, ops = cc.build_multifunction(traced, SHAPES, precision, target, "logits", path,
                                                      common.MODELS)
            entry.update({"conversion_seconds": round(seconds, 1), "operations": ops})
            print(f"{name}: converted in {seconds:.1f} s", flush=True)
        entry["package_bytes"] = common.directory_size(path)
        entry["parity"] = {}
        for unit in units:
            runner = cc.CoreMLRunner(path, kind, getattr(ct.ComputeUnit, unit), BATCHES, LENGTHS, 2, "logits")
            for batch in BATCHES:
                model = CoreMLVerdict(runner, pad_id)
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
                print(f"{name} {unit} batch {batch}: max |dp| {result['max_abs_probability_difference']:.2e}, "
                      f"top labels {result['top_label_agreement']}", flush=True)
            entry.setdefault("python_load_seconds", {})[unit] = round(runner.load_seconds, 2)
            runner.close()
        report["packages"][name] = entry
        save(report)
    report["onnx_route"] = onnx_route()
    print(f"onnx route: {report['onnx_route']['coremltools_convert']}", flush=True)
    save(report)


if __name__ == "__main__":
    sys.exit(main())
