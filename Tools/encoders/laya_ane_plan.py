#!/usr/bin/env python3
"""Where Core ML plans each piece of Laya under CPU_AND_NE, as convert_laya.py found it (spike #56).

Converts Laya's encoder with each part after it, at one fixed shape (batch 1 by 128 tokens, float16,
iOS 18), and prints how many operations Core ML's compute plan puts on each device:

    encoder alone
    encoder + type embedding (a gather indexed by the question type)
    encoder + head (convert_laya.head_layer)
    encoder + scorer
    encoder + type embedding + head             every operation lands on the CPU
    encoder + one-hot type embedding + head + scorer, as convert_laya.LayaCoreML converts it

    Tools/encoders/.venv/bin/python Tools/encoders/laya_ane_plan.py

It writes one temporary package to common.MODELS and removes it. It takes about five minutes.
"""

import shutil

import common
import coreml_common as cc
import coremltools as ct
import torch
from convert_laya import head_layer
from coremltools.models.compute_plan import MLComputePlan
from torch import nn


class Piece(nn.Module):
    """Laya's encoder followed by some of the parts DecisionModel applies after it."""

    def __init__(self, model, parts):
        super().__init__()
        self.model, self.parts = model, parts
        self.core = cc.ModernBertCore(model.encoder, 128)

    def forward(self, tokens):
        input_ids, attention_mask, qtype = tokens[:, 0, :], tokens[:, 1, :], tokens[:, 2, 0]
        h = self.core(input_ids, attention_mask)
        if "type" in self.parts:
            h = h + self.model.type_emb(qtype)[:, None, :]
        if "one-hot type" in self.parts:
            onehot = (qtype[:, None] == torch.arange(3, dtype=qtype.dtype)[None, :]).to(h.dtype)
            h = h + (onehot @ self.model.type_emb.weight)[:, None, :]
        padding = (1.0 - attention_mask.to(torch.float32)) * cc.NEG
        if "head" in self.parts:
            for layer in self.model.head.layers:
                h = head_layer(layer, h, padding)
        if "scorer" in self.parts:
            return self.model.scorer(h).squeeze(-1).float()
        return h.mean(-1)


def plan(module, path):
    mlmodel = cc.convert(cc.trace(module.eval(), 3, 128), (1, 3, 128), "fp16", ct.target.iOS18, "out")
    mlmodel.save(str(path))
    compiled = ct.utils.compile_model(str(path))
    counts = {}
    try:
        computed = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        for function in computed.model_structure.program.functions.values():
            for op in function.block.operations:
                usage = computed.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    continue  # constants have no device
                device = type(usage.preferred_compute_device).__name__
                device = device.replace("ML", "").replace("ComputeDevice", "")
                counts[device] = counts.get(device, 0) + 1
    finally:
        shutil.rmtree(compiled, ignore_errors=True)
        shutil.rmtree(path, ignore_errors=True)
    return counts


def main():
    common.require_upstream()
    torch.backends.mha.set_fastpath_enabled(False)
    model = common.laya_engine().agent.model.eval()
    common.MODELS.mkdir(parents=True, exist_ok=True)
    path = common.MODELS / "laya-plan.mlpackage"
    for parts in ((), ("type",), ("head",), ("scorer",), ("type", "head"), ("one-hot type", "head", "scorer")):
        label = " + ".join(("encoder",) + parts)
        print(f"{label}: {plan(Piece(model, parts), path)}", flush=True)


if __name__ == "__main__":
    main()
