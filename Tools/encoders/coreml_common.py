"""The Core ML conversion pieces shared by convert_verdict.py and convert_laya.py (spike #56).

Both models are ModernBERT encoders. ModernBertCore runs a transformers ModernBertModel's own
modules (embeddings, layers, final norm) with two changes that make the graph convert cleanly:

- The attention masks are additive float masks built from the attention mask with plain tensor
  operations (-1e4 where a key is padding, and in the local layers where |i - j| exceeds the
  window of 64), instead of transformers' masking_utils. -1e4 is finite in float16, so a padded
  query row whose whole window is padding gives finite values instead of NaN.
- The rotary cos and sin tables and the local-window band are computed once for the longest
  sequence and sliced to the input's length, so the converted program holds no position
  arithmetic.

PyTorch runs the wrapped models to within float32 rounding of the original ones; each converter
checks that on the reference prompts before converting.

Core ML allows only one input with enumerated shapes before iOS 18, so each model takes one int32
input, `tokens`, of shape [batch, planes, sequence]: plane 0 holds the token ids, plane 1 the
attention mask, and Laya's plane 2 the question type.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

NEG = -1e4


class ModernBertCore(nn.Module):
    """A ModernBertModel's forward pass with tabulated positions and additive float masks."""

    def __init__(self, encoder, max_length):
        super().__init__()
        self.encoder = encoder
        config = encoder.config
        pos = torch.arange(max_length)
        far = (pos[None, :] - pos[:, None]).abs() > config.sliding_window
        self.register_buffer("band", far.float() * NEG, persistent=False)
        dummy = torch.zeros(1, max_length, config.hidden_size)
        with torch.no_grad():
            for kind in ("full_attention", "sliding_attention"):
                cos, sin = encoder.rotary_emb(dummy, pos[None], kind)
                self.register_buffer(f"cos_{kind}", cos, persistent=False)
                self.register_buffer(f"sin_{kind}", sin, persistent=False)

    def forward(self, input_ids, attention_mask):
        s = input_ids.shape[1]
        full = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * NEG
        masks = {"full_attention": full, "sliding_attention": full + self.band[None, None, :s, :s]}
        positions = {kind: (getattr(self, f"cos_{kind}")[:, :s], getattr(self, f"sin_{kind}")[:, :s]) for kind in masks}
        h = self.encoder.embeddings(input_ids=input_ids)
        for layer in self.encoder.layers:
            h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=positions[layer.attention_type])
        return self.encoder.final_norm(h)


def function_name(batch, length):
    return f"b{batch}_s{length}"


def trace(module, planes, length):
    example = torch.zeros((1, planes, length), dtype=torch.int32)
    with torch.no_grad():
        return torch.jit.trace(module.eval(), (example,), check_trace=False)


def convert(traced, shape, precision, target, output):
    """One Core ML program. shape is a fixed tuple or a list of tuples to enumerate.

    precision is "fp32", "fp16", or "w8": float16 computation with the weights stored as int8,
    linearly quantized per output channel (symmetric), which halves the package.
    """
    import coremltools as ct

    if isinstance(shape, list):
        shape = ct.EnumeratedShapes(shapes=shape, default=shape[0])
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=shape, dtype=np.int32)],
        outputs=[ct.TensorType(name=output, dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16,
        convert_to="mlprogram",
        skip_model_load=True,
    )
    if precision == "w8":
        import coremltools.optimize as cto

        config = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config=config)
    return mlmodel


def op_histogram(mlmodel):
    """How many operations of each type the first function of a converted program holds."""
    counts = {}
    for fn in mlmodel._mil_program.functions.values():
        for op in fn.operations:
            counts[op.op_type] = counts.get(op.op_type, 0) + 1
        break
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_enumerated(traced, shapes, precision, target, output, path):
    """One program; a single shape converts as a fixed shape rather than an enumerated one."""
    started = time.perf_counter()
    mlmodel = convert(traced, shapes if len(shapes) > 1 else shapes[0], precision, target, output)
    ops = op_histogram(mlmodel)
    mlmodel.save(str(path))
    return time.perf_counter() - started, ops


def build_multifunction(traced, shapes, precision, target, output, path, scratch):
    """One function per fixed shape; save_multifunction keeps one copy of the shared weights."""
    import coremltools as ct

    started = time.perf_counter()
    descriptor = ct.utils.MultiFunctionDescriptor()
    ops = None
    parts = []
    for b, planes, s in shapes:
        mlmodel = convert(traced, (b, planes, s), precision, target, output)
        if ops is None:
            ops = op_histogram(mlmodel)
        part = Path(scratch) / f"{path.stem}.{function_name(b, s)}.mlpackage"
        mlmodel.save(str(part))
        descriptor.add_function(str(part), src_function_name="main", target_function_name=function_name(b, s))
        parts.append(part)
    descriptor.default_function_name = function_name(shapes[0][0], shapes[0][2])
    ct.utils.save_multifunction(descriptor, str(path))
    for part in parts:
        _remove_tree(part)
    return time.perf_counter() - started, ops


def _remove_tree(path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)


class CoreMLRunner:
    """Pads a batch to a shape the package accepts and runs it.

    The package is compiled once. An enumerated package is one loaded model. A multifunction
    package loads each function when a batch first needs it and keeps at most `capacity` loaded:
    every loaded function holds its own copy of the weights once it has run, and with all eight
    Laya functions loaded Core ML refused further loads with a misleading error ("functionName
    must be nil unless the model type is ML Program").
    """

    def __init__(self, path, kind, compute_units, batches, lengths, planes, output, capacity=2):
        import coremltools as ct

        self.batches, self.lengths, self.planes, self.output = batches, lengths, planes, output
        self.kind, self.units, self.capacity = kind, compute_units, capacity
        started = time.perf_counter()
        self.compiled = ct.utils.compile_model(str(path))
        self.loaded = {}
        if kind == "enumerated":
            self.loaded[None] = ct.models.CompiledMLModel(self.compiled, compute_units=compute_units)
        # compile_model, and for an enumerated package its load; a multifunction package loads each
        # function in model(), when a batch first needs it.
        self.load_seconds = time.perf_counter() - started
        # coremltools 9.0 can release an input's buffer from a Core ML thread after predict
        # returns, without the Python lock, which crashed the interpreter once in a multifunction
        # run. Holding every input until the runner goes away keeps that release from freeing it.
        self.inputs = []

    def model(self, b, s):
        import coremltools as ct

        if self.kind == "enumerated":
            return self.loaded[None]
        name = function_name(b, s)
        if name not in self.loaded:
            while len(self.loaded) >= self.capacity:
                self.loaded.pop(next(iter(self.loaded)))
            self.loaded[name] = ct.models.CompiledMLModel(self.compiled, compute_units=self.units,
                                                          function_name=name)
        return self.loaded[name]

    def close(self):
        self.loaded.clear()
        _remove_tree(self.compiled)

    def shape_for(self, rows, length):
        b = next(x for x in self.batches if x >= rows)
        s = next(x for x in self.lengths if x >= length)
        return b, s

    def run(self, planes_by_row, pad_id):
        """planes_by_row: per row, the list of planes as int lists of one length."""
        rows = len(planes_by_row)
        length = max(len(p[0]) for p in planes_by_row)
        b, s = self.shape_for(rows, length)
        tokens = np.zeros((b, self.planes, s), dtype=np.int32)
        tokens[:, 0, :] = pad_id
        for r, planes in enumerate(planes_by_row):
            n = len(planes[0])
            for k, plane in enumerate(planes):
                tokens[r, k, :n] = plane
            if self.planes > 2:
                tokens[r, 2, :] = planes[2][0]
        self.inputs.append(tokens)
        out = self.model(b, s).predict({"tokens": tokens})[self.output]
        return np.asarray(out[:rows], dtype=np.float32), (b, s)


def compare(reference, candidate):
    """Largest and mean absolute differences between two lists of equal-length float lists."""
    diffs = [abs(a - b) for r, c in zip(reference, candidate) for a, b in zip(r, c)]
    return max(diffs), float(np.mean(diffs))
