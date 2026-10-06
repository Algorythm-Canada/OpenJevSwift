// The Gemma 4 vision tower of mlx-vlm 0.6.15 (mlx_vlm/models/gemma4/vision.py) and the multimodal
// embedder and masked_scatter of mlx_vlm/models/gemma4/gemma4.py and language.py's
// RMSNormNoScale, adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT. The operations,
// their order, shapes and dtypes are mlx-vlm's, so that a read with an image is bit for bit in
// D-014's exact tier (D-054). MLXVLM's own Gemma 4 tower is internal to mlx-swift-lm and departs
// from mlx-vlm's arithmetic, so it is not used.

import Foundation
import MLX
import MLXNN

/// vision.py's `ClippableLinear`: a linear layer, under `linear`, that clamps its input and
/// output to the checkpoint's calibration bounds when `use_clipped_linears` is set.
public final class ClippableLinear: Module, UnaryLayer {
    /// The projection.
    @ModuleInfo public var linear: Linear
    @ParameterInfo(key: "input_min") var inputMin: MLXArray?
    @ParameterInfo(key: "input_max") var inputMax: MLXArray?
    @ParameterInfo(key: "output_min") var outputMin: MLXArray?
    @ParameterInfo(key: "output_max") var outputMax: MLXArray?

    /// A linear layer without bias; with `clipping`, bounds of ±infinity until the weights load.
    public init(_ inputs: Int, _ outputs: Int, clipping: Bool) {
        _linear.wrappedValue = Linear(inputs, outputs, bias: false)
        if clipping {
            _inputMin.wrappedValue = MLXArray(-Float.infinity)
            _inputMax.wrappedValue = MLXArray(Float.infinity)
            _outputMin.wrappedValue = MLXArray(-Float.infinity)
            _outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    /// `x`, `[..., inputs]`, projected to `[..., outputs]`. When the layer clips, `x` is clamped
    /// to `input_min` and `input_max` first and the result to `output_min` and `output_max`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let inputMin, let inputMax {
            x = clip(x, min: inputMin, max: inputMax)
        }
        x = linear(x)
        if let outputMin, let outputMax {
            x = clip(x, min: outputMin, max: outputMax)
        }
        return x
    }
}

/// MLX's `power` on float32 as the Python wheel computes it: `metal::precise::pow`, elementwise,
/// with `base` and `exponent` broadcast to one shape.
///
/// The wheel's precompiled `Power` kernel is built without fast math, so it calls the precise
/// `pow`. mlx-swift compiles `Power` from source at run time with fast math, which rounds
/// differently: on the hot dog's first vision block a third of `x ** 2` differ from the
/// wheel's, and the three RMS norms and the RoPE timescale with them, even under D-014's exact
/// tier. This kernel calls the precise function explicitly, so the tower's `pow`s are the
/// wheel's in both tiers (D-054).
func precisePow(_ base: MLXArray, _ exponent: MLXArray) -> MLXArray {
    let shape = broadcastShapes(base.shape, exponent.shape)
    let size = shape.reduce(1, *)
    guard size > 0 else { return MLXArray.zeros(shape, dtype: .float32) }
    let output = precisePowKernel(
        [
            broadcast(base.asType(.float32), to: shape).flattened(),
            broadcast(exponent.asType(.float32), to: shape).flattened(),
        ],
        grid: (size, 1, 1), threadGroup: (min(256, size), 1, 1),
        outputShapes: [[size]], outputDTypes: [.float32])[0]
    return output.reshaped(shape)
}

/// The kernel of ``precisePow(_:_:)``, compiled once.
private let precisePowKernel = MLXFast.metalKernel(
    name: "openjev_precise_pow", inputNames: ["base", "exponent"], outputNames: ["out"],
    source: """
        uint i = thread_position_in_grid.x;
        out[i] = metal::precise::pow(base[i], exponent[i]);
        """)

/// The shape two shapes broadcast to, as NumPy's rules give it.
private func broadcastShapes(_ a: [Int], _ b: [Int]) -> [Int] {
    let count = max(a.count, b.count)
    let left = Array(repeating: 1, count: count - a.count) + a
    let right = Array(repeating: 1, count: count - b.count) + b
    return zip(left, right).map { $0 == 1 ? $1 : $0 }
}

/// vision.py's `VisionRMSNorm`: the norm in float32, times the weight in float32, back to the
/// input's dtype.
public final class VisionRMSNorm: Module, UnaryLayer {
    /// The learned scale, `weight`.
    @ParameterInfo public var weight: MLXArray
    let eps: Float

    /// A norm over `dimensions` with a scale of ones until the weights load.
    public init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    /// `x`, `[..., dimensions]`, normed over its last axis and scaled by `weight`, in its own
    /// shape and dtype.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let float = x.asType(.float32)
        // `x_float**2` is MLX's power, not square, and the wheel's power is the precise pow.
        let variance = mean(precisePow(float, MLXArray(Float(2))), axis: -1, keepDims: true)
        let normed = float * rsqrt(variance + eps)
        return (normed * weight.asType(.float32)).asType(x.dtype)
    }
}

/// vision.py's `VisionRMSNormNoScale`, the values' norm: ``VisionRMSNorm`` without a weight.
func visionRMSNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let float = x.asType(.float32)
    let variance = mean(precisePow(float, MLXArray(Float(2))), axis: -1, keepDims: true)
    return (float * rsqrt(variance + eps)).asType(x.dtype)
}

/// vision.py's `one_hot`: float32 ones where `indices` equals the class.
func visionOneHot(_ indices: MLXArray, classes: Int) -> MLXArray {
    (expandedDimensions(indices, axis: -1) .== MLXArray(Int32(0)..<Int32(classes)))
        .asType(.float32)
}

/// vision.py's `_rotate_half`: `[-x2, x1]` over the last axis.
func visionRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// vision.py's `apply_multidimensional_rope` for `[B, L, n]` positions: the head splits into
/// `n` parts, each rotated by its own axis's position. The tower always passes 2D positions, so
/// mlx-vlm's 1D fallback is not ported.
///
/// - Parameters:
///   - inputs: `[B, L, heads, headDim]`.
///   - positions: `[B, L, n]` int32 patch coordinates, x first.
///   - baseFrequency: `rope_theta`.
func visionMultidimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float
) -> MLXArray {
    let headDim = inputs.dim(-1)
    let dimensions = positions.dim(-1)
    let channels = 2 * (headDim / (2 * dimensions))
    let half = channels / 2
    // Python divides in double, then MLX rounds the weak scalar to float32.
    let step = Float(2.0 / Double(channels))
    let exponents = step * MLXArray(Int32(0)..<Int32(half)).asType(.float32)
    let timescale = precisePow(MLXArray(baseFrequency), exponents)
    var parts: [MLXArray] = []
    for d in 0..<dimensions {
        let part = inputs[.ellipsis, (d * channels)..<((d + 1) * channels)]
        let sinusoid = positions[.ellipsis, d..<(d + 1)].asType(.float32) / timescale
        var cosine = cos(sinusoid)
        var sine = sin(sinusoid)
        cosine = expandedDimensions(
            concatenated([cosine, cosine], axis: -1).asType(inputs.dtype), axis: 2)
        sine = expandedDimensions(
            concatenated([sine, sine], axis: -1).asType(inputs.dtype), axis: 2)
        parts.append(part * cosine + visionRotateHalf(part) * sine)
    }
    return concatenated(parts, axis: -1)
}

/// mlx-vlm's `ensure_fused_sdpa` (models/base.py): the head size padded with zeros to the next
/// size MLX's fused kernel takes (64, 80 or 128), attention, and the padding cut off.
func visionFusedAttention(
    _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, mask: MLXArray?
) -> MLXArray {
    let size = queries.dim(-1)
    let target = [64, 80, 128].first { size <= $0 } ?? size
    var (q, k, v) = (queries, keys, values)
    if target != size {
        let widths =
            Array(repeating: IntOrPair(0), count: q.ndim - 1) + [IntOrPair((0, target - size))]
        q = padded(q, widths: widths)
        k = padded(k, widths: widths)
        v = padded(v, widths: widths)
    }
    let output = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v, scale: 1.0, mask: mask.map { .array($0) } ?? .none)
    return output[.ellipsis, ..<size]
}

/// vision.py's `VisionAttention`: bidirectional attention over the patches with 2D RoPE and
/// normed queries, keys and values, at scale 1.
public final class VisionAttention: Module {
    /// The query projection, `q_proj`.
    @ModuleInfo(key: "q_proj") public var qProj: ClippableLinear
    /// The key projection, `k_proj`.
    @ModuleInfo(key: "k_proj") public var kProj: ClippableLinear
    /// The value projection, `v_proj`.
    @ModuleInfo(key: "v_proj") public var vProj: ClippableLinear
    /// The output projection, `o_proj`.
    @ModuleInfo(key: "o_proj") public var oProj: ClippableLinear
    /// The query norm over each head, `q_norm`.
    @ModuleInfo(key: "q_norm") public var qNorm: VisionRMSNorm
    /// The key norm over each head, `k_norm`.
    @ModuleInfo(key: "k_norm") public var kNorm: VisionRMSNorm

    let heads: Int
    let keyValueHeads: Int
    let headDim: Int
    let ropeBase: Float

    /// The attention of one block.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        heads = config.numAttentionHeads
        keyValueHeads = config.numKeyValueHeads
        headDim = config.headDim
        ropeBase = config.ropeParameters.ropeTheta
        let hidden = config.hiddenSize
        let clip = config.useClippedLinears
        _qProj.wrappedValue = ClippableLinear(hidden, heads * headDim, clipping: clip)
        _kProj.wrappedValue = ClippableLinear(hidden, keyValueHeads * headDim, clipping: clip)
        _vProj.wrappedValue = ClippableLinear(hidden, keyValueHeads * headDim, clipping: clip)
        _oProj.wrappedValue = ClippableLinear(heads * headDim, hidden, clipping: clip)
        // mlx-vlm builds these norms with their default eps, 1e-6, not the configuration's.
        _qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim, eps: 1e-6)
        _kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim, eps: 1e-6)
        super.init()
    }

    /// vision.py's `VisionAttention.__call__`: queries and keys normed and rotated by the
    /// patches' positions, values normed without a scale, attention at scale 1, and the output
    /// projection.
    ///
    /// - Parameters:
    ///   - x: `[B, L, hidden]`, one row per patch.
    ///   - positions: `[B, L, 2]` int32 patch coordinates, x first.
    ///   - mask: the attention mask, `[B, 1, L, L]`, or nil for none.
    /// - Returns: `[B, L, hidden]`.
    public func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qNorm(qProj(x).reshaped(batch, length, heads, headDim))
        var keys = kNorm(kProj(x).reshaped(batch, length, keyValueHeads, headDim))
        var values = visionRMSNormNoScale(
            vProj(x).reshaped(batch, length, keyValueHeads, headDim), eps: 1e-6)
        queries = visionMultidimensionalRoPE(queries, positions: positions, baseFrequency: ropeBase)
        keys = visionMultidimensionalRoPE(keys, positions: positions, baseFrequency: ropeBase)
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        let output = visionFusedAttention(queries, keys, values, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

/// vision.py's `VisionMLP`: `down(gelu_approx(gate(x)) * up(x))`.
public final class VisionMLP: Module, UnaryLayer {
    /// The gate projection, `gate_proj`.
    @ModuleInfo(key: "gate_proj") public var gateProj: ClippableLinear
    /// The up projection, `up_proj`.
    @ModuleInfo(key: "up_proj") public var upProj: ClippableLinear
    /// The down projection back to the hidden size, `down_proj`.
    @ModuleInfo(key: "down_proj") public var downProj: ClippableLinear

    /// The MLP of one block.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        let clip = config.useClippedLinears
        _gateProj.wrappedValue = ClippableLinear(
            config.hiddenSize, config.intermediateSize, clipping: clip)
        _upProj.wrappedValue = ClippableLinear(
            config.hiddenSize, config.intermediateSize, clipping: clip)
        _downProj.wrappedValue = ClippableLinear(
            config.intermediateSize, config.hiddenSize, clipping: clip)
        super.init()
    }

    /// `down_proj(gelu_approx(gate_proj(x)) * up_proj(x))`, in `x`'s shape, `[..., hidden]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLXNN's geluApproximate is mlx.nn.gelu_approx, compiled shapeless as there.
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// vision.py's `VisionTransformerBlock`: attention and MLP, each between two norms, with
/// residuals.
public final class VisionBlock: Module {
    /// The self-attention, `self_attn`.
    @ModuleInfo(key: "self_attn") public var selfAttention: VisionAttention
    /// The MLP, `mlp`.
    @ModuleInfo public var mlp: VisionMLP
    /// The norm before the attention, `input_layernorm`.
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    /// The norm of the attention's output, `post_attention_layernorm`.
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    /// The norm before the MLP, `pre_feedforward_layernorm`.
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFeedforwardLayerNorm: RMSNorm
    /// The norm of the MLP's output, `post_feedforward_layernorm`.
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFeedforwardLayerNorm: RMSNorm

    /// One block.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        _selfAttention.wrappedValue = VisionAttention(config)
        _mlp.wrappedValue = VisionMLP(config)
        let (size, eps) = (config.hiddenSize, config.rmsNormEps)
        _inputLayerNorm.wrappedValue = rmsNorm(dimensions: size, eps: eps)
        _postAttentionLayerNorm.wrappedValue = rmsNorm(dimensions: size, eps: eps)
        _preFeedforwardLayerNorm.wrappedValue = rmsNorm(dimensions: size, eps: eps)
        _postFeedforwardLayerNorm.wrappedValue = rmsNorm(dimensions: size, eps: eps)
        super.init()
    }

    /// vision.py's `VisionTransformerBlock.__call__`: the attention's residual
    /// `h = x + post_attention_layernorm(self_attn(input_layernorm(x)))`, then the MLP's,
    /// `h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))`.
    ///
    /// - Parameters:
    ///   - x: `[B, L, hidden]`.
    ///   - positions: `[B, L, 2]` int32 patch coordinates, x first, for the attention's RoPE.
    ///   - mask: the attention mask, `[B, 1, L, L]`, or nil for none.
    /// - Returns: `[B, L, hidden]`.
    public func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let attention = postAttentionLayerNorm(
            selfAttention(inputLayerNorm(x), positions: positions, mask: mask))
        let h = x + attention
        return h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
    }
}

/// vision.py's `VisionPatchEmbedder`: each 16 by 16 patch projected, plus a learned embedding
/// per axis of its position.
public final class VisionPatchEmbedder: Module {
    /// The projection of each patch's `3·p·p` pixel values to the hidden size, `input_proj`.
    @ModuleInfo(key: "input_proj") public var inputProj: Linear
    /// The position embeddings, `position_embedding_table`: one table per axis, x first, with a
    /// row per coordinate, `[2, position_embedding_size, hidden]`.
    @ParameterInfo(key: "position_embedding_table") public var positionEmbeddingTable: MLXArray

    let patchSize: Int
    let positionEmbeddingSize: Int

    /// The embedder, with ones for the position table until the weights load.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        patchSize = config.patchSize
        positionEmbeddingSize = config.positionEmbeddingSize
        _inputProj.wrappedValue = Linear(
            3 * config.patchSize * config.patchSize, config.hiddenSize, bias: false)
        _positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, config.positionEmbeddingSize, config.hiddenSize,
        ])
        super.init()
    }

    /// `_position_embeddings`: one-hot positions times the table, summed over the two axes,
    /// zero at padding.
    func positionEmbeddings(_ positions: MLXArray, padding: MLXArray) -> MLXArray {
        let table = positionEmbeddingTable
        let oneHot = visionOneHot(positions, classes: positionEmbeddingSize)
            .transposed(0, 2, 1, 3).asType(table.dtype)
        let embeddings = matmul(oneHot, table).sum(axis: 1)
        return which(
            expandedDimensions(padding, axis: -1), MLXArray(Float(0)).asType(embeddings.dtype),
            embeddings)
    }

    /// `_patchify`: `[B, C, H, W]` pixels to `[B, patches, C·p·p]` rows, scaled to [-1, 1] and
    /// projected.
    func patchify(_ pixels: MLXArray) -> MLXArray {
        let (batch, channels) = (pixels.dim(0), pixels.dim(1))
        let (height, width) = (pixels.dim(2), pixels.dim(3))
        let p = patchSize
        let (rows, columns) = (height / p, width / p)
        var patches = pixels.reshaped(batch, channels, rows, p, columns, p)
            .transposed(0, 2, 4, 3, 5, 1)
            .reshaped(batch, rows * columns, channels * p * p)
        patches = Float(2) * (patches - Float(0.5))
        return inputProj(patches.asType(inputProj.weight.dtype))
    }

    /// vision.py's `VisionPatchEmbedder.__call__`: each patch's pixels scaled to [-1, 1] and
    /// projected, plus the embedding of its position, which is zero for a padding patch.
    ///
    /// - Parameters:
    ///   - pixels: `[B, C, H, W]`.
    ///   - positions: `[B, patches, 2]` int32 patch coordinates, x first.
    ///   - padding: `[B, patches]`, true for a padding patch.
    /// - Returns: `[B, patches, hidden]`.
    public func callAsFunction(_ pixels: MLXArray, positions: MLXArray, padding: MLXArray)
        -> MLXArray
    {
        patchify(pixels) + positionEmbeddings(positions, padding: padding)
    }
}

/// vision.py's `VisionModel` with its `VisionPooler` and `VisionTransformerModel`
/// (`encoder.layers`): patches, 27 bidirectional blocks, a 3 by 3 average pool into soft
/// tokens, times √hidden, and the standardization.
public final class VisionModel: Module {
    /// The patch embedder, `patch_embedder`.
    @ModuleInfo(key: "patch_embedder") public var patchEmbedder: VisionPatchEmbedder
    /// The transformer blocks, `encoder`.
    @ModuleInfo public var encoder: VisionEncoder
    /// The shift the standardization subtracts from the soft tokens, `std_bias`, or nil when
    /// the configuration does not standardize.
    @ParameterInfo(key: "std_bias") public var stdBias: MLXArray?
    /// The scale the standardization then multiplies them by, `std_scale`, or nil when the
    /// configuration does not standardize.
    @ParameterInfo(key: "std_scale") public var stdScale: MLXArray?

    /// The configuration the tower was built from.
    public let configuration: DiffusionGemmaVisionConfiguration
    /// `hidden_size ** 0.5`, which mlx-vlm computes in double and MLX rounds to the activations'
    /// dtype when it multiplies.
    let rootHiddenSize: Float

    /// The tower, with MLXNN's initial values until the weights load.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        configuration = config
        rootHiddenSize = Float(pow(Double(config.hiddenSize), 0.5))
        _patchEmbedder.wrappedValue = VisionPatchEmbedder(config)
        _encoder.wrappedValue = VisionEncoder(config)
        if config.standardize {
            _stdBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            _stdScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    /// The soft tokens of the images, `[1, total soft tokens, hidden]`, as mlx-vlm's
    /// `VisionModel.__call__` gives them for the processor's `pixel_values`: one `(n, 3, H, W)`
    /// array when the images share a size, read as a batch, else one `(3, H, W)` array per
    /// image, each read on its own, with the results concatenated in order.
    ///
    /// - Parameters:
    ///   - pixelValues: the processor's `pixel_values`, as ``ImageReadInputs`` holds them.
    ///   - stages: receives `vision.patches`, each block's output as `vision.layer.N`,
    ///     `vision.pooled` and `vision.out`, for the parity tests (the last image's, when they
    ///     are read one by one).
    public func callAsFunction(_ pixelValues: [MLXArray], stages: StageObserver? = nil)
        -> MLXArray
    {
        if pixelValues.count == 1, pixelValues[0].ndim == 4 {
            return features(pixelValues[0], stages: stages)
        }
        let each = pixelValues.map { image in
            features(image.ndim == 3 ? image[.newAxis] : image, stages: stages)[0]
        }
        return concatenated(each, axis: 0)[.newAxis]
    }

    /// One batch of same-sized images, `[B, 3, H, W]`, without position ids: every patch is
    /// real, so nothing is padded.
    func features(_ pixels: MLXArray, stages: StageObserver? = nil) -> MLXArray {
        let (batch, height, width) = (pixels.dim(0), pixels.dim(2), pixels.dim(3))
        let p = configuration.patchSize
        let (rows, columns) = (height / p, width / p)
        let patches = rows * columns
        let outputLength =
            patches / (configuration.poolingKernelSize * configuration.poolingKernelSize)

        // `_patch_positions_single`: x varies fastest, as numpy's meshgrid with "xy" indexing.
        var grid = [Int32]()
        grid.reserveCapacity(batch * patches * 2)
        for _ in 0..<batch {
            for y in 0..<rows {
                for x in 0..<columns {
                    grid.append(Int32(x))
                    grid.append(Int32(y))
                }
            }
        }
        let positions = MLXArray(grid, [batch, patches, 2])
        let padding = MLXArray.zeros([batch, patches], dtype: .bool)

        let embeddings = patchEmbedder(pixels, positions: positions, padding: padding)
        stages?("vision.patches", embeddings)

        // A bidirectional additive mask, 0 between real patches and -1e4 elsewhere, in the
        // activations' dtype, as mlx-vlm passes it to the fused kernel.
        let valid = logicalNot(padding)
        let pairs = expandedDimensions(valid, axis: 1) * expandedDimensions(valid, axis: 2)
        let mask = expandedDimensions(
            which(
                pairs, MLXArray(Float(0)).asType(embeddings.dtype),
                MLXArray(Float(-1e4)).asType(embeddings.dtype)),
            axis: 1)

        let hidden = encoder(embeddings, positions: positions, mask: mask, stages: stages)
        // The pooler's `output_length or self.default_output_length`.
        let (pooled, poolMask) = pool(
            hidden, positions: positions, padding: padding,
            length: outputLength == 0 ? configuration.defaultOutputLength : outputLength)
        stages?("vision.pooled", pooled)
        let validRows = poolMask.dim(1) == outputLength ? poolMask : logicalNot(poolMask)

        var rowsKept: [MLXArray] = []
        for index in 0..<batch {
            let count = validRows[index].asType(.int32).sum().item(Int.self)
            rowsKept.append(pooled[index, ..<count])
        }
        var output = concatenated(rowsKept, axis: 0)[.newAxis]
        if let stdBias, let stdScale {
            output = (output - stdBias) * stdScale
        }
        stages?("vision.out", output)
        return output
    }

    /// `VisionPooler.__call__`: padding zeroed, then each `k` by `k` square of patches averaged
    /// into one soft token by its position, and the result times √hidden.
    func pool(_ hidden: MLXArray, positions: MLXArray, padding: MLXArray, length: Int)
        -> (MLXArray, MLXArray)
    {
        var hidden = which(
            expandedDimensions(padding, axis: -1), MLXArray(Float(0)).asType(hidden.dtype), hidden)
        var mask = padding
        if hidden.dim(1) != length {
            // `_avg_pool_by_positions`.
            let k = Int(Double(hidden.dim(1) / length).squareRoot())
            let clamped = clip(positions, min: MLXArray(Int32(0)))
            let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
            let kernel = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
            let indices = kernel[.ellipsis, 0] + floorDivide(maxX, k) * kernel[.ellipsis, 1]
            let weights = visionOneHot(indices, classes: length).asType(.float32) / Float(k * k)
            hidden = einsum("bLl,bLd->bld", weights, hidden).asType(hidden.dtype)
            mask = logicalNot(all(weights .== Float(0), axis: 1))
        }
        return (hidden * rootHiddenSize, mask)
    }
}

/// vision.py's `VisionTransformerModel`, `vision_tower.encoder`: the blocks.
public final class VisionEncoder: Module {
    /// The blocks, `layers`, in the order they run.
    @ModuleInfo public var layers: [VisionBlock]

    /// `num_hidden_layers` blocks.
    public init(_ config: DiffusionGemmaVisionConfiguration) {
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in VisionBlock(config) }
        super.init()
    }

    /// vision.py's `VisionTransformerModel.__call__`: `x` through every block in order.
    ///
    /// - Parameters:
    ///   - x: `[B, L, hidden]`, the patch embeddings.
    ///   - positions: `[B, L, 2]` int32 patch coordinates, x first.
    ///   - mask: the attention mask, `[B, 1, L, L]`, or nil for none.
    ///   - stages: receives each block's output as `vision.layer.N`, for the parity tests.
    /// - Returns: `[B, L, hidden]`, the last block's output.
    public func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray?, stages: StageObserver? = nil
    ) -> MLXArray {
        var h = x
        for (index, layer) in layers.enumerated() {
            h = layer(h, positions: positions, mask: mask)
            stages?("vision.layer.\(index)", h)
        }
        return h
    }
}

/// gemma4.py's `MultimodalEmbedder`, `embed_vision`: the soft tokens normed without a scale
/// (language.py's `RMSNormNoScale`) and projected to the text model's width.
public final class MultimodalEmbedder: Module, UnaryLayer {
    /// The projection from the tower's width to the text model's, `embedding_projection`.
    @ModuleInfo(key: "embedding_projection") public var embeddingProjection: Linear
    let eps: Float

    /// The embedder from the tower's width to the text model's.
    public init(embeddingDimensions: Int, textHiddenSize: Int, eps: Float) {
        _embeddingProjection.wrappedValue = Linear(
            embeddingDimensions, textHiddenSize, bias: false)
        self.eps = eps
        super.init()
    }

    /// The soft tokens `x`, `[..., embeddingDimensions]`, normed without a scale and projected to
    /// the text model's width, `[..., textHiddenSize]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(rmsNormNoScale(x, eps: eps))
    }
}

/// gemma4.py's `masked_scatter`: the `source` values, in order, at the positions `mask` sets,
/// and `input` elsewhere. When the mask sets more positions than `source` has values, the
/// values wrap around, as mlx-vlm's `indices % source.size` makes them.
func maskedScatter(_ input: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray {
    let flat = mask.flattened().asType(.int32)
    let indices = cumsum(flat) - 1
    let aligned = source.flattened()[indices % source.size]
    return which(flat, aligned, input.flattened()).reshaped(input.shape)
}
