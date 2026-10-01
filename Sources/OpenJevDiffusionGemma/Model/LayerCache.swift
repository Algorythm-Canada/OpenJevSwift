// The encoder KV cache of one layer, as mlx-vlm 0.6.15 fills it in a one-piece prefill
// (mlx_vlm/models/cache.py, adapted from mlx-vlm, Copyright © 2025 Prince Canuma, MIT). The prefill
// API (#25) may refine it.

import MLX

/// The keys and values one layer's encoder pass leaves for the decoder.
///
/// Sliding layers keep every prompt position, as `RotatingKVCache._update_concat` does on an empty
/// cache. Full layers write into a zero buffer of a multiple of 256 positions and keep a view of
/// the filled part, as `KVCache.update_and_fetch` does; attention reads that view, so the layout
/// is mlx-vlm's. Not Sendable: it holds MLX arrays, and its caller serialises its use.
public final class LayerCache {
    /// The block size mlx-vlm's `KVCache` grows by.
    static let fullLayerStep = 256

    /// Whether this is a full-attention layer's cache.
    public let isFullAttention: Bool
    /// The cached keys, `[batch, kvHeads, positions, headDim]`, or nil before the prefill.
    public private(set) var keys: MLXArray?
    /// The cached values, shaped as ``keys``.
    public private(set) var values: MLXArray?
    /// The number of positions written, the RoPE offset of what follows.
    public private(set) var offset = 0

    public init(isFullAttention: Bool) {
        self.isFullAttention = isFullAttention
    }

    /// Stores one prefill's keys and values and returns what attention reads.
    ///
    /// A cache holds one prefill: a second update is a programming error.
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys == nil, "a LayerCache holds one prefill")
        let count = newKeys.dim(2)
        offset += count
        guard isFullAttention else {
            keys = newKeys
            values = newValues
            return (newKeys, newValues)
        }
        let steps = (Self.fullLayerStep + count - 1) / Self.fullLayerStep
        let length = steps * Self.fullLayerStep
        let keyBuffer = MLXArray.zeros(
            [newKeys.dim(0), newKeys.dim(1), length, newKeys.dim(3)], dtype: newKeys.dtype)
        let valueBuffer = MLXArray.zeros(
            [newValues.dim(0), newValues.dim(1), length, newValues.dim(3)], dtype: newValues.dtype)
        keyBuffer[.ellipsis, 0..<count, 0...] = newKeys
        valueBuffer[.ellipsis, 0..<count, 0...] = newValues
        let keyView = keyBuffer[.ellipsis, ..<count, 0...]
        let valueView = valueBuffer[.ellipsis, ..<count, 0...]
        keys = keyView
        values = valueView
        return (keyView, valueView)
    }
}
