// The encoder KV cache of one layer, as mlx-vlm 0.6.15 fills it in a one-piece prefill and extends
// it between generated blocks (mlx_vlm/models/cache.py `KVCache` and `RotatingKVCache`, adapted
// from mlx-vlm, Copyright © 2025 Prince Canuma, MIT).

import MLX

/// The keys and values one layer's encoder pass leaves for the decoder.
///
/// Full layers write into a zero buffer of a multiple of 256 positions and keep a view of the
/// filled part, as `KVCache.update_and_fetch` does; an update that does not fit grows the buffer
/// by the multiples of 256 the new positions need, first cutting it to the filled part when that
/// is not a multiple of 256. Sliding layers keep every prompt position, as
/// `RotatingKVCache._update_concat` does on an empty cache; an update keeps the last
/// `slidingWindow − 1` positions and appends the new ones, as it does on a filled cache.
/// Attention reads ``keys`` and ``values``, so the layout is mlx-vlm's.
///
/// A prefill fills a cache once; ``extended()`` copies it, so a generation can append the blocks
/// it commits (`diffusion_update_cache`) without changing the prefill reads share. Not Sendable:
/// it holds MLX arrays, and its caller serialises its use.
public final class LayerCache {
    /// The block size mlx-vlm's `KVCache` grows by.
    static let fullLayerStep = 256

    /// Whether this is a full-attention layer's cache.
    public let isFullAttention: Bool
    /// The sliding window, `RotatingKVCache`'s `max_size`, which an update of a sliding layer's
    /// cache needs; nil for a full layer, or for a cache that only ever holds a prefill.
    public let slidingWindow: Int?
    /// The cached keys, `[batch, kvHeads, positions, headDim]`, or nil before the prefill.
    public private(set) var keys: MLXArray?
    /// The cached values, shaped as ``keys``.
    public private(set) var values: MLXArray?
    /// The number of positions written, the RoPE offset of what follows.
    public private(set) var offset = 0
    /// A full layer's whole buffer, of which ``keys`` and ``values`` are the filled part.
    private var keyBuffer: MLXArray?
    private var valueBuffer: MLXArray?

    /// An empty cache for a full-attention layer, or for a sliding one with its window.
    public init(isFullAttention: Bool, slidingWindow: Int? = nil) {
        self.isFullAttention = isFullAttention
        self.slidingWindow = slidingWindow
    }

    /// A cache holding the same arrays, which an update changes without changing this one.
    public func extended() -> LayerCache {
        let copy = LayerCache(isFullAttention: isFullAttention, slidingWindow: slidingWindow)
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        copy.keyBuffer = keyBuffer
        copy.valueBuffer = valueBuffer
        return copy
    }

    /// Stores new keys and values after the cached ones and returns what attention reads.
    ///
    /// - Precondition: a sliding layer's cache that already holds positions has a
    ///   ``slidingWindow``.
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let count = newKeys.dim(2)
        guard isFullAttention else {
            guard let keys, let values else {
                offset += count
                self.keys = newKeys
                self.values = newValues
                return (newKeys, newValues)
            }
            guard let window = slidingWindow else {
                preconditionFailure("a sliding layer's cache needs its window to take an update")
            }
            // `_update_concat` on a filled cache: the largest size is the window + new − 1.
            let trim = keys.dim(2) - window + 1
            let keptKeys = trim > 0 ? keys[.ellipsis, trim..., 0...] : keys
            let keptValues = trim > 0 ? values[.ellipsis, trim..., 0...] : values
            self.keys = concatenated([keptKeys, newKeys], axis: 2)
            self.values = concatenated([keptValues, newValues], axis: 2)
            offset += count
            return (self.keys!, self.values!)
        }
        let previous = offset
        if keyBuffer == nil || previous + count > keyBuffer!.dim(2) {
            let steps = (Self.fullLayerStep + count - 1) / Self.fullLayerStep
            let length = steps * Self.fullLayerStep
            let newKeyBuffer = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), length, newKeys.dim(3)], dtype: newKeys.dtype)
            let newValueBuffer = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), length, newValues.dim(3)],
                dtype: newValues.dtype)
            if var keyBuffer, var valueBuffer {
                if previous % Self.fullLayerStep != 0 {
                    keyBuffer = keyBuffer[.ellipsis, ..<previous, 0...]
                    valueBuffer = valueBuffer[.ellipsis, ..<previous, 0...]
                }
                self.keyBuffer = concatenated([keyBuffer, newKeyBuffer], axis: 2)
                self.valueBuffer = concatenated([valueBuffer, newValueBuffer], axis: 2)
            } else {
                keyBuffer = newKeyBuffer
                valueBuffer = newValueBuffer
            }
        }
        offset += count
        keyBuffer![.ellipsis, previous..<offset, 0...] = newKeys
        valueBuffer![.ellipsis, previous..<offset, 0...] = newValues
        let keyView = keyBuffer![.ellipsis, ..<offset, 0...]
        let valueView = valueBuffer![.ellipsis, ..<offset, 0...]
        keys = keyView
        values = valueView
        return (keyView, valueView)
    }
}
