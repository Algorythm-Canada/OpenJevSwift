// `vision_config` of a DiffusionGemma checkpoint, with the defaults of mlx-vlm 0.6.15's
// mlx_vlm/models/gemma4/config.py `VisionConfig` (adapted from mlx-vlm, Copyright © 2025 Prince
// Canuma, MIT). The port's own type, so the library does not link MLXVLM for one struct (R18).

import Foundation

/// `vision_config`: the Gemma 4 vision tower's settings, with the defaults of mlx-vlm's
/// `VisionConfig`. Keys the type does not know are ignored, as mlx-vlm drops them.
public struct DiffusionGemmaVisionConfiguration: Codable, Equatable, Sendable {
    /// `model_type`, `gemma4_vision`.
    public let modelType: String
    /// `hidden_size`, the width of the tower's layers.
    public let hiddenSize: Int
    /// `intermediate_size`, the width of each layer's MLP.
    public let intermediateSize: Int
    /// `num_hidden_layers`, the number of transformer blocks.
    public let numHiddenLayers: Int
    /// `num_attention_heads`.
    public let numAttentionHeads: Int
    /// `num_key_value_heads`.
    public let numKeyValueHeads: Int
    /// `head_dim`, 72 for the pinned checkpoint, which the attention pads to 80 for MLX's fused
    /// kernel.
    public let headDim: Int
    /// `rms_norm_eps`.
    public let rmsNormEps: Float
    /// `default_output_length`, the most soft tokens an image gives (280).
    public let defaultOutputLength: Int
    /// `patch_size`, the side of a patch in pixels.
    public let patchSize: Int
    /// `position_embedding_size`, the entries of each axis's position table.
    public let positionEmbeddingSize: Int
    /// `pooling_kernel_size`, the side of the square of patches pooled into one soft token.
    public let poolingKernelSize: Int
    /// `use_clipped_linears`: whether the linears clamp their input and output to the
    /// checkpoint's calibration bounds.
    public let useClippedLinears: Bool
    /// `standardize`: whether the pooled output is shifted by `std_bias` and scaled by
    /// `std_scale`.
    public let standardize: Bool
    /// `rope_parameters`, the 2D RoPE's base frequency (`rope_theta`).
    public let ropeParameters: DiffusionGemmaTextConfiguration.RoPEParameters

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    /// Decodes `vision_config` with `VisionConfig`'s defaults; an absent `rope_parameters` is
    /// `{"rope_theta": 100.0, "rope_type": "default"}`, as `__post_init__` sets it.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3_072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        defaultOutputLength =
            try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10_240
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        ropeParameters =
            try c.decodeIfPresent(
                DiffusionGemmaTextConfiguration.RoPEParameters.self, forKey: .ropeParameters)
            ?? .init(ropeType: "default", ropeTheta: 100)
    }

    /// Encodes the configuration under config.json's keys.
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(defaultOutputLength, forKey: .defaultOutputLength)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(positionEmbeddingSize, forKey: .positionEmbeddingSize)
        try c.encode(poolingKernelSize, forKey: .poolingKernelSize)
        try c.encode(useClippedLinears, forKey: .useClippedLinears)
        try c.encode(standardize, forKey: .standardize)
        try c.encode(ropeParameters, forKey: .ropeParameters)
    }
}
