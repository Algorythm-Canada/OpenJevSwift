import Foundation
import MLX
import MLXLMCommon
import MLXNN
import OpenJevDiffusionGemma
import Testing

/// The tiny tree, with float32 weights drawn from a fixed seed.
private func tinyModel() throws -> DiffusionGemmaModel {
    MetalLibrary.configure()
    MLXRandom.seed(24)
    let model = DiffusionGemmaModel(try ModelFixtures.tinyText())
    eval(model)
    return model
}

/// Random token ids, `[1, count]`.
private func tokens(_ count: Int) -> MLXArray {
    MLXRandom.randInt(0..<128, [1, count]).asType(.int32)
}

/// The largest absolute difference between two arrays.
private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// The tiny tree's parameter names after the tiny per-layer quantization, written out from the
/// structure of language.py, not from the Swift tree.
private let tinyQuantizedKeys: [String] = [
    "model.decoder.embed_tokens.biases",
    "model.decoder.embed_tokens.scales",
    "model.decoder.embed_tokens.weight",
    "model.decoder.layers.0.experts.down_proj.weight",
    "model.decoder.layers.0.experts.gate_up_proj.biases",
    "model.decoder.layers.0.experts.gate_up_proj.scales",
    "model.decoder.layers.0.experts.gate_up_proj.weight",
    "model.decoder.layers.0.input_layernorm.weight",
    "model.decoder.layers.0.layer_scalar",
    "model.decoder.layers.0.mlp.down_proj.biases",
    "model.decoder.layers.0.mlp.down_proj.scales",
    "model.decoder.layers.0.mlp.down_proj.weight",
    "model.decoder.layers.0.mlp.gate_proj.biases",
    "model.decoder.layers.0.mlp.gate_proj.scales",
    "model.decoder.layers.0.mlp.gate_proj.weight",
    "model.decoder.layers.0.mlp.up_proj.biases",
    "model.decoder.layers.0.mlp.up_proj.scales",
    "model.decoder.layers.0.mlp.up_proj.weight",
    "model.decoder.layers.0.post_attention_layernorm.weight",
    "model.decoder.layers.0.post_feedforward_layernorm.weight",
    "model.decoder.layers.0.post_feedforward_layernorm_1.weight",
    "model.decoder.layers.0.post_feedforward_layernorm_2.weight",
    "model.decoder.layers.0.pre_feedforward_layernorm.weight",
    "model.decoder.layers.0.pre_feedforward_layernorm_2.weight",
    "model.decoder.layers.0.router.per_expert_scale",
    "model.decoder.layers.0.router.proj.biases",
    "model.decoder.layers.0.router.proj.scales",
    "model.decoder.layers.0.router.proj.weight",
    "model.decoder.layers.0.router.scale",
    "model.decoder.layers.0.self_attn.k_norm.weight",
    "model.decoder.layers.0.self_attn.k_proj.biases",
    "model.decoder.layers.0.self_attn.k_proj.scales",
    "model.decoder.layers.0.self_attn.k_proj.weight",
    "model.decoder.layers.0.self_attn.o_proj.biases",
    "model.decoder.layers.0.self_attn.o_proj.scales",
    "model.decoder.layers.0.self_attn.o_proj.weight",
    "model.decoder.layers.0.self_attn.q_norm.weight",
    "model.decoder.layers.0.self_attn.q_proj.biases",
    "model.decoder.layers.0.self_attn.q_proj.scales",
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layers.0.self_attn.v_proj.biases",
    "model.decoder.layers.0.self_attn.v_proj.scales",
    "model.decoder.layers.0.self_attn.v_proj.weight",
    "model.decoder.layers.1.experts.down_proj.weight",
    "model.decoder.layers.1.experts.gate_up_proj.biases",
    "model.decoder.layers.1.experts.gate_up_proj.scales",
    "model.decoder.layers.1.experts.gate_up_proj.weight",
    "model.decoder.layers.1.input_layernorm.weight",
    "model.decoder.layers.1.layer_scalar",
    "model.decoder.layers.1.mlp.down_proj.biases",
    "model.decoder.layers.1.mlp.down_proj.scales",
    "model.decoder.layers.1.mlp.down_proj.weight",
    "model.decoder.layers.1.mlp.gate_proj.biases",
    "model.decoder.layers.1.mlp.gate_proj.scales",
    "model.decoder.layers.1.mlp.gate_proj.weight",
    "model.decoder.layers.1.mlp.up_proj.biases",
    "model.decoder.layers.1.mlp.up_proj.scales",
    "model.decoder.layers.1.mlp.up_proj.weight",
    "model.decoder.layers.1.post_attention_layernorm.weight",
    "model.decoder.layers.1.post_feedforward_layernorm.weight",
    "model.decoder.layers.1.post_feedforward_layernorm_1.weight",
    "model.decoder.layers.1.post_feedforward_layernorm_2.weight",
    "model.decoder.layers.1.pre_feedforward_layernorm.weight",
    "model.decoder.layers.1.pre_feedforward_layernorm_2.weight",
    "model.decoder.layers.1.router.per_expert_scale",
    "model.decoder.layers.1.router.proj.biases",
    "model.decoder.layers.1.router.proj.scales",
    "model.decoder.layers.1.router.proj.weight",
    "model.decoder.layers.1.router.scale",
    "model.decoder.layers.1.self_attn.k_norm.weight",
    "model.decoder.layers.1.self_attn.k_proj.biases",
    "model.decoder.layers.1.self_attn.k_proj.scales",
    "model.decoder.layers.1.self_attn.k_proj.weight",
    "model.decoder.layers.1.self_attn.o_proj.biases",
    "model.decoder.layers.1.self_attn.o_proj.scales",
    "model.decoder.layers.1.self_attn.o_proj.weight",
    "model.decoder.layers.1.self_attn.q_norm.weight",
    "model.decoder.layers.1.self_attn.q_proj.biases",
    "model.decoder.layers.1.self_attn.q_proj.scales",
    "model.decoder.layers.1.self_attn.q_proj.weight",
    "model.decoder.norm.weight",
    "model.decoder.self_conditioning.down_proj.biases",
    "model.decoder.self_conditioning.down_proj.scales",
    "model.decoder.self_conditioning.down_proj.weight",
    "model.decoder.self_conditioning.gate_proj.biases",
    "model.decoder.self_conditioning.gate_proj.scales",
    "model.decoder.self_conditioning.gate_proj.weight",
    "model.decoder.self_conditioning.pre_norm.weight",
    "model.decoder.self_conditioning.up_proj.biases",
    "model.decoder.self_conditioning.up_proj.scales",
    "model.decoder.self_conditioning.up_proj.weight",
    "model.encoder.language_model.layers.0.layer_scalar",
    "model.encoder.language_model.layers.1.layer_scalar",
]

extension MLXTests {
    @Suite("DiffusionGemma text blocks on a tiny configuration")
    struct BlockTests {
        @Test("Encoder mode over 12 tokens fills both caches")
        func encoderFillsCaches() throws {
            let model = try tinyModel()
            let caches = model.prefill(tokens(12))
            eval(caches.compactMap(\.keys) + caches.compactMap(\.values))
            #expect(caches.count == 2)
            #expect(!caches[0].isFullAttention)
            #expect(caches[1].isFullAttention)
            // Sliding: 1 KV head of 16; full: 1 KV head of 32 (num_global_key_value_heads).
            #expect(caches[0].keys?.shape == [1, 1, 12, 16])
            #expect(caches[0].values?.shape == [1, 1, 12, 16])
            #expect(caches[1].keys?.shape == [1, 1, 12, 32])
            #expect(caches[1].values?.shape == [1, 1, 12, 32])
            #expect(caches.map(\.offset) == [12, 12])
        }

        @Test("Encoder masks: the boolean band past the window, causal otherwise")
        func encoderMasks() throws {
            let model = try tinyModel()
            let slidingMask = model.encoderMask(for: .slidingAttention, length: 12)
            guard case .array(let band) = slidingMask else {
                Issue.record("the sliding mask past the window is not an array")
                return
            }
            #expect(band.shape == [12, 12])
            #expect(band.dtype == .bool)
            let values = band.asType(.int32).asArray(Int32.self)
            for row in 0..<12 {
                for column in 0..<12 {
                    let expected = row >= column && row < column + 8
                    #expect(
                        (values[row * 12 + column] != 0) == expected,
                        "row \(row), column \(column)")
                }
            }
            if case .causal = model.encoderMask(for: .fullAttention, length: 12) {
            } else {
                Issue.record("the full mask is not causal")
            }
            if case .causal = model.encoderMask(for: .slidingAttention, length: 8) {
            } else {
                Issue.record("the sliding mask within the window is not causal")
            }
            if case .none = model.encoderMask(for: .fullAttention, length: 1) {
            } else {
                Issue.record("a single token has a mask")
            }
        }

        @Test("Decoder mode over a 4-token canvas at offset 12")
        func decoderMode() throws {
            let model = try tinyModel()
            let caches = model.prefill(tokens(12))
            let canvas = MLXRandom.normal([1, 4, 64])
            var h = canvas
            for (index, layer) in model.decoder.layers.enumerated() {
                h = layer(h, mask: .none, cache: caches[index], decoder: true, offset: 12)
                #expect(h.shape == [1, 4, 64])
            }
            eval(h)
            #expect(caches.map(\.offset) == [12, 12])

            // The sliding layer reads only the last sliding_window - 1 = 7 encoder positions: a cache
            // holding just those gives the same output.
            let sliding = model.decoder.layers[0].selfAttention
            let cut = LayerCache(isFullAttention: false)
            _ = cut.update(
                keys: try #require(caches[0].keys)[.ellipsis, 5..., 0...],
                values: try #require(caches[0].values)[.ellipsis, 5..., 0...])
            let whole = sliding(canvas, mask: .none, cache: caches[0], decoder: true, offset: 12)
            let last7 = sliding(canvas, mask: .none, cache: cut, decoder: true, offset: 12)
            #expect(maxDifference(whole, last7) == 0)

            // An array mask over all 12 + 4 columns is cut to the last 7 + 4; all true, it is `.none`.
            let mask = MLXArray.ones([1, 1, 4, 16], dtype: .bool)
            let masked = sliding(
                canvas, mask: .array(mask), cache: caches[0], decoder: true, offset: 12)
            #expect(masked.shape == [1, 4, 64])
            #expect(maxDifference(whole, masked) <= 1e-6)

            // The full layer is not cut: dropping its first 5 positions changes the output.
            let full = model.decoder.layers[1].selfAttention
            let fullCut = LayerCache(isFullAttention: true)
            _ = fullCut.update(
                keys: try #require(caches[1].keys)[.ellipsis, 5..., 0...],
                values: try #require(caches[1].values)[.ellipsis, 5..., 0...])
            let fullWhole = full(canvas, mask: .none, cache: caches[1], decoder: true, offset: 12)
            let fullLast7 = full(canvas, mask: .none, cache: fullCut, decoder: true, offset: 12)
            #expect(maxDifference(fullWhole, fullLast7) > 1e-4)
        }

        @Test("The router's top-2 indices and per_expert_scale-weighted softmax")
        func router() throws {
            let model = try tinyModel()
            let router = model.decoder.layers[0].router
            let perExpertScale = MLXRandom.uniform(low: Float(0.5), high: Float(2), [8])
            try router.update(
                parameters: ModuleParameters.unflattened(["per_expert_scale": perExpertScale]),
                verify: .none)
            let x = MLXRandom.normal([5, 64])
            let (indices, weights) = router(x)
            #expect(indices.shape == [5, 2])
            #expect(weights.shape == [5, 2])

            let normed = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: 1e-6)
            let scores = router.proj(normed * router.scale * Float(pow(64.0, -0.5)))
            let scoreRows = scores.asArray(Float.self)
            let indexRows = indices.asType(.int32).asArray(Int32.self)
            let weightRows = weights.asArray(Float.self)
            let scales = router.perExpertScale.asArray(Float.self)
            for row in 0..<5 {
                let rowScores = Array(scoreRows[(row * 8)..<(row * 8 + 8)])
                let chosen = Array(indexRows[(row * 2)..<(row * 2 + 2)]).map(Int.init)
                let top = rowScores.indices.sorted { rowScores[$0] > rowScores[$1] }.prefix(2)
                #expect(Set(chosen) == Set(top), "row \(row)")
                let exps = chosen.map { exp(Double(rowScores[$0])) }
                let total = exps.reduce(0, +)
                var unscaled = 0.0
                for (slot, expert) in chosen.enumerated() {
                    let weight = Double(weightRows[row * 2 + slot])
                    let expected = exps[slot] / total * Double(scales[expert])
                    #expect(abs(weight - expected) <= 1e-5, "row \(row), expert \(expert)")
                    unscaled += weight / Double(scales[expert])
                }
                #expect(abs(unscaled - 1) <= 1e-5, "row \(row)")
            }
        }

        @Test("The experts give the same values with and without the sort")
        func expertsSort() throws {
            let model = try tinyModel()
            let layer = model.decoder.layers[0]
            // 40 tokens × top 2 = 80 assignments, past the threshold of 64: the default sorts.
            let x = MLXRandom.normal([40, 64])
            let (indices, weights) = layer.router(x)
            #expect(indices.size >= Experts.sortThreshold)
            let sorted = layer.experts(x, indices: indices, weights: weights)
            let unsorted = layer.experts(x, indices: indices, weights: weights, sort: false)
            #expect(sorted.shape == [40, 64])
            #expect(maxDifference(sorted, unsorted) <= 1e-5)

            // 4 tokens: 8 assignments, so the default does not sort.
            let small = x[0..<4]
            let (smallIndices, smallWeights) = layer.router(small)
            let automatic = layer.experts(small, indices: smallIndices, weights: smallWeights)
            let forced = layer.experts(
                small, indices: smallIndices, weights: smallWeights, sort: true)
            let plain = layer.experts(
                small, indices: smallIndices, weights: smallWeights, sort: false)
            #expect(maxDifference(automatic, plain) == 0)
            #expect(maxDifference(forced, plain) <= 1e-5)
        }

        @Test("The quantized tiny tree's parameter names are exactly the expected list")
        func quantizedKeySet() throws {
            MetalLibrary.configure()
            let model = DiffusionGemmaModel(try ModelFixtures.tinyText())
            model.quantize(try ModelFixtures.tinyPerLayerQuantization())
            let keys = model.parameters().flattened().map(\.0).sorted()
            #expect(keys == tinyQuantizedKeys)
            // The embedding, 4 + 3 + 1 + 1 in the sliding layer, 3 + 3 + 1 + 1 in the full one, and
            // self-conditioning's three projections.
            #expect(model.quantizedModuleCount == 21)
            #expect(model.decoder.embedTokens is QuantizedEmbedding)
            #expect(model.decoder.layers[0].experts.gateUpProj is QuantizedSwitchLinear)
            #expect(!(model.decoder.layers[0].experts.downProj is QuantizedSwitchLinear))
        }

        @Test("The quantized tiny tree runs both modes")
        func quantizedForward() throws {
            MetalLibrary.configure()
            MLXRandom.seed(27)
            let model = DiffusionGemmaModel(try ModelFixtures.tinyText())
            model.quantize(try ModelFixtures.tinyPerLayerQuantization())
            eval(model)
            let caches = model.prefill(tokens(12))
            var h = model.decoder.embed(tokens(4))
            for (index, layer) in model.decoder.layers.enumerated() {
                h = layer(h, mask: .none, cache: caches[index], decoder: true, offset: 12)
            }
            eval(h)
            #expect(h.shape == [1, 4, 64])
            #expect(h.asType(.float32).abs().max().item(Float.self).isFinite)
        }

        @Test("The full-attention RoPE table: 64 finite entries for the checkpoint, settable")
        func ropeTable() throws {
            MetalLibrary.configure()
            let model = DiffusionGemmaModel(try ModelFixtures.tinyText())
            #expect(model.decoder.layers[0].selfAttention.fullAttentionFrequencies == nil)
            let attention = model.decoder.layers[1].selfAttention
            let table = try #require(attention.fullAttentionFrequencies)
            // global_head_dim 32, partial_rotary_factor 0.25: 4 finite of 16.
            #expect(table.shape == [16])
            #expect(table.asArray(Float.self).filter(\.isFinite).count == 4)
            attention.fullAttentionFrequencies = MLXArray.ones([16])
            let installed = attention.fullAttentionFrequencies?.asArray(Float.self)
            #expect(installed == Array(repeating: 1, count: 16))
        }
    }
}
