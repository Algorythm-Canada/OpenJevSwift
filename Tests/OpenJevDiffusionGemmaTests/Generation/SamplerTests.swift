import Foundation
import MLX
import Testing

@testable import OpenJevDiffusionGemma

/// The bit patterns of a float32 array.
private func bits(_ array: MLXArray) -> [UInt32] {
    array.asType(.float32).asArray(Float.self).map(\.bitPattern)
}

/// The values of an integer array.
private func ints(_ array: MLXArray) -> [Int] {
    array.asType(.int32).asArray(Int32.self).map(Int.init)
}

/// The values of a boolean array.
private func bools(_ array: MLXArray) -> [Bool] {
    array.asArray(Bool.self)
}

extension MLXTests {
    /// Each sampling function of mlx-vlm's diffusion loop against the vectors
    /// Tools/fixtures/generation_oracle.py recorded with mlx-vlm on the CPU
    /// (Fixtures/generation/generation.json `sampler`), on the CPU, bit for bit.
    @Suite("The diffusion sampler against mlx-vlm's vectors")
    struct SamplerTests {
        let sampler: GenerationOracle.Sampler

        init() throws {
            MetalLibrary.configure()
            sampler = try GenerationOracle.load().sampler
        }

        @Test("A seeded RandomState draws mlx-vlm's canvases after mx.random.seed")
        func initialCanvas() {
            Device.withDefaultDevice(.cpu) {
                for record in sampler.initializeCanvas {
                    let random = MLXRandom.RandomState(seed: record.seed)
                    for draw in record.draws {
                        let canvas = DiffusionSampler.initialCanvas(
                            length: draw.shape[1], vocabularySize: record.vocabSize,
                            random: random)
                        #expect(canvas.shape == draw.shape)
                        #expect(canvas.dtype == .int32)
                        #expect(ints(canvas) == draw.values, "seed \(record.seed)")
                    }
                }
            }
        }

        @Test("The linear temperature schedule runs from 0.8 to 0.4 in mlx-vlm's float32 values")
        func linearTemperature() {
            let record = sampler.linearTemperature
            let steps = record.maxDenoisingSteps
            let values = stride(from: steps, through: 1, by: -1).map {
                DiffusionSampler.linearTemperature(
                    step: $0, maxSteps: steps, schedule: (tMin: 0.4, tMax: 0.8))
            }
            #expect(values.map { $0?.bitPattern } == record.float32Bits)
            #expect(values.first == 0.8 && values.last == Float(0.4 + 0.4 / 48))
            #expect(
                DiffusionSampler.linearTemperature(step: 48, maxSteps: 48, schedule: nil) == nil)
        }

        @Test("Sampling is the argmax at temperature 0 and mlx-vlm's categorical draw above it")
        func sampleCanvas() {
            Device.withDefaultDevice(.cpu) {
                let logits = sampler.sampleCanvas.logits.array
                for record in sampler.sampleCanvas.cases {
                    let random = MLXRandom.RandomState(seed: record.seed ?? 0)
                    let ids = DiffusionSampler.sampleCanvas(
                        logits, temperature: Float(record.temperature), random: random)
                    #expect(ints(ids) == record.ids.values, "temperature \(record.temperature)")
                }
            }
        }

        @Test("Token probabilities and entropies are mlx-vlm's bits")
        func probabilityAndEntropy() {
            Device.withDefaultDevice(.cpu) {
                let record = sampler.tokenProbability
                let probability = DiffusionSampler.tokenProbability(
                    record.logits.array, tokenIDs: record.tokenIDs.array)
                #expect(bits(probability) == record.probability.float32Bits)
                for entropy in sampler.tokenEntropy {
                    let computed = DiffusionSampler.tokenEntropy(entropy.logits.array)
                    #expect(computed.shape == entropy.entropy.shape)
                    #expect(bits(computed) == entropy.entropy.float32Bits)
                }
            }
        }

        @Test("The entropy-bound mask accepts mlx-vlm's positions, ties included")
        func entropyTransferMask() {
            Device.withDefaultDevice(.cpu) {
                for record in sampler.entropyTransferMask {
                    let mask = DiffusionSampler.entropyTransferMask(
                        entropy: record.entropy.array, bound: Float(record.entropyBound))
                    #expect(bools(mask) == record.mask.values, "bound \(record.entropyBound)")
                }
            }
        }

        @Test("The confidence mask accepts, forces the best position, and forces all on request")
        func confidenceTransferMask() {
            Device.withDefaultDevice(.cpu) {
                let record = sampler.confidenceTransferMask
                for item in record.cases {
                    let mask = DiffusionSampler.confidenceTransferMask(
                        confidence: record.confidence.array, unrevealed: record.unrevealed.array,
                        threshold: Float(item.threshold), forceAll: item.forceAll)
                    #expect(
                        bools(mask) == item.mask.values,
                        "threshold \(item.threshold), force all \(item.forceAll)")
                }
            }
        }

        @Test("Stable-and-confident follows mlx-vlm's history over a sequence of steps")
        func stableAndConfident() throws {
            try Device.withDefaultDevice(.cpu) {
                let record = sampler.stableAndConfident
                for item in record.cases {
                    let rule = item.config.map {
                        DiffusionSampler.StoppingRule(
                            stabilityThreshold: $0.stabilityThreshold,
                            confidenceThreshold: $0.confidenceThreshold)
                    }
                    var history: [MLXArray] = []
                    var results: [Bool] = []
                    for step in record.sequence {
                        let canvas = try #require(record.canvases[step[0]]).array
                        let logits = try #require(record.logits[step[1]]).array
                        results.append(
                            DiffusionSampler.stableAndConfident(
                                canvas: canvas, logits: logits, history: &history, rule: rule))
                    }
                    #expect(results == item.results, "\(String(describing: item.config))")
                }
            }
        }
    }
}
