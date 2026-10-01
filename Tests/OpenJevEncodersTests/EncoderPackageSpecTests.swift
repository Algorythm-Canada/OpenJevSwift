import Testing

@testable import OpenJevEncoders

/// The model-free parts of the Core ML runner: which function a call uses, how its input is
/// padded, and which functions stay loaded.
@Suite("Encoder package spec")
struct EncoderPackageSpecTests {
    let spec = EncoderPackageSpec.verdict

    @Test("A call uses the smallest function that holds its rows and its longest row")
    func functionChoice() {
        let cases: [(rows: Int, longest: Int, function: String?)] = [
            (1, 1, "b1_s128"), (1, 128, "b1_s128"), (1, 129, "b1_s256"), (1, 512, "b1_s512"),
            (2, 10, "b16_s128"), (16, 300, "b16_s512"), (16, 256, "b16_s256"),
            (17, 10, nil), (1, 513, nil),
        ]
        for (rows, longest, function) in cases {
            #expect(
                spec.function(rows: rows, longestRow: longest)?.name == function,
                "\(rows) rows of \(longest) tokens")
        }
        #expect(spec.planes == 2)
        #expect(spec.inputName == "tokens")
        #expect(spec.outputName == "logits")
    }

    @Test("The input pads the ids with [PAD], the mask with 0, and fills padding rows")
    func inputPadding() {
        let function = EncoderPackageSpec.Function(batchSize: 16, sequenceLength: 128)
        let rows: [[[Int32]]] = [[[7, 8, 9], [1, 1, 1]], [[5], [1]]]
        let input = spec.input(rows, for: function)
        #expect(input.count == 16 * 2 * 128)
        let pad: Int32 = 50_283
        func plane(_ row: Int, _ plane: Int) -> ArraySlice<Int32> {
            let start = (row * 2 + plane) * 128
            return input[start..<(start + 128)]
        }
        #expect(Array(plane(0, 0)) == [7, 8, 9] + Array(repeating: pad, count: 125))
        #expect(Array(plane(0, 1)) == [1, 1, 1] + Array(repeating: 0, count: 125))
        #expect(Array(plane(1, 0)) == [5] + Array(repeating: pad, count: 127))
        #expect(Array(plane(1, 1)) == [1] + Array(repeating: 0, count: 127))
        for row in 2..<16 {
            #expect(plane(row, 0).allSatisfy { $0 == pad })
            #expect(plane(row, 1).allSatisfy { $0 == 0 })
        }
    }

    @Test("A plane padded with its first value repeats it, for Laya's question type")
    func firstValuePadding() {
        let laya = EncoderPackageSpec(
            name: "laya", batchSizes: [2], sequenceLengths: [4],
            padding: [.value(0), .value(0), .firstValue], outputName: "scores")
        let input = laya.input(
            [[[3, 4], [1, 1], [2, 2]]], for: .init(batchSize: 2, sequenceLength: 4))
        #expect(input == [3, 4, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2] + Array(repeating: 0, count: 12))
    }

    @Test("At most the capacity stays loaded, the least recently used released first")
    func leastRecentlyUsed() {
        var cache = LeastRecentlyUsed<String, Int>()
        cache.insert(1, forKey: "b16_s128")
        cache.insert(2, forKey: "b16_s512")
        #expect(cache.value(forKey: "b16_s128") == 1)
        #expect(cache.keys == ["b16_s512", "b16_s128"])
        cache.trim(to: 1)
        #expect(cache.keys == ["b16_s128"])
        #expect(cache.value(forKey: "b16_s512") == nil)
        cache.insert(3, forKey: "b16_s128")
        #expect(cache.keys == ["b16_s128"])
        #expect(cache.value(forKey: "b16_s128") == 3)
        cache.trim(to: 0)
        #expect(cache.keys.isEmpty)
    }

    @Test("The default compute units follow D-011 and never include .all")
    func computeUnits() {
        #if os(macOS)
            #expect(EncoderComputeUnits.platformDefault == .cpuAndGPU)
            #expect(VerdictBackend.Configuration.defaultMaxBatchRows == 16)
            #expect(VerdictBackend.Configuration.defaultFunctionCapacity == 2)
        #else
            #expect(EncoderComputeUnits.platformDefault == .cpuAndNeuralEngine)
            #expect(VerdictBackend.Configuration.defaultMaxBatchRows == 1)
            #expect(VerdictBackend.Configuration.defaultFunctionCapacity == 1)
        #endif
        #expect(EncoderComputeUnits.allCases.map(\.rawValue).contains("all") == false)
    }

    @Test("The module reports the package version")
    func version() {
        #expect(openJevEncodersVersion == "0.1.0-dev")
    }
}
