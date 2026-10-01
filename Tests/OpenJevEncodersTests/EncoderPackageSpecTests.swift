import Foundation
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

    @Test("Laya's packages: the Mac's one function per shape, the iPhone's one program each")
    func layaSpecs() {
        let mac = EncoderPackageSpec.layaMultifunction
        #expect(mac.name == "laya-m18-fp16")
        #expect(mac.layout == .functionPerShape)
        #expect(mac.batchSizes == [1, 16])
        #expect(mac.sequenceLengths == [128, 256, 512, 1024])
        #expect(mac.padding == [.value(50_283), .value(0), .firstValue])
        #expect(mac.outputName == "scores")
        #expect(mac.function(rows: 16, longestRow: 1000)?.name == "b16_s1024")
        #expect(mac.function(rows: 3, longestRow: 300)?.name == "b16_s512")
        #expect(mac.function(rows: 1, longestRow: 128)?.name == "b1_s128")
        #expect(mac.function(rows: 1, longestRow: 1025) == nil)
        for length in EncoderPackageSpec.layaSequenceLengths {
            let phone = EncoderPackageSpec.laya(sequenceLength: length)
            #expect(phone.name == "laya-f18-b1s\(length)-fp16")
            #expect(phone.layout == .singleShape)
            #expect(phone.batchSizes == [1])
            #expect(phone.sequenceLengths == [length])
            #expect(phone.padding == mac.padding)
            #expect(phone.outputName == "scores")
            #expect(phone.function(rows: 1, longestRow: length)?.name == "b1_s\(length)")
            #expect(phone.function(rows: 2, longestRow: 1) == nil)
            #expect(phone.function(rows: 1, longestRow: length + 1) == nil)
        }
        #expect(EncoderPackageSpec.verdict.layout == .functionPerShape)
        // A Laya row pads its ids with [PAD], its mask with 0 and its question type with itself.
        let input = EncoderPackageSpec.laya(sequenceLength: 128).input(
            [[[7, 8], [1, 1], [2, 2]]], for: .init(batchSize: 1, sequenceLength: 128))
        #expect(Array(input[0..<128]) == [7, 8] + [Int32](repeating: 50_283, count: 126))
        #expect(Array(input[128..<256]) == [1, 1] + [Int32](repeating: 0, count: 126))
        #expect(Array(input[256..<384]) == [Int32](repeating: 2, count: 128))
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

    @Test("A key that is absent loads after the least recently used are released")
    func loadingWithCapacity() {
        var cache = LeastRecentlyUsed<String, Int>()
        var loads: [String] = []
        func value(_ key: String) -> Int {
            cache.value(forKey: key, capacity: 2) {
                loads.append(key)
                return loads.count
            }
        }
        #expect(value("b16_s128") == 1)
        #expect(value("b1_s128") == 2)
        #expect(value("b16_s128") == 1)
        #expect(loads == ["b16_s128", "b1_s128"])
        #expect(value("b1_s512") == 3)
        #expect(cache.keys == ["b16_s128", "b1_s512"])
        // The room is made before the load, so a load that fails adds nothing and what was
        // released stays released: the cache never holds more than its capacity.
        struct Failure: Error {}
        #expect(throws: Failure.self) {
            try cache.value(forKey: "b1_s1024", capacity: 2) { throw Failure() }
        }
        #expect(cache.keys == ["b1_s512"])
    }

    @Test("Every function of a package, by batch size and then by length")
    func functions() {
        #expect(
            EncoderPackageSpec.verdict.functions.map(\.name) == [
                "b1_s128", "b1_s256", "b1_s512", "b16_s128", "b16_s256", "b16_s512",
            ])
        #expect(
            EncoderPackageSpec.layaMultifunction.functions.map(\.name) == [
                "b1_s128", "b1_s256", "b1_s512", "b1_s1024",
                "b16_s128", "b16_s256", "b16_s512", "b16_s1024",
            ])
        #expect(EncoderPackageSpec.laya(sequenceLength: 512).functions.map(\.name) == ["b1_s512"])
    }

    /// Requests whose shapes alternate, as the JevBench runs' did (D-042): the warm-up's three
    /// questions, then one-question requests of every length and a few longer batches.
    @Test("Keeping every function, a server whose requests change shape loads each one once")
    func everyFunctionLoadsEachOnce() {
        let calls = [
            "b16_s128", "b1_s128", "b1_s128", "b1_s1024", "b1_s512", "b1_s1024", "b1_s256",
            "b1_s512", "b16_s512", "b1_s256", "b1_s1024", "b16_s1024", "b1_s128", "b16_s128",
            "b1_s512", "b16_s256", "b1_s1024", "b1_s256", "b16_s512",
        ]
        func loads(capacity: Int) -> Int {
            var cache = LeastRecentlyUsed<String, Int>()
            var count = 0
            for call in calls {
                _ = cache.value(forKey: call, capacity: capacity) {
                    count += 1
                    return count
                }
            }
            return count
        }
        let shapes = Set(calls).count
        #expect(shapes == EncoderPackageSpec.layaMultifunction.functions.count)
        #if os(macOS)
            #expect(loads(capacity: LayaBackend.Configuration.defaultFunctionCapacity) == shapes)
        #endif
        #expect(loads(capacity: shapes) == shapes)
        // One fewer loads a function again; two, the earlier default, loads 9 again in 19 calls.
        #expect(loads(capacity: shapes - 1) == shapes + 1)
        #expect(loads(capacity: 2) == 17)
    }

    @Test("The default compute units follow D-011 and never include .all")
    func computeUnits() {
        #if os(macOS)
            #expect(EncoderComputeUnits.platformDefault == .cpuAndGPU)
            #expect(VerdictBackend.Configuration.defaultMaxBatchRows == 16)
            // Every function of the package stays loaded once a call has needed it (D-042).
            #expect(VerdictBackend.Configuration.defaultFunctionCapacity == 6)
            #expect(LayaBackend.Configuration.defaultMaxBatchRows == 16)
            #expect(LayaBackend.Configuration.defaultFunctionCapacity == 8)
            #expect(LayaPackageSet.platformDefault == .multifunction)
        #else
            #expect(EncoderComputeUnits.platformDefault == .cpuAndNeuralEngine)
            #expect(VerdictBackend.Configuration.defaultMaxBatchRows == 1)
            #expect(VerdictBackend.Configuration.defaultFunctionCapacity == 1)
            #expect(LayaBackend.Configuration.defaultMaxBatchRows == 1)
            #expect(LayaBackend.Configuration.defaultFunctionCapacity == 1)
            #expect(LayaPackageSet.platformDefault == .byLength)
        #endif
        #expect(EncoderComputeUnits.allCases.map(\.rawValue).contains("all") == false)
        // Laya's defaults follow its packages, not the platform: Core ML does not load the
        // multifunction package for the Neural Engine, and a per-length package holds one row.
        let folder = URL(fileURLWithPath: "/nonexistent", isDirectory: true)
        let mac = LayaBackend.Configuration(
            packages: .multifunction(folder), tokenizerDirectory: folder,
            configurationFile: folder)
        #expect(mac.computeUnits == .cpuAndGPU)
        #expect(mac.maxBatchRows == LayaBackend.Configuration.defaultMaxBatchRows)
        let phone = LayaBackend.Configuration(
            packages: .byLength([:]), tokenizerDirectory: folder, configurationFile: folder)
        #expect(phone.computeUnits == .cpuAndNeuralEngine)
        #expect(phone.maxBatchRows == 1)
        let chosen = LayaBackend.Configuration(
            packages: .byLength([:]), tokenizerDirectory: folder, configurationFile: folder,
            computeUnits: .cpuAndGPU, maxBatchRows: 1)
        #expect(chosen.computeUnits == .cpuAndGPU)
    }

    @Test("The module reports the package version")
    func version() {
        #expect(openJevEncodersVersion == "0.1.0-dev")
    }
}
