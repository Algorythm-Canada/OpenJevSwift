import CoreML
import Foundation

/// How a converted package holds its input shapes.
public enum PackageKind: String, Codable, Sendable {
    /// One program with an input of enumerated shapes (iOS 17 and later).
    case enumerated
    /// One function per input shape, named `b<batch>_s<length>`, sharing the weights (iOS 18 and
    /// later).
    case multifunction
}

/// A converted Verdict or Laya package: its shapes, its input planes and its output.
public struct PackageSpec: Codable, Sendable {
    public let name: String
    public let model: String
    public let kind: PackageKind
    public let batches: [Int]
    public let lengths: [Int]
    public let planes: Int
    public let output: String

    public static let verdictLengths = [128, 256, 512]
    public static let layaLengths = [128, 256, 512, 1024]

    public static func verdict(_ name: String, kind: PackageKind) -> PackageSpec {
        PackageSpec(
            name: name, model: "verdict", kind: kind, batches: [1, 16], lengths: verdictLengths,
            planes: 2,
            output: "logits")
    }

    public static func laya(
        _ name: String, kind: PackageKind, batches: [Int] = [1, 16], lengths: [Int] = layaLengths
    ) -> PackageSpec {
        PackageSpec(
            name: name, model: "laya", kind: kind, batches: batches, lengths: lengths, planes: 3,
            output: "scores")
    }

    /// Every package the converters write.
    public static let all: [PackageSpec] = [
        .verdict("verdict-e17-fp16", kind: .enumerated),
        .verdict("verdict-e17-fp32", kind: .enumerated),
        .verdict("verdict-m18-fp16", kind: .multifunction),
        .laya("laya-e17-fp16", kind: .enumerated),
        .laya("laya-m18-fp16", kind: .multifunction),
        // int8 weights, float16 computation (convert_*.py --only ...-w8).
        .verdict("verdict-m18-w8", kind: .multifunction),
        .laya("laya-m18-w8", kind: .multifunction),
        // One program for one fixed shape (convert_laya.py --only laya-f18-b1s128-fp16, ...); the
        // harness measures the questions that fit.
        .laya("laya-f18-b1s128-fp16", kind: .enumerated, batches: [1], lengths: [128]),
        .laya("laya-f18-b1s1024-fp16", kind: .enumerated, batches: [1], lengths: [1024]),
    ]

    public static func named(_ name: String) -> PackageSpec? {
        all.first { $0.name == name }
    }

    public func functionName(batch: Int, length: Int) -> String {
        "b\(batch)_s\(length)"
    }

    /// The smallest shape that holds `rows` rows of `length` tokens.
    public func shape(rows: Int, length: Int) -> (batch: Int, length: Int)? {
        guard let b = batches.first(where: { $0 >= rows }),
            let s = lengths.first(where: { $0 >= length })
        else { return nil }
        return (b, s)
    }
}

public enum ComputeUnitsName: String, Codable, Sendable, CaseIterable {
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    public var units: MLComputeUnits {
        switch self {
        case .cpuOnly: return .cpuOnly
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        case .all: return .all
        }
    }
}

public enum HarnessError: Error, CustomStringConvertible {
    case missing(String)
    case shape(String)
    case output(String)
    case mismatch(String)

    public var description: String {
        switch self {
        case .missing(let what): return "missing: \(what)"
        case .shape(let what): return "no shape: \(what)"
        case .output(let what): return "output: \(what)"
        case .mismatch(let what): return "mismatch: \(what)"
        }
    }
}

/// A package ready to run: one MLModel for an enumerated package; for a multifunction package, the
/// functions it has needed so far, at most `capacity` of them at once.
///
/// Each loaded function keeps its own copy of the weights once it has run (on the Mac GPU, six
/// Verdict functions took 1.5 GB for a 290 MB package), so a multifunction package is loaded one
/// function at a time and the least recently loaded one is released first.
public final class EncoderModel {
    public let spec: PackageSpec
    let compiled: URL
    let units: MLComputeUnits
    let padId: Int32
    let capacity: Int
    private var loaded: [(name: String, model: MLModel)] = []
    /// How long each function took to load, the time its first use costs.
    public private(set) var functionLoadSeconds: [String: Double] = [:]

    public init(
        spec: PackageSpec, compiled: URL, units: MLComputeUnits, padId: Int, capacity: Int = 1
    ) throws {
        self.spec = spec
        self.compiled = compiled
        self.units = units
        self.padId = Int32(padId)
        self.capacity = max(1, capacity)
        if spec.kind == .enumerated {
            let started = now()
            loaded = [("main", try Self.load(compiled: compiled, units: units, function: nil))]
            functionLoadSeconds["main"] = now() - started
        }
    }

    /// One MLModel for a package, or for one function of a multifunction package.
    public static func load(compiled: URL, units: MLComputeUnits, function: String?) throws
        -> MLModel
    {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        if let function {
            guard #available(macOS 15.0, iOS 18.0, *) else {
                throw HarnessError.missing("multifunction models need iOS 18 or macOS 15")
            }
            configuration.functionName = function
        }
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    func model(for name: String) throws -> MLModel {
        if spec.kind == .enumerated { return loaded[0].model }
        if let found = loaded.first(where: { $0.name == name }) { return found.model }
        while loaded.count >= capacity { loaded.removeFirst() }
        let started = now()
        let model = try Self.load(compiled: compiled, units: units, function: name)
        functionLoadSeconds[name] = now() - started
        loaded.append((name, model))
        return model
    }

    /// Runs rows of planes (each plane one Int32 per token, all planes of a row the same length),
    /// padded to the smallest shape that holds them, and returns the output rows for the real rows.
    public func run(_ rows: [[[Int32]]]) throws -> (outputs: [[Float]], batch: Int, length: Int) {
        let longest = rows.map { $0[0].count }.max() ?? 0
        guard let shape = spec.shape(rows: rows.count, length: longest) else {
            throw HarnessError.shape("\(rows.count) rows of \(longest) tokens for \(spec.name)")
        }
        let model = try model(for: spec.functionName(batch: shape.batch, length: shape.length))
        let input = try MLMultiArray(
            shape: [shape.batch, spec.planes, shape.length].map { NSNumber(value: $0) },
            dataType: .int32)
        let strides = input.strides.map(\.intValue)
        let pointer = input.dataPointer.bindMemory(to: Int32.self, capacity: input.count)
        for b in 0..<shape.batch {
            for p in 0..<spec.planes {
                let base = b * strides[0] + p * strides[1]
                for s in 0..<shape.length {
                    pointer[base + s * strides[2]] = p == 0 ? padId : 0
                }
            }
        }
        for (b, planes) in rows.enumerated() {
            for (p, plane) in planes.enumerated() {
                let base = b * strides[0] + p * strides[1]
                for (s, value) in plane.enumerated() { pointer[base + s * strides[2]] = value }
            }
            if spec.planes > 2, let qtype = planes[2].first {
                // Laya reads the question type at position 0; fill the row so it does not matter.
                let base = b * strides[0] + 2 * strides[1]
                for s in 0..<shape.length { pointer[base + s * strides[2]] = qtype }
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: input)
        ])
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: spec.output)?.multiArrayValue else {
            throw HarnessError.output("\(spec.output) missing from \(spec.name)")
        }
        let outStrides = output.strides.map(\.intValue)
        let width = output.shape[1].intValue
        var result: [[Float]] = []
        result.reserveCapacity(rows.count)
        switch output.dataType {
        case .float32:
            let values = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for b in 0..<rows.count {
                result.append((0..<width).map { values[b * outStrides[0] + $0 * outStrides[1]] })
            }
        case .float16:
            let values = output.dataPointer.bindMemory(to: Float16.self, capacity: output.count)
            for b in 0..<rows.count {
                result.append(
                    (0..<width).map { Float(values[b * outStrides[0] + $0 * outStrides[1]]) })
            }
        default:
            throw HarnessError.output("unexpected output type \(output.dataType.rawValue)")
        }
        return (result, shape.batch, shape.length)
    }
}

/// Which compute device Core ML plans each operation on, from MLComputePlan (iOS 17.4, macOS 14.4).
public struct ComputePlanSummary: Codable, Sendable {
    public let function: String
    public let operations: [String: Int]
    /// The share of the plan's estimated cost on each device, where Core ML gives an estimate.
    public let costShare: [String: Double]
}

public func computePlanSummary(compiled: URL, spec: PackageSpec, units: MLComputeUnits) async throws
    -> ComputePlanSummary?
{
    guard #available(macOS 14.4, iOS 17.4, *) else { return nil }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = units
    var function = "main"
    if spec.kind == .multifunction {
        guard #available(macOS 15.0, iOS 18.0, *) else { return nil }
        function = spec.functionName(batch: 1, length: spec.lengths[0])
        configuration.functionName = function
    }
    let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
    guard case .program(let program) = plan.modelStructure else { return nil }
    guard let body = program.functions[function] ?? program.functions.values.first else {
        return nil
    }
    var operations: [String: Int] = [:]
    var cost: [String: Double] = [:]
    func device(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neuralEngine"
        @unknown default: return "unknown"
        }
    }
    func walk(_ block: MLModelStructure.Program.Block) {
        for operation in block.operations {
            if let usage = plan.deviceUsage(for: operation) {
                let name = device(usage.preferred)
                operations[name, default: 0] += 1
                if let weight = plan.estimatedCost(of: operation)?.weight {
                    cost[name, default: 0] += weight
                }
            } else {
                operations["none", default: 0] += 1
            }
            for inner in operation.blocks { walk(inner) }
        }
    }
    walk(body.block)
    let total = cost.values.reduce(0, +)
    return ComputePlanSummary(
        function: function, operations: operations,
        costShare: total > 0 ? cost.mapValues { $0 / total } : [:])
}
