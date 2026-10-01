/// Runs rows of token planes through an encoder in one call and returns one output row per
/// input row.
///
/// ``CoreMLEncoderModel`` runs a converted package. Tests hand ``VerdictBackend`` a runner that
/// returns recorded logits, so the backend's batching, ordering and billing are checked without
/// Core ML.
public protocol EncoderModelRunner: Sendable {
    /// Runs the rows in one call.
    ///
    /// - Parameter rows: Each row is a list of planes (Verdict's token ids and attention mask),
    ///   all of the row's length.
    /// - Returns: The output rows of the rows given, in order.
    func run(_ rows: [[[Int32]]]) async throws -> [[Float]]
}

/// Where Core ML may run an encoder.
///
/// `MLComputeUnits.all` is left out on purpose: with it, Core ML fails to load Verdict's batch-16
/// functions on macOS 27.0.1 and loads flakily on iOS 27.0 (spike #56, D-011).
public enum EncoderComputeUnits: String, Sendable, Hashable, CaseIterable {
    /// The CPU alone.
    case cpuOnly
    /// The CPU and the GPU: the Mac's setting.
    case cpuAndGPU
    /// The CPU and the Neural Engine: the iPhone's setting.
    case cpuAndNeuralEngine

    /// The setting D-011 chose for this platform: ``cpuAndGPU`` on macOS,
    /// ``cpuAndNeuralEngine`` on iOS.
    public static var platformDefault: EncoderComputeUnits {
        #if os(macOS)
            return .cpuAndGPU
        #else
            return .cpuAndNeuralEngine
        #endif
    }
}

/// A call an encoder cannot run, or an output it did not expect.
public enum EncoderModelError: Error, Sendable, Hashable, CustomStringConvertible {
    /// No function of the package holds the rows.
    case noFunction(package: String, rows: Int, longestRow: Int)
    /// The rows do not have the planes the package takes.
    case malformedRows(String)
    /// A function's input or output is not what the spec says: the wrong package.
    case unexpectedModel(String)
    /// The prediction's output is missing or has the wrong shape.
    case unexpectedOutput(String)

    /// What went wrong, naming the package or function.
    public var description: String {
        switch self {
        case .noFunction(let package, let rows, let longestRow):
            return "\(package) has no function that takes \(rows) rows of \(longestRow) tokens"
        case .malformedRows(let message), .unexpectedModel(let message),
            .unexpectedOutput(let message):
            return message
        }
    }
}
