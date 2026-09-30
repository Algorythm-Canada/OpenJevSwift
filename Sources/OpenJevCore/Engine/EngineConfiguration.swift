// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/config.py`, the settings
// `Engine` reads, and `openjev/engine.py`, `DEFAULT_OPTIONS` and the option handling of
// `Engine.decide`. Apache-2.0. See THIRD_PARTY.md.

/// The settings of a ``DecisionEngine``, with upstream's defaults and names.
public struct EngineConfiguration: Sendable, Hashable {
    /// The canvas length and step, `OPENJEV_CANVAS` (64) and `OPENJEV_CANVAS_STEP` (16).
    public var geometry: CanvasGeometry
    /// The top-k entropy above which a read is repeated, `OPENJEV_AUTO_THRESHOLD` (0.1).
    public var autoThreshold: Double
    /// The total reads of a group under the automatic policy, `OPENJEV_AUTO_MAX` (4). 1 turns
    /// the re-reads off.
    public var autoMax: Int
    /// The most backend calls in flight at once, `OPENJEV_MAX_INFLIGHT` (64).
    public var maxInflight: Int
    /// The most requests inside `decide` at once, `OPENJEV_MAX_QUEUE` (512); one more is refused
    /// with ``OverloadedError``.
    public var maxQueue: Int
    /// How many images a request may carry and how large each may be.
    public var imageLimits: ImageLimits
    /// The resolved templates kept before the cache is emptied, upstream's 4,096.
    public var templateCacheLimit: Int
    /// The `model` a response names. The server owns model naming; this default is for building
    /// a response outside it, as the tests do.
    public var servedModelVersion: String

    /// Creates a configuration; every argument defaults to upstream's value.
    public init(
        geometry: CanvasGeometry = .standard,
        autoThreshold: Double = 0.1,
        autoMax: Int = 4,
        maxInflight: Int = 64,
        maxQueue: Int = 512,
        imageLimits: ImageLimits = ImageLimits(),
        templateCacheLimit: Int = 4096,
        servedModelVersion: String = "openjev-0.1"
    ) {
        self.geometry = geometry
        self.autoThreshold = autoThreshold
        self.autoMax = autoMax
        self.maxInflight = maxInflight
        self.maxQueue = maxQueue
        self.imageLimits = imageLimits
        self.templateCacheLimit = templateCacheLimit
        self.servedModelVersion = servedModelVersion
    }

    /// Upstream's defaults.
    public static let `default` = EngineConfiguration()
}

extension CanvasGeometry {
    /// Upstream's default geometry, a canvas of 64 tokens in steps of 16.
    public static let standard: CanvasGeometry = {
        do {
            return try CanvasGeometry(canvas: 64, step: 16)
        } catch {
            preconditionFailure("the default canvas geometry is valid: \(error)")
        }
    }()
}

/// The read options of one request, with upstream's defaults filled in.
///
/// Upstream's `DEFAULT_OPTIONS` is `steps 1, samples None, think 0, sequential False`, and
/// `decide` overrides it with every option the request sets to a value other than `null`.
public struct ReadOptions: Sendable, Hashable {
    /// Denoise passes per read.
    public var steps: Int
    /// A fixed number of billed reads per group, or `nil` for the automatic policy.
    public var samples: Int?
    /// The thought token budget; 0 is no thought.
    public var think: Int
    /// Whether groups are read in series, each conditioned on the earlier answers.
    public var sequential: Bool

    /// Creates options.
    public init(steps: Int, samples: Int?, think: Int, sequential: Bool) {
        self.steps = steps
        self.samples = samples
        self.think = think
        self.sequential = sequential
    }

    /// Upstream's `DEFAULT_OPTIONS`.
    public static let `default` = ReadOptions(steps: 1, samples: nil, think: 0, sequential: false)

    /// The options of `request`; each field the request leaves unset takes the default.
    public init(_ request: SystemOneRequest) {
        self.init(
            steps: request.steps ?? Self.default.steps,
            samples: request.samples ?? Self.default.samples,
            think: request.think ?? Self.default.think,
            sequential: request.sequential ?? Self.default.sequential)
    }
}
