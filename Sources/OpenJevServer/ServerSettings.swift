// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/config.py` (`Settings`,
// `_env_num`, `parse_routes`) and the host, port and log level `openjev/__main__.py` reads.
// Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore

/// Upstream's `Settings`, for everything that is not vLLM-specific: the same names, defaults and
/// startup validation, so deployment documentation and compose files transfer (decision D-013).
/// Two settings are this port's own: `OPENJEV_ENCODER_FUNCTIONS` (D-042), read and checked as
/// upstream's `_env_num` settings are, and `OPENJEV_JEVK5_MODEL` (D-052), the JevK5 checkpoint,
/// which upstream reads from its vLLM server's `OPENJEV_MODEL`.
///
/// The fields are constants, as upstream's frozen dataclass is: a value exists only once the
/// checks have passed, so no later change can reach an engine's preconditions.
///
/// The library never reads the process environment. The CLI passes its environment to
/// ``init(environment:)``; everything else uses the memberwise initializer, whose arguments all
/// default to upstream's values. Both refuse the values upstream refuses, with upstream's
/// messages, so a bad environment fails at startup rather than as a 500 or a hang on the first
/// request.
public struct ServerSettings: Sendable, Hashable {
    /// A log level, uvicorn's names plus swift-log's `notice`.
    public enum LogLevel: String, Sendable, Hashable, CaseIterable {
        /// What `info` writes, and the most detail the libraries log.
        case trace
        /// What `info` writes, and the libraries' debugging detail.
        case debug
        /// The default: the settings, the phases and one line per request, with the refusals
        /// and failures.
        case info
        /// swift-log's level between `info` and `warning`, which uvicorn does not have. The
        /// server writes nothing at it, so it keeps what `warning` keeps.
        case notice
        /// The refusals and failures, without the phase and request lines.
        case warning
        /// The failures alone, such as the 503 of a backend or of a routed server.
        case error
        /// Nothing the server writes; only the libraries' critical messages.
        case critical
    }

    /// The address to bind, `OPENJEV_HOST`.
    public let host: String
    /// The port to bind, `OPENJEV_PORT`.
    public let port: Int
    /// The backend to load, `OPENJEV_BACKEND`: `mlx` or an encoder model's backend name.
    public let backend: String
    /// The MLX weights and tokenizer, `OPENJEV_MLX_MODEL`.
    public let mlxModel: String
    /// The longest prompt in tokens on MLX, `OPENJEV_MLX_MAX_PROMPT`.
    public let mlxMaxPrompt: Int
    /// The MLX buffer pool ceiling in GB, `OPENJEV_MLX_CACHE_LIMIT_GB`. `nil` leaves MLX alone;
    /// 0 disables the cache, which is a real choice and not the same as leaving it unset.
    public let mlxCacheLimitGB: Double?
    /// The prefill cache size in entries, `OPENJEV_MLX_PROMPT_CACHE`.
    public let mlxPromptCache: Int
    /// The canvas length in tokens, `OPENJEV_CANVAS`.
    public let canvas: Int
    /// The canvas width rounding, `OPENJEV_CANVAS_STEP`.
    public let canvasStep: Int
    /// Reads in flight at once, `OPENJEV_MAX_INFLIGHT`.
    public let maxInflight: Int
    /// Decisions waiting before a 529, `OPENJEV_MAX_QUEUE`.
    public let maxQueue: Int
    /// Questions per request before a 400, `OPENJEV_MAX_QUESTIONS`.
    public let maxQuestions: Int
    /// The request body cap before a 413, `OPENJEV_MAX_BODY_BYTES`.
    public let maxBodyBytes: Int
    /// Seconds before a forwarded request is a 503, `OPENJEV_FORWARD_TIMEOUT`.
    public let forwardTimeout: Double
    /// The Bearer token clients must send, `OPENJEV_API_KEY`; empty means no key is required.
    public let apiKey: String
    /// The `X-Origin-Secret` a front proxy sends, `OPENJEV_ORIGIN_SECRET`; empty means none.
    public let originSecret: String
    /// The entropy above which a group is re-read, `OPENJEV_AUTO_THRESHOLD`.
    public let autoThreshold: Double
    /// The most reads of one group, `OPENJEV_AUTO_MAX`.
    public let autoMax: Int
    /// Images per request, `OPENJEV_MAX_IMAGES`.
    public let maxImages: Int
    /// Bytes per decoded image, `OPENJEV_MAX_IMAGE_BYTES`.
    public let maxImageBytes: Int
    /// Generations in flight at once, `OPENJEV_GEN_MAX_INFLIGHT`.
    public let genMaxInflight: Int
    /// Generations waiting before a 529, `OPENJEV_GEN_MAX_QUEUE`.
    public let genMaxQueue: Int
    /// The longest generation in tokens, `OPENJEV_GEN_MAX_TOKENS`.
    public let genMaxTokens: Int
    /// Whether to warm the model up before serving, `OPENJEV_WARMUP`; anything but `0` is true.
    public let warmup: Bool
    /// The Laya checkpoint, `OPENJEV_LAYA_MODEL`.
    public let layaModel: String
    /// The Verdict checkpoint, `OPENJEV_VERDICT_MODEL`.
    public let verdictModel: String
    /// The JevK5 checkpoint, `OPENJEV_JEVK5_MODEL`, this port's (D-052): a folder holding a
    /// conversion (when it starts with `/`, `~` or `.`) or a Hub repository, optionally followed
    /// by `@revision`. The default is the 8-bit conversion's repository.
    public let jevk5Model: String
    /// The device an encoder runs on, `OPENJEV_DEVICE`; empty picks the default.
    public let device: String
    /// Questions per encoder batch, `OPENJEV_ENCODER_BATCH`.
    public let encoderBatch: Int
    /// The most Core ML functions an encoder keeps loaded, `OPENJEV_ENCODER_FUNCTIONS`, this
    /// port's (D-042). `nil` keeps every function of the package once a read has needed it.
    public let encoderFunctions: Int?
    /// Other System One models served by other OpenJev servers, `OPENJEV_MODEL_ROUTES`
    /// (`name=url,name=url`), in the order given. A request for one of them that this server does
    /// not serve is passed through unchanged, and `GET /v1/models` lists them.
    public let modelRoutes: OrderedMap<String>
    /// The log level, `OPENJEV_LOG_LEVEL`.
    public let logLevel: LogLevel

    /// Creates settings; every argument defaults to upstream's value.
    ///
    /// - Throws: ``ServerSettingsError`` with upstream's `__post_init__` message for a value it
    ///   refuses: `canvas`, `canvasStep`, `maxInflight`, `maxQuestions`, `maxBodyBytes`,
    ///   `maxImageBytes`, `genMaxInflight`, `genMaxTokens`, `mlxMaxPrompt`, `encoderBatch` and
    ///   `forwardTimeout` must be at least 1; `maxQueue`, `genMaxQueue` and `maxImages` must not
    ///   be negative; `mlxCacheLimitGB` and `mlxPromptCache` must not be negative either, which
    ///   upstream checks while reading the environment, and `encoderFunctions` must be at least
    ///   1, with the same message.
    public init(
        host: String = "127.0.0.1",
        port: Int = 8080,
        backend: String = "mlx",
        mlxModel: String = "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        mlxMaxPrompt: Int = 32768,
        mlxCacheLimitGB: Double? = nil,
        mlxPromptCache: Int = 12,
        canvas: Int = 64,
        canvasStep: Int = 16,
        maxInflight: Int = 64,
        maxQueue: Int = 512,
        maxQuestions: Int = 256,
        maxBodyBytes: Int = 64 * 1024 * 1024,
        forwardTimeout: Double = 300,
        apiKey: String = "",
        originSecret: String = "",
        autoThreshold: Double = 0.1,
        autoMax: Int = 4,
        maxImages: Int = 8,
        maxImageBytes: Int = 5 * 1024 * 1024,
        genMaxInflight: Int = 8,
        genMaxQueue: Int = 32,
        genMaxTokens: Int = 8192,
        warmup: Bool = true,
        layaModel: String = "convaiinnovations/laya-typed-decisions",
        verdictModel: String = "heman10x/rlcd-modernbert-151m",
        jevk5Model: String = "Algorythm-Canada/jevk5-0.2-mlx-8bit",
        device: String = "",
        encoderBatch: Int = 16,
        encoderFunctions: Int? = nil,
        modelRoutes: OrderedMap<String> = [:],
        logLevel: LogLevel = .info
    ) throws(ServerSettingsError) {
        self.host = host
        self.port = port
        self.backend = backend
        self.mlxModel = mlxModel
        self.mlxMaxPrompt = mlxMaxPrompt
        self.mlxCacheLimitGB = mlxCacheLimitGB
        self.mlxPromptCache = mlxPromptCache
        self.canvas = canvas
        self.canvasStep = canvasStep
        self.maxInflight = maxInflight
        self.maxQueue = maxQueue
        self.maxQuestions = maxQuestions
        self.maxBodyBytes = maxBodyBytes
        self.forwardTimeout = forwardTimeout
        self.apiKey = apiKey
        self.originSecret = originSecret
        self.autoThreshold = autoThreshold
        self.autoMax = autoMax
        self.maxImages = maxImages
        self.maxImageBytes = maxImageBytes
        self.genMaxInflight = genMaxInflight
        self.genMaxQueue = genMaxQueue
        self.genMaxTokens = genMaxTokens
        self.warmup = warmup
        self.layaModel = layaModel
        self.verdictModel = verdictModel
        self.jevk5Model = jevk5Model
        self.device = device
        self.encoderBatch = encoderBatch
        self.encoderFunctions = encoderFunctions
        self.modelRoutes = modelRoutes
        self.logLevel = logLevel
        try validate()
    }

    /// Creates settings from `OPENJEV_*` variables, as upstream reads them. A missing variable
    /// means the default. A variable set to the empty string is kept for a string setting, means
    /// the default for the two MLX cache settings (upstream's `_env_num`) and
    /// `OPENJEV_ENCODER_FUNCTIONS`, and is refused for every other number, as Python's `int("")`
    /// refuses it. Numbers parse as Python's `int` and `float` parse them, surrounding whitespace,
    /// a `+` and `_` between digits included. Unlike upstream, a value that does not parse names
    /// its variable in the error.
    ///
    /// Only the CLI calls this, with the process environment; the library never reads it.
    ///
    /// - Throws: ``ServerSettingsError`` with upstream's texts: `{NAME}={raw!r} is not a int` (or
    ///   `float`) for a value that does not parse, `{NAME}={value} is below the minimum of
    ///   {minimum}` for the two MLX cache settings and `OPENJEV_ENCODER_FUNCTIONS`,
    ///   `OPENJEV_MODEL_ROUTES: {part!r} is not name=url` for a route without both parts, and the
    ///   memberwise initializer's messages for the values it refuses. An unknown
    ///   `OPENJEV_LOG_LEVEL` is refused with a message that lists the levels.
    public init(environment: [String: String]) throws(ServerSettingsError) {
        let env = EnvironmentReader(environment)
        try self.init(
            host: env.string("OPENJEV_HOST", default: "127.0.0.1"),
            port: env.integer("OPENJEV_PORT", default: 8080),
            backend: env.string("OPENJEV_BACKEND", default: "mlx"),
            mlxModel: env.string(
                "OPENJEV_MLX_MODEL", default: "mlx-community/diffusiongemma-26B-A4B-it-4bit"),
            mlxMaxPrompt: env.integer("OPENJEV_MLX_MAX_PROMPT", default: 32768),
            mlxCacheLimitGB: env.optionalDouble("OPENJEV_MLX_CACHE_LIMIT_GB", minimum: 0),
            mlxPromptCache: env.optionalInteger("OPENJEV_MLX_PROMPT_CACHE", minimum: 0) ?? 12,
            canvas: env.integer("OPENJEV_CANVAS", default: 64),
            canvasStep: env.integer("OPENJEV_CANVAS_STEP", default: 16),
            maxInflight: env.integer("OPENJEV_MAX_INFLIGHT", default: 64),
            maxQueue: env.integer("OPENJEV_MAX_QUEUE", default: 512),
            maxQuestions: env.integer("OPENJEV_MAX_QUESTIONS", default: 256),
            maxBodyBytes: env.integer("OPENJEV_MAX_BODY_BYTES", default: 64 * 1024 * 1024),
            forwardTimeout: env.double("OPENJEV_FORWARD_TIMEOUT", default: 300),
            apiKey: env.string("OPENJEV_API_KEY", default: ""),
            originSecret: env.string("OPENJEV_ORIGIN_SECRET", default: ""),
            autoThreshold: env.double("OPENJEV_AUTO_THRESHOLD", default: 0.1),
            autoMax: env.integer("OPENJEV_AUTO_MAX", default: 4),
            maxImages: env.integer("OPENJEV_MAX_IMAGES", default: 8),
            maxImageBytes: env.integer("OPENJEV_MAX_IMAGE_BYTES", default: 5 * 1024 * 1024),
            genMaxInflight: env.integer("OPENJEV_GEN_MAX_INFLIGHT", default: 8),
            genMaxQueue: env.integer("OPENJEV_GEN_MAX_QUEUE", default: 32),
            genMaxTokens: env.integer("OPENJEV_GEN_MAX_TOKENS", default: 8192),
            warmup: env.string("OPENJEV_WARMUP", default: "1") != "0",
            layaModel: env.string(
                "OPENJEV_LAYA_MODEL", default: "convaiinnovations/laya-typed-decisions"),
            verdictModel: env.string(
                "OPENJEV_VERDICT_MODEL", default: "heman10x/rlcd-modernbert-151m"),
            jevk5Model: env.string(
                "OPENJEV_JEVK5_MODEL", default: "Algorythm-Canada/jevk5-0.2-mlx-8bit"),
            device: env.string("OPENJEV_DEVICE", default: ""),
            encoderBatch: env.integer("OPENJEV_ENCODER_BATCH", default: 16),
            encoderFunctions: env.optionalInteger("OPENJEV_ENCODER_FUNCTIONS", minimum: 1),
            modelRoutes: Self.parseRoutes(env.string("OPENJEV_MODEL_ROUTES", default: "")),
            logLevel: env.logLevel("OPENJEV_LOG_LEVEL", default: .info)
        )
    }

    /// Upstream's `parse_routes`: `name=url,name=url` into the routes in the order given. Parts
    /// are trimmed, empty parts are skipped and every trailing `/` is dropped from a URL. A name
    /// given twice keeps its first place and takes its last URL, as a Python dict does.
    ///
    /// The text is read code point by code point, as Python reads it, and trimmed of the characters
    /// `str.strip()` removes (``/OpenJevCore/TextOf/isPythonWhitespace(_:)``), so a `,` or `=`
    /// followed by a combining mark still separates, and U+001C to U+001F are trimmed.
    ///
    /// - Throws: ``ServerSettingsError`` `OPENJEV_MODEL_ROUTES: {part!r} is not name=url` for a
    ///   part without a `=`, or with an empty name or URL.
    public static func parseRoutes(_ text: String) throws(ServerSettingsError) -> OrderedMap<String>
    {
        func stripped(_ scalars: Substring.UnicodeScalarView) -> Substring.UnicodeScalarView {
            guard let start = scalars.firstIndex(where: { !TextOf.isPythonWhitespace($0) }),
                let end = scalars.lastIndex(where: { !TextOf.isPythonWhitespace($0) })
            else {
                return Substring.UnicodeScalarView()
            }
            return scalars[start...end]
        }
        var routes = OrderedMap<String>()
        let pieces = Substring(text).unicodeScalars.split(
            separator: ",", omittingEmptySubsequences: false)
        for piece in pieces {
            let part = stripped(piece)
            if part.isEmpty {
                continue
            }
            let partText = String(String.UnicodeScalarView(part))
            guard let separator = part.firstIndex(of: "=") else {
                throw ServerSettingsError.notARoute(partText)
            }
            let name = stripped(part[..<separator])
            var url = stripped(part[part.index(after: separator)...])
            if name.isEmpty || url.isEmpty {
                throw ServerSettingsError.notARoute(partText)
            }
            while url.last == "/" {
                url.removeLast()
            }
            routes.updateValue(
                String(String.UnicodeScalarView(url)),
                forKey: String(String.UnicodeScalarView(name)))
        }
        return routes
    }

    /// The model routes as `serve` logs them, in order: each URL without the user name and
    /// password it may hold before its host, so that no log holds them.
    public var modelRoutesWithoutCredentials: OrderedMap<String> {
        var routes = OrderedMap<String>()
        for (name, url) in modelRoutes {
            routes.updateValue(Self.withoutCredentials(url), forKey: name)
        }
        return routes
    }

    /// `url` without the `user:password@` that may come before its host: everything up to the
    /// last `@` of the authority, which ends at the first `/`, `?` or `#` and starts after
    /// `scheme://`, after the `//` that opens a scheme-relative URL, or at the start of a URL
    /// with neither. The URL is read code point by code point, so a combining mark after a
    /// delimiter does not hide it.
    static func withoutCredentials(_ url: String) -> String {
        let scalars = Array(url.unicodeScalars)
        let schemeCharacters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-."
            .unicodeScalars
        let schemeLength = scalars.prefix { schemeCharacters.contains($0) }.count
        let slashes: [Unicode.Scalar] = ["/", "/"]
        var start = 0
        if schemeLength > 0, scalars[0].properties.isAlphabetic,
            scalars[schemeLength...].starts(with: [":"] + slashes)
        {
            start = schemeLength + 3
        } else if scalars.starts(with: slashes) {
            start = 2
        }
        let end =
            scalars[start...].firstIndex { "/?#".unicodeScalars.contains($0) }
            ?? scalars.endIndex
        guard let at = scalars[start..<end].lastIndex(of: "@") else { return url }
        var kept = String.UnicodeScalarView()
        kept.append(contentsOf: scalars[..<start])
        kept.append(contentsOf: scalars[(at + 1)...])
        return String(kept)
    }

    /// Upstream's `__post_init__` checks, plus the two minimums `_env_num` applies while reading
    /// and this port's `OPENJEV_ENCODER_FUNCTIONS` minimum.
    private func validate() throws(ServerSettingsError) {
        let positive: [(name: String, value: SettingNumber)] = [
            ("canvas", .integer(canvas)),
            ("canvas_step", .integer(canvasStep)),
            ("max_inflight", .integer(maxInflight)),
            ("max_questions", .integer(maxQuestions)),
            ("max_body_bytes", .integer(maxBodyBytes)),
            ("max_image_bytes", .integer(maxImageBytes)),
            ("gen_max_inflight", .integer(genMaxInflight)),
            ("gen_max_tokens", .integer(genMaxTokens)),
            ("mlx_max_prompt", .integer(mlxMaxPrompt)),
            ("encoder_batch", .integer(encoderBatch)),
            ("forward_timeout", .float(forwardTimeout)),
        ]
        for (name, value) in positive where value.isBelow(1) {
            throw ServerSettingsError(
                "\(name) must be at least 1, got \(value.pythonRepr) (OPENJEV_\(name.uppercased()))"
            )
        }
        let nonNegative: [(name: String, value: SettingNumber)] = [
            ("max_queue", .integer(maxQueue)),
            ("gen_max_queue", .integer(genMaxQueue)),
            ("max_images", .integer(maxImages)),
        ]
        for (name, value) in nonNegative where value.isBelow(0) {
            throw ServerSettingsError(
                "\(name) must not be negative, got \(value.pythonRepr) (OPENJEV_\(name.uppercased()))"
            )
        }
        if let limit = mlxCacheLimitGB, SettingNumber.float(limit).isBelow(0) {
            throw ServerSettingsError.belowMinimum(
                "OPENJEV_MLX_CACHE_LIMIT_GB", value: .float(limit), minimum: 0)
        }
        if SettingNumber.integer(mlxPromptCache).isBelow(0) {
            throw ServerSettingsError.belowMinimum(
                "OPENJEV_MLX_PROMPT_CACHE", value: .integer(mlxPromptCache), minimum: 0)
        }
        if let functions = encoderFunctions, SettingNumber.integer(functions).isBelow(1) {
            throw ServerSettingsError.belowMinimum(
                "OPENJEV_ENCODER_FUNCTIONS", value: .integer(functions), minimum: 1)
        }
    }
}

/// A setting the environment refused, with upstream's message.
public struct ServerSettingsError: Error, Sendable, Hashable, CustomStringConvertible {
    /// Upstream's message, which names the variable.
    public var message: String

    /// Creates the error with a message.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }

    /// `{NAME}={raw!r} is not a {type}`, upstream's `_env_num` message for a value that does not
    /// parse; `type` is Python's type name, `int` or `float`.
    static func notA(_ type: String, _ name: String, raw: String) -> ServerSettingsError {
        ServerSettingsError("\(name)=\(raw.pythonRepr) is not a \(type)")
    }

    /// `{NAME}={value} is below the minimum of {minimum}`, upstream's `_env_num` message.
    static func belowMinimum(
        _ name: String, value: SettingNumber, minimum: Int
    ) -> ServerSettingsError {
        ServerSettingsError("\(name)=\(value.pythonRepr) is below the minimum of \(minimum)")
    }

    /// `OPENJEV_MODEL_ROUTES: {part!r} is not name=url`, upstream's `parse_routes` message.
    static func notARoute(_ part: String) -> ServerSettingsError {
        ServerSettingsError("OPENJEV_MODEL_ROUTES: \(part.pythonRepr) is not name=url")
    }

    /// `unknown backend {name!r}; use one of {known} (OPENJEV_BACKEND)`: upstream's `create_app`
    /// message for a backend it does not have, listing the backends this port knows, with the
    /// variable named as `__post_init__` names it.
    public static func unknownBackend(_ name: String, known: [String]) -> ServerSettingsError {
        ServerSettingsError(
            "unknown backend \(name.pythonRepr); use one of \(known.joined(separator: ", ")) "
                + "(OPENJEV_BACKEND)")
    }
}

/// A numeric setting, kept as Python would type it so messages format it as Python does: an
/// `int` prints its digits, a `float` prints its `repr`, `0.0` for zero.
enum SettingNumber: Sendable {
    case integer(Int)
    case float(Double)

    /// Whether the value is below `bound`, as Python's `<` compares it. NaN is below nothing.
    func isBelow(_ bound: Int) -> Bool {
        switch self {
        case .integer(let value):
            return value < bound
        case .float(let value):
            return value < Double(bound)
        }
    }

    /// The value as Python's `repr` or `str` writes it.
    var pythonRepr: String {
        switch self {
        case .integer(let value):
            return String(value)
        case .float(let value):
            if value.isNaN {
                return "nan"
            }
            if value.isInfinite {
                return value < 0 ? "-inf" : "inf"
            }
            // The Python JSON writer renders a finite float as `repr(float)` does.
            return (try? PythonJSONWriter().string(.float(value))) ?? String(value)
        }
    }
}

/// Reads `OPENJEV_*` variables as upstream's `_env` and `_env_num` do.
struct EnvironmentReader {
    let environment: [String: String]

    init(_ environment: [String: String]) {
        self.environment = environment
    }

    /// A string setting, `_env`: the default only when the variable is missing.
    func string(_ name: String, default defaultValue: String) -> String {
        environment[name] ?? defaultValue
    }

    /// An integer setting, `int(_env(name, default))`: an empty value does not parse.
    func integer(_ name: String, default defaultValue: Int) throws(ServerSettingsError) -> Int {
        guard let text = environment[name] else { return defaultValue }
        return try parseInteger(name, text)
    }

    /// A float setting, `float(_env(name, default))`: an empty value does not parse.
    func double(_ name: String, default defaultValue: Double) throws(ServerSettingsError) -> Double
    {
        guard let text = environment[name] else { return defaultValue }
        return try parseDouble(name, text)
    }

    /// An integer setting read by `_env_num`: `nil` when missing or empty, and refused below
    /// `minimum`.
    func optionalInteger(_ name: String, minimum: Int) throws(ServerSettingsError) -> Int? {
        guard let text = environment[name], !text.isEmpty else { return nil }
        let value = try parseInteger(name, text)
        if value < minimum {
            throw ServerSettingsError.belowMinimum(name, value: .integer(value), minimum: minimum)
        }
        return value
    }

    /// A float setting read by `_env_num`: `nil` when missing or empty, and refused below
    /// `minimum`. NaN passes, as it does upstream, because NaN is below nothing.
    func optionalDouble(_ name: String, minimum: Int) throws(ServerSettingsError) -> Double? {
        guard let text = environment[name], !text.isEmpty else { return nil }
        let value = try parseDouble(name, text)
        if value < Double(minimum) {
            throw ServerSettingsError.belowMinimum(name, value: .float(value), minimum: minimum)
        }
        return value
    }

    /// A log level by name, uvicorn's `log_level`: case-sensitive, and the default only when the
    /// variable is missing.
    func logLevel(
        _ name: String, default defaultValue: ServerSettings.LogLevel
    ) throws(ServerSettingsError) -> ServerSettings.LogLevel {
        guard let text = environment[name] else { return defaultValue }
        guard let level = ServerSettings.LogLevel(rawValue: text) else {
            let names = ServerSettings.LogLevel.allCases.map(\.rawValue).joined(separator: ", ")
            throw ServerSettingsError(
                "\(name)=\(text.pythonRepr) is not a log level; use one of \(names)")
        }
        return level
    }

    private func parseInteger(_ name: String, _ text: String) throws(ServerSettingsError) -> Int {
        guard let value = PythonNumber.integer(text) else {
            throw ServerSettingsError.notA("int", name, raw: text)
        }
        return value
    }

    private func parseDouble(_ name: String, _ text: String) throws(ServerSettingsError) -> Double {
        guard let value = PythonNumber.float(text) else {
            throw ServerSettingsError.notA("float", name, raw: text)
        }
        return value
    }
}

/// Python's `int(text)` and `float(text)` for the ASCII forms a setting uses.
///
/// Both strip surrounding whitespace and accept a sign and single `_` between digits. `float`
/// also accepts a fraction, an exponent, and `inf`, `infinity` and `nan` in any case. `int`
/// refuses more than 4,300 digits, leading zeros included, CPython's default
/// `sys.int_max_str_digits`. Python accepts non-ASCII decimal digits too; here they are refused.
enum PythonNumber {
    private static let digits = #"[0-9](?:_?[0-9])*"#

    /// `int(text)`, or `nil` where Python raises or the value does not fit an `Int`.
    static func integer(_ text: String) -> Int? {
        guard isInteger(text) else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return Int(trimmed.replacingOccurrences(of: "_", with: ""))
    }

    /// Whether `int(text)` succeeds, however large the result. Past 4,300 digits CPython raises
    /// the `ValueError` of its limit, whose count leaves out the sign and the underscores.
    static func isInteger(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let decimal = UInt8(ascii: "0")...UInt8(ascii: "9")
        let digitCount = trimmed.utf8.lazy.filter { decimal.contains($0) }.count
        return matches(trimmed, #"[+-]?"# + digits)
            && digitCount <= PythonJSONLoads.maximumIntegerDigits
    }

    /// Whether the text, without its surrounding whitespace, starts with `-`.
    static func isNegative(_ text: String) -> Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("-")
    }

    /// `float(text)`, or `nil` where Python raises. Out of range is infinity, as in Python.
    static func float(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if matches(trimmed.lowercased(), #"[+-]?(?:inf|infinity|nan)"#) {
            let negative = trimmed.hasPrefix("-")
            if trimmed.lowercased().hasSuffix("nan") {
                return negative ? -Double.nan : Double.nan
            }
            return negative ? -Double.infinity : Double.infinity
        }
        let mantissa = "(?:\(digits)(?:\\.(?:\(digits))?)?|\\.\(digits))"
        guard matches(trimmed, "[+-]?" + mantissa + "(?:[eE][+-]?\(digits))?") else { return nil }
        return Double(trimmed.replacingOccurrences(of: "_", with: ""))
    }

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        text.range(of: "^(?:" + pattern + ")$", options: .regularExpression) != nil
    }
}
