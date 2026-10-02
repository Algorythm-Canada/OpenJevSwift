import OpenJevCore
import OpenJevTestSupport
import Testing

@testable import OpenJevServer

/// ``ServerSettings`` against upstream's `Settings`, `_env_num` and `parse_routes` in
/// `openjev/config.py`, and `test_settings_are_checked_at_startup` in `tests/test_api.py`.
@Suite("Server settings")
struct ServerSettingsTests {
    @Test("The defaults are upstream's, with the MLX backend")
    func defaults() throws {
        let settings = try ServerSettings()
        #expect(settings == (try ServerSettings(environment: [:])))
        #expect(settings.host == "127.0.0.1")
        #expect(settings.port == 8080)
        #expect(settings.backend == "mlx")
        #expect(settings.mlxModel == "mlx-community/diffusiongemma-26B-A4B-it-4bit")
        #expect(settings.mlxMaxPrompt == 32768)
        #expect(settings.mlxCacheLimitGB == nil)
        #expect(settings.mlxPromptCache == 12)
        #expect(settings.canvas == 64)
        #expect(settings.canvasStep == 16)
        #expect(settings.maxInflight == 64)
        #expect(settings.maxQueue == 512)
        #expect(settings.maxQuestions == 256)
        #expect(settings.maxBodyBytes == 64 * 1024 * 1024)
        #expect(settings.forwardTimeout == 300)
        #expect(settings.apiKey == "")
        #expect(settings.originSecret == "")
        #expect(settings.autoThreshold == 0.1)
        #expect(settings.autoMax == 4)
        #expect(settings.maxImages == 8)
        #expect(settings.maxImageBytes == 5 * 1024 * 1024)
        #expect(settings.genMaxInflight == 8)
        #expect(settings.genMaxQueue == 32)
        #expect(settings.genMaxTokens == 8192)
        #expect(settings.warmup)
        #expect(settings.layaModel == "convaiinnovations/laya-typed-decisions")
        #expect(settings.verdictModel == "heman10x/rlcd-modernbert-151m")
        #expect(settings.device == "")
        #expect(settings.encoderBatch == 16)
        #expect(settings.encoderFunctions == nil)
        #expect(settings.modelRoutes.isEmpty)
        #expect(settings.logLevel == .info)
    }

    /// Upstream's test refuses these eight; the messages are `__post_init__`'s.
    @Test(
        "Invalid settings fail at startup, as test_settings_are_checked_at_startup checks",
        arguments: [
            ("OPENJEV_CANVAS", "0", "canvas must be at least 1, got 0 (OPENJEV_CANVAS)"),
            (
                "OPENJEV_CANVAS_STEP", "0",
                "canvas_step must be at least 1, got 0 (OPENJEV_CANVAS_STEP)"
            ),
            (
                "OPENJEV_MAX_INFLIGHT", "0",
                "max_inflight must be at least 1, got 0 (OPENJEV_MAX_INFLIGHT)"
            ),
            (
                "OPENJEV_GEN_MAX_INFLIGHT", "0",
                "gen_max_inflight must be at least 1, got 0 (OPENJEV_GEN_MAX_INFLIGHT)"
            ),
            (
                "OPENJEV_MAX_QUESTIONS", "0",
                "max_questions must be at least 1, got 0 (OPENJEV_MAX_QUESTIONS)"
            ),
            (
                "OPENJEV_MAX_BODY_BYTES", "0",
                "max_body_bytes must be at least 1, got 0 (OPENJEV_MAX_BODY_BYTES)"
            ),
            (
                "OPENJEV_MAX_QUEUE", "-1",
                "max_queue must not be negative, got -1 (OPENJEV_MAX_QUEUE)"
            ),
            (
                "OPENJEV_FORWARD_TIMEOUT", "0",
                "forward_timeout must be at least 1, got 0.0 (OPENJEV_FORWARD_TIMEOUT)"
            ),
        ])
    func checkedAtStartup(name: String, value: String, message: String) {
        #expect(throws: ServerSettingsError(message)) {
            try ServerSettings(environment: [name: value])
        }
    }

    @Test(
        "Every other bounded setting is refused with its variable",
        arguments: [
            ("OPENJEV_MAX_IMAGE_BYTES", "0", "max_image_bytes must be at least 1, got 0"),
            ("OPENJEV_GEN_MAX_TOKENS", "0", "gen_max_tokens must be at least 1, got 0"),
            ("OPENJEV_MLX_MAX_PROMPT", "-5", "mlx_max_prompt must be at least 1, got -5"),
            ("OPENJEV_ENCODER_BATCH", "0", "encoder_batch must be at least 1, got 0"),
            ("OPENJEV_GEN_MAX_QUEUE", "-1", "gen_max_queue must not be negative, got -1"),
            ("OPENJEV_MAX_IMAGES", "-2", "max_images must not be negative, got -2"),
            ("OPENJEV_FORWARD_TIMEOUT", "0.5", "forward_timeout must be at least 1, got 0.5"),
        ])
    func bounded(name: String, value: String, message: String) {
        #expect(throws: ServerSettingsError("\(message) (\(name))")) {
            try ServerSettings(environment: [name: value])
        }
    }

    @Test("The memberwise initializer refuses the same values")
    func memberwise() {
        #expect(
            throws: ServerSettingsError(
                "canvas_step must be at least 1, got 0 (OPENJEV_CANVAS_STEP)")
        ) {
            try ServerSettings(canvasStep: 0)
        }
        #expect(
            throws: ServerSettingsError(
                "max_images must not be negative, got -1 (OPENJEV_MAX_IMAGES)")
        ) {
            try ServerSettings(maxImages: -1)
        }
        #expect(
            throws: ServerSettingsError("OPENJEV_MLX_PROMPT_CACHE=-1 is below the minimum of 0")
        ) {
            try ServerSettings(mlxPromptCache: -1)
        }
    }

    @Test("A value that does not parse names its variable and Python's type")
    func notANumber() {
        #expect(throws: ServerSettingsError("OPENJEV_CANVAS='sixty' is not a int")) {
            try ServerSettings(environment: ["OPENJEV_CANVAS": "sixty"])
        }
        #expect(throws: ServerSettingsError("OPENJEV_AUTO_THRESHOLD='0x1p-3' is not a float")) {
            try ServerSettings(environment: ["OPENJEV_AUTO_THRESHOLD": "0x1p-3"])
        }
        #expect(throws: ServerSettingsError("OPENJEV_PORT='80 80' is not a int")) {
            try ServerSettings(environment: ["OPENJEV_PORT": "80 80"])
        }
        #expect(throws: ServerSettingsError("OPENJEV_MLX_CACHE_LIMIT_GB='lots' is not a float")) {
            try ServerSettings(environment: ["OPENJEV_MLX_CACHE_LIMIT_GB": "lots"])
        }
    }

    @Test("An empty value is refused for a plain number, as Python's int('') is")
    func emptyPlainNumber() {
        #expect(throws: ServerSettingsError("OPENJEV_CANVAS='' is not a int")) {
            try ServerSettings(environment: ["OPENJEV_CANVAS": ""])
        }
        #expect(throws: ServerSettingsError("OPENJEV_FORWARD_TIMEOUT='' is not a float")) {
            try ServerSettings(environment: ["OPENJEV_FORWARD_TIMEOUT": ""])
        }
    }

    @Test("An empty value is the default for the _env_num settings and kept for strings")
    func emptyValues() throws {
        let settings = try ServerSettings(environment: [
            "OPENJEV_MLX_CACHE_LIMIT_GB": "", "OPENJEV_MLX_PROMPT_CACHE": "",
            "OPENJEV_MLX_MODEL": "", "OPENJEV_WARMUP": "",
        ])
        #expect(settings.mlxCacheLimitGB == nil)
        #expect(settings.mlxPromptCache == 12)
        #expect(settings.mlxModel == "")
        #expect(settings.warmup)
    }

    @Test("The MLX cache settings have _env_num's minimum and keep 0 distinct from unset")
    func cacheMinimums() throws {
        #expect(
            throws: ServerSettingsError("OPENJEV_MLX_CACHE_LIMIT_GB=-0.5 is below the minimum of 0")
        ) {
            try ServerSettings(environment: ["OPENJEV_MLX_CACHE_LIMIT_GB": "-0.5"])
        }
        #expect(
            throws: ServerSettingsError("OPENJEV_MLX_PROMPT_CACHE=-1 is below the minimum of 0")
        ) {
            try ServerSettings(environment: ["OPENJEV_MLX_PROMPT_CACHE": "-1"])
        }
        let zero = try ServerSettings(environment: [
            "OPENJEV_MLX_CACHE_LIMIT_GB": "0", "OPENJEV_MLX_PROMPT_CACHE": "0",
        ])
        #expect(zero.mlxCacheLimitGB == 0)
        #expect(zero.mlxPromptCache == 0)
    }

    @Test(
        "Integers parse as Python's int parses them",
        arguments: [
            (" 42 ", 42), ("+7", 7), ("-0", 0), ("1_000", 1000), ("0010", 10), ("\t9\n", 9),
        ])
    func pythonIntegers(text: String, value: Int) {
        #expect(PythonNumber.integer(text) == value)
    }

    @Test(
        "Strings Python's int refuses are refused",
        arguments: ["", " ", "1.0", "_1", "1_", "1__0", "0x10", "+-3", "1e3", "abc"])
    func pythonIntegerRefusals(text: String) {
        #expect(PythonNumber.integer(text) == nil)
    }

    /// CPython's `int()` refuses a string of more than 4,300 digits, `sys.int_max_str_digits`,
    /// counting leading zeros but not the sign, the underscores or the whitespace.
    @Test("Past 4,300 digits Python's int refuses, however small the value")
    func pythonIntegerDigitLimit() {
        let zeros = String(repeating: "0", count: 4299)
        #expect(PythonNumber.integer(zeros + "5") == 5)
        #expect(PythonNumber.integer(zeros + "05") == nil)
        #expect(PythonNumber.integer("-" + zeros + "5") == -5)
        #expect(PythonNumber.integer(" +" + zeros + "5\t") == 5)
        let underscored = Array(repeating: String(repeating: "0", count: 100), count: 43)
            .joined(separator: "_")
        #expect(PythonNumber.integer(underscored) == 0)
        #expect(PythonNumber.integer(underscored + "_1") == nil)
        #expect(PythonNumber.isInteger(String(repeating: "9", count: 4300)))
        #expect(!PythonNumber.isInteger(String(repeating: "9", count: 4301)))
    }

    @Test(
        "Floats parse as Python's float parses them",
        arguments: [
            ("0.25", 0.25), (" 3 ", 3), ("1e3", 1000), ("1.", 1), (".5", 0.5), ("1_0.5", 10.5),
            ("-2.5E-1", -0.25), ("inf", .infinity), ("-Infinity", -.infinity), ("1e999", .infinity),
        ])
    func pythonFloats(text: String, value: Double) {
        #expect(PythonNumber.float(text) == value)
    }

    @Test("NaN parses, and like upstream passes the minimum check")
    func nan() throws {
        #expect(PythonNumber.float("nan")?.isNaN == true)
        #expect(PythonNumber.float("-NaN")?.isNaN == true)
        let settings = try ServerSettings(environment: ["OPENJEV_MLX_CACHE_LIMIT_GB": "nan"])
        #expect(settings.mlxCacheLimitGB?.isNaN == true)
    }

    @Test(
        "Strings Python's float refuses are refused",
        arguments: ["", ".", "e3", "1e", "0x1p3", "1__0", "_1.0", "1._5", "infinit", "nan1"])
    func pythonFloatRefusals(text: String) {
        #expect(PythonNumber.float(text) == nil)
    }

    /// This port's setting (D-042), read as `_env_num` reads upstream's MLX cache settings.
    @Test("OPENJEV_ENCODER_FUNCTIONS is a number of at least 1; unset or empty is every function")
    func encoderFunctions() throws {
        func functions(_ value: String) throws -> Int? {
            try ServerSettings(environment: ["OPENJEV_ENCODER_FUNCTIONS": value]).encoderFunctions
        }
        #expect(try functions("2") == 2)
        #expect(try functions(" 8 ") == 8)
        #expect(try functions("") == nil)
        #expect(try ServerSettings(encoderFunctions: 1).encoderFunctions == 1)
        for value in ["0", "-1"] {
            #expect(
                throws: ServerSettingsError(
                    "OPENJEV_ENCODER_FUNCTIONS=\(value) is below the minimum of 1")
            ) {
                try functions(value)
            }
        }
        #expect(throws: ServerSettingsError("OPENJEV_ENCODER_FUNCTIONS='all' is not a int")) {
            try functions("all")
        }
        #expect(
            throws: ServerSettingsError("OPENJEV_ENCODER_FUNCTIONS=0 is below the minimum of 1")
        ) {
            try ServerSettings(encoderFunctions: 0)
        }
    }

    @Test("OPENJEV_WARMUP is off only for 0")
    func warmup() throws {
        #expect(try ServerSettings(environment: ["OPENJEV_WARMUP": "0"]).warmup == false)
        #expect(try ServerSettings(environment: ["OPENJEV_WARMUP": "false"]).warmup)
        #expect(try ServerSettings(environment: ["OPENJEV_WARMUP": "00"]).warmup)
    }

    @Test("Routes parse as parse_routes does, in order, trimmed, without trailing slashes")
    func routes() throws {
        let routes = try ServerSettings.parseRoutes(
            "verdict-1.4=http://verdict:8080/, laya-1.0=http://laya:8080,, ")
        #expect(Array(routes.keys) == ["verdict-1.4", "laya-1.0"])
        #expect(routes["verdict-1.4"] == "http://verdict:8080")
        #expect(routes["laya-1.0"] == "http://laya:8080")
        #expect(try ServerSettings.parseRoutes("").isEmpty)
        #expect(try ServerSettings.parseRoutes("a = http://x=y ")["a"] == "http://x=y")
    }

    /// `test_parse_routes`'s example, and what Python's dict, `str.split`, `str.strip` and
    /// `str.rstrip` do with the rest, each checked against upstream's `parse_routes`.
    @Test("Routes parse code point by code point, trimmed as str.strip() trims")
    func routesAsPythonReadsThem() throws {
        let example = try ServerSettings.parseRoutes("a=http://x/, b = http://y")
        #expect(Array(example.keys) == ["a", "b"])
        #expect(example["a"] == "http://x")
        #expect(example["b"] == "http://y")
        // A name given twice keeps its first place and takes its last URL; every slash goes.
        let twice = try ServerSettings.parseRoutes("a=http://1,b=http://2,a=http://3//")
        #expect(Array(twice.keys) == ["a", "b"])
        #expect(twice["a"] == "http://3")
        // U+001C to U+001F are Python's whitespace too, which Foundation's set leaves.
        let spaced = try ServerSettings.parseRoutes("\u{1C}a\u{3000}=\u{A0}http://x\u{85}\u{1F}")
        #expect(Array(spaced.keys) == ["a"])
        #expect(spaced["a"] == "http://x")
        // A comma followed by a combining mark still separates two routes.
        let combining = try ServerSettings.parseRoutes("a=http://x,\u{301}b=http://y")
        #expect(
            Array(combining.keys).map { Array($0.unicodeScalars) } == [["a"], ["\u{301}", "b"]])
        #expect(try ServerSettings.parseRoutes("a=/")["a"] == "")
        #expect(try ServerSettings.parseRoutes("\u{1C}").isEmpty)
    }

    @Test("Routes are logged without the credentials a URL holds")
    func routesWithoutCredentials() throws {
        let settings = try ServerSettings(environment: [
            "OPENJEV_MODEL_ROUTES":
                "a=http://u:p@h:1/x?y=@z, b=https://h/, c=u:p@h/x, d=http://@h, e=h/x@y, "
                + "f=//user:password@host/path, g=//u:p@h/x?y=http://a, h=http://u:p@\u{301}h"
        ])
        #expect(
            Array(settings.modelRoutesWithoutCredentials.keys) == [
                "a", "b", "c", "d", "e", "f", "g", "h",
            ])
        // A URL without a scheme cannot be forwarded to, but its credentials stay out of the log;
        // so do a scheme-relative URL's, and an @ that a combining mark follows still ends them.
        #expect(
            Array(settings.modelRoutesWithoutCredentials.values) == [
                "http://h:1/x?y=@z", "https://h", "h/x", "http://h", "h/x@y", "//host/path",
                "//h/x?y=http://a", "http://\u{301}h",
            ])
        #expect(settings.modelRoutes["a"] == "http://u:p@h:1/x?y=@z")
    }

    @Test(
        "A route without a name or URL is refused with parse_routes's message",
        arguments: [
            ("laya", "'laya'"), ("=http://x", "'=http://x'"), ("a=", "'a='"), (" b = ", "'b ='"),
        ])
    func badRoutes(text: String, repr: String) {
        #expect(throws: ServerSettingsError("OPENJEV_MODEL_ROUTES: \(repr) is not name=url")) {
            try ServerSettings(environment: ["OPENJEV_MODEL_ROUTES": text])
        }
    }

    @Test("Log levels are uvicorn's names; others are refused with the list")
    func logLevels() throws {
        #expect(try ServerSettings(environment: ["OPENJEV_LOG_LEVEL": "debug"]).logLevel == .debug)
        #expect(
            throws: ServerSettingsError(
                "OPENJEV_LOG_LEVEL='DEBUG' is not a log level; use one of trace, debug, info, "
                    + "notice, warning, error, critical")
        ) {
            try ServerSettings(environment: ["OPENJEV_LOG_LEVEL": "DEBUG"])
        }
    }

    @Test("Engine configurations take the settings the engines read")
    func engineConfigurations() throws {
        let settings = try ServerSettings(
            canvas: 32, canvasStep: 8, maxInflight: 3, maxQueue: 7, autoThreshold: 0.2,
            autoMax: 2, maxImages: 1, maxImageBytes: 99, warmup: false, encoderBatch: 5)
        let engine = try EngineConfiguration(settings)
        #expect(engine.geometry.canvas == 32)
        #expect(engine.geometry.step == 8)
        #expect(engine.maxInflight == 3)
        #expect(engine.maxQueue == 7)
        #expect(engine.autoThreshold == 0.2)
        #expect(engine.autoMax == 2)
        #expect(engine.imageLimits == ImageLimits(maxImages: 1, maxImageBytes: 99))
        let encoder = EncoderEngineConfiguration(settings)
        #expect(encoder.batchSize == 5)
        #expect(encoder.maxQueue == 7)
        #expect(encoder.maxInflight == 1)
        #expect(encoder.warmUp == false)
    }

    @Test("OPENJEV_AUTO_MAX reaches the engine: 1 never re-reads an uncertain read (#44)")
    func autoMaxReachesTheEngine() async throws {
        let recorded = try WireFixtures.recordedCase(named: "quickstart")
        let body = try #require(recorded["request"]?["body_text"]?.stringValue)
        let request = try SystemOneRequest(json: JSONParser().parse(body))
        // Every stub read has entropy 0.5, above the 0.1 threshold.
        for (environment, reads) in [([:], 4), (["OPENJEV_AUTO_MAX": "1"], 1)] {
            let stub = StubBackend(tokenizer: AnyPromptTokenizer(), entropy: 0.5)
            let service = try await DecisionBackendProvider { _ in stub }
                .makeService(settings: try ServerSettings(environment: environment))
            let decision = try await service.decide(request)
            #expect(stub.reads.count == reads, "\(environment)")
            #expect(decision.inputTokens == stub.readPromptTokens, "\(environment)")
        }
    }
}
