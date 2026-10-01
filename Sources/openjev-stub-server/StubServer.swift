// The stub-backed OpenJev server the SDK compatibility suite runs (Tools/sdk-compat, issue #39,
// decision D-039): the real application over OpenJevTestSupport's stub backends, which answer as
// upstream's `tests/test_api.py` and `tests/test_encoders.py` stub their reads. It is not a
// product and never ships. Apache-2.0. See THIRD_PARTY.md.

import Foundation
import Logging
import OpenJevCore
import OpenJevServer
import OpenJevTestSupport
import ServiceLifecycle
import UnixSignals

/// `openjev-stub-server`: `openjev serve` over a stub backend, for tests that need a real server
/// without a model, such as the SDK suite on Linux, where Core ML does not exist.
///
/// The settings are the `OPENJEV_*` variables `openjev serve` reads, checked the same way.
/// `OPENJEV_BACKEND` picks the stub: `mlx`, the default, is ``StubBackend`` behind the
/// DiffusionGemma engine, which gives each question's first label 0.7 and serves `openjev-0.1`;
/// `laya` and `verdict` are ``StubQuestionReadBackend`` behind the encoder engine, which gives
/// the second option 0.7 and serves that model's name. Prompts the tokenizer fixtures never
/// recorded get stand-in ids (``AnyPromptTokenizer``), which the stubs never read.
///
/// Once it listens, the server prints the port it bound to standard output, alone on a line,
/// which is how a caller learns the port when `OPENJEV_PORT` is 0. Its log goes to standard
/// error. SIGTERM and SIGINT stop it gracefully with exit status 0; invalid settings exit 2 and
/// any other failure 1, as `openjev serve` does.
@main
enum StubServer {
    /// The backends `OPENJEV_BACKEND` may name.
    static let backends = ["mlx", "laya", "verdict"]

    static func main() async {
        let environment = ProcessInfo.processInfo.environment
        let settings: ServerSettings
        do {
            settings = try ServerSettings(environment: environment)
            guard backends.contains(settings.backend) else {
                throw ServerSettingsError.unknownBackend(settings.backend, known: backends)
            }
        } catch {
            fail("\(error)", status: 2)
        }
        var logger = Logger(label: "openjev-stub-server") { label in
            StreamLogHandler.standardError(label: label)
        }
        logger.logLevel = Logger.Level(rawValue: settings.logLevel.rawValue) ?? .info
        do {
            let service = try await provider(for: settings.backend).makeService(settings: settings)
            let server = DecisionServer(
                settings: settings, service: service, logger: logger,
                onServerRunning: { port in
                    FileHandle.standardOutput.write(Data("\(port)\n".utf8))
                })
            var configuration = ServiceGroupConfiguration(
                services: [server], gracefulShutdownSignals: [.sigterm, .sigint], logger: logger)
            configuration.maximumGracefulShutdownDuration = .seconds(10)
            try await ServiceGroup(configuration: configuration).run()
        } catch {
            fail("\(error)", status: 1)
        }
    }

    /// The stub `backend` names.
    static func provider(for backend: String) -> any BackendProvider {
        switch backend {
        case "laya":
            return QuestionReadBackendProvider { _ in
                StubQuestionReadBackend(modelInfo: KnownEncoderModels.laya)
            }
        case "verdict":
            return QuestionReadBackendProvider { _ in
                StubQuestionReadBackend(modelInfo: KnownEncoderModels.verdict, maxChoices: 24)
            }
        default:
            return DecisionBackendProvider { _ in StubBackend(tokenizer: AnyPromptTokenizer()) }
        }
    }

    /// Writes `openjev-stub-server: {message}` to standard error and exits with `status`.
    static func fail(_ message: String, status: Int32) -> Never {
        FileHandle.standardError.write(Data("openjev-stub-server: \(message)\n".utf8))
        exit(status)
    }
}
