import Foundation
import MLX
import Testing

/// Points MLX at the Metal library that Swift Build copies into this test bundle, the helper of
/// docs/development.md ("MLX in tests", issue #8).
///
/// MLX looks for the library through `Bundle` objects, and the Swift Testing runner creates none
/// for the test bundle. Call `configure()` before the first MLX call in a test.
///
/// `OPENJEV_MLX_METALLIB`, when set to an existing file, is loaded instead, as the DiffusionGemma
/// tests take it: pointed at the Python mlx-metal wheel's `mlx.metallib`, the Swift model runs
/// the kernels mlx-lm ran when it recorded the fixture. It applies only if no MLX call came first
/// in the process; ``override`` says whether it was taken.
enum MetalLibrary {
    private final class Token {}

    /// The environment variable naming a metallib to load instead of the bundle's.
    static let overrideVariable = "OPENJEV_MLX_METALLIB"

    /// The library loaded instead of the bundle's, or nil.
    static var override: URL? { configured }

    /// Sets `GPU.metallib` once and returns the override it set, if any.
    private static let configured: URL? = {
        if let path = ProcessInfo.processInfo.environment[overrideVariable], !path.isEmpty,
            FileManager.default.fileExists(atPath: path)
        {
            let url = URL(fileURLWithPath: path)
            GPU.metallib = url
            return url
        }
        let url = Bundle(for: Token.self).resourceURL?.appending(
            path: "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        if let url, FileManager.default.fileExists(atPath: url.path) {
            GPU.metallib = url
        }
        return nil
    }()

    static func configure() {
        _ = configured
    }
}

/// The parent of every suite in this target that runs MLX. Serialized, as the DiffusionGemma
/// tests' parent is, so that no two MLX tests evaluate at once: MLX's streams are not meant to be
/// driven from several threads. Each test target is its own test bundle, run in its own process,
/// so the MLX suites of other targets cannot overlap these.
@Suite("MLX", .serialized)
enum MLXTests {}
