import Foundation
import MLX
import Testing

/// Points MLX at the Metal library that Swift Build copies into this test bundle.
///
/// MLX looks for the library through `Bundle` objects, and the Swift Testing runner creates none
/// for the test bundle. Call `configure()` before the first MLX call in a test.
///
/// `OPENJEV_MLX_METALLIB`, when set to an existing file, is loaded instead: D-014's exact tier
/// points it at the Python mlx-metal wheel's `mlx.metallib`, whose precompiled kernels are the
/// oracle's. It takes effect only if no MLX call came first in the process, so it applies to the
/// whole test run; ``override`` reports whether it was taken.
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

/// The parent of every suite that runs MLX. Serialized, so that no two MLX tests evaluate at
/// once: MLX's streams are not meant to be driven from several threads, and the live tests hold
/// a 16 GB model.
@Suite("MLX", .serialized)
enum MLXTests {}
