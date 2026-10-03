import Foundation
import MLX

/// Points MLX at the Metal library that Swift Build copies into this test bundle, the helper of
/// docs/development.md ("MLX in tests", issue #8).
///
/// MLX looks for the library through `Bundle` objects, and the Swift Testing runner creates none
/// for the test bundle. Call `configure()` before the first MLX call in a test.
enum MetalLibrary {
    private final class Token {}

    private static let configured: Void = {
        let url = Bundle(for: Token.self).resourceURL?.appending(
            path: "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        if let url, FileManager.default.fileExists(atPath: url.path) {
            GPU.metallib = url
        }
    }()

    static func configure() {
        configured
    }
}
