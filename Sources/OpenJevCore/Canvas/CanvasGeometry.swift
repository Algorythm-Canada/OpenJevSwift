// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.canvas_width`, and the `canvas` and `canvas_step` settings of `openjev/config.py`.
// Apache-2.0. See THIRD_PARTY.md.

/// A canvas setting that upstream's startup validation refuses.
public struct CanvasGeometryError: Error, Sendable, Hashable, CustomStringConvertible {
    /// Which setting was refused and its value.
    public var message: String

    /// Creates an error with a message.
    public init(_ message: String) {
        self.message = message
    }

    /// The message.
    public var description: String { message }
}

/// The canvas length and the step its width is rounded to: upstream's `OPENJEV_CANVAS` (64) and
/// `OPENJEV_CANVAS_STEP` (16).
public struct CanvasGeometry: Sendable, Hashable {
    /// The most tokens a canvas may hold. A template and its turn close must fit.
    public let canvas: Int
    /// The multiple a canvas width is rounded up to.
    public let step: Int

    /// Creates a geometry.
    ///
    /// - Throws: ``CanvasGeometryError`` when either value is below 1. Upstream's `Settings`
    ///   refuses these at startup: a step of 0 would divide by zero on the first read.
    public init(canvas: Int = 64, step: Int = 16) throws(CanvasGeometryError) {
        guard canvas >= 1 else {
            throw CanvasGeometryError("canvas must be a positive integer, got \(canvas)")
        }
        guard step >= 1 else {
            throw CanvasGeometryError("canvas step must be a positive integer, got \(step)")
        }
        self.canvas = canvas
        self.step = step
    }

    /// The width of a canvas for a template of `n` tokens: `n + 1` (the turn close) rounded up
    /// to a multiple of ``step``, capped at ``canvas``.
    ///
    /// Upstream computes the rounding as `-(-need // step) * step`, Python's floor division
    /// negated to get a ceiling. With `need` and `step` positive that ceiling is
    /// `(need + step - 1) / step` in Swift's truncating division, which is what this uses.
    public func width(templateCount n: Int) -> Int {
        let need = n + 1
        return min(canvas, (need + step - 1) / step * step)
    }
}
