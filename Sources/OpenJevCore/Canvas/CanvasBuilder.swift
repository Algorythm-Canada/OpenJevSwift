// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, method
// `Engine.build_canvas`. Apache-2.0. See THIRD_PARTY.md.

/// A seeded canvas: the template with noise at the slots, and the noise itself.
public struct SeededCanvas: Sendable, Hashable {
    /// The canvas tokens, ``CanvasGeometry/width(templateCount:)`` of them.
    public var tokens: [Int]
    /// The noise token written at each slot, in slot order, for diagnostics and tests.
    public var noise: [Int]

    /// Creates a canvas.
    public init(tokens: [Int], noise: [Int]) {
        self.tokens = tokens
        self.noise = noise
    }

    /// The canvas width, `tokens.count`.
    public var width: Int { tokens.count }
}

/// Seeds a canvas from a resolved template, as upstream's `Engine.build_canvas` does.
public enum CanvasBuilder {
    /// The canvas for `template`: the template, the turn close ``EngineTokens/turnClose``, then
    /// ``EngineTokens/pad`` up to the geometry's width, with each slot position overwritten by
    /// the next `random.Random(seed).randrange(262144)` of one generator, in slot order.
    ///
    /// The noise token at a slot is what the model denoises; its distribution over the slot's
    /// label ids is the answer.
    ///
    /// - Precondition: `template.count + 1` is at most the geometry's canvas, which
    ///   ``TemplateResolver`` has checked, and every slot position is inside the template.
    public static func build(
        template: [Int], slots: [ResolvedTemplate.Slot], seed: UInt64, geometry: CanvasGeometry
    ) -> SeededCanvas {
        let width = geometry.width(templateCount: template.count)
        precondition(
            template.count + 1 <= width,
            "a template of \(template.count) tokens does not fit a canvas of \(geometry.canvas)")
        var rng = PythonRandom(seed: seed)
        var canvas = template + [EngineTokens.turnClose]
        canvas += [Int](repeating: EngineTokens.pad, count: width - canvas.count)
        var noise: [Int] = []
        noise.reserveCapacity(slots.count)
        for slot in slots {
            precondition(
                template.indices.contains(slot.position),
                "slot position \(slot.position) is outside a template of \(template.count) tokens")
            let token = rng.randrange(EngineTokens.vocabularySize)
            canvas[slot.position] = token
            noise.append(token)
        }
        return SeededCanvas(tokens: canvas, noise: noise)
    }
}
