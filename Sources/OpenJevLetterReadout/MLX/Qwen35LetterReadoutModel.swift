import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// JevK5 on MLX: mlx-swift-lm's Qwen3.5 text model (`Qwen35TextModel`, model type
/// `qwen3_5_text`) loaded from a converted checkpoint (`Tools/jevk5/convert.py`), read at the
/// prompt's last position only.
///
/// A pass runs the model over the prompt and applies the head (the embeddings, which JevK5 ties
/// to it) to the last position's post-norm hidden state, never to the other positions, so a
/// 16,000-token prompt does not compute 16,000 rows of 248,320 logits. A prompt longer than
/// ``prefillStepSize`` is read in chunks of that many tokens through the model's caches, as
/// mlx-lm prefills a prompt, which bounds the attention scores (MLX does not fuse attention over
/// Qwen3.5's 256-wide heads, so a pass holds them in full); a shorter prompt is one pass with no
/// cache.
///
/// `@unchecked Sendable`: the MLX modules are not `Sendable`, and every call comes from
/// ``JevK5Backend``'s actor, one pass at a time, which is what keeps them safe.
public final class Qwen35LetterReadoutModel: LetterReadoutModel, @unchecked Sendable {
    /// The longest chunk read in one call: 2,048 tokens, mlx-lm's prefill step.
    public let prefillStepSize: Int

    private let model: Qwen35TextModel
    /// The output head: the embeddings as a linear layer, or `lm_head` when the checkpoint does
    /// not tie them.
    private let head: (MLXArray) -> MLXArray

    private init(
        model: Qwen35TextModel, head: @escaping (MLXArray) -> MLXArray, prefillStepSize: Int
    ) {
        self.model = model
        self.head = head
        self.prefillStepSize = prefillStepSize
    }

    /// Loads the model from a converted checkpoint folder: `config.json` (model type
    /// `qwen3_5_text`, with mlx-lm's quantization entries) and the safetensors its index names.
    ///
    /// - Throws: ``JevK5LoadError/unsupportedModel(_:)`` for another model type or a checkpoint
    ///   whose head cannot be found, and the decoder's and mlx-swift-lm's weight loader's errors.
    public static func load(directory: URL, prefillStepSize: Int = 2048) async throws
        -> Qwen35LetterReadoutModel
    {
        precondition(prefillStepSize >= 1, "a prefill chunk holds at least one token")
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "qwen3_5_text" else {
            throw JevK5LoadError.unsupportedModel(
                "\(directory.path) holds a \(base.modelType) model; JevK5 is qwen3_5_text "
                    + "(convert it with Tools/jevk5/convert.py)")
        }
        let configuration = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
        let model = Qwen35TextModel(configuration)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        // MLX modules start in training mode, and Qwen3.5's linear attention takes its Metal
        // kernel only outside it; mlx-swift-lm's ModelContext and mlx-lm's load_model switch to
        // evaluation mode, and so does this loader, which uses neither.
        model.train(false)
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let head: (MLXArray) -> MLXArray
        if let linear = modules["lm_head"] as? any UnaryLayer {
            head = { linear($0) }
        } else if let embedding = modules["model.embed_tokens"] as? Embedding {
            head = { embedding.asLinear($0) }
        } else {
            throw JevK5LoadError.unsupportedModel(
                "\(directory.path): the model has neither lm_head nor model.embed_tokens")
        }
        return Qwen35LetterReadoutModel(
            model: model, head: head, prefillStepSize: prefillStepSize)
    }

    /// One pass: the logits of `letterIDs` at the last position of `tokens`, as Float.
    public func letterLogits(tokens: [Int], letterIDs: [Int]) throws -> [Float] {
        precondition(!tokens.isEmpty, "a prompt holds at least one token")
        let ids = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let cache: [KVCache]? =
            tokens.count > prefillStepSize ? try model.newCache(parameters: nil) : nil
        var start = 0
        while tokens.count - start > prefillStepSize {
            // Only the caches are evaluated, so the head never runs over these positions.
            _ = model(ids[0..., start..<(start + prefillStepSize)], cache: cache)
            eval(cache ?? [])
            start += prefillStepSize
        }
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let output = model(LMInput.Text(tokens: ids[0..., start...]), cache: cache, state: state)
        guard let hidden = output.state?[mtpLastHiddenStatesKey] else {
            throw JevK5ModelError("the model returned no hidden states")
        }
        let last = hidden.dim(1) - 1
        let logits = head(hidden[0..., last..., 0...]).reshaped(-1)
        let letters = take(logits, MLXArray(letterIDs.map { Int32($0) })).asType(.float32)
        eval(letters)
        return letters.asArray(Float.self)
    }
}
