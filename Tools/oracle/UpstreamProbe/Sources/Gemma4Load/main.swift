// Risk R20: does upstream mlx-swift-lm (c043fb3) load the Gemma 4 26B-A4B MoE checkpoint, expert
// tensors included? Loads mlx-community/gemma-4-26B-A4B-it-4bit with LLMModelFactory, lists the
// expert parameters the model ended up with, and answers one short question greedily so the
// experts are exercised. The main package's SwiftTransformersTokenizer renders the prompt and
// decodes the answer; the factory only needs a tokenizer to finish loading.
//
//     swift run --package-path Tools/oracle/UpstreamProbe -c release Gemma4Load

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import OpenJevDiffusionGemma

struct UnusedTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

struct UnusedTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer { UnusedTokenizer() }
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let environment = ProcessInfo.processInfo.environment
let hub = environment["HF_HUB_CACHE"] ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
let path =
    CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : hub + "/models--mlx-community--gemma-4-26B-A4B-it-4bit/snapshots/"
        + "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
let directory = URL(fileURLWithPath: path)

let started = now()
let context: ModelContext
do {
    context = try await LLMModelFactory.shared.load(from: directory, using: UnusedTokenizerLoader())
} catch {
    print("load failed after \(String(format: "%.1f", now() - started)) s: \(error)")
    exit(1)
}
print(String(format: "loaded %@ in %.1f s", String(describing: type(of: context.model)), now() - started))

// The expert parameters as loaded: every name under an `experts` module, with shape and dtype.
let parameters = context.model.parameters().flattened()
let experts = parameters.filter { $0.0.contains(".experts.") }
print("parameters: \(parameters.count), of which experts: \(experts.count)")
for (name, array) in experts where name.contains("layers.0.") {
    print("  \(name) \(array.shape) \(array.dtype)")
}
let quantizedExperts = experts.filter { $0.0.hasSuffix(".scales") }.count
print("quantized expert projections (with scales): \(quantizedExperts)")

// One greedy answer through every layer, experts included.
let tokenizer = try await SwiftTransformersTokenizer.load(from: TokenizerFiles(directory: directory))
let prompt = try tokenizer.chatPromptIDs(
    system: "Answer with one word.", user: "What is the capital of France?", thinking: false)
guard let model = context.model as? any LLMModel else { fatalError("not an LLMModel") }
let cache = try model.newCache(parameters: nil)
var logits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: cache)
var answer: [Int] = []
let generationStart = now()
for _ in 0 ..< 12 {
    let next = argMax(logits[0, -1], axis: -1).item(Int.self)
    if next == 1 || next == 106 || next == 50 { break }  // <eos>, <turn|>, the checkpoint's EOS set
    answer.append(next)
    logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
}
let text = try tokenizer.decode(answer, skipSpecialTokens: true)
print(String(format: "greedy answer in %.2f s: %@ %@", now() - generationStart, String(describing: answer), text))
