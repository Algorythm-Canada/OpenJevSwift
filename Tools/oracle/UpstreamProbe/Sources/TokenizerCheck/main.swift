// Renders every prompt of Fixtures/oracle/reads.json with the main package's
// SwiftTransformersTokenizer (issue #20) and compares the ids with the oracle's, which upstream's
// Engine.chat_prompt_ids produced in Python. Five of the twelve prompts are not in
// Fixtures/chat-prompts, the 2,939-token long_state prompt among them.

import Foundation
import OpenJevCore
import OpenJevDiffusionGemma

struct Prompt: Decodable {
    let system: String
    let user: String
    let ids: [Int]
}

struct Oracle: Decodable {
    let prompts: [String: Prompt]
}

let arguments = CommandLine.arguments
let oraclePath = arguments.count > 1 ? arguments[1] : "Fixtures/oracle/reads.json"
let environment = ProcessInfo.processInfo.environment
let hub = environment["HF_HUB_CACHE"] ?? (NSHomeDirectory() + "/.cache/huggingface/hub")
let model =
    environment["OPENJEV_TEST_MODEL"]
    ?? (hub + "/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/"
        + "a7a81407613811e8ba63af92ac0d852b809e191f")

let oracle = try JSONDecoder().decode(
    Oracle.self, from: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
let tokenizer = try await SwiftTransformersTokenizer.load(
    from: TokenizerFiles(directory: URL(fileURLWithPath: model)))
var matched = 0
for (key, prompt) in oracle.prompts.sorted(by: { $0.key < $1.key }) {
    let ids = try tokenizer.chatPromptIDs(system: prompt.system, user: prompt.user, thinking: false)
    let same = ids == prompt.ids
    matched += same ? 1 : 0
    let first = zip(ids, prompt.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    print(
        key.padding(toLength: 22, withPad: " ", startingAt: 0),
        "\(prompt.ids.count) tokens", same ? "match" : "DIFFER (swift \(ids.count), first at \(first ?? -1))")
}
print("prompt ids matched: \(matched)/\(oracle.prompts.count)")
if matched != oracle.prompts.count { exit(1) }
