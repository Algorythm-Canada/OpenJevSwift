// openjev-bench profile: where a read's GPU time goes, stage by stage, through the model's public
// stage observer. Each stage is evaluated where the observer sees it, so the stages run one after
// another instead of in one command stream; the unstaged totals beside them show what that
// serialisation costs.

import ArgumentParser
import Foundation
import MLX
import OpenJevCore
import OpenJevDiffusionGemma

/// The stage an observer name belongs to.
func profileStage(_ name: String) -> String {
    if name.hasPrefix("attn.") { return "attention" }
    if name.hasPrefix("mlp.") { return "dense MLP" }
    if name.hasPrefix("router.") { return "router" }
    if name.hasPrefix("experts.") { return "experts (gathered quantized matmuls)" }
    if name.hasPrefix("layer.") { return "norms, residuals, layer scalar" }
    return name
}

/// Sums the time between successive evaluated stages into named buckets.
final class StageClock {
    private let clock = ContinuousClock()
    private var last: ContinuousClock.Instant
    private(set) var buckets: [String: Double] = [:]
    /// The evaluation points of each bucket, summed over every read timed.
    private(set) var marks: [String: Int] = [:]
    private(set) var order: [String] = []

    init() {
        last = clock.now
    }

    func restart() {
        last = clock.now
    }

    /// Evaluates `values` and adds the time since the last mark to `stage`.
    func mark(_ stage: String, _ values: [MLXArray]) {
        eval(values)
        let now = clock.now
        if buckets[stage] == nil {
            order.append(stage)
        }
        buckets[stage, default: 0] += milliseconds(now - last)
        marks[stage, default: 0] += 1
        last = now
    }

    /// The observer the model's layers call.
    var observer: StageObserver {
        { [unowned self] name, value in mark(profileStage(name), [value]) }
    }
}

struct Profile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Where a 3-question read's time goes: prefill and decoder pass stage by stage.")

    @OptionGroup var common: CommonOptions

    @Option(help: "Profile a state of about this many tokens instead of the README state.")
    var stateTokens: Int?

    func validate() throws {
        if let stateTokens, stateTokens < 1 {
            throw ValidationError("--state-tokens must be at least 1, got \(stateTokens)")
        }
    }

    func run() async throws {
        let directory = common.modelDirectory()
        common.applyMetallib()
        let recorded = common.recordedSettings
        FileHandle.standardError.write(Data("loading \(directory.path)\n".utf8))
        let tokenizer = try await SwiftTransformersTokenizer.load(
            from: TokenizerFiles(directory: directory))
        let model = try await DiffusionGemmaModel.load(from: directory).model
        let labels = try LabelDiscovery.choiceLabels(using: tokenizer)
        let request = try SystemOneRequest(
            json: JSONParser().parse(Workload.body(state: "", questions: Workload.questions(3))))
        let schema = try QuestionSchemaBuilder(choiceLabels: labels.labels).build(
            request.questions)
        let resolved = try TemplateResolver(
            tokenizer: tokenizer, tokens: try EngineTokens(tokenizer: tokenizer)
        ).resolve(schema.questions, format: schema.format)
        let system = SystemText.render(schema.questions, format: schema.format, chunked: false)
        let canvas = CanvasBuilder.build(
            template: resolved.template, slots: resolved.slots, seed: 0, geometry: .standard)
        let slots = resolved.slots.map { SlotRequest(position: $0.position, labelIDs: $0.labelIDs) }
        let softcap = makeSoftcap(model.configuration.finalLogitSoftcapping)

        let prefillClock = StageClock()
        let decoderClock = StageClock()
        let clock = ContinuousClock()
        var prefillTotals: [Double] = []
        var readTotals: [Double] = []
        var promptTokens = 0
        var identical = true
        for index in 0..<(common.warmup + common.runs) {
            let timed = index >= common.warmup
            let state =
                stateTokens.map {
                    Workload.longState(tokens: $0, tag: "\(Bench.tag)-p", index: index)
                } ?? Workload.uniqueState(tag: "\(Bench.tag)-p", index: index)
            let promptIDs = try tokenizer.chatPromptIDs(
                system: system, user: state, thinking: false)
            promptTokens = promptIDs.count

            // The unstaged read, as the runtime runs it.
            var start = clock.now
            let cache = try model.prefill(promptIDs: promptIDs)
            let prefillTime = milliseconds(clock.now - start)
            start = clock.now
            let read = try model.read(
                canvas: canvas.tokens, slots: slots, cache: cache, steps: 1,
                topK: DiffusionGemmaRuntime.topK)
            let readTime = milliseconds(clock.now - start)

            // The same prefill, stage by stage; its caches are dropped.
            let staged = timed ? prefillClock : StageClock()
            let ids = MLXArray(promptIDs.map(Int32.init)).reshaped(1, promptIDs.count)
            staged.restart()
            let caches = model.prefill(
                ids,
                stages: { name, value in
                    staged.mark(name == "embeddings" ? "embedding" : profileStage(name), [value])
                })
            staged.mark(
                "cache evaluation", caches.flatMap { [$0.keys, $0.values].compactMap { $0 } })

            // The same decoder pass over the unstaged cache, stage by stage.
            let decoder = timed ? decoderClock : StageClock()
            let canvasIDs = MLXArray(canvas.tokens.map(Int32.init)).reshaped(1, canvas.tokens.count)
            let masks = model.decoderMasks(canvasLength: canvas.tokens.count, cache: cache)
            decoder.restart()
            let embeddings = model.decoder.embed(canvasIDs)
            var h = model.decoder.selfConditioning(embeddings, signal: zeros(like: embeddings))
            decoder.mark("embedding and self-conditioning", [h])
            for (layerIndex, layer) in model.decoder.layers.enumerated() {
                h = layer(
                    h, mask: masks[layer.layerType] ?? .none, cache: cache.layers[layerIndex],
                    decoder: true, offset: cache.offset, stages: decoder.observer)
            }
            h = model.decoder.norm(h)
            decoder.mark("final norm", [h])
            let projected = model.decoder.embedTokens.asLinear(h)
            decoder.mark("output projection (tied embedding)", [projected])
            let logits = softcap(projected)
            decoder.mark("softcap", [logits])
            let maps = slots.map {
                DiffusionGemmaModel.slotLogprobs(
                    row: logits[0, $0.position], labelIDs: $0.labelIDs,
                    topK: DiffusionGemmaRuntime.topK)
            }
            decoder.mark("slot log-softmax, top 20, host copies", [])

            if timed {
                prefillTotals.append(prefillTime)
                readTotals.append(readTime)
                identical =
                    identical
                    && zip(maps, read.slots).allSatisfy { a, b in
                        a.map(\.tokenID) == b.map(\.tokenID)
                            && a.map { Float($0.logprob).bitPattern }
                                == b.map { Float($0.logprob).bitPattern }
                    }
            }
        }

        func rows(_ phase: String, _ stageClock: StageClock, total: [Double]) -> [StageRow] {
            let runs = Double(common.runs)
            return StageRow.phase(
                phase,
                stageClock.order.map {
                    StageRow.Measured(
                        stage: $0, meanMilliseconds: stageClock.buckets[$0, default: 0] / runs,
                        marks: stageClock.marks[$0, default: 0] / common.runs)
                }, unstagedMilliseconds: total.reduce(0, +) / Double(total.count))
        }
        var run = BenchRun(
            mode: "profile", target: "engine", url: nil, server: "swift",
            modelDirectory: directory.path, started: Bench.now(),
            settings: [
                "runs": "\(common.runs)", "warmup": "\(common.warmup)", "questions": "3",
                "prompt tokens": "\(promptTokens)", "canvas": "\(canvas.tokens.count)",
                "staged read identical to unstaged": identical ? "yes" : "no",
            ].merging(recorded) { mode, _ in mode })
        run.stages =
            rows("prefill", prefillClock, total: prefillTotals)
            + rows("decoder pass", decoderClock, total: readTotals)
        try Bench.finish(run, common)
    }
}
