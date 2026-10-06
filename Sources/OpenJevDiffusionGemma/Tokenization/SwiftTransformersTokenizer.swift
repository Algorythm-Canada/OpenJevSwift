import Darwin
import Foundation
import Hub
import Jinja
import OpenJevCore
import Tokenizers

/// The production ``/OpenJevCore/DecisionTokenizer``: the DiffusionGemma tokenizer loaded through
/// swift-transformers' `Tokenizers`, with the shipped Gemma 4 chat template rendered by
/// swift-jinja (decision D-008, spikes #20 and #21).
///
/// `encode` goes straight to swift-transformers, which agrees with Python's `tokenizers` on every
/// fixture text. `decode` goes through swift-transformers too, with two documented departures
/// (docs/spikes/tokenizer-parity.md): byte tokens at the end of a sequence are decoded here
/// because swift-transformers 1.3.4 drops them, and `clean_up_tokenization_spaces` is off, as it
/// is in Python's `transformers`, unless `tokenizer_config.json` turns it on.
///
/// `chatPromptIDs` is upstream's `Engine.chat_prompt_ids`: `apply_chat_template` over
/// `[system, user]` with the generation prompt and `enable_thinking`.
/// ``chatPromptText(system:user:thinking:)`` renders the template with swift-jinja and the
/// context swift-transformers builds, and `chatPromptIDs` tokenizes that text without adding
/// special tokens (the template writes `<bos>` itself), which is what swift-transformers'
/// `applyChatTemplate` does. `applyChatTemplate` itself is not called: it renders in
/// swift-jinja's own environment, whose `trim` strips Foundation's whitespace set, and the
/// template here renders in `templateEnvironment()`, whose `trim` is jinja2's, Python's
/// `str.strip()` (decision D-056). The tests check the text and the ids against the Python
/// fixtures.
///
/// Loading is asynchronous because swift-transformers' loader is. The value is immutable after
/// loading and shares one parsed vocabulary, so it is cheap to copy and safe to use from any
/// task.
public struct SwiftTransformersTokenizer: DecisionTokenizer {
    /// The files the tokenizer was loaded from.
    public let files: TokenizerFiles
    /// How long loading took and what it cost in memory.
    public let loadMetrics: LoadMetrics
    /// The text of `chat_template.jinja`, as loaded.
    public let chatTemplateSource: String
    /// The ids of the special tokens `tokenizer_config.json` names, which
    /// `decode(_:skipSpecialTokens: true)` drops: `bos_token`, `eos_token` and the other
    /// `*_token` entries, `additional_special_tokens`, `extra_special_tokens` and
    /// `model_specific_special_tokens`. For the pinned checkpoint these are the 24 added tokens
    /// `tokenizer.json` marks special.
    public let specialTokenIDs: Set<Int>

    private let tokenizer: any Tokenizers.Tokenizer
    private let chatTemplate: Jinja.Template
    /// The special token attributes of `tokenizer_config.json` that swift-transformers puts in
    /// the template context (`bos_token`, `eos_token`, and so on), converted once.
    private let specialTokenContext: [String: Jinja.Value]

    /// The `tokenizer_config.json` keys swift-transformers 1.3.4 exposes to the template.
    static let specialTokenAttributes = [
        "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
        "mask_token", "additional_special_tokens",
    ]

    /// The template options swift-transformers compiles chat templates with, which are also the
    /// options Python's `transformers` uses.
    static let templateOptions = Jinja.Template.Options(lstripBlocks: true, trimBlocks: true)

    /// The environment a chat template renders in: swift-jinja's, with
    /// ``pythonTrim(_:kwargs:env:)`` as `trim` (decision D-056).
    ///
    /// swift-jinja 2.5.1's own `trim` strips Foundation's `whitespacesAndNewlines`, which keeps
    /// U+001C to U+001F and removes U+200B, where upstream's jinja2 strips what Python's
    /// `str.strip()` strips. swift-jinja looks a filter up in the environment before its
    /// built-in filters, so a `trim` function here takes the built-in's place without a change
    /// to the package. A template that set a variable named `trim` would hide it; Gemma 4's
    /// does not. Every render gets a new environment, since an environment holds the render's
    /// variables.
    static func templateEnvironment() -> Jinja.Environment {
        let environment = Jinja.Environment()
        environment["trim"] = .function(pythonTrim)
        environment["sequence"] = .function(isPythonSequence)
        return environment
    }

    /// jinja2's `sequence` test, which a value passes when Python's `len()` and `__getitem__`
    /// accept it: a list, a string or a dict. swift-jinja 2.5.1's own test leaves dicts out, and
    /// the template asks it of a message's `content` and a tool's result, which a chat request
    /// may give as an object. swift-jinja looks a test up in the environment before its built-in
    /// tests, as it does a filter (D-058).
    @Sendable static func isPythonSequence(
        _ args: [Jinja.Value], kwargs: [String: Jinja.Value], env: Jinja.Environment
    ) throws -> Jinja.Value {
        switch args.first ?? .undefined {
        case .array, .string, .object:
            return .boolean(true)
        default:
            return .boolean(false)
        }
    }

    /// jinja2's `trim` filter, `soft_str(value).strip(chars)`, which is Python's `str.strip`.
    ///
    /// Without `chars`, or with `none`, it strips the scalars CPython's `str.isspace()` accepts
    /// (``/OpenJevCore/TextOf/isPythonWhitespace(_:)``); with a string, the scalars the string
    /// holds, so an empty string strips nothing. Scalars are compared one by one, as Python
    /// compares code points. Any other `chars`, an undefined variable included, throws, where
    /// Python raises `TypeError: strip arg must be None or str`. A value that is not a string is
    /// written as Python's `str()` writes it first (``pythonString(_:)``), as `soft_str` does: a
    /// chat request's text part can hold a number, a Boolean or null (D-058).
    @Sendable static func pythonTrim(
        _ args: [Jinja.Value], kwargs: [String: Jinja.Value], env: Jinja.Environment
    ) throws -> Jinja.Value {
        guard args.count <= 2, kwargs.keys.allSatisfy({ $0 == "chars" }),
            args.count < 2 || kwargs.isEmpty
        else {
            throw Jinja.JinjaError.runtime("trim takes one argument, chars")
        }
        let text = pythonString(args.first ?? .undefined)
        let strips: (Unicode.Scalar) -> Bool
        switch args.count == 2 ? args[1] : kwargs["chars"] ?? .null {
        case .null:
            strips = TextOf.isPythonWhitespace
        case .string(let chars):
            let set = Set(chars.unicodeScalars)
            strips = { set.contains($0) }
        default:
            throw Jinja.JinjaError.runtime("trim: strip arg must be None or str")
        }
        let scalars = text.unicodeScalars
        guard let start = scalars.firstIndex(where: { !strips($0) }),
            let end = scalars.lastIndex(where: { !strips($0) })
        else {
            return .string("")
        }
        return .string(String(scalars[start...end]))
    }

    /// Python's `str()` of a value the template reads, as jinja2 writes it: a string as itself,
    /// an undefined value as nothing, `None`, `True`, `False`, an integer's digits, a float as
    /// Python's `repr`, and a list or dict as Python writes it, its strings quoted.
    static func pythonString(_ value: Jinja.Value) -> String {
        switch value {
        case .string(let string):
            return string
        case .undefined:
            return ""
        case .null:
            return "None"
        case .boolean(let flag):
            return flag ? "True" : "False"
        case .int(let integer):
            return String(integer)
        case .double(let number):
            return number.pythonRepr
        case .array(let elements):
            return "[" + elements.map(pythonRepresentation).joined(separator: ", ") + "]"
        case .object(let object):
            let entries = object.map { key, value in
                pythonRepresentation(Jinja.Value(key)) + ": " + pythonRepresentation(value)
            }
            return "{" + entries.joined(separator: ", ") + "}"
        case .function, .macro:
            return value.description
        }
    }

    /// Python's `repr()` of a value inside a list or dict: a string quoted, the rest as
    /// ``pythonString(_:)``.
    static func pythonRepresentation(_ value: Jinja.Value) -> String {
        if case .string(let string) = value {
            return string.pythonRepr
        }
        return pythonString(value)
    }

    /// Renders `source`, compiled with ``templateOptions``, over `context` in
    /// ``templateEnvironment()``: the chat template's path, for a template of one's own. The
    /// tests check the environment's filters with it, which needs none of the checkpoint's
    /// files.
    ///
    /// - Throws: swift-jinja's error when the template does not compile or render.
    static func renderTemplate(_ source: String, context: [String: any Sendable]) throws -> String {
        let template = try Jinja.Template(source, with: templateOptions)
        return try template.render(
            context.mapValues { try Jinja.Value(any: $0) }, environment: templateEnvironment())
    }

    /// Loads the tokenizer from `files`.
    ///
    /// The steps are those of `Tokenizers.AutoTokenizer.from(modelFolder:)`, the entry point
    /// mlx-swift-lm's `#huggingFaceTokenizerLoader()` calls: `LanguageModelConfigurationFromHub`
    /// reads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (merged into the
    /// configuration as `chat_template`) and `config.json` when present, then
    /// `PreTrainedTokenizer` is built from the two configurations. One thing is added in
    /// between: when `tokenizer_config.json` has no `clean_up_tokenization_spaces`, it is set to
    /// false, Python's default since transformers 4.45, where swift-transformers would default to
    /// true and rewrite `" ."` to `"."` in every decode.
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` describing what swift-transformers or the
    ///   template compiler refused, or the error reading a file gave.
    public static func load(from files: TokenizerFiles) async throws -> Self {
        let before = ResourceUsage.current()
        let clock = ContinuousClock()
        let start = clock.now
        let tokenizer: any Tokenizers.Tokenizer
        do {
            let configuration = LanguageModelConfigurationFromHub(modelFolder: files.directory)
            guard var tokenizerConfig = try await configuration.tokenizerConfig?.dictionary()
            else {
                throw OpenJevCore.TokenizerError(
                    "\(files.tokenizerConfig.path) did not load as a tokenizer configuration")
            }
            if tokenizerConfig["clean_up_tokenization_spaces"] == nil {
                tokenizerConfig["clean_up_tokenization_spaces"] = Config(false)
            }
            tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: Config(tokenizerConfig),
                tokenizerData: try await configuration.tokenizerData)
        } catch let error as OpenJevCore.TokenizerError {
            throw error
        } catch {
            throw OpenJevCore.TokenizerError(
                "swift-transformers could not load \(files.directory.path): \(error)")
        }
        let wallTime = clock.now - start
        let after = ResourceUsage.current()

        let source = try String(contentsOf: files.chatTemplate, encoding: .utf8)
        let template: Jinja.Template
        do {
            template = try Jinja.Template(source, with: templateOptions)
        } catch {
            throw OpenJevCore.TokenizerError(
                "swift-jinja could not compile chat_template.jinja: \(error)")
        }
        let (context, specialTokens) = try specialTokens(from: files.tokenizerConfig)

        return SwiftTransformersTokenizer(
            files: files,
            loadMetrics: LoadMetrics(
                wallTime: wallTime,
                residentBytesBefore: before.residentBytes,
                residentBytesAfter: after.residentBytes,
                peakResidentBytes: after.peakResidentBytes),
            chatTemplateSource: source,
            specialTokenIDs: Set(specialTokens.compactMap { tokenizer.convertTokenToId($0) }),
            tokenizer: tokenizer,
            chatTemplate: template,
            specialTokenContext: context)
    }

    /// Reads `tokenizer_config.json` for the special token attributes the template sees, the
    /// way swift-transformers hands them over (a string as itself, an added-token object as its
    /// `content`, an array of strings as a list), and for every special token text it names.
    private static func specialTokens(from url: URL) throws -> (
        context: [String: Jinja.Value], tokens: [String]
    ) {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OpenJevCore.TokenizerError("\(url.path) is not a JSON object")
        }
        var context: [String: Jinja.Value] = [:]
        var tokens: [String] = []
        for (key, value) in object {
            let texts: [String]
            switch value {
            case let string as String:
                texts = [string]
            case let token as [String: Any]:
                texts = (token["content"] as? String).map { [$0] } ?? []
            case let strings as [String]:
                texts = strings
            default:
                continue
            }
            let isSpecialTokenKey =
                key.hasSuffix("_token") || key == "additional_special_tokens"
                || key == "extra_special_tokens"
            if isSpecialTokenKey {
                tokens += texts
            }
            if specialTokenAttributes.contains(key) {
                if value is [String] {
                    context[key] = .array(texts.map { .string($0) })
                } else if let text = texts.first {
                    context[key] = .string(text)
                }
            }
        }
        if let modelSpecific = object["model_specific_special_tokens"] as? [String: Any] {
            tokens += modelSpecific.values.compactMap { $0 as? String }
        }
        return (context, tokens)
    }

    /// True when swift-transformers found a chat template, here the one it read from
    /// `chat_template.jinja`.
    public var hasChatTemplate: Bool { tokenizer.hasChatTemplate }

    /// The id of `token`, an entry of the vocabulary such as `<turn|>`, or nil when the
    /// vocabulary has no such entry.
    public func tokenID(of token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    /// The vocabulary entry with id `id`, or nil when there is none.
    public func token(of id: Int) -> String? {
        tokenizer.convertIdToToken(id)
    }

    /// The ids of `text`, through swift-transformers' `encode(text:addSpecialTokens:)`, which the
    /// tokenizer parity spike matched with Python's `tokenizers` on every recorded text (D-008).
    public func encode(_ text: String, addSpecialTokens: Bool) throws -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    /// The text of `ids`.
    ///
    /// swift-transformers 1.3.4's byte-fallback decoder gathers consecutive `<0xNN>` tokens and
    /// decodes them as UTF-8 when an ordinary token follows, but never flushes the bytes a
    /// sequence ends with, so `decode([238])` (`<0x00>`) gives `""` where Python gives `"\0"`.
    /// Here the special tokens are dropped first when asked, as Python does, then the trailing
    /// run of byte tokens is split off, the rest decoded by swift-transformers and the run
    /// decoded as UTF-8 with U+FFFD for an invalid sequence, as Python's `tokenizers` does.
    public func decode(_ ids: [Int], skipSpecialTokens: Bool) throws -> String {
        let kept = skipSpecialTokens ? ids.filter { !specialTokenIDs.contains($0) } : ids
        var trailingBytes: [UInt8] = []
        var end = kept.endIndex
        while end > kept.startIndex, let byte = Self.byteValue(of: token(of: kept[end - 1])) {
            trailingBytes.append(byte)
            end -= 1
        }
        let head = tokenizer.decode(tokens: Array(kept[..<end]), skipSpecialTokens: false)
        if trailingBytes.isEmpty {
            return head
        }
        return head + String(decoding: trailingBytes.reversed(), as: UTF8.self)
    }

    /// The byte a byte-fallback token such as `<0xE2>` stands for, or nil for any other token.
    static func byteValue(of token: String?) -> UInt8? {
        guard let token, token.utf8.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">")
        else {
            return nil
        }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16)
    }

    /// The prompt ids of `[system, user]` with the generation prompt, the shipped chat template
    /// rendered with `enable_thinking` set to `thinking`, as upstream's `apply_chat_template` gives
    /// them (D-008, D-056).
    public func chatPromptIDs(system: String, user: String, thinking: Bool) throws -> [Int] {
        try applyChatTemplate(
            messages: Self.messages(system: system, user: user), thinking: thinking)
    }

    /// The text `chatPromptIDs(system:user:thinking:)` tokenizes: the chat template rendered
    /// over `[system, user]` with the generation prompt and `enable_thinking` set to `thinking`,
    /// upstream's `apply_chat_template(..., tokenize=False)`.
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` when the template does not render.
    public func chatPromptText(system: String, user: String, thinking: Bool) throws -> String {
        try renderChatTemplate(
            messages: Self.messages(system: system, user: user), addGenerationPrompt: true,
            thinking: thinking)
    }

    /// The ids of `messages` with the generation prompt and `enable_thinking` set to `thinking`,
    /// upstream's `apply_chat_template(..., tokenize=True)`: the text
    /// ``renderChatTemplate(messages:addGenerationPrompt:thinking:)`` renders, tokenized without
    /// special tokens and without truncation. That is what swift-transformers'
    /// `applyChatTemplate` does, in `templateEnvironment()` rather than swift-jinja's own,
    /// whose `trim` is not Python's (D-056). `messages` are chat messages as dictionaries, `role`
    /// and `content`, with `content` a string or a list of parts.
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` when the template does not render.
    public func applyChatTemplate(messages: [[String: any Sendable]], thinking: Bool) throws
        -> [Int]
    {
        try encode(
            renderChatTemplate(messages: messages, addGenerationPrompt: true, thinking: thinking),
            addSpecialTokens: false)
    }

    /// Renders `chat_template.jinja` over `messages` with swift-jinja in
    /// `templateEnvironment()`, with the context swift-transformers builds: `messages`,
    /// `add_generation_prompt`, `enable_thinking` and the special token attributes of
    /// `tokenizer_config.json`.
    ///
    /// `messages` are chat messages as dictionaries, `role` and `content`, with `content` a
    /// string or a list of parts such as `{"type": "image"}` and `{"type": "text", "text": ...}`.
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` when the template does not render.
    public func renderChatTemplate(
        messages: [[String: any Sendable]], addGenerationPrompt: Bool, thinking: Bool
    ) throws -> String {
        var context = specialTokenContext
        do {
            context["messages"] = try .array(messages.map { try Jinja.Value(any: $0) })
            context["add_generation_prompt"] = .boolean(addGenerationPrompt)
            context["enable_thinking"] = .boolean(thinking)
            return try chatTemplate.render(context, environment: Self.templateEnvironment())
        } catch {
            throw OpenJevCore.TokenizerError(
                "swift-jinja could not render the chat template: \(error)")
        }
    }

    /// The prompt ids of a chat request, upstream's `MlxGenerator.prompt_ids`: the chat template
    /// over `messages` with the generation prompt and `enable_thinking` set to `thinking`
    /// (``renderChatTemplate(chatMessages:thinking:)``), tokenized without special tokens, then
    /// the ids of the empty thought scaffold `<|channel>thought\n<channel|>`, which has the model
    /// start past a thought on most replies (issue #53).
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` when the template does not render, or when
    ///   `messages` nest deeper than ``/OpenJevCore/ChatCompletionRequest/maximumNesting``.
    public func generationPromptIDs(messages: [JSONValue], thinking: Bool) async throws -> [Int] {
        let text = try await renderChatTemplate(chatMessages: messages, thinking: thinking)
        return try encode(text, addSpecialTokens: false)
            + encode(EngineTokens.scaffoldText, addSpecialTokens: false)
    }

    /// The chat template rendered over a chat request's messages, as upstream's
    /// `apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking)` renders
    /// them: each message is the value `json.loads` gave, its keys in the request's order, so the
    /// template reads `role`, `content`, `tool_calls` and the rest as it does upstream.
    ///
    /// Two things differ from jinja2 with values a chat request can hold, both in tool calls: a
    /// `null` that the template writes with `{{ }}`, a tool call's argument or a tool's missing
    /// result, is written as nothing where jinja2 writes `None`, and an integer too large for
    /// `Int` is a float (D-058).
    ///
    /// The template renders on a thread of its own with 16 MB of stack:
    /// swift-jinja recurses into tool call arguments, and in a debug build, on a thread with a
    /// task's 512 KB of stack, 7 levels of them rendered and 16 overflowed it, while the 64 levels
    /// ``/OpenJevCore/ChatCompletionRequest/maximumNesting`` allows needed 8 MB.
    ///
    /// - Throws: An ``/OpenJevCore/TokenizerError`` when the template does not render, or when
    ///   `messages` nest deeper than ``/OpenJevCore/ChatCompletionRequest/maximumNesting``, which
    ///   bounds the template's recursion.
    public func renderChatTemplate(chatMessages messages: [JSONValue], thinking: Bool) async throws
        -> String
    {
        let depth = ChatCompletionRequest.nesting(of: .array(messages))
        guard depth <= ChatCompletionRequest.maximumNesting else {
            throw OpenJevCore.TokenizerError(
                "the messages nest \(depth) levels deep; the limit is "
                    + "\(ChatCompletionRequest.maximumNesting)")
        }
        let template = chatTemplate
        var context = specialTokenContext
        context["add_generation_prompt"] = .boolean(true)
        context["enable_thinking"] = .boolean(thinking)
        let base = context
        return try await Self.onRenderingThread {
            var context = base
            context["messages"] = .array(messages.map(Self.templateValue))
            do {
                return try template.render(context, environment: Self.templateEnvironment())
            } catch {
                throw OpenJevCore.TokenizerError(
                    "swift-jinja could not render the chat template: \(error)")
            }
        }
    }

    /// The stack of the thread a chat request's template renders on: 16 MB, twice what a debug
    /// build needs at the deepest messages allowed.
    static let renderingStackSize = 16 << 20

    /// Runs `body` on a new thread with ``renderingStackSize`` bytes of stack and returns what it
    /// returns or throws. The task waits without blocking its thread.
    static func onRenderingThread<T: Sendable>(
        _ body: @escaping @Sendable () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            let thread = Thread {
                continuation.resume(with: Result { try body() })
            }
            thread.stackSize = renderingStackSize
            thread.start()
        }
    }

    /// A JSON value as the template reads it, with an object's keys in order.
    static func templateValue(_ value: JSONValue) -> Jinja.Value {
        switch value {
        case .null:
            return .null
        case .bool(let flag):
            return .boolean(flag)
        case .integer(let digits):
            return Int(digits).map(Jinja.Value.int) ?? .double(Double(digits) ?? .nan)
        case .float(let number):
            return .double(number)
        case .string(let string):
            return .string(string)
        case .array(let elements):
            return .array(elements.map(templateValue))
        case .object(let object):
            var entries = OrderedDictionary<Jinja.ObjectKey, Jinja.Value>()
            for (key, element) in object {
                entries[.string(key)] = templateValue(element)
            }
            return .object(entries)
        }
    }

    /// The `[system, user]` messages of a read, as upstream's `chat_prompt_ids` builds them.
    static func messages(system: String, user: String) -> [[String: any Sendable]] {
        [
            ["role": "system", "content": system],
            ["role": "user", "content": user],
        ]
    }
}

/// What loading a tokenizer cost, measured by ``SwiftTransformersTokenizer/load(from:)``.
///
/// The memory figures are of the whole process, so they mean most in a process that has done
/// little else, such as a fresh test run; a tokenizer loaded twice shows a much smaller second
/// increase because the first load's allocations are reused.
public struct LoadMetrics: Sendable, Hashable {
    /// The wall time the swift-transformers load took, by `ContinuousClock`.
    public var wallTime: Duration
    /// The process's resident memory in bytes just before loading (`task_info` resident size).
    public var residentBytesBefore: Int
    /// The process's resident memory in bytes just after loading.
    public var residentBytesAfter: Int
    /// The process's peak resident memory in bytes after loading (`getrusage` `ru_maxrss`).
    public var peakResidentBytes: Int

    /// Creates a metrics value.
    public init(
        wallTime: Duration, residentBytesBefore: Int, residentBytesAfter: Int,
        peakResidentBytes: Int
    ) {
        self.wallTime = wallTime
        self.residentBytesBefore = residentBytesBefore
        self.residentBytesAfter = residentBytesAfter
        self.peakResidentBytes = peakResidentBytes
    }

    /// The resident memory loading added, in bytes.
    public var residentBytesAdded: Int { residentBytesAfter - residentBytesBefore }
}

/// The process's memory use, read from the kernel.
struct ResourceUsage {
    /// Resident memory in bytes, `mach_task_basic_info.resident_size`.
    var residentBytes: Int
    /// Peak resident memory in bytes, `rusage.ru_maxrss`, which macOS reports in bytes.
    var peakResidentBytes: Int

    /// The current figures. A failed kernel call gives zero for its figure.
    static func current() -> ResourceUsage {
        var usage = rusage()
        let peak = getrusage(RUSAGE_SELF, &usage) == 0 ? Int(usage.ru_maxrss) : 0

        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let resident = result == KERN_SUCCESS ? Int(info.resident_size) : 0
        return ResourceUsage(residentBytes: resident, peakResidentBytes: peak)
    }
}
