/// A converted encoder package, as Tools/encoders' converters write them: the shapes it holds, how
/// it holds them, its input planes and their padding, and its output.
///
/// A package holds either one function per input shape, each named `b{batch}_s{length}`, or one
/// program for one fixed shape (``Layout``). Each shape takes one int32 input of shape
/// [batch, planes, length]. A row is a list of planes, all the length of the row's tokens; the
/// padded part of each plane is filled as ``padding`` says. The output is float32
/// [batch, width], one row per input row. Nothing here names a model, so Verdict's backend and
/// Laya's share it.
public struct EncoderPackageSpec: Sendable, Hashable {
    /// How a package holds its shapes.
    public enum Layout: Sendable, Hashable {
        /// One function per shape, named `b{batch}_s{length}`, sharing one copy of the weights:
        /// the iOS 18 multifunction packages, loaded with `MLModelConfiguration.functionName`.
        case functionPerShape
        /// One program for one fixed shape, loaded without a function name: Laya's per-length
        /// packages, the only Laya packages Core ML loads for the Neural Engine (D-011).
        case singleShape
    }

    /// How the padded part of one input plane is filled.
    public enum Padding: Sendable, Hashable {
        /// A fixed value: the padding token's id for the ids, 0 for the attention mask.
        case value(Int32)
        /// The row's first value in that plane, for a plane that holds one value per row; 0 in
        /// a padding row.
        case firstValue
    }

    /// One function of the package: the shape it takes.
    public struct Function: Sendable, Hashable, CustomStringConvertible {
        /// The rows one call takes.
        public var batchSize: Int
        /// The tokens of each row, padding included.
        public var sequenceLength: Int

        /// Creates a function's shape.
        public init(batchSize: Int, sequenceLength: Int) {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
        }

        /// The function's name in the package, `b{batch}_s{length}`.
        public var name: String { "b\(batchSize)_s\(sequenceLength)" }

        /// The name.
        public var description: String { name }
    }

    /// The package's name, which is also its folder's name without `.mlpackage`.
    public var name: String
    /// The batch sizes of its functions, ascending.
    public var batchSizes: [Int]
    /// The sequence lengths of its functions, ascending.
    public var sequenceLengths: [Int]
    /// The name of the input, `tokens`.
    public var inputName: String
    /// How each plane is padded; its count is the number of planes.
    public var padding: [Padding]
    /// The name of the output, such as Verdict's `logits`.
    public var outputName: String
    /// How the package holds its shapes.
    public var layout: Layout

    /// Creates a spec. The batch sizes and lengths are sorted.
    ///
    /// - Precondition: A ``Layout/singleShape`` package has one batch size and one length.
    public init(
        name: String, batchSizes: [Int], sequenceLengths: [Int], inputName: String = "tokens",
        padding: [Padding], outputName: String, layout: Layout = .functionPerShape
    ) {
        precondition(
            layout == .functionPerShape || (batchSizes.count == 1 && sequenceLengths.count == 1),
            "a package of one program holds one shape")
        self.name = name
        self.batchSizes = batchSizes.sorted()
        self.sequenceLengths = sequenceLengths.sorted()
        self.inputName = inputName
        self.padding = padding
        self.outputName = outputName
        self.layout = layout
    }

    /// The planes of every row.
    public var planes: Int { padding.count }

    /// Verdict's float16 package with one function per shape, `verdict-m18-fp16` (D-011):
    /// batch 1 and 16 by 128, 256 and 512 tokens; the token ids padded with ModernBERT's
    /// `[PAD]` (50283) and the attention mask with 0; 25 logits per row.
    public static let verdict = EncoderPackageSpec(
        name: "verdict-m18-fp16", batchSizes: [1, 16], sequenceLengths: [128, 256, 512],
        padding: [.value(50_283), .value(0)], outputName: "logits")

    /// The sequence lengths Laya's packages hold: 128, 256, 512 and 1,024 tokens.
    public static let layaSequenceLengths = [128, 256, 512, 1024]

    /// The planes of a Laya row, padded: the token ids with ModernBERT's `[PAD]` (50283), the
    /// attention mask with 0, and the question type (0 choice, 1 score, 2 noul), which the model
    /// reads at the first position, with the row's own type.
    private static let layaPadding: [Padding] = [.value(50_283), .value(0), .firstValue]

    /// Laya's float16 package with one function per shape, `laya-m18-fp16` (D-011): batch 1 and
    /// 16 by 128 to 1,024 tokens, the Mac's package. Its output `scores` is the scorer at every
    /// position, [batch, length], which the backend reads at the option markers.
    public static let layaMultifunction = EncoderPackageSpec(
        name: "laya-m18-fp16", batchSizes: [1, 16], sequenceLengths: layaSequenceLengths,
        padding: layaPadding, outputName: "scores")

    /// Laya's float16 package of one program for one fixed shape, batch 1 by `length` tokens,
    /// `laya-f18-b1s{length}-fp16` (D-011): the iPhone's packages, one per sequence length, which
    /// run on the Neural Engine.
    public static func laya(sequenceLength length: Int) -> EncoderPackageSpec {
        EncoderPackageSpec(
            name: "laya-f18-b1s\(length)-fp16", batchSizes: [1], sequenceLengths: [length],
            padding: layaPadding, outputName: "scores", layout: .singleShape)
    }

    /// The smallest function that holds `rows` rows of at most `longestRow` tokens, or `nil`
    /// when the package has none.
    public func function(rows: Int, longestRow: Int) -> Function? {
        guard let batchSize = batchSizes.first(where: { $0 >= rows }),
            let sequenceLength = sequenceLengths.first(where: { $0 >= longestRow })
        else {
            return nil
        }
        return Function(batchSize: batchSize, sequenceLength: sequenceLength)
    }

    /// The input of one call, row-major [batch, planes, length]: each row's planes, padded as
    /// ``padding`` says, then padding rows up to the batch size.
    ///
    /// - Precondition: The function holds the rows, and every row has ``planes`` planes of one
    ///   length.
    public func input(_ rows: [[[Int32]]], for function: Function) -> [Int32] {
        precondition(rows.count <= function.batchSize, "more rows than the function takes")
        let length = function.sequenceLength
        var values = [Int32]()
        values.reserveCapacity(function.batchSize * planes * length)
        for index in 0..<function.batchSize {
            let row = index < rows.count ? rows[index] : nil
            for (plane, pad) in padding.enumerated() {
                let tokens = row?[plane] ?? []
                precondition(tokens.count <= length, "a row longer than the function takes")
                values += tokens
                let fill: Int32
                switch pad {
                case .value(let value):
                    fill = value
                case .firstValue:
                    fill = tokens.first ?? 0
                }
                values += repeatElement(fill, count: length - tokens.count)
            }
        }
        return values
    }
}
