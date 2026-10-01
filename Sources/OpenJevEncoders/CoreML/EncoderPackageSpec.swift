/// A converted encoder package with one function per input shape, as Tools/encoders' converters
/// write them: the shapes it holds, its input planes and their padding, and its output.
///
/// Each function is named `b{batch}_s{length}` and takes one int32 input of shape
/// [batch, planes, length]. A row is a list of planes, all the length of the row's tokens; the
/// padded part of each plane is filled as ``padding`` says. The output is float32
/// [batch, width], one row per input row. Nothing here names a model, so Verdict's backend and
/// Laya's (issue #58) share it.
public struct EncoderPackageSpec: Sendable, Hashable {
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

    /// Creates a spec. The batch sizes and lengths are sorted.
    public init(
        name: String, batchSizes: [Int], sequenceLengths: [Int], inputName: String = "tokens",
        padding: [Padding], outputName: String
    ) {
        self.name = name
        self.batchSizes = batchSizes.sorted()
        self.sequenceLengths = sequenceLengths.sorted()
        self.inputName = inputName
        self.padding = padding
        self.outputName = outputName
    }

    /// The planes of every row.
    public var planes: Int { padding.count }

    /// Verdict's float16 package with one function per shape, `verdict-m18-fp16` (D-011):
    /// batch 1 and 16 by 128, 256 and 512 tokens; the token ids padded with ModernBERT's
    /// `[PAD]` (50283) and the attention mask with 0; 25 logits per row.
    public static let verdict = EncoderPackageSpec(
        name: "verdict-m18-fp16", batchSizes: [1, 16], sequenceLengths: [128, 256, 512],
        padding: [.value(50_283), .value(0)], outputName: "logits")

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
