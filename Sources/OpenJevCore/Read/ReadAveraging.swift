// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the averaging
// in `read_group`. Apache-2.0. See THIRD_PARTY.md.

/// Combines the label distributions of several reads of one question.
public enum ReadAveraging {
    /// The arithmetic mean of each label's probability over the reads.
    ///
    /// Upstream averages every read it made, the automatic re-reads and the `samples` reads
    /// alike, with Python's `sum`, then divides by the number of reads.
    ///
    /// - Parameter reads: One probability vector per read, each in label order.
    /// - Returns: One mean per label.
    /// - Precondition: There is at least one read and every read has the same number of labels.
    public static func mean(_ reads: [[Double]]) -> [Double] {
        precondition(!reads.isEmpty, "ReadAveraging.mean needs at least one read")
        let labels = reads[0].count
        precondition(
            reads.allSatisfy { $0.count == labels }, "every read needs \(labels) probabilities")
        let count = Double(reads.count)
        return (0..<labels).map { label in
            pythonSum(reads.lazy.map { $0[label] }) / count
        }
    }
}
