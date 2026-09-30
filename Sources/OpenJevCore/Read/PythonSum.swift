/// Adds floats exactly as CPython 3.12 and later's built-in `sum` does.
///
/// Since Python 3.12, `sum` over floats uses Neumaier's compensated summation, so
/// `sum([0.1] * 10)` is `1.0` rather than `0.9999999999999999`. Upstream's probabilities, scores
/// and confidences are all sums, and the wire fixtures were recorded with Python 3.14, so a plain
/// running total would differ from them in the last bits. The compensation is added at the end
/// only when it is non-zero and finite, as CPython does, so an infinite sum stays infinite.
func pythonSum<S: Sequence<Double>>(_ values: S) -> Double {
    var total = 0.0
    var compensation = 0.0
    for value in values {
        let next = total + value
        if abs(total) >= abs(value) {
            compensation += (total - next) + value
        } else {
            compensation += (value - next) + total
        }
        total = next
    }
    if compensation != 0, compensation.isFinite {
        total += compensation
    }
    return total
}
