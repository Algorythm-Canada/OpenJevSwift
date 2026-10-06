// Matches what CPython makes of a value `json.loads` gave it: `bool(value)` and `repr(value)`
// (lists and dicts around the scalar reprs of `Errors/PythonRepr.swift`), which upstream OpenJev's
// chat route relies on for `if out.get("stream")` and for `{max_tokens!r}`. Written from the
// documented behaviour.

extension JSONValue {
    /// Python's `bool()` of the value: false for `None`, `False`, `0`, `0.0` (either sign), and
    /// empty strings, lists and dicts; true otherwise, NaN included.
    public var isPythonTruthy: Bool {
        switch self {
        case .null:
            return false
        case .bool(let value):
            return value
        case .integer(let digits):
            return digits != "0"
        case .float(let value):
            return value != 0
        case .string(let text):
            return !text.isEmpty
        case .array(let elements):
            return !elements.isEmpty
        case .object(let object):
            return !object.isEmpty
        }
    }

    /// Python's `repr()` of the value: `None`, `True`, `False`, an integer's digits, a float as
    /// `Double.pythonRepr`, a string as `String.pythonRepr`, `[a, b]` for a list and `{'k': v}`
    /// for a dict, with `, ` and `: ` between items.
    ///
    /// The value is walked with an explicit stack, so deep nesting cannot overflow the thread's
    /// stack. CPython itself raises `RecursionError` past about a thousand levels.
    public var pythonRepr: String {
        var out = ""
        // A value still to write, or the text that closes or separates what was opened.
        enum Step {
            case value(JSONValue)
            case text(String)
        }
        var pending: [Step] = [.value(self)]
        while let step = pending.popLast() {
            switch step {
            case .text(let text):
                out += text
            case .value(let value):
                switch value {
                case .null:
                    out += "None"
                case .bool(let flag):
                    out += flag ? "True" : "False"
                case .integer(let digits):
                    out += digits
                case .float(let number):
                    out += number.pythonRepr
                case .string(let text):
                    out += text.pythonRepr
                case .array(let elements):
                    out += "["
                    var steps: [Step] = []
                    for (index, element) in elements.enumerated() {
                        if index > 0 {
                            steps.append(.text(", "))
                        }
                        steps.append(.value(element))
                    }
                    steps.append(.text("]"))
                    pending.append(contentsOf: steps.reversed())
                case .object(let object):
                    out += "{"
                    var steps: [Step] = []
                    for (index, entry) in object.enumerated() {
                        if index > 0 {
                            steps.append(.text(", "))
                        }
                        steps.append(.text(entry.key.pythonRepr + ": "))
                        steps.append(.value(entry.value))
                    }
                    steps.append(.text("}"))
                    pending.append(contentsOf: steps.reversed())
                }
            }
        }
        return out
    }
}
