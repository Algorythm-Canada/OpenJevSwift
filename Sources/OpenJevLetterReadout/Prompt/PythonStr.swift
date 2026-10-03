// Python's `str()` and truthiness of the values `json.loads` produces, which the `jevk5` package's
// `decision_options` relies on when it writes an option as an f-string (github.com/allebee/jevk5
// at v0.2.2, `jevk5/prompt.py`). Apache-2.0. See THIRD_PARTY.md.

import OpenJevCore

extension JSONValue {
    /// Python's truth value of the value `json.loads` makes of this one: `None`, `False`, `0`,
    /// `0.0`, `""`, `[]` and `{}` are false, everything else is true.
    ///
    /// `decision_options` replaces a false description with a default (`v or k`), so an empty
    /// string, an empty object, `0` or `false` as a description reads as no description.
    var isPythonTruthy: Bool {
        switch self {
        case .null:
            return false
        case .bool(let value):
            return value
        case .integer(let digits):
            return digits.contains { $0 != "0" && $0 != "-" }
        case .float(let value):
            // NaN is true in Python; the request parser never produces one (D-016).
            return value != 0 || value.isNaN
        case .string(let text):
            return !text.isEmpty
        case .array(let elements):
            return !elements.isEmpty
        case .object(let object):
            return !object.isEmpty
        }
    }

    /// Python's `str()` of the value `json.loads` makes of this one, which an f-string
    /// replacement field without a format spec writes: a string as itself, anything else as
    /// ``pythonRepr``.
    var pythonStr: String {
        if case .string(let text) = self {
            return text
        }
        return pythonRepr
    }

    /// Python's `repr()` of the value `json.loads` makes of this one: `None`, `True` and
    /// `False`; an integer's digits; a float as `repr(float)`; a string as `repr(str)`, in single
    /// quotes unless it holds `'` and no `"`; a list as `[a, b]` and a dict as `{'k': v}`, each
    /// element written as its `repr` with `", "` and `": "` between them, keys in the order sent.
    ///
    /// The text is built with an explicit stack, so the deepest nesting the request parser takes
    /// (1,024 levels) cannot overflow the thread's stack.
    var pythonRepr: String {
        enum Step {
            case value(JSONValue)
            case text(String)
        }
        var output = ""
        var stack: [Step] = [.value(self)]
        while let step = stack.popLast() {
            switch step {
            case .text(let text):
                output += text
            case .value(.null):
                output += "None"
            case .value(.bool(let value)):
                output += value ? "True" : "False"
            case .value(.integer(let digits)):
                output += digits
            case .value(.float(let value)):
                output += value.pythonRepr
            case .value(.string(let text)):
                output += text.pythonRepr
            case .value(.array(let elements)):
                // Pushed in reverse, so the elements come off the stack in order.
                stack.append(.text("]"))
                for (index, element) in elements.enumerated().reversed() {
                    stack.append(.value(element))
                    if index > 0 {
                        stack.append(.text(", "))
                    }
                }
                stack.append(.text("["))
            case .value(.object(let object)):
                stack.append(.text("}"))
                for (index, entry) in object.enumerated().reversed() {
                    stack.append(.value(entry.value))
                    stack.append(.text(entry.key.pythonRepr + ": "))
                    if index > 0 {
                        stack.append(.text(", "))
                    }
                }
                stack.append(.text("{"))
            }
        }
        return output
    }
}
