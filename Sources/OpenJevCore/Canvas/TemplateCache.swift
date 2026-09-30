// A port of upstream OpenJev (razorback16/openjev at dcd2094), `openjev/engine.py`, the
// `Engine._templates` dictionary and its handling in `resolve_template`. Apache-2.0. See
// THIRD_PARTY.md.

import Foundation

/// The resolved templates of one engine, keyed as upstream keys them.
///
/// Upstream keeps `_templates` as a plain dictionary and, before inserting, clears the whole
/// dictionary when it holds more than 4,096 entries. This does the same behind a lock, so one
/// cache can be shared by the engine actor of #17, the group reads it runs concurrently and any
/// caller outside the actor without a copy. A struct with a mutating dictionary would need every
/// user to hold the same mutable value, and a copy would fork the memo.
public final class TemplateCache: @unchecked Sendable {
    /// What upstream's `json.dumps([fmt, head, lead] + [(q["id"], q["labels"]) for q in qs])`
    /// identifies a template by.
    public struct Key: Sendable, Hashable {
        /// One question's part of the key: the id the model sees and its labels.
        public struct Question: Sendable, Hashable {
            /// The question id, `q1` to `qN`.
            public var id: String
            /// The question's labels in order.
            public var labels: [String]

            /// Creates a question key.
            public init(id: String, labels: [String]) {
                self.id = id
                self.labels = labels
            }
        }

        /// The answer format.
        public var format: AnswerFormat
        /// The head as used: the scaffold for a plain read, empty after a thought.
        public var head: [Int]
        /// The text before the first answer.
        public var lead: String
        /// The questions' ids and labels, in order.
        public var questions: [Question]

        /// Creates a key.
        public init(format: AnswerFormat, head: [Int], lead: String, questions: [Question]) {
            self.format = format
            self.head = head
            self.lead = lead
            self.questions = questions
        }
    }

    /// The entry count past which the cache is emptied before the next insert, upstream's 4096.
    public let limit: Int

    private let lock = NSLock()
    private var entries: [Key: ResolvedTemplate] = [:]

    /// Creates an empty cache. `limit` is upstream's 4,096 unless a test needs a smaller one.
    public init(limit: Int = 4096) {
        self.limit = limit
    }

    /// The number of templates held.
    public var count: Int {
        lock.withLock { entries.count }
    }

    /// The stored template for `key`, if any.
    public func template(for key: Key) -> ResolvedTemplate? {
        lock.withLock { entries[key] }
    }

    /// Stores `template` under `key`, emptying the cache first when it holds more than ``limit``
    /// entries, as upstream does.
    public func insert(_ template: ResolvedTemplate, for key: Key) {
        lock.withLock {
            if entries.count > limit {
                entries.removeAll()
            }
            entries[key] = template
        }
    }
}
