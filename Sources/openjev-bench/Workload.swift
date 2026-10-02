// The request sets openjev-bench sends, as `/v1/systemone` bodies, so the in-process engine and an
// HTTP server get the same bytes.

import Foundation

enum Workload {
    /// Upstream's README questions (tests/test_mlx_model.py `QUESTIONS`).
    static let readmeQuestions = """
        {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}, \
        "team": {"type": "choice", "instructions": "Which team should handle it?", \
        "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}}, \
        "tone": {"type": "score", "instructions": "How upset is the customer?", \
        "criteria": ["calm", "annoyed", "furious"]}}
        """

    /// The topics of the nine extra noul questions of the 12-question set.
    static let extraTopics = [
        "a payment", "a login problem", "a deadline", "a named person", "a refund",
        "a security concern", "a mobile app", "a competitor", "a feature request",
    ]

    /// The questions of a read of `count` (1, 3 or 12): the README's urgent question alone, the
    /// three README questions, or those three and nine nouls.
    static func questions(_ count: Int) -> String {
        switch count {
        case 1:
            return
                #"{"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}"#
        case 3:
            return readmeQuestions
        default:
            precondition(count == 12, "the request sets have 1, 3 or 12 questions")
            let extra = extraTopics.enumerated().map { index, topic in
                #""m\#(index)": {"type": "noul", "instructions": "Does the message mention \#(topic)?"}"#
            }
            return String(readmeQuestions.dropLast()) + ", " + extra.joined(separator: ", ")
                + "}"
        }
    }

    /// A state no earlier request sent, so its prefill is not cached: upstream's README state
    /// after a ticket number made of `tag` and `index`.
    static func uniqueState(tag: String, index: Int) -> String {
        "Ticket \(tag)-\(index): Everything is down and we have a demo with our biggest client "
            + "at noon."
    }

    /// A state of about `tokens` prompt tokens: a ticket number, then a sentence of about 12
    /// tokens repeated. `tag` and `index` make it unique.
    static func longState(tokens: Int, tag: String, index: Int) -> String {
        let sentence = "The checkout page times out after the card form is submitted. "
        return "Ticket \(tag)-\(index). "
            + String(repeating: sentence, count: Swift.max(1, tokens / 12))
    }

    /// A request body with `"samples": 1`, the read the bench times.
    static func body(state: String, questions: String) -> String {
        let quoted = String(decoding: (try? JSONEncoder().encode(state)) ?? Data(), as: UTF8.self)
        return
            #"{"model": "openjev-latest", "state": \#(quoted), "questions": \#(questions), "samples": 1}"#
    }
}
