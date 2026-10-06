import OpenJevCore
import Testing
import TriageDemo

/// The view model must send upstream OpenJev's README example ("Try it") byte for byte.
@MainActor
struct RequestTests {
    /// The body TypeSafe's SDK posts for the README's `client.system_one(...)` call: the SDK's
    /// default model, the state, and the three questions in the README's order.
    static let upstreamExample = """
        {"model": "jev-latest",
         "state": "Everything is down and we have a demo with our biggest client at noon.",
         "questions": {
           "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
           "team":   {"type": "choice", "instructions": "Which team should handle it?",
                      "criteria": {"outage": "service down", "billing": "charges, refunds", "feature": "requests, how-to"}},
           "tone":   {"type": "score", "instructions": "How upset is the customer?",
                      "criteria": ["calm", "annoyed", "furious"]}}}
        """

    @Test func buildsUpstreamsReadmeRequest() throws {
        let model = TriageModel()
        let request = model.request(
            for: "Everything is down and we have a demo with our biggest client at noon.")
        // Decoded and validated as the server decodes a body. Equality compares the questions and
        // each choice's options in order, the orders that decide the answers.
        let expected = try SystemOneRequest(json: JSONParser().parse(Self.upstreamExample))
        #expect(request == expected)
        // The same JSON, byte for byte. The library writes the top-level keys in pydantic's field
        // order (state, model, questions), so the README's own key order is not compared.
        let encoder = WireEncoder()
        #expect(try encoder.string(request) == encoder.string(expected))
        #expect(request.json == expected.json)
    }

    @Test func outageSampleIsTheReadmeState() {
        let outage = TriageQuestions.samples.first { $0.name == "Outage report" }
        #expect(
            outage?.text == "Everything is down and we have a demo with our biggest client at noon."
        )
    }
}
