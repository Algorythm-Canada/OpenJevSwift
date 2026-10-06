import OpenJevCore

/// The three questions of upstream OpenJev's README example ("Try it"), in its order, and the
/// sample messages the app offers.
public enum TriageQuestions {
    /// The model name TypeSafe's SDK sends by default, which every OpenJev backend accepts.
    public static let model = "jev-latest"

    /// `urgent`, `team` and `tone`, exactly as upstream's README sends them.
    public static let all: OrderedMap<Question> = [
        "urgent": .noul(
            instructions: "Does the customer need a reply within the hour?", criteria: nil),
        "team": .choice(
            instructions: "Which team should handle it?",
            criteria: [
                "outage": "service down",
                "billing": "charges, refunds",
                "feature": "requests, how-to",
            ]),
        "tone": .score(
            instructions: "How upset is the customer?",
            criteria: ["calm", "annoyed", "furious"]),
    ]

    /// A message to try, with the name the sample picker shows.
    public struct Sample: Identifiable, Hashable, Sendable {
        public var name: String
        public var text: String
        public var id: String { name }
    }

    /// A support ticket, an outage report (upstream's README state) and a billing complaint.
    public static let samples: [Sample] = [
        Sample(
            name: "Support ticket",
            text: "Hi! Is there a way to export my invoices as a CSV file? "
                + "If not, could you add it at some point? No rush, thanks."),
        Sample(
            name: "Outage report",
            text: "Everything is down and we have a demo with our biggest client at noon."),
        Sample(
            name: "Billing complaint",
            text: "You charged my card twice this month and nobody has answered my last two "
                + "emails. Refund me today or I am cancelling."),
    ]
}
