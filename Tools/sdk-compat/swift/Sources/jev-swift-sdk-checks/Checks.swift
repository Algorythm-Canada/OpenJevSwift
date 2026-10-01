// Drives NSStudent's JevSwiftSDK through one scenario of the SDK compatibility suite and prints
// what it observed as JSON on standard output, as python_checks.py and checks.mjs do for the
// official SDKs. run.py starts it once per scenario:
//
//     jev-swift-sdk-checks quickstart
//
// The client reads TYPESAFE_BASE_URL and TYPESAFE_API_KEY, as JevConfiguration.fromEnvironment
// documents. SDK_COMPAT_OVERLOADED_URL is the server that refuses every request with 529. The SDK
// sends state, model and questions only, so the samples scenario does not apply to it.

import Foundation
import JevSwiftSDK

@main
enum Checks {
    /// Jev's quickstart request.
    static let state: JSONValue =
        "Hi, I've been trying to connect my Stripe account but keep getting a 403 error."
    static let department = ChoiceQuestion<ChoiceOption>(
        id: "department", instructions: "Which team should handle this",
        criteria: [
            "billing": "Payment or subscription issues",
            "technical": "Bugs or integration problems",
            "sales": "Pricing or account questions",
        ])
    static let frustration = ScoreQuestion(
        id: "frustration", instructions: "How frustrated the customer appears",
        levels: ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"])
    static let urgent = NoulQuestion(
        id: "is_urgent", instructions: "The message conveys urgency or time-sensitivity")
    static let questions = [
        AnyJevQuestion(department), AnyJevQuestion(frustration), AnyJevQuestion(urgent),
    ]

    static func main() async {
        let name = CommandLine.arguments.dropFirst().first ?? ""
        do {
            let observed: [String: Any]
            switch name {
            case "quickstart": observed = try await answers(model: nil)
            case "models": observed = try await models()
            case "wrong_key": observed = try await failure(client(apiKey: "sk-wrong"))
            case "overloaded": observed = try await failure(client(baseURL: overloadedURL()))
            case "routed": observed = try await answers(model: "laya-1.0")
            default:
                fail(
                    "unknown scenario \(name); use quickstart, models, wrong_key, overloaded, routed"
                )
            }
            let data = try JSONSerialization.data(withJSONObject: observed, options: [.sortedKeys])
            FileHandle.standardOutput.write(data)
        } catch {
            fail("\(name): \(error)")
        }
    }

    /// A client from the environment, with the key or the base URL replaced when given.
    static func client(apiKey: String? = nil, baseURL: URL? = nil) throws -> JevClient {
        JevClient(
            configuration: try JevConfiguration.fromEnvironment(apiKey: apiKey, baseURL: baseURL))
    }

    static func overloadedURL() throws -> URL {
        guard let text = ProcessInfo.processInfo.environment["SDK_COMPAT_OVERLOADED_URL"],
            let url = URL(string: text)
        else {
            fail("SDK_COMPAT_OVERLOADED_URL is not set")
        }
        return url
    }

    /// The quickstart's answers, typed by the SDK, as run.py compares them.
    static func answers(model: String?) async throws -> [String: Any] {
        let response = try await client().evaluate(
            state: state, questions: questions, model: model)
        let choice = try response.answer(for: department)
        let score = try response.answer(for: frustration)
        let noul = try response.answer(for: urgent)
        var probabilities: [String: Double] = [:]
        for (option, probability) in choice.probabilities {
            probabilities[option.rawValue] = probability
        }
        var legend: [String: Any] = [:]
        for (key, value) in score.legend {
            if case .string(let text) = value { legend[key] = text }
        }
        return [
            "model": response.model,
            "usage": [
                "input_tokens": response.usage.inputTokens,
                "output_tokens": response.usage.outputTokens,
            ],
            "department": [
                "choice": choice.choice.rawValue, "probabilities": probabilities,
                "confidence": choice.confidence,
            ],
            "frustration": [
                "score": score.score, "legend": legend, "probabilities": score.probabilities,
            ],
            "is_urgent": ["noul": noul.noul],
        ]
    }

    static func models() async throws -> [String: Any] {
        let models = try await client().listModels()
        return [
            "models": models.map {
                ["name": $0.name, "description": $0.description, "release_date": $0.releaseDate]
            }
        ]
    }

    /// The quickstart through `client`, which must fail with an HTTP error, as run.py compares it.
    static func failure(_ client: JevClient) async throws -> [String: Any] {
        do {
            _ = try await client.evaluate(state: state, questions: questions)
        } catch let error as JevError {
            guard let details = error.httpDetails else {
                fail("expected an HTTP error, got \(error)")
            }
            return [
                "error": "http", "api_error": true, "status": details.statusCode,
                "request_id": details.requestID ?? NSNull(),
                "message": details.message ?? "",
            ]
        }
        fail("expected an HTTP error, but the request succeeded")
    }

    static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        exit(1)
    }
}
