// Where openjev-bench sends a request: the engine in this process, or any `/v1/systemone` server.

import Foundation
import OpenJevCore
import OpenJevDiffusionGemma

/// What one request cost.
struct Sample: Sendable {
    /// The caller's time: the engine call, or the HTTP exchange from send to the last byte.
    var milliseconds: Double
    /// The decision's model time (engine) or `server-timing`'s `model` (HTTP).
    var modelMilliseconds: Double?
    /// The billed input tokens.
    var inputTokens: Int?
}

/// Something that answers a request body.
protocol DecisionTarget: Sendable {
    func decide(body: String) async throws -> Sample
}

/// Why a request failed.
struct BenchFailure: Error, CustomStringConvertible {
    var description: String
}

/// The engine over the DiffusionGemma runtime, in this process.
struct EngineTarget: DecisionTarget {
    let engine: DecisionEngine

    func decide(body: String) async throws -> Sample {
        let request = try SystemOneRequest(json: JSONParser().parse(body))
        let clock = ContinuousClock()
        let start = clock.now
        let decision = try await engine.decide(request)
        return Sample(
            milliseconds: milliseconds(clock.now - start),
            modelMilliseconds: milliseconds(decision.modelTime),
            inputTokens: decision.inputTokens)
    }
}

/// A server's `/v1/systemone` over HTTP/1.1, on its own URLSession so concurrent requests get
/// their own connections.
struct HTTPTarget: DecisionTarget {
    let endpoint: URL
    let session: URLSession
    let apiKey: String?

    init(baseURL: URL, apiKey: String?) {
        endpoint = baseURL.appendingPathComponent("v1/systemone")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpMaximumConnectionsPerHost = 64
        configuration.timeoutIntervalForRequest = 600
        session = URLSession(configuration: configuration)
        self.apiKey = apiKey
    }

    func decide(body: String) async throws -> Sample {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "authorization")
        }
        request.httpBody = Data(body.utf8)
        let clock = ContinuousClock()
        let start = clock.now
        let (data, response) = try await session.data(for: request)
        let elapsed = milliseconds(clock.now - start)
        guard let http = response as? HTTPURLResponse else {
            throw BenchFailure(description: "\(endpoint) did not answer over HTTP")
        }
        guard http.statusCode == 200 else {
            throw BenchFailure(
                description: "\(endpoint) answered \(http.statusCode): "
                    + String(decoding: data.prefix(500), as: UTF8.self))
        }
        let timing = (http.value(forHTTPHeaderField: "server-timing")).map(ServerTiming.parse)
        let usage =
            (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["usage"]
            as? [String: Any]
        return Sample(
            milliseconds: elapsed, modelMilliseconds: timing?["model"],
            inputTokens: usage?["input_tokens"] as? Int)
    }
}
