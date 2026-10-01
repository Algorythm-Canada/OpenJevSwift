// A port of the `client` and `models` fixtures and the `ask` helper of upstream OpenJev's
// `tests/test_live.py` (razorback16/openjev at dcd2094). Apache-2.0. See THIRD_PARTY.md.

import Foundation
import OpenJevCore
import Testing

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// One HTTP exchange's answer: the status, the header fields and the body.
struct LiveResponse: Sendable {
    /// The status code.
    let status: Int
    /// The header fields, under lowercase names.
    let headers: [String: String]
    /// The body bytes.
    let body: Data

    /// The body as text, for failure messages, as upstream's `r.text` shows it.
    var text: String { String(decoding: body, as: UTF8.self) }

    /// The value of the header field `name`, matched without regard to case.
    func header(_ name: String) -> String? {
        headers[name.lowercased()]
    }

    /// The body, parsed.
    func json() throws -> JSONValue {
        try JSONParser().parse(body)
    }
}

/// A failed exchange or an answer the suite cannot read.
struct LiveFailure: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

/// Upstream's `httpx.Client(base_url=URL, headers=headers, timeout=300)` on URLSession: the
/// suite's `Authorization` and `X-Origin-Secret` headers on every request, httpx's 300-second
/// timeouts, and room for `test_concurrent_reads`' 32 requests in flight.
final class LiveClient: Sendable {
    /// httpx's `timeout=300` sets 300 s for connecting, for each read and for each write, and no
    /// deadline for the whole exchange. URLSession's request timeout is the same kind of limit:
    /// the longest wait for the connection or the next bytes.
    static let timeout: TimeInterval = 300
    /// The width of upstream's `ThreadPoolExecutor(32)`. URLSession opens at most 6
    /// connections to one host unless told otherwise, which would cap the requests in flight.
    static let maximumConnections = 32

    /// The settings the client was made from.
    let settings: LiveSettings
    private let session: URLSession

    /// The suite's client, made once from the environment.
    static let shared: Result<LiveClient, LiveSettingsError> = LiveSettings.current.map {
        LiveClient(settings: $0)
    }

    /// The suite's client, or the reason there is none.
    static func make() throws -> LiveClient {
        try shared.get()
    }

    /// A client for the server `settings` name.
    init(settings: LiveSettings) {
        self.settings = settings
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = Self.timeout
        configuration.httpMaximumConnectionsPerHost = Self.maximumConnections
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        configuration.httpShouldSetCookies = false
        session = URLSession(configuration: configuration)
    }

    /// `GET path`.
    func get(_ path: String) async throws -> LiveResponse {
        try await send(request(path, method: "GET"))
    }

    /// `POST path` with `body` as JSON, as httpx's `json=` sends it.
    func post(_ path: String, json body: JSONValue) async throws -> LiveResponse {
        var request = try request(path, method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(try PythonJSONWriter(options: .compact).bytes(body))
        return try await send(request)
    }

    /// A request for `path` with the suite's headers.
    private func request(_ path: String, method: String) throws -> URLRequest {
        guard let url = URL(string: settings.baseURL + path) else {
            throw LiveFailure("\(settings.baseURL + path) is not a URL")
        }
        var request = URLRequest(
            url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: Self.timeout)
        request.httpMethod = method
        if let key = settings.apiKey {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        if let secret = settings.originSecret {
            request.setValue(secret, forHTTPHeaderField: "X-Origin-Secret")
        }
        return request
    }

    /// Sends `request` and waits for the whole answer.
    private func send(_ request: URLRequest) async throws -> LiveResponse {
        let label = "\(request.httpMethod ?? "GET") \(request.url?.absoluteString ?? "")"
        return try await withCheckedThrowingContinuation { continuation in
            let task = session.dataTask(with: request) { data, response, error in
                if let error {
                    continuation.resume(throwing: LiveFailure("\(label) failed: \(error)"))
                    return
                }
                guard let http = response as? HTTPURLResponse else {
                    continuation.resume(throwing: LiveFailure("\(label) got no HTTP response"))
                    return
                }
                var headers: [String: String] = [:]
                for (name, value) in http.allHeaderFields {
                    headers["\(name)".lowercased()] = "\(value)"
                }
                continuation.resume(
                    returning: LiveResponse(
                        status: http.statusCode, headers: headers, body: data ?? Data()))
            }
            task.resume()
        }
    }
}

/// Upstream's module-scoped `models` fixture: the names `GET /v1/models` lists, asked once per
/// test process and shared by every test that depends on them.
actor ModelListing {
    /// The process's listing.
    static let shared = ModelListing()

    private var names: Task<Set<String>, any Error>?

    /// The listed names. A listing that cannot be read throws, so the tests that depend on it
    /// fail, as a failed fixture fails them in pytest.
    func names(from client: LiveClient) async throws -> Set<String> {
        if let names {
            return try await names.value
        }
        let task = Task { try await Self.fetch(from: client) }
        names = task
        return try await task.value
    }

    /// Whether the server lists `model`. False without asking when no server is named.
    static func lists(_ model: String) async throws -> Bool {
        guard LiveSettings.configured else { return false }
        return try await shared.names(from: LiveClient.make()).contains(model)
    }

    private static func fetch(from client: LiveClient) async throws -> Set<String> {
        let response = try await client.get("/v1/models")
        guard response.status == 200 else {
            throw LiveFailure("GET /v1/models answered \(response.status): \(response.text)")
        }
        guard let models = try response.json()["models"]?.arrayValue else {
            throw LiveFailure("GET /v1/models has no models array: \(response.text)")
        }
        return Set(models.compactMap { $0["name"]?.stringValue })
    }
}
