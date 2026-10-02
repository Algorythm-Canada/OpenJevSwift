// A port of the settings of upstream OpenJev's `tests/test_live.py` (razorback16/openjev at
// dcd2094): the module's `URL` and `GATEWAY` and the headers of its `client` fixture. Apache-2.0.
// See THIRD_PARTY.md.

import Foundation
import Testing

/// What the live suite reads from the environment, as upstream's `test_live.py` reads it.
///
/// - `OPENJEV_LIVE_URL`: the server under test, such as `http://127.0.0.1:8080`. Unset or empty,
///   every test skips with a comment naming it, which CI's test log check accepts.
/// - `OPENJEV_LIVE_KEY`, else upstream's `OPENJEV_API_KEY`: the key sent as
///   `Authorization: Bearer <key>`. Unset or empty, no `Authorization` header is sent.
/// - `OPENJEV_ORIGIN_SECRET`: sent as `X-Origin-Secret` when set and not empty.
/// - `OPENJEV_LIVE_GATEWAY=1`: a gateway in front of the server strips `server-timing`, so the
///   suite does not require it. Any other value requires it, as upstream's `== "1"` has it.
struct LiveSettings: Sendable {
    /// The server's base URL, without a trailing slash; request paths start with `/`.
    let baseURL: String
    /// The bearer key, if one is sent.
    let apiKey: String?
    /// The origin secret, if one is sent.
    let originSecret: String?
    /// Whether a gateway strips `server-timing`.
    let gateway: Bool

    /// The variable that names the server.
    static let urlVariable = "OPENJEV_LIVE_URL"

    /// `OPENJEV_LIVE_URL` when it is set and not empty, upstream's `if not URL` skip.
    static let configuredURL: String? = nonEmpty(
        ProcessInfo.processInfo.environment[urlVariable])

    /// Whether a server is named, so the suite runs.
    static var configured: Bool { configuredURL != nil }

    /// The skip comment of every test when no server is named. CI's test log check accepts a
    /// skip whose comment names `OPENJEV_LIVE_URL`.
    static let unsetMessage = Comment(
        rawValue: "OPENJEV_LIVE_URL is unset; set it to a running OpenJev server, such as "
            + "http://127.0.0.1:8080")

    /// The settings of this process's environment, made once.
    static let current: Result<LiveSettings, LiveSettingsError> = Result {
        () throws(LiveSettingsError) in
        try LiveSettings(environment: ProcessInfo.processInfo.environment)
    }

    /// Reads the settings from `environment`.
    ///
    /// - Throws: ``LiveSettingsError`` when `OPENJEV_LIVE_URL` is unset or is not an `http` or
    ///   `https` URL with a host.
    init(environment: [String: String]) throws(LiveSettingsError) {
        guard let text = Self.nonEmpty(environment[Self.urlVariable]) else {
            throw .unset
        }
        guard let components = URLComponents(string: text),
            let scheme = components.scheme?.lowercased(), ["http", "https"].contains(scheme),
            let host = components.host, !host.isEmpty,
            components.query == nil, components.fragment == nil
        else {
            throw .invalidURL(text)
        }
        // httpx joins a base URL's path and a request's path with one slash.
        var base = text
        while base.hasSuffix("/") {
            base.removeLast()
        }
        baseURL = base
        apiKey =
            Self.nonEmpty(environment["OPENJEV_LIVE_KEY"])
            ?? Self.nonEmpty(environment["OPENJEV_API_KEY"])
        originSecret = Self.nonEmpty(environment["OPENJEV_ORIGIN_SECRET"])
        gateway = environment["OPENJEV_LIVE_GATEWAY"] == "1"
    }

    /// `value`, unless it is missing or empty.
    private static func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}

/// Why the suite's settings could not be read.
enum LiveSettingsError: Error, CustomStringConvertible, Equatable {
    /// `OPENJEV_LIVE_URL` is unset or empty.
    case unset
    /// `OPENJEV_LIVE_URL` is not an `http` or `https` URL with a host.
    case invalidURL(String)

    var description: String {
        switch self {
        case .unset:
            return "OPENJEV_LIVE_URL is unset"
        case .invalidURL(let text):
            return "OPENJEV_LIVE_URL is \(text), which is not an http or https URL with a host "
                + "and no query"
        }
    }
}
