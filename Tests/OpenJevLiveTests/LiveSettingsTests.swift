import Testing

/// ``LiveSettings`` from environments given here, so these run everywhere, CI included, where
/// `OPENJEV_LIVE_URL` is unset and the live tests skip before any setting is read.
@Suite("Live suite settings")
struct LiveSettingsTests {
    /// The environment of a server at `url`, with `extra` on top.
    private func environment(
        _ extra: [String: String] = [:], url: String = "http://127.0.0.1:8080"
    ) -> [String: String] {
        ["OPENJEV_LIVE_URL": url].merging(extra) { $1 }
    }

    @Test(
        "An unset or empty OPENJEV_LIVE_URL names no server",
        arguments: [[:], ["OPENJEV_LIVE_URL": ""]])
    func unset(_ environment: [String: String]) {
        #expect(throws: LiveSettingsError.unset) { try LiveSettings(environment: environment) }
    }

    @Test(
        "An http or https URL with a host is accepted, without its trailing slashes",
        arguments: [
            ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
            ("http://127.0.0.1:8080/", "http://127.0.0.1:8080"),
            ("https://openjev.example/prefix//", "https://openjev.example/prefix"),
            ("HTTP://localhost:8080", "HTTP://localhost:8080"),
        ])
    func accepted(_ url: String, _ base: String) throws {
        #expect(try LiveSettings(environment: environment(url: url)).baseURL == base)
    }

    @Test(
        "Any other OPENJEV_LIVE_URL is refused, so the tests fail instead of skipping",
        arguments: [
            "localhost:8080", "127.0.0.1:8080", "ftp://127.0.0.1:8080", "http://",
            "http://127.0.0.1:8080?debug=1", "http://127.0.0.1:8080#top", "not a url",
        ])
    func refused(_ url: String) {
        #expect(throws: LiveSettingsError.invalidURL(url)) {
            try LiveSettings(environment: environment(url: url))
        }
    }

    @Test("OPENJEV_LIVE_KEY wins over OPENJEV_API_KEY, and an empty value counts as unset")
    func keyPrecedence() throws {
        let both = ["OPENJEV_LIVE_KEY": "live", "OPENJEV_API_KEY": "server"]
        #expect(try LiveSettings(environment: environment(both)).apiKey == "live")
        let upstreams = ["OPENJEV_API_KEY": "server"]
        #expect(try LiveSettings(environment: environment(upstreams)).apiKey == "server")
        let emptyLive = ["OPENJEV_LIVE_KEY": "", "OPENJEV_API_KEY": "server"]
        #expect(try LiveSettings(environment: environment(emptyLive)).apiKey == "server")
        let empty = ["OPENJEV_LIVE_KEY": "", "OPENJEV_API_KEY": ""]
        #expect(try LiveSettings(environment: environment(empty)).apiKey == nil)
        #expect(try LiveSettings(environment: environment()).apiKey == nil)
    }

    @Test("OPENJEV_ORIGIN_SECRET is sent when set and not empty")
    func originSecret() throws {
        let secret = ["OPENJEV_ORIGIN_SECRET": "front"]
        #expect(try LiveSettings(environment: environment(secret)).originSecret == "front")
        let empty = ["OPENJEV_ORIGIN_SECRET": ""]
        #expect(try LiveSettings(environment: environment(empty)).originSecret == nil)
        #expect(try LiveSettings(environment: environment()).originSecret == nil)
    }

    @Test(
        "Only OPENJEV_LIVE_GATEWAY=1 drops the server-timing check, as upstream compares it",
        arguments: [("1", true), ("0", false), ("true", false), ("yes", false), ("", false)])
    func gateway(_ value: String, _ dropped: Bool) throws {
        let flag = ["OPENJEV_LIVE_GATEWAY": value]
        #expect(try LiveSettings(environment: environment(flag)).gateway == dropped)
    }

    @Test("Without OPENJEV_LIVE_GATEWAY the server-timing check applies")
    func gatewayUnset() throws {
        #expect(try LiveSettings(environment: environment()).gateway == false)
    }
}
