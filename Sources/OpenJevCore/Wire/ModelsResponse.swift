// The shapes follow upstream OpenJev (razorback16/openjev at dcd2094), `openjev/config.py`,
// `MODELS` and `ENCODER_MODELS`, and the `/v1/models` and `/health` routes in `openjev/api.py`.
// Apache-2.0. See THIRD_PARTY.md.

/// One entry of the `GET /v1/models` listing.
///
/// `Codable` is offered for convenience; the wire form is ``json``, written by ``WireEncoder``.
public struct ModelInfo: Sendable, Hashable, Codable, WireEncodable {
    /// The name a request uses, such as `openjev-latest`.
    public var name: String
    /// What the model is. Empty for a routed model upstream does not describe.
    public var description: String
    /// The release date as `YYYY-MM-DD`, or empty.
    public var releaseDate: String

    /// Creates an entry.
    public init(name: String, description: String, releaseDate: String) {
        self.name = name
        self.description = description
        self.releaseDate = releaseDate
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case description
        case releaseDate = "release_date"
    }

    /// `{"name", "description", "release_date"}`.
    public var json: JSONValue {
        [
            "name": .string(name),
            "description": .string(description),
            "release_date": .string(releaseDate),
        ]
    }

    /// Decodes an entry, requiring its exact key set.
    public init(json: JSONValue) throws(WireDecodingError) {
        try self.init(json: json, path: [])
    }

    init(json: JSONValue, path: [LocComponent]) throws(WireDecodingError) {
        let object = try json.requireObject(at: path)
        try object.requireKeys(["name", "description", "release_date"], at: path)
        name = try object.require("name", at: path).requireString(at: path + ["name"])
        description = try object.require("description", at: path)
            .requireString(at: path + ["description"])
        releaseDate = try object.require("release_date", at: path)
            .requireString(at: path + ["release_date"])
    }
}

/// The `GET /v1/models` response body, `{"models": [...]}`.
public struct ModelsResponse: Sendable, Hashable, WireEncodable {
    /// The listed models, in upstream's order.
    public var models: [ModelInfo]

    /// Creates a listing.
    public init(models: [ModelInfo]) {
        self.models = models
    }

    /// `{"models": [...]}`.
    public var json: JSONValue {
        ["models": .array(models.map(\.json))]
    }

    /// Decodes a listing, requiring its exact key set.
    public init(json: JSONValue) throws(WireDecodingError) {
        let object = try json.requireObject(at: [])
        try object.requireKeys(["models"], at: [])
        let entries = try object.require("models", at: []).requireArray(at: ["models"])
        var models: [ModelInfo] = []
        for (index, entry) in entries.enumerated() {
            models.append(try ModelInfo(json: entry, path: ["models", .index(index)]))
        }
        self.models = models
    }
}

/// The `GET /health` response body, `{"status": "ok"}`.
public struct HealthResponse: Sendable, Hashable, WireEncodable {
    /// The status text. Upstream only ever sends `ok`.
    public var status: String

    /// Creates a health body.
    public init(status: String) {
        self.status = status
    }

    /// The body upstream sends.
    public static let ok = HealthResponse(status: "ok")

    /// `{"status": ...}`.
    public var json: JSONValue {
        ["status": .string(status)]
    }

    /// Decodes a health body, requiring its exact key set.
    public init(json: JSONValue) throws(WireDecodingError) {
        let object = try json.requireObject(at: [])
        try object.requireKeys(["status"], at: [])
        status = try object.require("status", at: []).requireString(at: ["status"])
    }
}
