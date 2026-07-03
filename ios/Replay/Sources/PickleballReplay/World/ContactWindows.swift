import Foundation

/// Swift mirror of `contact_windows.json`, mirroring `ContactWindows` /
/// `ContactWindowEvent` in `web/replay/src/viewerData.ts`. Used both for
/// timeline markers and for gating which player gets the MESH tier at a
/// given time (mesh only renders inside an active contact window for that
/// player).
public struct ContactWindows: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var events: [Event]

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case events
    }

    public init(schemaVersion: Int, events: [Event]) {
        self.schemaVersion = schemaVersion
        self.events = events
    }

    public struct Event: Codable, Equatable, Sendable {
        public var type: String
        public var t: Double
        public var playerID: Int?
        public var confidence: Double
        public var window: Window

        private enum CodingKeys: String, CodingKey {
            case type, t
            case playerID = "player_id"
            case confidence, window
        }

        public init(type: String, t: Double, playerID: Int?, confidence: Double, window: Window) {
            self.type = type
            self.t = t
            self.playerID = playerID
            self.confidence = confidence
            self.window = window
        }

        public struct Window: Codable, Equatable, Sendable {
            public var t0: Double
            public var t1: Double
            public var importance: Double

            public init(t0: Double, t1: Double, importance: Double) {
                self.t0 = t0
                self.t1 = t1
                self.importance = importance
            }
        }

        public func contains(_ timeSeconds: Double) -> Bool {
            window.t0 <= timeSeconds && timeSeconds <= window.t1
        }
    }
}

extension ContactWindows {
    public static func load(from url: URL, decoder: JSONDecoder = JSONDecoder()) throws -> ContactWindows {
        let data = try Data(contentsOf: url)
        return try decoder.decode(ContactWindows.self, from: data)
    }

    /// Contact events active (`type == "contact"`) at `timeSeconds`, mirroring
    /// `contactEventsForTime` in `web/replay/src/viewerData.ts`.
    public func activeContactEvents(at timeSeconds: Double) -> [Event] {
        events.filter { $0.type == "contact" && $0.contains(timeSeconds) }
    }
}
