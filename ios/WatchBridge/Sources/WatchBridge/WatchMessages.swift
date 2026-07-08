import Foundation

/// Commands the watch app sends to the phone. Codable via a `type`
/// discriminator string so instances round-trip through the
/// `[String: Any]` dictionaries WCSession moves (not raw `Data`).
public enum WatchCommand: Codable, Equatable, Sendable {
    case startRecording
    case stopRecording
    case requestState

    private enum CodingKeys: String, CodingKey {
        case type
    }

    private enum Kind: String, Codable {
        case startRecording
        case stopRecording
        case requestState
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .type)
        switch kind {
        case .startRecording:
            self = .startRecording
        case .stopRecording:
            self = .stopRecording
        case .requestState:
            self = .requestState
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .startRecording:
            try container.encode(Kind.startRecording, forKey: .type)
        case .stopRecording:
            try container.encode(Kind.stopRecording, forKey: .type)
        case .requestState:
            try container.encode(Kind.requestState, forKey: .type)
        }
    }
}

/// The phone's recording status, pushed to the watch so it can render a
/// live Start/Stop control without polling.
public struct PhoneRecordingState: Codable, Equatable, Sendable {
    public var isRecording: Bool
    public var canRecord: Bool
    public var clipName: String?
    public var elapsedSeconds: Double

    public init(
        isRecording: Bool,
        canRecord: Bool,
        clipName: String? = nil,
        elapsedSeconds: Double = 0
    ) {
        self.isRecording = isRecording
        self.canRecord = canRecord
        self.clipName = clipName
        self.elapsedSeconds = elapsedSeconds
    }
}

/// Dictionary encode/decode helpers, since `WCSession.sendMessage` and
/// `updateApplicationContext` move `[String: Any]` payloads rather than
/// `Data`. Both directions are lossless for well-formed input and
/// defensive (return `nil`, never throw/crash) on malformed input.
public enum WatchMessageCoding {
    private static let encoder: JSONEncoder = JSONEncoder()
    private static let decoder: JSONDecoder = JSONDecoder()

    public static func toMessage<T: Encodable>(_ value: T) -> [String: Any] {
        guard let data = try? encoder.encode(value),
              let object = try? JSONSerialization.jsonObject(with: data, options: []),
              let dictionary = object as? [String: Any] else {
            return [:]
        }
        return dictionary
    }

    public static func decode<T: Decodable>(_ type: T.Type, message: [String: Any]) -> T? {
        guard JSONSerialization.isValidJSONObject(message),
              let data = try? JSONSerialization.data(withJSONObject: message, options: []) else {
            return nil
        }
        return try? decoder.decode(type, from: data)
    }
}

extension WatchCommand {
    public func toMessage() -> [String: Any] {
        WatchMessageCoding.toMessage(self)
    }

    public init?(message: [String: Any]) {
        guard let decoded = WatchMessageCoding.decode(WatchCommand.self, message: message) else {
            return nil
        }
        self = decoded
    }
}

extension PhoneRecordingState {
    public func toMessage() -> [String: Any] {
        WatchMessageCoding.toMessage(self)
    }

    public init?(message: [String: Any]) {
        guard let decoded = WatchMessageCoding.decode(PhoneRecordingState.self, message: message) else {
            return nil
        }
        self = decoded
    }
}
