import Foundation
import SwayCore

public enum GuidanceFlag: String, Codable, Sendable {
    case framing
    case tracking
    case exposure
    case blur
    case level
    case shake
}

public struct GuidanceState: Codable, Equatable, Sendable {
    public var flags: [GuidanceFlag]
    public var captureQuality: CaptureQuality

    public init(flags: [GuidanceFlag] = [], captureQuality: CaptureQuality) {
        self.flags = flags
        self.captureQuality = captureQuality
    }
}
