import Foundation

public struct CaptureQuality: Codable, Equatable, Sendable {
    public enum Grade: String, Codable, Sendable {
        case good
        case warn
        case poor
    }

    public var grade: Grade
    public var reasons: [String]

    public init(grade: Grade, reasons: [String] = []) {
        self.grade = grade
        self.reasons = reasons
    }
}
