import Foundation

public struct ManualCourtTaps: Codable, Equatable, Sendable {
    public var imagePoints: [[Double]]

    public init(imagePoints: [[Double]]) {
        self.imagePoints = imagePoints
    }
}
