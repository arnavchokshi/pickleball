import Foundation

public struct OnDevicePoseTrack: Codable, Equatable, Sendable {
    public var relativePath: String
    public var fps: Double
    public var previewOnly: Bool

    public init(relativePath: String, fps: Double, previewOnly: Bool = true) {
        self.relativePath = relativePath
        self.fps = fps
        self.previewOnly = previewOnly
    }
}
