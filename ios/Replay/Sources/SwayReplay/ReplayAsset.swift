import Foundation

public struct ReplayAsset: Codable, Equatable, Sendable {
    public var usdzURL: URL?
    public var glbURL: URL?
    public var durationSeconds: Double

    public init(usdzURL: URL? = nil, glbURL: URL? = nil, durationSeconds: Double) {
        self.usdzURL = usdzURL
        self.glbURL = glbURL
        self.durationSeconds = durationSeconds
    }
}
