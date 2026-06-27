import Foundation

public enum DeviceTier: String, Codable, Sendable {
    case lidar = "A_lidar"
    case standard = "B_standard"
    case fallback
}

public enum CaptureFormat: String, Codable, Sendable {
    case hevc
    case prores422lt
}

public enum CaptureOrientation: String, Codable, Sendable {
    case landscape
}
