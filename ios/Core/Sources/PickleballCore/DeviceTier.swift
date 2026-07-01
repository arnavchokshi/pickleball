import Foundation

// DeviceTier is a capture capability label, not the canonical product tier
// split. LiDAR is a near-field (~5 m) bonus only; ON-DEVICE LIVE works on the
// standard vision baseline, and SERVER OFFLINE owns deep reconstruction.
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
    case portrait
    case landscape
}

public enum CaptureDeviceOrientation: String, Codable, Sendable {
    case landscapeRight
    case landscapeLeft
    case portrait
    case portraitUpsideDown
}
