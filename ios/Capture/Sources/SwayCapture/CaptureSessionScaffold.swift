import Foundation
import SwayCore

public struct CaptureSessionScaffold: Equatable, Sendable {
    public var mode: CaptureMode
    public var format: CaptureFormat
    public var orientation: CaptureOrientation
    public var requestedFPS: Int
    public var resolution: CaptureResolution
    public var hdrEnabled: Bool
    public var cinematicEnabled: Bool
    public var liveFramesEnabled: Bool

    public init(mode: CaptureMode) {
        self.init(mode: mode, deviceTier: .standard, capabilities: .hevcOnly)
    }

    public init(mode: CaptureMode, deviceTier: DeviceTier, capabilities: CaptureCodecCapabilities) {
        let policy = CapturePolicy.recommended(for: mode, deviceTier: deviceTier, capabilities: capabilities)
        self.mode = mode
        self.format = policy.codec
        self.orientation = policy.orientation
        self.requestedFPS = policy.fps
        self.resolution = policy.resolution
        self.hdrEnabled = policy.hdrEnabled
        self.cinematicEnabled = policy.cinematicEnabled
        self.liveFramesEnabled = policy.liveFramesEnabled
    }
}
