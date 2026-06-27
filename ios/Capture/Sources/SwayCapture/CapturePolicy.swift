import Foundation
import SwayCore

public enum CaptureResolution: String, Codable, Equatable, Sendable {
    case hd720p
    case hd1080p
    case uhd4K

    public var dimensions: [Int] {
        switch self {
        case .hd720p:
            return [1280, 720]
        case .hd1080p:
            return [1920, 1080]
        case .uhd4K:
            return [3840, 2160]
        }
    }
}

public struct CaptureCodecCapabilities: Equatable, Sendable {
    public var supportsHEVC: Bool
    public var supportsProRes422LT: Bool

    public init(supportsHEVC: Bool, supportsProRes422LT: Bool) {
        self.supportsHEVC = supportsHEVC
        self.supportsProRes422LT = supportsProRes422LT
    }

    public static let allCodecs = CaptureCodecCapabilities(supportsHEVC: true, supportsProRes422LT: true)
    public static let hevcOnly = CaptureCodecCapabilities(supportsHEVC: true, supportsProRes422LT: false)
}

public struct CapturePolicy: Equatable, Sendable {
    public var mode: CaptureMode
    public var fps: Int
    public var resolution: CaptureResolution
    public var codec: CaptureFormat
    public var orientation: CaptureOrientation
    public var hdrEnabled: Bool
    public var cinematicEnabled: Bool
    public var liveFramesEnabled: Bool

    public init(
        mode: CaptureMode,
        fps: Int,
        resolution: CaptureResolution,
        codec: CaptureFormat,
        orientation: CaptureOrientation = .landscape,
        hdrEnabled: Bool = false,
        cinematicEnabled: Bool = false,
        liveFramesEnabled: Bool = true
    ) {
        self.mode = mode
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        self.orientation = orientation
        self.hdrEnabled = hdrEnabled
        self.cinematicEnabled = cinematicEnabled
        self.liveFramesEnabled = liveFramesEnabled
    }

    public static func recommended(
        for mode: CaptureMode,
        deviceTier: DeviceTier,
        capabilities: CaptureCodecCapabilities
    ) -> CapturePolicy {
        switch mode {
        case .standard60:
            return CapturePolicy(mode: mode, fps: 60, resolution: .hd1080p, codec: .hevc)
        case .swing120:
            return CapturePolicy(mode: mode, fps: 120, resolution: .hd1080p, codec: .hevc)
        case .ballPhysics240:
            return CapturePolicy(mode: mode, fps: 240, resolution: .hd720p, codec: .hevc)
        case .quality4K60:
            if deviceTier == .lidar && capabilities.supportsProRes422LT {
                return CapturePolicy(mode: mode, fps: 60, resolution: .uhd4K, codec: .prores422lt)
            }

            return CapturePolicy(mode: mode, fps: 60, resolution: .hd1080p, codec: .hevc)
        }
    }
}

public struct ISOClampBounds: Equatable, Sendable {
    public var min: Double
    public var max: Double

    public init(min: Double, max: Double) {
        self.min = Swift.min(min, max)
        self.max = Swift.max(min, max)
    }
}

public enum LockedCapturePolicy {
    public static let fastestAllowedShutterSeconds = 1.0 / 1_000.0
    public static let slowestAllowedShutterSeconds = 1.0 / 500.0

    public static func request(
        requestedShutterSeconds: Double,
        requestedISO: Double,
        requestedFocusLensPosition: Double,
        isoBounds: ISOClampBounds
    ) -> LockedCapture {
        LockedCapture(
            exposureS: clamp(
                requestedShutterSeconds,
                min: fastestAllowedShutterSeconds,
                max: slowestAllowedShutterSeconds
            ),
            iso: clamp(requestedISO, min: isoBounds.min, max: isoBounds.max),
            focus: clamp(requestedFocusLensPosition, min: 0.0, max: 1.0),
            wbLocked: true
        )
    }

    private static func clamp(_ value: Double, min lowerBound: Double, max upperBound: Double) -> Double {
        Swift.max(lowerBound, Swift.min(upperBound, value))
    }
}

public enum CaptureDeviceOrientation: String, Codable, Sendable {
    case landscapeRight
    case landscapeLeft
    case portrait
    case portraitUpsideDown
}

public enum CaptureOrientationPolicy {
    public static func rotationAngleDegrees(for deviceOrientation: CaptureDeviceOrientation) -> Int? {
        switch deviceOrientation {
        case .landscapeRight:
            return 90
        case .landscapeLeft:
            return 270
        case .portrait, .portraitUpsideDown:
            return nil
        }
    }
}
