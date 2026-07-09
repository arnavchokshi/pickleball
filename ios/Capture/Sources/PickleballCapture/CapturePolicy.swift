import Foundation
import PickleballCore

public enum CaptureResolution: String, Codable, Equatable, Sendable {
    case hd720p
    case hd1080p
    case uhd4K

    public var dimensions: [Int] {
        dimensions(for: .landscape)
    }

    public func dimensions(for orientation: CaptureOrientation) -> [Int] {
        switch self {
        case .hd720p:
            return orientation == .portrait ? [720, 1280] : [1280, 720]
        case .hd1080p:
            return orientation == .portrait ? [1080, 1920] : [1920, 1080]
        case .uhd4K:
            return orientation == .portrait ? [2160, 3840] : [3840, 2160]
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
        capabilities: CaptureCodecCapabilities,
        orientation: CaptureOrientation = .landscape
    ) -> CapturePolicy {
        switch mode {
        case .standard60:
            return CapturePolicy(mode: mode, fps: 60, resolution: .hd1080p, codec: .hevc, orientation: orientation)
        case .swing120:
            return CapturePolicy(mode: mode, fps: 120, resolution: .hd1080p, codec: .hevc, orientation: orientation)
        case .ballPhysics240:
            return CapturePolicy(mode: mode, fps: 240, resolution: .hd720p, codec: .hevc, orientation: orientation)
        case .quality4K60:
            if deviceTier == .lidar && capabilities.supportsProRes422LT {
                return CapturePolicy(mode: mode, fps: 60, resolution: .uhd4K, codec: .prores422lt, orientation: orientation)
            }

            return CapturePolicy(mode: mode, fps: 60, resolution: .hd1080p, codec: .hevc, orientation: orientation)
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

public enum CaptureFocusExposureMode: Equatable, Sendable {
    case continuous
    case locked
    case unavailable
}

public struct CaptureFocusExposurePlan: Equatable, Sendable {
    public var focus: CaptureFocusExposureMode
    public var exposure: CaptureFocusExposureMode

    public init(focus: CaptureFocusExposureMode, exposure: CaptureFocusExposureMode) {
        self.focus = focus
        self.exposure = exposure
    }
}

public enum CaptureFocusExposurePolicy {
    public static func preview(
        supportsContinuousFocus: Bool,
        supportsContinuousExposure: Bool
    ) -> CaptureFocusExposurePlan {
        CaptureFocusExposurePlan(
            focus: supportsContinuousFocus ? .continuous : .unavailable,
            exposure: supportsContinuousExposure ? .continuous : .unavailable
        )
    }

    public static func recording(
        supportsLockedFocus: Bool,
        supportsContinuousFocus: Bool,
        supportsLockedExposure: Bool,
        supportsContinuousExposure: Bool
    ) -> CaptureFocusExposurePlan {
        CaptureFocusExposurePlan(
            focus: recordingMode(
                supportsLocked: supportsLockedFocus,
                supportsContinuous: supportsContinuousFocus
            ),
            exposure: recordingMode(
                supportsLocked: supportsLockedExposure,
                supportsContinuous: supportsContinuousExposure
            )
        )
    }

    private static func recordingMode(
        supportsLocked: Bool,
        supportsContinuous: Bool
    ) -> CaptureFocusExposureMode {
        if supportsLocked {
            return .locked
        }
        if supportsContinuous {
            return .continuous
        }
        return .unavailable
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

public enum CaptureOrientationPolicy {
    public static func captureOrientation(for deviceOrientation: CaptureDeviceOrientation) -> CaptureOrientation {
        switch deviceOrientation {
        case .portrait, .portraitUpsideDown:
            return .portrait
        case .landscapeRight, .landscapeLeft:
            return .landscape
        }
    }

    public static func rotationAngleDegrees(for deviceOrientation: CaptureDeviceOrientation) -> Int {
        switch deviceOrientation {
        case .landscapeRight:
            return 90
        case .landscapeLeft:
            return 270
        case .portrait:
            return 0
        case .portraitUpsideDown:
            return 180
        }
    }
}
