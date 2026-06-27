import Foundation
import SwayCore

public enum ARKitSetupTrackingState: String, Codable, Equatable, Sendable {
    case normal
    case limited
    case unavailable
}

public struct ARKitSetupPassSidecar: Codable, Equatable, Sendable {
    public var intrinsics: CameraIntrinsics
    public var cameraPose: RigidPose
    public var courtPlane: Plane
    public var trackingState: ARKitSetupTrackingState
    public var timestampS: Double?

    public init(
        intrinsics: CameraIntrinsics,
        cameraPose: RigidPose,
        courtPlane: Plane,
        trackingState: ARKitSetupTrackingState,
        timestampS: Double? = nil
    ) {
        self.intrinsics = intrinsics
        self.cameraPose = cameraPose
        self.courtPlane = courtPlane
        self.trackingState = trackingState
        self.timestampS = timestampS
    }

    public func calibrationSeed() -> CalibrationSeed {
        CalibrationSeed(
            intrinsics: intrinsics,
            arkitCameraPose: cameraPose,
            courtPlane: courtPlane
        )
    }

    private enum CodingKeys: String, CodingKey {
        case intrinsics
        case cameraPose = "camera_pose"
        case courtPlane = "court_plane"
        case trackingState = "tracking_state"
        case timestampS = "timestamp_s"
    }
}

public struct CalibrationSidecarMetadata: Equatable, Sendable {
    public var deviceTier: DeviceTier
    public var deviceModel: String
    public var fps: Int
    public var format: CaptureFormat
    public var locked: LockedCapture
    public var gravity: [Double]
    public var lidarDepthRefs: [String]
    public var ondevicePoseTrack: String?

    public init(
        deviceTier: DeviceTier,
        deviceModel: String,
        fps: Int,
        format: CaptureFormat,
        locked: LockedCapture,
        gravity: [Double],
        lidarDepthRefs: [String] = [],
        ondevicePoseTrack: String? = nil
    ) {
        self.deviceTier = deviceTier
        self.deviceModel = deviceModel
        self.fps = fps
        self.format = format
        self.locked = locked
        self.gravity = gravity
        self.lidarDepthRefs = lidarDepthRefs
        self.ondevicePoseTrack = ondevicePoseTrack
    }
}

public enum CalibrationSidecarPackager {
    public enum PackagingError: Error, Equatable, Sendable {
        case invalidManualTaps(ManualCourtTaps.ValidationError)
        case invalidSeed(CalibrationSeedValidationReport)
    }

    public static func package(
        seed: CalibrationSeed,
        imageSize: ImageSize,
        metadata: CalibrationSidecarMetadata,
        manualTaps: ManualCourtTaps? = nil
    ) throws -> CaptureSidecar {
        let orderedManualTaps = try orderedManualTaps(from: seed, explicitManualTaps: manualTaps, imageSize: imageSize)
        let seedForValidation = CalibrationSeed(
            intrinsics: seed.intrinsics,
            arkitCameraPose: seed.arkitCameraPose,
            courtPlane: seed.courtPlane,
            manualCourtTaps: orderedManualTaps?.imagePoints ?? []
        )
        let report = seedForValidation.validationReport(imageSize: imageSize)
        guard report.isUsable else {
            throw PackagingError.invalidSeed(report)
        }

        let usesManualFallback = !report.hasARKitSeed && report.hasManualFallback
        return CaptureSidecar(
            deviceTier: metadata.deviceTier,
            deviceModel: metadata.deviceModel,
            fps: metadata.fps,
            format: metadata.format,
            resolution: [Int(imageSize.width), Int(imageSize.height)],
            locked: metadata.locked,
            intrinsics: seed.intrinsics,
            arkitCameraPose: seed.arkitCameraPose,
            courtPlane: seed.courtPlane,
            manualCourtTaps: orderedManualTaps?.imagePoints ?? [],
            gravity: metadata.gravity,
            lidarDepthRefs: metadata.lidarDepthRefs,
            ondevicePoseTrack: metadata.ondevicePoseTrack,
            captureQuality: CaptureQuality(
                grade: usesManualFallback ? .warn : .good,
                reasons: usesManualFallback ? ["manual_calibration_fallback"] : []
            )
        )
    }

    private static func orderedManualTaps(
        from seed: CalibrationSeed,
        explicitManualTaps: ManualCourtTaps?,
        imageSize: ImageSize
    ) throws -> ManualCourtTaps? {
        let taps = explicitManualTaps ?? (seed.manualCourtTaps.isEmpty ? nil : ManualCourtTaps(imagePoints: seed.manualCourtTaps))
        do {
            return try taps?.orderedFourCorners(imageSize: imageSize)
        } catch let error as ManualCourtTaps.ValidationError {
            throw PackagingError.invalidManualTaps(error)
        }
    }
}
