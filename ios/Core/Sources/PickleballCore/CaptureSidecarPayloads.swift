import Foundation

public enum ARTrackingState: String, Codable, Equatable, Sendable {
    case normal
    case limited
    case unavailable
}

public enum ARTrackingQuality: String, Codable, Equatable, Sendable {
    case good
    case limited
    case unavailable
}

public struct ARTrackingSnapshot: Codable, Equatable, Sendable {
    public var state: ARTrackingState
    public var quality: ARTrackingQuality
    public var reason: String?

    public init(state: ARTrackingState, quality: ARTrackingQuality, reason: String? = nil) {
        self.state = state
        self.quality = quality
        self.reason = reason
    }
}

public struct ARKitFrameSample: Codable, Equatable, Sendable {
    public var videoPTSS: Double
    public var arkitTimestampS: Double
    public var cameraPose: RigidPose
    public var intrinsics: CameraIntrinsics
    public var tracking: ARTrackingSnapshot

    public init(
        videoPTSS: Double,
        arkitTimestampS: Double,
        cameraPose: RigidPose,
        intrinsics: CameraIntrinsics,
        tracking: ARTrackingSnapshot
    ) {
        self.videoPTSS = videoPTSS
        self.arkitTimestampS = arkitTimestampS
        self.cameraPose = cameraPose
        self.intrinsics = intrinsics
        self.tracking = tracking
    }

    private enum CodingKeys: String, CodingKey {
        case videoPTSS = "video_pts_s"
        case arkitTimestampS = "arkit_timestamp_s"
        case cameraPose = "camera_pose"
        case intrinsics
        case tracking
    }
}

public struct CapturePolicyRequestedState: Codable, Equatable, Sendable {
    public var fps: Int
    public var resolution: [Int]
    public var format: CaptureFormat
    public var orientation: CaptureOrientation
    public var electronicStabilizationEnabled: Bool
    public var exposureLocked: Bool
    public var focusLocked: Bool
    public var whiteBalanceLocked: Bool

    public init(
        fps: Int,
        resolution: [Int],
        format: CaptureFormat,
        orientation: CaptureOrientation,
        electronicStabilizationEnabled: Bool,
        exposureLocked: Bool,
        focusLocked: Bool,
        whiteBalanceLocked: Bool
    ) {
        self.fps = fps
        self.resolution = resolution
        self.format = format
        self.orientation = orientation
        self.electronicStabilizationEnabled = electronicStabilizationEnabled
        self.exposureLocked = exposureLocked
        self.focusLocked = focusLocked
        self.whiteBalanceLocked = whiteBalanceLocked
    }

    private enum CodingKeys: String, CodingKey {
        case fps
        case resolution
        case format
        case orientation
        case electronicStabilizationEnabled = "electronic_stabilization_enabled"
        case exposureLocked = "exposure_locked"
        case focusLocked = "focus_locked"
        case whiteBalanceLocked = "white_balance_locked"
    }
}

public struct CapturePolicyAchievedState: Codable, Equatable, Sendable {
    public var fps: Int?
    public var resolution: [Int]?
    public var format: CaptureFormat?
    public var orientation: CaptureOrientation?
    public var electronicStabilizationEnabled: Bool?
    public var exposureLocked: Bool?
    public var focusLocked: Bool?
    public var whiteBalanceLocked: Bool?

    public init(
        fps: Int?,
        resolution: [Int]?,
        format: CaptureFormat?,
        orientation: CaptureOrientation?,
        electronicStabilizationEnabled: Bool?,
        exposureLocked: Bool?,
        focusLocked: Bool?,
        whiteBalanceLocked: Bool?
    ) {
        self.fps = fps
        self.resolution = resolution
        self.format = format
        self.orientation = orientation
        self.electronicStabilizationEnabled = electronicStabilizationEnabled
        self.exposureLocked = exposureLocked
        self.focusLocked = focusLocked
        self.whiteBalanceLocked = whiteBalanceLocked
    }

    private enum CodingKeys: String, CodingKey {
        case fps
        case resolution
        case format
        case orientation
        case electronicStabilizationEnabled = "electronic_stabilization_enabled"
        case exposureLocked = "exposure_locked"
        case focusLocked = "focus_locked"
        case whiteBalanceLocked = "white_balance_locked"
    }
}

public struct CapturePolicyEnforcementReport: Codable, Equatable, Sendable {
    public var requested: CapturePolicyRequestedState
    public var achieved: CapturePolicyAchievedState?
    public var violations: [String]

    public var isCompliant: Bool {
        violations.isEmpty
    }

    public init(
        requested: CapturePolicyRequestedState,
        achieved: CapturePolicyAchievedState?,
        violations: [String]
    ) {
        self.requested = requested
        self.achieved = achieved
        self.violations = violations
    }
}

public enum ProfileCaptureStepKind: String, Codable, CaseIterable, Equatable, Sendable {
    case emptyCourtClip = "empty_court_clip"
    case calibrationGridSweep = "calibration_grid_sweep"
    case paddleOrbit = "paddle_orbit"
    case playerHeightEntry = "player_height_entry"
    case ballPick = "ball_pick"

    public static let h0Order: [ProfileCaptureStepKind] = [
        .emptyCourtClip,
        .calibrationGridSweep,
        .paddleOrbit,
        .playerHeightEntry,
        .ballPick,
    ]
}

public enum ProfileCaptureStepStatus: String, Codable, Equatable, Sendable {
    case pending
    case complete
}

public struct ProfileCaptureStepRecord: Codable, Equatable, Sendable {
    public var kind: ProfileCaptureStepKind
    public var status: ProfileCaptureStepStatus
    public var artifactRef: String?
    public var metadata: [String: String]

    public init(
        kind: ProfileCaptureStepKind,
        status: ProfileCaptureStepStatus,
        artifactRef: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.kind = kind
        self.status = status
        self.artifactRef = artifactRef
        self.metadata = metadata
    }

    private enum CodingKeys: String, CodingKey {
        case kind
        case status
        case artifactRef = "artifact_ref"
        case metadata
    }
}

public struct ProfileCapturePayload: Codable, Equatable, Sendable {
    public var steps: [ProfileCaptureStepRecord]

    public init(steps: [ProfileCaptureStepRecord]) {
        self.steps = steps
    }
}
