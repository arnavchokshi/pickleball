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

public enum ARKitFrameSampleProvenance: String, Codable, Equatable, Sendable {
    case arkit = "arkit"
    case coreMotionOnly = "coremotion_only"
}

public struct ARKitFrameSample: Codable, Equatable, Sendable {
    public var videoPTSS: Double
    public var arkitTimestampS: Double?
    public var cameraPose: RigidPose?
    public var intrinsics: CameraIntrinsics?
    public var tracking: ARTrackingSnapshot?
    public var gravity: [Double]?
    public var provenance: ARKitFrameSampleProvenance
    public var unavailableReason: String?

    public init(
        videoPTSS: Double,
        arkitTimestampS: Double,
        cameraPose: RigidPose,
        intrinsics: CameraIntrinsics,
        tracking: ARTrackingSnapshot,
        gravity: [Double]? = nil,
        provenance: ARKitFrameSampleProvenance = .arkit,
        unavailableReason: String? = nil
    ) {
        self.videoPTSS = videoPTSS
        self.arkitTimestampS = arkitTimestampS
        self.cameraPose = cameraPose
        self.intrinsics = intrinsics
        self.tracking = tracking
        self.gravity = gravity
        self.provenance = provenance
        self.unavailableReason = unavailableReason
    }

    public init(
        videoPTSS: Double,
        gravity: [Double],
        provenance: ARKitFrameSampleProvenance,
        unavailableReason: String
    ) {
        self.videoPTSS = videoPTSS
        self.arkitTimestampS = nil
        self.cameraPose = nil
        self.intrinsics = nil
        self.tracking = nil
        self.gravity = gravity
        self.provenance = provenance
        self.unavailableReason = unavailableReason
    }

    private enum CodingKeys: String, CodingKey {
        case videoPTSS = "video_pts_s"
        case arkitTimestampS = "arkit_timestamp_s"
        case cameraPose = "camera_pose"
        case intrinsics
        case tracking
        case gravity
        case provenance
        case unavailableReason = "unavailable_reason"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        videoPTSS = try container.decode(Double.self, forKey: .videoPTSS)
        arkitTimestampS = try container.decodeIfPresent(Double.self, forKey: .arkitTimestampS)
        cameraPose = try container.decodeIfPresent(RigidPose.self, forKey: .cameraPose)
        intrinsics = try container.decodeIfPresent(CameraIntrinsics.self, forKey: .intrinsics)
        tracking = try container.decodeIfPresent(ARTrackingSnapshot.self, forKey: .tracking)
        gravity = try container.decodeIfPresent([Double].self, forKey: .gravity)
        provenance = try container.decodeIfPresent(ARKitFrameSampleProvenance.self, forKey: .provenance)
            ?? (cameraPose == nil ? .coreMotionOnly : .arkit)
        unavailableReason = try container.decodeIfPresent(String.self, forKey: .unavailableReason)
    }
}

public enum ARKitSetupPassStatus: String, Codable, Equatable, Sendable {
    case available
    case unavailable
}

public enum ARKitSetupTrackingState: String, Codable, Equatable, Sendable {
    case normal
    case limited
    case unavailable
}

public struct ARKitSetupPassSidecar: Codable, Equatable, Sendable {
    public var status: ARKitSetupPassStatus
    public var provenance: String
    public var intrinsics: CameraIntrinsics?
    public var cameraPose: RigidPose?
    public var courtPlane: Plane?
    public var gravity: [Double]?
    public var trackingState: ARKitSetupTrackingState
    public var timestampS: Double?
    public var durationS: Double?
    public var unavailableReason: String?

    public init(
        intrinsics: CameraIntrinsics,
        cameraPose: RigidPose,
        courtPlane: Plane,
        trackingState: ARKitSetupTrackingState,
        timestampS: Double? = nil,
        durationS: Double? = nil,
        gravity: [Double]? = nil,
        provenance: String = "arkit_setup_pass"
    ) {
        self.status = .available
        self.provenance = provenance
        self.intrinsics = intrinsics
        self.cameraPose = cameraPose
        self.courtPlane = courtPlane
        self.gravity = gravity
        self.trackingState = trackingState
        self.timestampS = timestampS
        self.durationS = durationS
        self.unavailableReason = nil
    }

    public static func unavailable(
        reason: String,
        gravity: [Double]? = nil,
        durationS: Double? = nil,
        provenance: String = "arkit_setup_pass"
    ) -> ARKitSetupPassSidecar {
        ARKitSetupPassSidecar(
            status: .unavailable,
            provenance: provenance,
            intrinsics: nil,
            cameraPose: nil,
            courtPlane: nil,
            gravity: gravity,
            trackingState: .unavailable,
            timestampS: nil,
            durationS: durationS,
            unavailableReason: reason
        )
    }

    private init(
        status: ARKitSetupPassStatus,
        provenance: String,
        intrinsics: CameraIntrinsics?,
        cameraPose: RigidPose?,
        courtPlane: Plane?,
        gravity: [Double]?,
        trackingState: ARKitSetupTrackingState,
        timestampS: Double?,
        durationS: Double?,
        unavailableReason: String?
    ) {
        self.status = status
        self.provenance = provenance
        self.intrinsics = intrinsics
        self.cameraPose = cameraPose
        self.courtPlane = courtPlane
        self.gravity = gravity
        self.trackingState = trackingState
        self.timestampS = timestampS
        self.durationS = durationS
        self.unavailableReason = unavailableReason
    }

    private enum CodingKeys: String, CodingKey {
        case status
        case provenance
        case intrinsics
        case cameraPose = "camera_pose"
        case courtPlane = "court_plane"
        case gravity
        case trackingState = "tracking_state"
        case timestampS = "timestamp_s"
        case durationS = "duration_s"
        case unavailableReason = "unavailable_reason"
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
