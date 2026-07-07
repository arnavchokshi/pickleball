import Foundation

public struct LockedCapture: Codable, Equatable, Sendable {
    public var exposureS: Double
    public var iso: Double
    public var focus: Double
    public var wbLocked: Bool

    public init(exposureS: Double, iso: Double, focus: Double, wbLocked: Bool) {
        self.exposureS = exposureS
        self.iso = iso
        self.focus = focus
        self.wbLocked = wbLocked
    }

    private enum CodingKeys: String, CodingKey {
        case exposureS = "exposure_s"
        case iso
        case focus
        case wbLocked = "wb_locked"
    }
}

public struct CameraIntrinsics: Codable, Equatable, Sendable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var dist: [Double]
    public var source: String

    public init(fx: Double, fy: Double, cx: Double, cy: Double, dist: [Double] = [], source: String) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist = dist
        self.source = source
    }
}

public struct RigidPose: Codable, Equatable, Sendable {
    public var R: [[Double]]
    public var t: [Double]

    public init(R: [[Double]], t: [Double]) {
        self.R = R
        self.t = t
    }
}

public struct Plane: Codable, Equatable, Sendable {
    public var point: [Double]
    public var normal: [Double]

    public init(point: [Double], normal: [Double]) {
        self.point = point
        self.normal = normal
    }
}

public enum CaptureProvenance: String, Codable, Equatable, Sendable {
    case liveRecording = "live_recording"
    case cameraRollImport = "camera_roll_import"
}

public struct CaptureSidecar: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var provenance: CaptureProvenance
    public var deviceTier: DeviceTier
    public var deviceModel: String
    public var fps: Int
    public var format: CaptureFormat
    public var resolution: [Int]
    public var orientation: CaptureOrientation
    public var captureDeviceOrientation: CaptureDeviceOrientation?
    public var videoRotationAngleDegrees: Int?
    public var recordingStartedAt: String?
    public var recordingDurationS: Double?
    public var cameraPosition: String?
    public var cameraLens: String?
    public var locked: LockedCapture?
    public var intrinsics: CameraIntrinsics?
    public var arkitCameraPose: RigidPose?
    public var courtPlane: Plane?
    public var setupPass: ARKitSetupPassSidecar?
    public var manualCourtTaps: [[Double]]
    public var gravity: [Double]?
    public var arkitFrameSamples: [ARKitFrameSample]
    public var lidarDepthRefs: [String]
    public var ondevicePoseTrack: String?
    public var unavailableSensorReasons: [String: String]
    public var policyEnforcement: CapturePolicyEnforcementReport?
    public var profileCapture: ProfileCapturePayload?
    public var captureQuality: CaptureQuality

    public init(
        schemaVersion: Int = 1,
        provenance: CaptureProvenance = .liveRecording,
        deviceTier: DeviceTier,
        deviceModel: String,
        fps: Int,
        format: CaptureFormat,
        resolution: [Int],
        orientation: CaptureOrientation = .landscape,
        captureDeviceOrientation: CaptureDeviceOrientation? = nil,
        videoRotationAngleDegrees: Int? = nil,
        recordingStartedAt: String? = nil,
        recordingDurationS: Double? = nil,
        cameraPosition: String? = nil,
        cameraLens: String? = nil,
        locked: LockedCapture?,
        intrinsics: CameraIntrinsics?,
        arkitCameraPose: RigidPose? = nil,
        courtPlane: Plane? = nil,
        setupPass: ARKitSetupPassSidecar? = nil,
        manualCourtTaps: [[Double]] = [],
        gravity: [Double]?,
        arkitFrameSamples: [ARKitFrameSample] = [],
        lidarDepthRefs: [String] = [],
        ondevicePoseTrack: String? = nil,
        unavailableSensorReasons: [String: String] = [:],
        policyEnforcement: CapturePolicyEnforcementReport? = nil,
        profileCapture: ProfileCapturePayload? = nil,
        captureQuality: CaptureQuality
    ) {
        self.schemaVersion = schemaVersion
        self.provenance = provenance
        self.deviceTier = deviceTier
        self.deviceModel = deviceModel
        self.fps = fps
        self.format = format
        self.resolution = resolution
        self.orientation = orientation
        self.captureDeviceOrientation = captureDeviceOrientation
        self.videoRotationAngleDegrees = videoRotationAngleDegrees
        self.recordingStartedAt = recordingStartedAt
        self.recordingDurationS = recordingDurationS
        self.cameraPosition = cameraPosition
        self.cameraLens = cameraLens
        self.locked = locked
        self.intrinsics = intrinsics
        self.arkitCameraPose = arkitCameraPose
        self.courtPlane = courtPlane
        self.setupPass = setupPass
        self.manualCourtTaps = manualCourtTaps
        self.gravity = gravity
        self.arkitFrameSamples = arkitFrameSamples
        self.lidarDepthRefs = lidarDepthRefs
        self.ondevicePoseTrack = ondevicePoseTrack
        self.unavailableSensorReasons = unavailableSensorReasons
        self.policyEnforcement = policyEnforcement
        self.profileCapture = profileCapture
        self.captureQuality = captureQuality
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case provenance
        case deviceTier = "device_tier"
        case deviceModel = "device_model"
        case fps
        case format
        case resolution
        case orientation
        case captureDeviceOrientation = "capture_device_orientation"
        case videoRotationAngleDegrees = "video_rotation_angle_degrees"
        case recordingStartedAt = "recording_started_at"
        case recordingDurationS = "recording_duration_s"
        case cameraPosition = "camera_position"
        case cameraLens = "camera_lens"
        case locked
        case intrinsics
        case arkitCameraPose = "arkit_camera_pose"
        case courtPlane = "court_plane"
        case setupPass = "setup_pass"
        case manualCourtTaps = "manual_court_taps"
        case gravity
        case arkitFrameSamples = "arkit_frame_samples"
        case lidarDepthRefs = "lidar_depth_refs"
        case ondevicePoseTrack = "ondevice_pose_track"
        case unavailableSensorReasons = "unavailable_sensor_reasons"
        case policyEnforcement = "policy_enforcement"
        case profileCapture = "profile_capture"
        case captureQuality = "capture_quality"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        self.provenance = try container.decodeIfPresent(CaptureProvenance.self, forKey: .provenance) ?? .liveRecording
        self.deviceTier = try container.decode(DeviceTier.self, forKey: .deviceTier)
        self.deviceModel = try container.decode(String.self, forKey: .deviceModel)
        self.fps = try container.decode(Int.self, forKey: .fps)
        self.format = try container.decode(CaptureFormat.self, forKey: .format)
        self.resolution = try container.decode([Int].self, forKey: .resolution)
        self.orientation = try container.decode(CaptureOrientation.self, forKey: .orientation)
        self.captureDeviceOrientation = try container.decodeIfPresent(CaptureDeviceOrientation.self, forKey: .captureDeviceOrientation)
        self.videoRotationAngleDegrees = try container.decodeIfPresent(Int.self, forKey: .videoRotationAngleDegrees)
        self.recordingStartedAt = try container.decodeIfPresent(String.self, forKey: .recordingStartedAt)
        self.recordingDurationS = try container.decodeIfPresent(Double.self, forKey: .recordingDurationS)
        self.cameraPosition = try container.decodeIfPresent(String.self, forKey: .cameraPosition)
        self.cameraLens = try container.decodeIfPresent(String.self, forKey: .cameraLens)
        self.locked = try container.decodeIfPresent(LockedCapture.self, forKey: .locked)
        self.intrinsics = try container.decodeIfPresent(CameraIntrinsics.self, forKey: .intrinsics)
        self.arkitCameraPose = try container.decodeIfPresent(RigidPose.self, forKey: .arkitCameraPose)
        self.courtPlane = try container.decodeIfPresent(Plane.self, forKey: .courtPlane)
        self.setupPass = try container.decodeIfPresent(ARKitSetupPassSidecar.self, forKey: .setupPass)
        self.manualCourtTaps = try container.decodeIfPresent([[Double]].self, forKey: .manualCourtTaps) ?? []
        self.gravity = try container.decodeIfPresent([Double].self, forKey: .gravity)
        self.arkitFrameSamples = try container.decodeIfPresent([ARKitFrameSample].self, forKey: .arkitFrameSamples) ?? []
        self.lidarDepthRefs = try container.decodeIfPresent([String].self, forKey: .lidarDepthRefs) ?? []
        self.ondevicePoseTrack = try container.decodeIfPresent(String.self, forKey: .ondevicePoseTrack)
        self.unavailableSensorReasons = try container.decodeIfPresent([String: String].self, forKey: .unavailableSensorReasons) ?? [:]
        self.policyEnforcement = try container.decodeIfPresent(CapturePolicyEnforcementReport.self, forKey: .policyEnforcement)
        self.profileCapture = try container.decodeIfPresent(ProfileCapturePayload.self, forKey: .profileCapture)
        self.captureQuality = try container.decode(CaptureQuality.self, forKey: .captureQuality)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(provenance, forKey: .provenance)
        try container.encode(deviceTier, forKey: .deviceTier)
        try container.encode(deviceModel, forKey: .deviceModel)
        try container.encode(fps, forKey: .fps)
        try container.encode(format, forKey: .format)
        try container.encode(resolution, forKey: .resolution)
        try container.encode(orientation, forKey: .orientation)
        try container.encodeIfPresent(captureDeviceOrientation, forKey: .captureDeviceOrientation)
        try container.encodeIfPresent(videoRotationAngleDegrees, forKey: .videoRotationAngleDegrees)
        try container.encodeIfPresent(recordingStartedAt, forKey: .recordingStartedAt)
        try container.encodeIfPresent(recordingDurationS, forKey: .recordingDurationS)
        try container.encodeIfPresent(cameraPosition, forKey: .cameraPosition)
        try container.encodeIfPresent(cameraLens, forKey: .cameraLens)
        try container.encodeIfPresent(locked, forKey: .locked)
        try container.encodeIfPresent(intrinsics, forKey: .intrinsics)
        try container.encodeIfPresent(arkitCameraPose, forKey: .arkitCameraPose)
        try container.encodeIfPresent(courtPlane, forKey: .courtPlane)
        try container.encodeIfPresent(setupPass, forKey: .setupPass)
        try container.encode(manualCourtTaps, forKey: .manualCourtTaps)
        try container.encodeIfPresent(gravity, forKey: .gravity)
        try container.encode(arkitFrameSamples, forKey: .arkitFrameSamples)
        try container.encode(lidarDepthRefs, forKey: .lidarDepthRefs)
        try container.encodeIfPresent(ondevicePoseTrack, forKey: .ondevicePoseTrack)
        try container.encode(unavailableSensorReasons, forKey: .unavailableSensorReasons)
        try container.encodeIfPresent(policyEnforcement, forKey: .policyEnforcement)
        try container.encodeIfPresent(profileCapture, forKey: .profileCapture)
        try container.encode(captureQuality, forKey: .captureQuality)
    }
}
