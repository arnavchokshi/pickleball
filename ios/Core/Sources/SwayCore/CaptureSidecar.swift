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

public struct CaptureSidecar: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var deviceTier: DeviceTier
    public var deviceModel: String
    public var fps: Int
    public var format: CaptureFormat
    public var resolution: [Int]
    public var orientation: CaptureOrientation
    public var locked: LockedCapture
    public var intrinsics: CameraIntrinsics
    public var arkitCameraPose: RigidPose?
    public var courtPlane: Plane?
    public var manualCourtTaps: [[Double]]
    public var gravity: [Double]
    public var lidarDepthRefs: [String]
    public var ondevicePoseTrack: String?
    public var captureQuality: CaptureQuality

    public init(
        schemaVersion: Int = 1,
        deviceTier: DeviceTier,
        deviceModel: String,
        fps: Int,
        format: CaptureFormat,
        resolution: [Int],
        orientation: CaptureOrientation = .landscape,
        locked: LockedCapture,
        intrinsics: CameraIntrinsics,
        arkitCameraPose: RigidPose? = nil,
        courtPlane: Plane? = nil,
        manualCourtTaps: [[Double]] = [],
        gravity: [Double],
        lidarDepthRefs: [String] = [],
        ondevicePoseTrack: String? = nil,
        captureQuality: CaptureQuality
    ) {
        self.schemaVersion = schemaVersion
        self.deviceTier = deviceTier
        self.deviceModel = deviceModel
        self.fps = fps
        self.format = format
        self.resolution = resolution
        self.orientation = orientation
        self.locked = locked
        self.intrinsics = intrinsics
        self.arkitCameraPose = arkitCameraPose
        self.courtPlane = courtPlane
        self.manualCourtTaps = manualCourtTaps
        self.gravity = gravity
        self.lidarDepthRefs = lidarDepthRefs
        self.ondevicePoseTrack = ondevicePoseTrack
        self.captureQuality = captureQuality
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case deviceTier = "device_tier"
        case deviceModel = "device_model"
        case fps
        case format
        case resolution
        case orientation
        case locked
        case intrinsics
        case arkitCameraPose = "arkit_camera_pose"
        case courtPlane = "court_plane"
        case manualCourtTaps = "manual_court_taps"
        case gravity
        case lidarDepthRefs = "lidar_depth_refs"
        case ondevicePoseTrack = "ondevice_pose_track"
        case captureQuality = "capture_quality"
    }
}
