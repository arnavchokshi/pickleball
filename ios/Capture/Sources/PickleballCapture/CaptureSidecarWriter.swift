import Foundation
import PickleballCore

public struct CaptureSidecarWriteContext: Equatable, Sendable {
    public var deviceTier: DeviceTier
    public var deviceModel: String
    public var cameraPosition: String?
    public var cameraLens: String?
    public var locked: LockedCapture
    public var intrinsics: CameraIntrinsics
    public var gravity: [Double]
    public var arkit: ARCaptureSidecarPayload?
    public var policyEnforcement: CapturePolicyEnforcementReport?
    public var profileCapture: ProfileCapturePayload?
    public var captureQuality: CaptureQuality

    public init(
        deviceTier: DeviceTier,
        deviceModel: String,
        cameraPosition: String?,
        cameraLens: String?,
        locked: LockedCapture,
        intrinsics: CameraIntrinsics,
        gravity: [Double],
        arkit: ARCaptureSidecarPayload? = nil,
        policyEnforcement: CapturePolicyEnforcementReport? = nil,
        profileCapture: ProfileCapturePayload? = nil,
        captureQuality: CaptureQuality
    ) {
        self.deviceTier = deviceTier
        self.deviceModel = deviceModel
        self.cameraPosition = cameraPosition
        self.cameraLens = cameraLens
        self.locked = locked
        self.intrinsics = intrinsics
        self.gravity = gravity
        self.arkit = arkit
        self.policyEnforcement = policyEnforcement
        self.profileCapture = profileCapture
        self.captureQuality = captureQuality
    }
}

public enum CaptureSidecarWriter {
    @discardableResult
    public static func write(
        descriptor: CapturePackageDescriptor,
        packageRootURL: URL,
        recordingStartedAt: Date,
        finishedAt: Date,
        context: CaptureSidecarWriteContext,
        fileManager: FileManager = .default
    ) throws -> URL {
        let sidecarURL = packageRootURL.appendingPathComponent(descriptor.sidecarRelativePath, isDirectory: false)
        try fileManager.createDirectory(
            at: sidecarURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let sidecar = makeSidecar(
            descriptor: descriptor,
            recordingStartedAt: recordingStartedAt,
            finishedAt: finishedAt,
            context: context
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(sidecar).write(to: sidecarURL, options: .atomic)
        return sidecarURL
    }

    public static func makeSidecar(
        descriptor: CapturePackageDescriptor,
        recordingStartedAt: Date,
        finishedAt: Date,
        context: CaptureSidecarWriteContext
    ) -> CaptureSidecar {
        let latestARFrame = context.arkit?.latestFrame
        let setupPass = context.arkit?.setupPass
        let setupPassAvailable = setupPass?.status == .available
        let setupPassUnavailableReason = setupPass?.unavailableReason ?? "arkit_setup_pass_unavailable"
        let courtPlane = setupPassAvailable ? setupPass?.courtPlane : context.arkit?.courtPlane
        let arkitFrameSamples = context.arkit?.frameSamples ?? []
        let unavailableSensorReasons = unavailableSensorReasons(
            setupPass: setupPass,
            latestARFrame: latestARFrame,
            courtPlane: courtPlane,
            setupPassUnavailableReason: setupPassUnavailableReason
        )
        return CaptureSidecar(
            deviceTier: context.deviceTier,
            deviceModel: context.deviceModel,
            fps: descriptor.expectedFPS,
            format: descriptor.expectedFormat,
            resolution: descriptor.expectedResolution,
            orientation: descriptor.expectedOrientation,
            captureDeviceOrientation: descriptor.captureDeviceOrientation,
            videoRotationAngleDegrees: descriptor.videoRotationAngleDegrees,
            recordingStartedAt: iso8601String(from: recordingStartedAt),
            recordingDurationS: max(0.0, finishedAt.timeIntervalSince(recordingStartedAt)),
            cameraPosition: context.cameraPosition,
            cameraLens: context.cameraLens,
            locked: context.locked,
            intrinsics: setupPassAvailable
                ? (setupPass?.intrinsics ?? context.intrinsics)
                : (latestARFrame?.intrinsics ?? context.intrinsics),
            arkitCameraPose: setupPassAvailable ? setupPass?.cameraPose : latestARFrame?.cameraPose,
            courtPlane: courtPlane,
            setupPass: setupPass,
            gravity: context.gravity,
            arkitFrameSamples: arkitFrameSamples,
            ondevicePoseTrack: nil,
            unavailableSensorReasons: unavailableSensorReasons,
            policyEnforcement: context.policyEnforcement,
            profileCapture: context.profileCapture,
            captureQuality: captureQuality(
                context.captureQuality,
                setupPass: setupPass,
                courtPlane: courtPlane,
                policyEnforcement: context.policyEnforcement
            )
        )
    }

    private static func unavailableSensorReasons(
        setupPass: ARKitSetupPassSidecar?,
        latestARFrame: ARKitFrameSample?,
        courtPlane: Plane?,
        setupPassUnavailableReason: String
    ) -> [String: String] {
        var reasons: [String: String] = [:]
        if setupPass?.status == .unavailable {
            reasons["arkit_camera_pose"] = setupPassUnavailableReason
        } else if setupPass?.cameraPose == nil && latestARFrame?.cameraPose == nil {
            reasons["arkit_camera_pose"] = "no_arkit_setup_pass_recorded"
        }
        if courtPlane == nil {
            reasons["court_plane"] = setupPass?.status == .unavailable
                ? setupPassUnavailableReason
                : "no_horizontal_arkit_plane_recorded"
        }
        return reasons
    }

    private static func captureQuality(
        _ original: CaptureQuality,
        setupPass: ARKitSetupPassSidecar?,
        courtPlane: Plane?,
        policyEnforcement: CapturePolicyEnforcementReport?
    ) -> CaptureQuality {
        var reasons = original.reasons.filter { reason in
            if setupPass?.status == .available && (reason == "arkit_seed_missing" || reason == "intrinsics_estimated_from_fov") {
                return false
            }
            if courtPlane != nil && reason == "court_plane_missing" {
                return false
            }
            return true
        }
        if let policyEnforcement, !policyEnforcement.isCompliant {
            reasons.append(contentsOf: policyEnforcement.violations.map { "policy_\($0)" })
        }
        let grade: CaptureQuality.Grade
        if reasons.isEmpty {
            grade = .good
        } else if original.grade == .poor {
            grade = .poor
        } else {
            grade = .warn
        }
        return CaptureQuality(grade: grade, reasons: reasons)
    }

    private static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }
}
