import Foundation
import PickleballCore

public struct CapturePackageDescriptor: Equatable, Sendable {
    public enum ValidationError: Error, Equatable, Sendable {
        case unsafeSessionID(String)
    }

    public var sessionID: String
    public var directoryRelativePath: String
    public var clipRelativePath: String
    public var sidecarRelativePath: String
    public var onDevicePoseTrackRelativePath: String
    public var onDevicePersonTracksRelativePath: String
    public var onDevicePersonTimingRelativePath: String
    public var expectedFPS: Int
    public var expectedResolution: [Int]
    public var expectedFormat: CaptureFormat
    public var expectedOrientation: CaptureOrientation
    public var captureDeviceOrientation: CaptureDeviceOrientation
    public var videoRotationAngleDegrees: Int
    public var startedAt: Date

    public var preferredUploadOrder: [CaptureUploadPart] {
        [
            CaptureUploadPart(kind: .captureSidecar, relativePath: sidecarRelativePath),
            CaptureUploadPart(kind: .clip, relativePath: clipRelativePath),
        ]
    }

    public init(
        sessionID: String,
        policy: CapturePolicy,
        startedAt: Date,
        captureDeviceOrientation: CaptureDeviceOrientation = .landscapeRight,
        rootDirectory: String = "captures"
    ) throws {
        guard Self.isSafeSessionID(sessionID) else {
            throw ValidationError.unsafeSessionID(sessionID)
        }

        let directory = "\(rootDirectory)/\(sessionID)"
        self.sessionID = sessionID
        self.directoryRelativePath = directory
        self.clipRelativePath = "\(directory)/clip.mov"
        self.sidecarRelativePath = "\(directory)/capture_sidecar.json"
        self.onDevicePoseTrackRelativePath = "\(directory)/ondevice_pose.json"
        self.onDevicePersonTracksRelativePath = "\(directory)/on_device_person_tracks.json"
        self.onDevicePersonTimingRelativePath = "\(directory)/timing.json"
        self.expectedFPS = policy.fps
        self.expectedResolution = policy.resolution.dimensions(for: policy.orientation)
        self.expectedFormat = policy.codec
        self.expectedOrientation = policy.orientation
        self.captureDeviceOrientation = captureDeviceOrientation
        self.videoRotationAngleDegrees = CaptureOrientationPolicy.rotationAngleDegrees(for: captureDeviceOrientation)
        self.startedAt = startedAt
    }

    private static func isSafeSessionID(_ value: String) -> Bool {
        guard !value.isEmpty else {
            return false
        }

        return value.unicodeScalars.allSatisfy { scalar in
            CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_"
        }
    }
}

public struct CaptureSessionIDFactory: Sendable {
    private var lastTimestampStem: String?
    private var sequence: Int

    public init() {
        self.lastTimestampStem = nil
        self.sequence = 0
    }

    public mutating func nextSessionID(now: Date = Date()) -> String {
        let stem = Self.timestampStem(now: now)
        if stem == lastTimestampStem {
            sequence += 1
        } else {
            lastTimestampStem = stem
            sequence = 0
        }
        return "\(stem)-\(String(format: "%03d", sequence))"
    }

    public static func uniqueSessionID(now: Date = Date()) -> String {
        "\(timestampStem(now: now))-\(UUID().uuidString.prefix(8))"
    }

    private static func timestampStem(now: Date) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "capture-\(formatter.string(from: now))"
    }
}

public struct CaptureUploadPart: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case captureSidecar
        case onDevicePoseTrack
        case onDevicePersonTracks
        case onDevicePersonTiming
        case clip
    }

    public var kind: Kind
    public var relativePath: String

    public init(kind: Kind, relativePath: String) {
        self.kind = kind
        self.relativePath = relativePath
    }
}

public enum CaptureSensorStream: String, CaseIterable, Hashable, Sendable {
    case videoFrames = "video_frames"
    case audioSamples = "audio_samples"
    case frameTiming = "frame_timing"
    case lockedCameraSettings = "locked_camera_settings"
    case cameraIntrinsics = "camera_intrinsics"
    case arkitCameraPose = "arkit_camera_pose"
    case courtPlane = "court_plane"
    case coreMotionGravity = "core_motion_gravity"
    case manualCourtTaps = "manual_court_taps"
    case onDevicePoseTrack = "ondevice_pose_track"
    case lidarDepthRefs = "lidar_depth_refs"
}

public enum CaptureSwiftCapability: String, CaseIterable, Hashable, Sendable {
    case avFoundationCamera = "avfoundation_camera"
    case avFoundationMicrophone = "avfoundation_microphone"
    case lockedExposureFocusWhiteBalance = "locked_exposure_focus_white_balance"
    case highFrameRateRecording = "high_frame_rate_recording"
    case arkitSetupPass = "arkit_setup_pass"
    case coreMotionGravity = "core_motion_gravity"
    case manualCourtTapFallback = "manual_court_tap_fallback"
    case visionFastTier = "vision_fast_tier"
    case coreMLFastTier = "coreml_fast_tier"
    case urlSessionBackgroundUpload = "urlsession_background_upload"
    case realityKitReplay = "realitykit_replay"
}

public struct CaptureSwiftCapabilityManifest: Equatable, Sendable {
    public var capabilities: [CaptureSwiftCapability]

    public var declaresCaptureCalibrationRequirements: Bool {
        let required: Set<CaptureSwiftCapability> = [
            .avFoundationCamera,
            .avFoundationMicrophone,
            .lockedExposureFocusWhiteBalance,
            .highFrameRateRecording,
            .arkitSetupPass,
            .coreMotionGravity,
            .manualCourtTapFallback,
        ]

        return required.isSubset(of: Set(capabilities))
    }

    public var declaresFastTierReplayRequirements: Bool {
        let required: Set<CaptureSwiftCapability> = [
            .visionFastTier,
            .coreMLFastTier,
            .urlSessionBackgroundUpload,
            .realityKitReplay,
        ]

        return required.isSubset(of: Set(capabilities))
    }

    public init(capabilities: [CaptureSwiftCapability]) {
        self.capabilities = capabilities
    }

    public static let pipelineRequired = CaptureSwiftCapabilityManifest(capabilities: [
        .avFoundationCamera,
        .avFoundationMicrophone,
        .lockedExposureFocusWhiteBalance,
        .highFrameRateRecording,
        .coreMotionGravity,
        .manualCourtTapFallback,
    ])

    public static let plannedPipelineRequired = CaptureSwiftCapabilityManifest(capabilities: [
        .avFoundationCamera,
        .avFoundationMicrophone,
        .lockedExposureFocusWhiteBalance,
        .highFrameRateRecording,
        .arkitSetupPass,
        .coreMotionGravity,
        .manualCourtTapFallback,
        .visionFastTier,
        .coreMLFastTier,
        .urlSessionBackgroundUpload,
        .realityKitReplay,
    ])
}

public struct CaptureSensorManifest: Equatable, Sendable {
    public var streams: [CaptureSensorStream]

    public var hasRequiredSidecarInputs: Bool {
        let streamSet = Set(streams)
        let commonRequired: Set<CaptureSensorStream> = [
            .videoFrames,
            .audioSamples,
            .frameTiming,
            .lockedCameraSettings,
            .cameraIntrinsics,
            .coreMotionGravity,
            .onDevicePoseTrack,
        ]
        let hasARKitSeed = streamSet.contains(.arkitCameraPose) && streamSet.contains(.courtPlane)
        let hasManualFallback = streamSet.contains(.manualCourtTaps)

        return commonRequired.isSubset(of: streamSet) && (hasARKitSeed || hasManualFallback)
    }

    public var hasOptionalDepthInput: Bool {
        streams.contains(.lidarDepthRefs)
    }

    public init(streams: [CaptureSensorStream]) {
        self.streams = streams
    }

    public static let pipelineRequired = CaptureSensorManifest(streams: [
        .videoFrames,
        .audioSamples,
        .frameTiming,
        .lockedCameraSettings,
        .cameraIntrinsics,
        .arkitCameraPose,
        .courtPlane,
        .coreMotionGravity,
        .manualCourtTaps,
        .onDevicePoseTrack,
    ])
}

public enum CapturePermissionState: Equatable, Sendable {
    case authorized
    case denied
    case restricted
    case notDetermined
}

public struct CapturePermissionSnapshot: Equatable, Sendable {
    public var camera: CapturePermissionState
    public var microphone: CapturePermissionState

    public init(camera: CapturePermissionState, microphone: CapturePermissionState) {
        self.camera = camera
        self.microphone = microphone
    }
}

public enum CaptureReadinessBlocker: Equatable, Sendable {
    case cameraPermissionDenied
    case cameraPermissionMissing
    case microphonePermissionDenied
    case microphonePermissionMissing
    case landscapeRequired
}

public struct CaptureReadiness: Equatable, Sendable {
    public var isReady: Bool
    public var blockers: [CaptureReadinessBlocker]

    public init(isReady: Bool, blockers: [CaptureReadinessBlocker]) {
        self.isReady = isReady
        self.blockers = blockers
    }
}

public enum CaptureReadinessEvaluator {
    public static func evaluate(
        permissions: CapturePermissionSnapshot,
        descriptor: CapturePackageDescriptor
    ) -> CaptureReadiness {
        var blockers: [CaptureReadinessBlocker] = []
        appendBlocker(for: permissions.camera, denied: .cameraPermissionDenied, missing: .cameraPermissionMissing, to: &blockers)
        appendBlocker(for: permissions.microphone, denied: .microphonePermissionDenied, missing: .microphonePermissionMissing, to: &blockers)
        if descriptor.expectedOrientation != .landscape {
            blockers.append(.landscapeRequired)
        }
        return CaptureReadiness(isReady: blockers.isEmpty, blockers: blockers)
    }

    private static func appendBlocker(
        for state: CapturePermissionState,
        denied: CaptureReadinessBlocker,
        missing: CaptureReadinessBlocker,
        to blockers: inout [CaptureReadinessBlocker]
    ) {
        switch state {
        case .authorized:
            break
        case .denied, .restricted:
            blockers.append(denied)
        case .notDetermined:
            blockers.append(missing)
        }
    }
}
