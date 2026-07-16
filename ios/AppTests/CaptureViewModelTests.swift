import XCTest
@preconcurrency import AVFoundation
import PickleballCapture
import PickleballCore
@testable import Pickleball

final class CaptureViewModelTests: XCTestCase {
    @MainActor
    func testInitialCaptureStateIsIdleAndCameraFree() {
        let model = CaptureViewModel()

        XCTAssertEqual(model.status, .idle)
        XCTAssertFalse(model.isRecording)
        XCTAssertNil(model.descriptor)
        XCTAssertEqual(model.selectedMode, .standard60)
        XCTAssertEqual(CaptureViewModel.modes, [.standard60, .swing120, .ballPhysics240, .quality4K60])
        XCTAssertEqual(model.captureOrientationTitle, "Landscape")
        XCTAssertEqual(model.videoRotationTitle, "90°")
        XCTAssertEqual(model.previewRotationAngle, 90)
        XCTAssertEqual(model.replayBenchmarkTitle, "Replay")
        XCTAssertEqual(model.replayBenchmarkDetail, "Vision")
        XCTAssertNil(model.replayBenchmarkOutputPath)
        XCTAssertTrue(model.modeSummary.contains("60 fps"))
        XCTAssertTrue(model.modeSummary.contains("hevc"))
        XCTAssertEqual(model.recordFlowPhase, .idle)
        XCTAssertEqual(model.policyChips.map(\.id), ["eis", "camera_locks", "landscape"])
    }

    @MainActor
    func testPrepareUsesInjectedAsyncCameraController() async throws {
        let controller = FakeCameraCaptureController()
        let descriptor = try Self.captureDescriptor(sessionID: "prepare-test")
        controller.configureDescriptor = descriptor
        controller.policyEnforcementReport = .compliant60FPS
        var permissionRequestCount = 0
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                permissionRequestCount += 1
                return CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()

        XCTAssertEqual(permissionRequestCount, 1)
        XCTAssertEqual(model.permissions, CapturePermissionSnapshot(camera: .authorized, microphone: .authorized))
        XCTAssertEqual(model.status, .ready)
        XCTAssertEqual(model.descriptor, descriptor)
        XCTAssertEqual(controller.configureCalls, [
            FakeCameraCaptureController.ConfigureCall(mode: .standard60, orientation: .landscapeRight),
        ])
        XCTAssertEqual(controller.startPreviewCallCount, 1)
        XCTAssertEqual(model.capturePolicyStatusText, "Capture policy locked")
        XCTAssertEqual(model.recordFlowPhase, .ready)
    }

    @MainActor
    func testPrepareRunsARKitSetupPassBeforeStartingAVCapturePreview() async throws {
        let controller = FakeCameraCaptureController()
        controller.setupPass = ARKitSetupPassSidecar(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            cameraPose: RigidPose(R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 1.4, 0]),
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0]),
            trackingState: .normal,
            gravity: [0, -1, 0]
        )
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()

        XCTAssertEqual(controller.events, ["configure", "setupPass", "startPreview"])
        XCTAssertEqual(model.setupPassStatus, .aligned)
        XCTAssertEqual(model.setupPassStatusText, "Aligned ✓")
        XCTAssertEqual(model.setupPassChipStatus, .pass)
    }

    @MainActor
    func testRefreshAfterFinishedRecordingRerunsStaleSetupPassAndRestartsPreview() async throws {
        let controller = FakeCameraCaptureController()
        let recordingDescriptor = try Self.captureDescriptor(sessionID: "recording")
        controller.configureDescriptor = try Self.captureDescriptor(sessionID: "configured")
        controller.startRecordingDescriptor = recordingDescriptor
        controller.setupPass = ARKitSetupPassSidecar.unavailable(reason: "first_timeout", gravity: [0, -1, 0])
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()
        await model.toggleRecording()
        await model.toggleRecording()
        controller.onRecordingFinished?(.success(CameraRecordingResult(
            descriptor: recordingDescriptor,
            clipURL: URL(fileURLWithPath: "/tmp/recording.mov")
        )))
        await Task.yield()

        controller.gravity = [0.4, -0.6, 0]
        controller.setupPass = ARKitSetupPassSidecar(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            cameraPose: RigidPose(R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 1.4, 0]),
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0]),
            trackingState: .normal,
            gravity: [0.4, -0.6, 0]
        )

        await model.refreshSetupPassIfNeeded()

        XCTAssertEqual(Array(controller.events.suffix(2)), ["setupPass", "startPreview"])
        XCTAssertEqual(model.setupPassStatus, .aligned)
    }

    @MainActor
    func testPrepareSurfacesCapturePolicyViolations() async throws {
        let controller = FakeCameraCaptureController()
        controller.policyEnforcementReport = CapturePolicyEnforcementReport(
            requested: CapturePolicyRequestedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            achieved: CapturePolicyAchievedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: true,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            violations: ["electronic_stabilization_enabled"]
        )
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()

        XCTAssertEqual(model.capturePolicyStatusText, "Policy issue: electronic_stabilization_enabled")
        XCTAssertEqual(model.capturePolicyEnforcement?.violations, ["electronic_stabilization_enabled"])
    }

    @MainActor
    func testProfileChecklistRecordsStepMetadataForSidecarPayload() {
        let model = CaptureViewModel()

        model.recordCurrentProfileStep(artifactRef: "captures/profile-empty/clip.mov", metadata: ["duration_s": "8.0"])
        model.recordProfileStep(.playerHeightEntry, artifactRef: nil, metadata: ["height_cm": "180"])

        XCTAssertEqual(model.profileFlow.steps[0].status, .complete)
        XCTAssertEqual(model.profileFlow.steps[0].artifactRef, "captures/profile-empty/clip.mov")
        XCTAssertEqual(model.profileFlow.payload.steps[3].metadata["height_cm"], "180")
    }

    @MainActor
    func testToggleRecordingAwaitsInjectedControllerAndStoresReturnedDescriptor() async throws {
        let controller = FakeCameraCaptureController()
        let configuredDescriptor = try Self.captureDescriptor(sessionID: "configured")
        let recordingDescriptor = try Self.captureDescriptor(sessionID: "recording")
        controller.configureDescriptor = configuredDescriptor
        controller.startRecordingDescriptor = recordingDescriptor
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()
        await model.toggleRecording()

        XCTAssertEqual(model.status, .recording)
        XCTAssertEqual(model.descriptor, recordingDescriptor)
        XCTAssertEqual(controller.startRecordingCallCount, 1)
        XCTAssertEqual(model.recordFlowPhase, .recording(startedAt: model.recordingStartedAt!))

        await model.toggleRecording()

        XCTAssertEqual(controller.stopRecordingCallCount, 1)
        XCTAssertEqual(model.recordFlowPhase, .saving)

        controller.onRecordingFinished?(.success(CameraRecordingResult(
            descriptor: recordingDescriptor,
            clipURL: URL(fileURLWithPath: "/tmp/recording.mov")
        )))
        await Task.yield()

        XCTAssertEqual(model.recordFlowPhase, .done(sessionID: "recording"))
    }

    @MainActor
    func testToggleRecordingDoesNotStartWhenCameraPreparationIsBlocked() async throws {
        let controller = FakeCameraCaptureController()
        controller.configureError = CameraCaptureControllerError.permissionDenied(
            CapturePermissionSnapshot(camera: .denied, microphone: .authorized)
        )
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .denied, microphone: .authorized)
            }
        )

        await model.prepare()
        await model.toggleRecording()

        XCTAssertEqual(model.status, .blocked("Enable Camera in Settings, then tap Retry."))
        XCTAssertEqual(model.recordFlowPhase, .permissionDenied("Enable Camera in Settings, then tap Retry."))
        XCTAssertFalse(model.isRecording)
        XCTAssertEqual(controller.startRecordingCallCount, 0)
    }

    @MainActor
    func testPortraitRecordTapCanRetryInLandscapeAndStartRecording() async throws {
        let controller = FakeCameraCaptureController()
        controller.configureDescriptor = try Self.captureDescriptor(sessionID: "configured")
        controller.startRecordingDescriptor = try Self.captureDescriptor(sessionID: "recording")
        controller.startRecordingError = CameraCaptureControllerError.landscapeRequired
        var permissionRequestCount = 0
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                permissionRequestCount += 1
                return CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()
        await model.handleRecordTap()

        XCTAssertEqual(model.status, .blocked("Rotate to landscape to record"))
        XCTAssertEqual(model.blockedReason, "Rotate to landscape to record")
        XCTAssertTrue(model.isRecordButtonEnabled)
        XCTAssertEqual(controller.startRecordingCallCount, 1)

        controller.startRecordingError = nil
        await model.handleRecordTap()

        XCTAssertEqual(model.status, .recording)
        XCTAssertEqual(model.descriptor?.sessionID, "recording")
        XCTAssertEqual(controller.startRecordingCallCount, 2)
        XCTAssertEqual(permissionRequestCount, 2)
        XCTAssertEqual(controller.configureCalls.count, 2)
    }

    @MainActor
    func testC1ConfigureWatchdogCannotLeaveRecordButtonDisabledForever() async throws {
        let controller = FakeCameraCaptureController()
        controller.configureDelayNanoseconds = 5_000_000_000
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            },
            preparationTimeoutNanoseconds: 25_000_000
        )

        await model.prepare()

        XCTAssertEqual(model.status, .blocked(CaptureViewModel.preparationTimeoutMessage))
        XCTAssertEqual(model.blockedReason, CaptureViewModel.preparationTimeoutMessage)
        XCTAssertTrue(model.isRecordButtonEnabled)
    }

    @MainActor
    func testC2ARKitSetupStallTimesOutToLoudBlockedState() async throws {
        let controller = FakeCameraCaptureController()
        controller.setupPassDelayNanoseconds = 5_000_000_000
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            },
            preparationTimeoutNanoseconds: 25_000_000
        )

        await model.prepare()

        XCTAssertEqual(model.status, .blocked(CaptureViewModel.preparationTimeoutMessage))
        XCTAssertTrue(model.isRecordButtonEnabled)
        XCTAssertEqual(controller.events, ["configure", "setupPass"])
    }

    @MainActor
    func testC3PreviewOwnershipFailureSurfacesBlockedReasonAndRetry() async throws {
        let controller = FakeCameraCaptureController()
        controller.startPreviewError = CameraCaptureControllerError.cameraResourceBusy(.arKitSetup)
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            }
        )

        await model.prepare()

        XCTAssertEqual(model.status, .blocked("Camera is still busy finishing alignment. Tap Retry."))
        XCTAssertEqual(model.blockedReason, "Camera is still busy finishing alignment. Tap Retry.")
        XCTAssertTrue(model.isRecordButtonEnabled)
    }

    @MainActor
    func testC4ConcurrentPrepareCallsCoalesceIntoOnePreparation() async throws {
        let controller = FakeCameraCaptureController()
        controller.configureDelayNanoseconds = 50_000_000
        var permissionRequestCount = 0
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                permissionRequestCount += 1
                return CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            },
            preparationTimeoutNanoseconds: 1_000_000_000
        )

        async let first: Void = model.prepare()
        async let second: Void = model.prepare()
        _ = await (first, second)

        XCTAssertEqual(permissionRequestCount, 1)
        XCTAssertEqual(controller.configureCalls.count, 1)
        XCTAssertEqual(controller.events, ["configure", "setupPass", "startPreview"])
        XCTAssertEqual(model.status, .ready)
    }

    @MainActor
    func testC5DeniedTCCPermissionsAreActionableAndRetryable() async throws {
        let controller = FakeCameraCaptureController()
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .denied, microphone: .authorized)
            }
        )

        await model.prepare()

        XCTAssertEqual(model.status, .blocked("Enable Camera in Settings, then tap Retry."))
        XCTAssertEqual(model.recordFlowPhase, .permissionDenied("Enable Camera in Settings, then tap Retry."))
        XCTAssertTrue(model.isRecordButtonEnabled)
        XCTAssertEqual(controller.configureCalls.count, 0)
    }

    @MainActor
    func testC6FirstAppearanceOrientationRefreshPrecedesConfiguration() async throws {
        let controller = FakeCameraCaptureController()
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            },
            orientationResolver: { isLandscapeViewport in
                isLandscapeViewport ? .landscapeRight : .portrait
            }
        )

        await model.prepare(isLandscapeViewport: false)

        XCTAssertEqual(model.captureDeviceOrientation, .portrait)
        XCTAssertEqual(controller.configureCalls, [
            FakeCameraCaptureController.ConfigureCall(mode: .standard60, orientation: .portrait),
        ])
    }

    @MainActor
    func testRecordingStartWatchdogCannotBecomeAnotherSilentTap() async throws {
        let controller = FakeCameraCaptureController()
        controller.startRecordingDelayNanoseconds = 5_000_000_000
        let model = CaptureViewModel(
            controller: controller,
            requestPermissions: {
                CapturePermissionSnapshot(camera: .authorized, microphone: .authorized)
            },
            preparationTimeoutNanoseconds: 25_000_000
        )
        await model.prepare()

        await model.toggleRecording()

        XCTAssertEqual(model.status, .blocked(CaptureViewModel.recordActionTimeoutMessage))
        XCTAssertEqual(model.blockedReason, CaptureViewModel.recordActionTimeoutMessage)
        XCTAssertTrue(model.isRecordButtonEnabled)
    }

    fileprivate static func captureDescriptor(sessionID: String) throws -> CapturePackageDescriptor {
        try CapturePackageDescriptor(
            sessionID: sessionID,
            policy: CapturePolicy.recommended(
                for: .standard60,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                orientation: .landscape
            ),
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .landscapeRight
        )
    }
}

private final class FakeCameraCaptureController: CameraCaptureControlling, @unchecked Sendable {
    struct ConfigureCall: Equatable {
        var mode: CaptureMode
        var orientation: CaptureDeviceOrientation
    }

    let session = AVCaptureSession()
    var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)?
    var configureDescriptor: CapturePackageDescriptor?
    var configureError: Error?
    var startRecordingDescriptor: CapturePackageDescriptor?
    var startRecordingError: Error?
    var startPreviewError: Error?
    var configureDelayNanoseconds: UInt64 = 0
    var setupPassDelayNanoseconds: UInt64 = 0
    var startRecordingDelayNanoseconds: UInt64 = 0
    var policyEnforcementReport: CapturePolicyEnforcementReport?
    var setupPass: ARKitSetupPassSidecar = .unavailable(reason: "test_setup_pass_unavailable", gravity: [0, -1, 0])
    var gravity: [Double] = [0, -1, 0]
    private(set) var events: [String] = []
    private(set) var configureCalls: [ConfigureCall] = []
    private(set) var startPreviewCallCount = 0
    private(set) var stopPreviewCallCount = 0
    private(set) var startRecordingCallCount = 0
    private(set) var stopRecordingCallCount = 0

    func configure(
        mode: CaptureMode,
        deviceTier _: DeviceTier,
        capabilities _: CaptureCodecCapabilities,
        captureDeviceOrientation: CaptureDeviceOrientation,
        sessionID _: String,
        packageRootURL _: URL
    ) async throws -> CapturePackageDescriptor {
        events.append("configure")
        configureCalls.append(ConfigureCall(mode: mode, orientation: captureDeviceOrientation))
        if configureDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: configureDelayNanoseconds)
        }
        if let configureError {
            throw configureError
        }
        if let configureDescriptor {
            return configureDescriptor
        }
        return try CaptureViewModelTests.captureDescriptor(sessionID: "fake-configured")
    }

    func startPreview() async throws {
        events.append("startPreview")
        startPreviewCallCount += 1
        if let startPreviewError {
            throw startPreviewError
        }
    }

    func stopPreview() async throws {
        events.append("stopPreview")
        stopPreviewCallCount += 1
    }

    func performARKitSetupPass(timeoutSeconds _: Double) async -> ARKitSetupPassSidecar {
        events.append("setupPass")
        if setupPassDelayNanoseconds > 0 {
            try? await Task.sleep(nanoseconds: setupPassDelayNanoseconds)
        }
        return setupPass
    }

    func latestGravity() async -> [Double] {
        gravity
    }

    func startRecording() async throws -> CapturePackageDescriptor {
        events.append("startRecording")
        startRecordingCallCount += 1
        if startRecordingDelayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: startRecordingDelayNanoseconds)
        }
        if let startRecordingError {
            throw startRecordingError
        }
        if let startRecordingDescriptor {
            return startRecordingDescriptor
        }
        return try CaptureViewModelTests.captureDescriptor(sessionID: "fake-recording")
    }

    func stopRecording() async throws {
        events.append("stopRecording")
        stopRecordingCallCount += 1
    }

    func currentPolicyEnforcementReport() async -> CapturePolicyEnforcementReport? {
        policyEnforcementReport
    }
}

private extension CapturePolicyEnforcementReport {
    static let compliant60FPS = CapturePolicyEnforcementReport(
        requested: CapturePolicyRequestedState(
            fps: 60,
            resolution: [1920, 1080],
            format: .hevc,
            orientation: .landscape,
            electronicStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            whiteBalanceLocked: true
        ),
        achieved: CapturePolicyAchievedState(
            fps: 60,
            resolution: [1920, 1080],
            format: .hevc,
            orientation: .landscape,
            electronicStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            whiteBalanceLocked: true
        ),
        violations: []
    )
}
