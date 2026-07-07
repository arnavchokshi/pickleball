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

        await model.toggleRecording()

        XCTAssertEqual(controller.stopRecordingCallCount, 1)
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

        XCTAssertEqual(model.status, .blocked("Camera or microphone access needed"))
        XCTAssertFalse(model.isRecording)
        XCTAssertEqual(controller.startRecordingCallCount, 0)
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
    var policyEnforcementReport: CapturePolicyEnforcementReport?
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
        configureCalls.append(ConfigureCall(mode: mode, orientation: captureDeviceOrientation))
        if let configureError {
            throw configureError
        }
        if let configureDescriptor {
            return configureDescriptor
        }
        return try CaptureViewModelTests.captureDescriptor(sessionID: "fake-configured")
    }

    func startPreview() async {
        startPreviewCallCount += 1
    }

    func stopPreview() async {
        stopPreviewCallCount += 1
    }

    func startRecording() async throws -> CapturePackageDescriptor {
        startRecordingCallCount += 1
        if let startRecordingDescriptor {
            return startRecordingDescriptor
        }
        return try CaptureViewModelTests.captureDescriptor(sessionID: "fake-recording")
    }

    func stopRecording() async throws {
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
