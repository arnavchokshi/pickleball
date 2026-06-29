import XCTest
import PickleballCore
@testable import PickleballCapture

final class CapturePackageTests: XCTestCase {
    func testCapturePackageDescriptorUsesPipelineSafePathsAndSidecarFirstUploadOrder() throws {
        let policy = CapturePolicy.recommended(
            for: .swing120,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: .landscape
        )

        let descriptor = try CapturePackageDescriptor(
            sessionID: "court-a-001",
            policy: policy,
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .landscapeRight
        )

        XCTAssertEqual(descriptor.sessionID, "court-a-001")
        XCTAssertEqual(descriptor.directoryRelativePath, "captures/court-a-001")
        XCTAssertEqual(descriptor.clipRelativePath, "captures/court-a-001/clip.mov")
        XCTAssertEqual(descriptor.sidecarRelativePath, "captures/court-a-001/capture_sidecar.json")
        XCTAssertEqual(descriptor.onDevicePoseTrackRelativePath, "captures/court-a-001/ondevice_pose.json")
        XCTAssertEqual(descriptor.onDevicePersonTracksRelativePath, "captures/court-a-001/on_device_person_tracks.json")
        XCTAssertEqual(descriptor.onDevicePersonTimingRelativePath, "captures/court-a-001/timing.json")
        XCTAssertEqual(
            descriptor.preferredUploadOrder.map(\.relativePath),
            [
                "captures/court-a-001/capture_sidecar.json",
                "captures/court-a-001/ondevice_pose.json",
                "captures/court-a-001/on_device_person_tracks.json",
                "captures/court-a-001/timing.json",
                "captures/court-a-001/clip.mov",
            ]
        )
        XCTAssertEqual(descriptor.expectedFPS, 120)
        XCTAssertEqual(descriptor.expectedResolution, [1920, 1080])
        XCTAssertEqual(descriptor.expectedFormat, .hevc)
        XCTAssertEqual(descriptor.expectedOrientation, .landscape)
        XCTAssertEqual(descriptor.captureDeviceOrientation, .landscapeRight)
        XCTAssertEqual(descriptor.videoRotationAngleDegrees, 90)
    }

    func testCapturePackageDescriptorTracksPortraitCaptureMetadata() throws {
        let policy = CapturePolicy.recommended(
            for: .standard60,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: .portrait
        )

        let descriptor = try CapturePackageDescriptor(
            sessionID: "portrait-001",
            policy: policy,
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .portrait
        )

        XCTAssertEqual(descriptor.expectedResolution, [1080, 1920])
        XCTAssertEqual(descriptor.expectedOrientation, .portrait)
        XCTAssertEqual(descriptor.captureDeviceOrientation, .portrait)
        XCTAssertEqual(descriptor.videoRotationAngleDegrees, 0)
    }

    func testCapturePackageDescriptorRejectsUnsafeSessionIdentifiers() {
        XCTAssertThrowsError(try CapturePackageDescriptor(
            sessionID: "../bad",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly),
            startedAt: Date(timeIntervalSince1970: 0)
        )) { error in
            XCTAssertEqual(error as? CapturePackageDescriptor.ValidationError, .unsafeSessionID("../bad"))
        }
    }

    func testDefaultCapturePackageDescriptorIsLandscapeRecordable() throws {
        let descriptor = try CapturePackageDescriptor(
            sessionID: "default-landscape",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly),
            startedAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(descriptor.expectedOrientation, .landscape)
        XCTAssertEqual(descriptor.captureDeviceOrientation, .landscapeRight)
        XCTAssertEqual(descriptor.videoRotationAngleDegrees, 90)
    }

    func testCaptureSessionIDFactoryAvoidsSameSecondPackageCollisions() throws {
        var factory = CaptureSessionIDFactory()
        let timestamp = Date(timeIntervalSince1970: 1_782_711_200)

        let first = factory.nextSessionID(now: timestamp)
        let second = factory.nextSessionID(now: timestamp)

        XCTAssertNotEqual(first, second)
        XCTAssertTrue(first.hasPrefix("capture-20260629-"))
        XCTAssertTrue(second.hasPrefix("capture-20260629-"))

        let policy = CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly)
        let firstDescriptor = try CapturePackageDescriptor(sessionID: first, policy: policy, startedAt: timestamp)
        let secondDescriptor = try CapturePackageDescriptor(sessionID: second, policy: policy, startedAt: timestamp)

        XCTAssertNotEqual(firstDescriptor.directoryRelativePath, secondDescriptor.directoryRelativePath)
        XCTAssertNotEqual(firstDescriptor.clipRelativePath, secondDescriptor.clipRelativePath)
        XCTAssertNotEqual(firstDescriptor.sidecarRelativePath, secondDescriptor.sidecarRelativePath)
    }

    func testPipelineSensorManifestDeclaresAllCaptureInputsNeededDownstream() {
        let manifest = CaptureSensorManifest.pipelineRequired

        XCTAssertEqual(
            Set(manifest.streams),
            [
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
                .lidarDepthRefs,
            ]
        )
        XCTAssertTrue(manifest.hasRequiredSidecarInputs)
        XCTAssertTrue(manifest.hasOptionalDepthInput)
    }

    func testCaptureReadinessRequiresCameraMicrophoneAndWritablePackage() throws {
        let descriptor = try CapturePackageDescriptor(
            sessionID: "ready",
            policy: CapturePolicy.recommended(
                for: .standard60,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                orientation: .landscape
            ),
            startedAt: Date(timeIntervalSince1970: 0)
        )

        let ready = CaptureReadinessEvaluator.evaluate(
            permissions: CapturePermissionSnapshot(camera: .authorized, microphone: .authorized),
            descriptor: descriptor
        )

        XCTAssertEqual(ready, CaptureReadiness(isReady: true, blockers: []))

        let blocked = CaptureReadinessEvaluator.evaluate(
            permissions: CapturePermissionSnapshot(camera: .denied, microphone: .notDetermined),
            descriptor: descriptor
        )

        XCTAssertEqual(blocked.isReady, false)
        XCTAssertEqual(blocked.blockers, [.cameraPermissionDenied, .microphonePermissionMissing])
    }

    func testCaptureReadinessRequiresLandscapePolicy() throws {
        let descriptor = try CapturePackageDescriptor(
            sessionID: "portrait-blocked",
            policy: CapturePolicy.recommended(
                for: .standard60,
                deviceTier: .standard,
                capabilities: .hevcOnly,
                orientation: .portrait
            ),
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .portrait
        )

        let readiness = CaptureReadinessEvaluator.evaluate(
            permissions: CapturePermissionSnapshot(camera: .authorized, microphone: .authorized),
            descriptor: descriptor
        )

        XCTAssertFalse(readiness.isReady)
        XCTAssertEqual(readiness.blockers, [.landscapeRequired])

        let permissionAndLandscapeBlocked = CaptureReadinessEvaluator.evaluate(
            permissions: CapturePermissionSnapshot(camera: .denied, microphone: .authorized),
            descriptor: descriptor
        )
        XCTAssertEqual(permissionAndLandscapeBlocked.blockers, [.cameraPermissionDenied, .landscapeRequired])
    }

    func testSwiftCapabilityManifestDeclaresRequiredCaptureCalibrationFastTierUploadAndReplayFrameworks() {
        let manifest = CaptureSwiftCapabilityManifest.pipelineRequired

        XCTAssertEqual(
            Set(manifest.capabilities),
            [
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
            ]
        )
        XCTAssertTrue(manifest.declaresCaptureCalibrationRequirements)
        XCTAssertTrue(manifest.declaresFastTierReplayRequirements)
    }
}
