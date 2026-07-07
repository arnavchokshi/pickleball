import XCTest
import PickleballCore
@testable import PickleballCapture

final class ARSessionProviderTests: XCTestCase {
    func testSetupPassStopsARBeforeAVCaptureOwnershipCanStart() async throws {
        let provider = DeterministicARSessionProvider(samples: [
            ARFrameSnapshot(
                timestampS: 10.0,
                cameraPose: Self.pose(tx: 0.1),
                intrinsics: CameraIntrinsics(fx: 1180, fy: 1190, cx: 960, cy: 540, source: "arkit"),
                tracking: ARTrackingSnapshot(state: .normal, quality: .good),
                courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0])
            ),
        ])
        let ownership = CameraResourceOwnership()
        let runner = ARKitSetupPassRunner(
            provider: provider,
            ownership: ownership,
            gravityProvider: { [0, -1, 0] },
            pollIntervalNanoseconds: 0
        )

        let setupPass = await runner.run(timeoutSeconds: 0.01)

        XCTAssertEqual(setupPass.status, .available)
        XCTAssertEqual(setupPass.cameraPose, Self.pose(tx: 0.1))
        XCTAssertEqual(setupPass.gravity, [0, -1, 0])
        XCTAssertFalse(provider.isRunning)
        XCTAssertNil(ownership.activeOwner)

        let captureToken = try ownership.beginAVCapture()
        XCTAssertEqual(ownership.activeOwner, .avCapture)
        captureToken.release()
        XCTAssertNil(ownership.activeOwner)
    }

    func testCameraOwnershipRejectsSimultaneousARAndAVCapture() throws {
        let ownership = CameraResourceOwnership()

        let captureToken = try ownership.beginAVCapture()
        XCTAssertThrowsError(try ownership.beginARKitSetup()) { error in
            XCTAssertEqual(error as? CameraResourceOwnershipError, .cameraAlreadyOwned(.avCapture))
        }
        captureToken.release()

        let arToken = try ownership.beginARKitSetup()
        XCTAssertThrowsError(try ownership.beginAVCapture()) { error in
            XCTAssertEqual(error as? CameraResourceOwnershipError, .cameraAlreadyOwned(.arKitSetup))
        }
        arToken.release()
    }

    func testSetupPassUnavailableStopsProviderAndReportsReason() async {
        let provider = DeterministicARSessionProvider(samples: [])
        let runner = ARKitSetupPassRunner(
            provider: provider,
            ownership: CameraResourceOwnership(),
            gravityProvider: { [0, -1, 0] },
            pollIntervalNanoseconds: 0
        )

        let setupPass = await runner.run(timeoutSeconds: 0)

        XCTAssertEqual(setupPass.status, .unavailable)
        XCTAssertEqual(setupPass.unavailableReason, "arkit_setup_pass_timeout")
        XCTAssertFalse(provider.isRunning)
    }

    func testCoreMotionOnlyFrameSamplesDoNotFabricateARKitPose() {
        let recorder = CoreMotionFrameSidecarRecorder(maxSamples: 2)

        recorder.beginRecording()
        recorder.recordVideoFrame(ptsS: 0.0, gravity: [0, -1, 0])
        recorder.recordVideoFrame(ptsS: 0.041, gravity: [0.01, -0.99, 0.02])
        recorder.endRecording()

        let samples = recorder.frameSamples()

        XCTAssertEqual(samples.map(\.provenance), [.coreMotionOnly, .coreMotionOnly])
        XCTAssertEqual(samples.map(\.videoPTSS), [0.0, 0.041])
        XCTAssertEqual(samples.first?.gravity, [0, -1, 0])
        XCTAssertNil(samples.first?.cameraPose)
        XCTAssertNil(samples.first?.intrinsics)
        XCTAssertNil(samples.first?.tracking)
    }

    func testSetupRefreshPolicyUsesAgeAndGravityDelta() {
        let completedAt = Date(timeIntervalSince1970: 100)

        XCTAssertFalse(ARKitSetupPassRefreshPolicy.shouldRefresh(
            now: Date(timeIntervalSince1970: 140),
            lastCompletedAt: completedAt,
            lastGravity: [0, -1, 0],
            currentGravity: [0.01, -0.99, 0.0]
        ))
        XCTAssertTrue(ARKitSetupPassRefreshPolicy.shouldRefresh(
            now: Date(timeIntervalSince1970: 221),
            lastCompletedAt: completedAt,
            lastGravity: [0, -1, 0],
            currentGravity: [0.01, -0.99, 0.0]
        ))
        XCTAssertTrue(ARKitSetupPassRefreshPolicy.shouldRefresh(
            now: Date(timeIntervalSince1970: 140),
            lastCompletedAt: completedAt,
            lastGravity: [0, -1, 0],
            currentGravity: [0.3, -0.7, 0.0]
        ))
    }

    func testFakeARProviderPoseSamplesPopulateSidecarWithSetupPassAndCoreMotionPTS() throws {
        let setupPass = ARKitSetupPassSidecar(
            intrinsics: CameraIntrinsics(fx: 1182, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            cameraPose: Self.pose(tx: 0.2),
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0]),
            trackingState: .normal,
            timestampS: 10.04,
            gravity: [0, -1, 0]
        )
        let recorder = CoreMotionFrameSidecarRecorder(maxSamples: 4)

        recorder.beginRecording()
        recorder.recordVideoFrame(ptsS: 0.0, gravity: [0, -1, 0])
        recorder.recordVideoFrame(ptsS: 0.041, gravity: [0, -1, 0])
        recorder.endRecording()

        let descriptor = try CapturePackageDescriptor(
            sessionID: "arkit-sidecar",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly),
            startedAt: Date(timeIntervalSince1970: 0)
        )
        let context = CaptureSidecarWriteContext(
            deviceTier: .standard,
            deviceModel: "iPhone16,2",
            cameraPosition: "back",
            cameraLens: "builtInWideAngleCamera",
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.7, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 900, fy: 900, cx: 960, cy: 540, source: "avfoundation_fov_estimate"),
            gravity: [0, -1, 0],
            arkit: ARCaptureSidecarPayload(setupPass: setupPass, frameSamples: recorder.frameSamples()),
            policyEnforcement: nil,
            profileCapture: nil,
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: ["arkit_seed_missing", "court_plane_missing", "intrinsics_estimated_from_fov"]
            )
        )

        let sidecar = CaptureSidecarWriter.makeSidecar(
            descriptor: descriptor,
            recordingStartedAt: Date(timeIntervalSince1970: 1),
            finishedAt: Date(timeIntervalSince1970: 3),
            context: context
        )

        XCTAssertEqual(sidecar.setupPass, setupPass)
        XCTAssertEqual(sidecar.intrinsics?.source, "arkit")
        XCTAssertEqual(sidecar.arkitCameraPose, Self.pose(tx: 0.2))
        XCTAssertEqual(sidecar.courtPlane, Plane(point: [0, 0, 0], normal: [0, 1, 0]))
        XCTAssertEqual(sidecar.arkitFrameSamples.map(\.provenance), [.coreMotionOnly, .coreMotionOnly])
        XCTAssertEqual(sidecar.arkitFrameSamples.map(\.videoPTSS), [0.0, 0.041])
        XCTAssertNil(sidecar.arkitFrameSamples.first?.cameraPose)
        XCTAssertEqual(sidecar.captureQuality.grade, .good)
        XCTAssertNil(sidecar.unavailableSensorReasons["arkit_camera_pose"])
        XCTAssertNil(sidecar.unavailableSensorReasons["court_plane"])
    }

    func testUnavailableSetupPassKeepsSidecarHonest() throws {
        let setupPass = ARKitSetupPassSidecar.unavailable(
            reason: "arkit_setup_pass_timeout",
            gravity: [0, -1, 0]
        )
        let descriptor = try CapturePackageDescriptor(
            sessionID: "arkit-unavailable",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly),
            startedAt: Date(timeIntervalSince1970: 0)
        )
        let context = CaptureSidecarWriteContext(
            deviceTier: .standard,
            deviceModel: "iPhone16,2",
            cameraPosition: "back",
            cameraLens: "builtInWideAngleCamera",
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.7, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 900, fy: 900, cx: 960, cy: 540, source: "avfoundation_fov_estimate"),
            gravity: [0, -1, 0],
            arkit: ARCaptureSidecarPayload(setupPass: setupPass, frameSamples: []),
            policyEnforcement: nil,
            profileCapture: nil,
            captureQuality: CaptureQuality(
                grade: .warn,
                reasons: ["arkit_seed_missing", "court_plane_missing", "intrinsics_estimated_from_fov"]
            )
        )

        let sidecar = CaptureSidecarWriter.makeSidecar(
            descriptor: descriptor,
            recordingStartedAt: Date(timeIntervalSince1970: 1),
            finishedAt: Date(timeIntervalSince1970: 3),
            context: context
        )

        XCTAssertEqual(sidecar.setupPass, setupPass)
        XCTAssertNil(sidecar.arkitCameraPose)
        XCTAssertNil(sidecar.courtPlane)
        XCTAssertEqual(sidecar.intrinsics?.source, "avfoundation_fov_estimate")
        XCTAssertEqual(sidecar.unavailableSensorReasons["arkit_camera_pose"], "arkit_setup_pass_timeout")
        XCTAssertEqual(sidecar.unavailableSensorReasons["court_plane"], "arkit_setup_pass_timeout")
        XCTAssertEqual(sidecar.captureQuality.reasons, ["arkit_seed_missing", "court_plane_missing", "intrinsics_estimated_from_fov"])
    }

    func testLegacyARProviderPoseSamplesStillRoundTripForFixtureCompatibility() throws {
        let provider = DeterministicARSessionProvider(samples: [
            ARFrameSnapshot(
                timestampS: 10.0,
                cameraPose: Self.pose(tx: 0.1),
                intrinsics: CameraIntrinsics(fx: 1180, fy: 1190, cx: 960, cy: 540, source: "arkit"),
                tracking: ARTrackingSnapshot(state: .normal, quality: .good)
            ),
            ARFrameSnapshot(
                timestampS: 10.04,
                cameraPose: Self.pose(tx: 0.2),
                intrinsics: CameraIntrinsics(fx: 1182, fy: 1192, cx: 960, cy: 540, source: "arkit"),
                tracking: ARTrackingSnapshot(state: .normal, quality: .good)
            ),
        ])
        let recorder = ARFrameSidecarRecorder(provider: provider)

        recorder.start()
        recorder.recordVideoFrame(ptsS: 0.0)
        recorder.recordVideoFrame(ptsS: 0.041)
        recorder.stop()

        let descriptor = try CapturePackageDescriptor(
            sessionID: "arkit-sidecar",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly),
            startedAt: Date(timeIntervalSince1970: 0)
        )
        let context = CaptureSidecarWriteContext(
            deviceTier: .standard,
            deviceModel: "iPhone16,2",
            cameraPosition: "back",
            cameraLens: "builtInWideAngleCamera",
            locked: LockedCapture(exposureS: 0.001, iso: 200, focus: 0.7, wbLocked: true),
            intrinsics: CameraIntrinsics(fx: 900, fy: 900, cx: 960, cy: 540, source: "avfoundation_fov_estimate"),
            gravity: [0, -1, 0],
            arkit: recorder.sidecarPayload(),
            policyEnforcement: nil,
            profileCapture: nil,
            captureQuality: CaptureQuality(grade: .good)
        )

        let sidecar = CaptureSidecarWriter.makeSidecar(
            descriptor: descriptor,
            recordingStartedAt: Date(timeIntervalSince1970: 1),
            finishedAt: Date(timeIntervalSince1970: 3),
            context: context
        )

        XCTAssertEqual(sidecar.intrinsics?.source, "arkit")
        XCTAssertEqual(sidecar.arkitCameraPose, Self.pose(tx: 0.2))
        XCTAssertEqual(sidecar.arkitFrameSamples.map(\.videoPTSS), [0.0, 0.041])
        XCTAssertEqual(sidecar.arkitFrameSamples.map(\.arkitTimestampS), [10.0, 10.04])
        XCTAssertEqual(sidecar.arkitFrameSamples.last?.tracking?.state, .normal)
        XCTAssertNil(sidecar.unavailableSensorReasons["arkit_camera_pose"])
    }

    private static func pose(tx: Double) -> RigidPose {
        RigidPose(
            R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            t: [tx, 1.4, 0]
        )
    }
}
