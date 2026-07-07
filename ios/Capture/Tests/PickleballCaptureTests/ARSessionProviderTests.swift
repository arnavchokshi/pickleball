import XCTest
import PickleballCore
@testable import PickleballCapture

final class ARSessionProviderTests: XCTestCase {
    func testFakeARProviderPoseSamplesPopulateSidecarWithVideoPTS() throws {
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
        XCTAssertEqual(sidecar.arkitFrameSamples.last?.tracking.state, .normal)
        XCTAssertNil(sidecar.unavailableSensorReasons["arkit_camera_pose"])
    }

    private static func pose(tx: Double) -> RigidPose {
        RigidPose(
            R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            t: [tx, 1.4, 0]
        )
    }
}
