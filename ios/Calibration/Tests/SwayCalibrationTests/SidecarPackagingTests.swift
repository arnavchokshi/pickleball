import XCTest
@testable import SwayCalibration
import SwayCore

final class SidecarPackagingTests: XCTestCase {
    func testARKitSetupPassCanPackageServerCaptureSidecar() throws {
        let setupPass = ARKitSetupPassSidecar(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            cameraPose: Self.identityPose,
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0]),
            trackingState: .normal
        )

        let sidecar = try CalibrationSidecarPackager.package(
            seed: setupPass.calibrationSeed(),
            imageSize: ImageSize(width: 1920, height: 1080),
            metadata: Self.metadata,
            manualTaps: nil
        )

        XCTAssertEqual(sidecar.intrinsics.source, "arkit")
        XCTAssertEqual(sidecar.arkitCameraPose, Self.identityPose)
        XCTAssertEqual(sidecar.courtPlane, Plane(point: [0, 0, 0], normal: [0, 1, 0]))
        XCTAssertTrue(sidecar.manualCourtTaps.isEmpty)
        XCTAssertEqual(sidecar.captureQuality.grade, .good)
    }

    func testManualFallbackPackagesOrderedTapCorrespondencesWithoutARKitPose() throws {
        let seed = CalibrationSeed(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "manual_estimate"),
            manualCourtTaps: [[1840, 1010], [110, 95], [1810, 120], [95, 1000]]
        )

        let sidecar = try CalibrationSidecarPackager.package(
            seed: seed,
            imageSize: ImageSize(width: 1920, height: 1080),
            metadata: Self.metadata,
            manualTaps: ManualCourtTaps(imagePoints: seed.manualCourtTaps)
        )

        XCTAssertNil(sidecar.arkitCameraPose)
        XCTAssertNil(sidecar.courtPlane)
        XCTAssertEqual(sidecar.manualCourtTaps, [[110, 95], [1810, 120], [1840, 1010], [95, 1000]])
        XCTAssertEqual(sidecar.captureQuality.grade, .warn)
        XCTAssertEqual(sidecar.captureQuality.reasons, ["manual_calibration_fallback"])
    }

    private static let metadata = CalibrationSidecarMetadata(
        deviceTier: .standard,
        deviceModel: "iPhone16,2",
        fps: 60,
        format: .hevc,
        locked: LockedCapture(exposureS: 0.001, iso: 320, focus: 0.7, wbLocked: true),
        gravity: [0, -1, 0],
        lidarDepthRefs: [],
        ondevicePoseTrack: "ondevice_pose.json"
    )

    private static let identityPose = RigidPose(
        R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t: [0, 1.4, 0]
    )
}
