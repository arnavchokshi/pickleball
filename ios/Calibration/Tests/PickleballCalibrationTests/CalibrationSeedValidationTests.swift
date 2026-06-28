import XCTest
@testable import PickleballCalibration
import PickleballCore

final class CalibrationSeedValidationTests: XCTestCase {
    func testARKitSeedIsValidWhenIntrinsicsPoseAndCourtPlaneArePlausible() {
        let seed = CalibrationSeed(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            arkitCameraPose: Self.identityPose,
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0])
        )

        let report = seed.validationReport(imageSize: ImageSize(width: 1920, height: 1080))

        XCTAssertTrue(report.isUsable)
        XCTAssertTrue(report.issues.isEmpty)
        XCTAssertTrue(report.hasARKitSeed)
        XCTAssertFalse(report.hasManualFallback)
    }

    func testSeedRequiresEitherCompleteARKitSeedOrValidManualFallback() {
        let report = CalibrationSeed(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            arkitCameraPose: Self.identityPose,
            courtPlane: nil
        ).validationReport(imageSize: ImageSize(width: 1920, height: 1080))

        XCTAssertFalse(report.isUsable)
        XCTAssertEqual(report.issues, [.missingCalibrationAnchor])
    }

    func testIntrinsicsMustMatchImageBoundsAndFocalPlausibility() {
        let report = CalibrationSeed(
            intrinsics: CameraIntrinsics(fx: 80, fy: 1192, cx: 3000, cy: 540, source: "arkit"),
            arkitCameraPose: Self.identityPose,
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 1, 0])
        ).validationReport(imageSize: ImageSize(width: 1920, height: 1080))

        XCTAssertFalse(report.isUsable)
        XCTAssertTrue(report.issues.contains(.implausibleIntrinsics))
    }

    func testCourtPlaneRejectsNonFiniteOrNonUnitNormals() {
        let report = CalibrationSeed(
            intrinsics: CameraIntrinsics(fx: 1180, fy: 1192, cx: 960, cy: 540, source: "arkit"),
            arkitCameraPose: Self.identityPose,
            courtPlane: Plane(point: [0, 0, 0], normal: [0, 4, 0])
        ).validationReport(imageSize: ImageSize(width: 1920, height: 1080))

        XCTAssertFalse(report.isUsable)
        XCTAssertTrue(report.issues.contains(.implausibleCourtPlane))
    }

    private static let identityPose = RigidPose(
        R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t: [0, 1.4, 0]
    )
}
