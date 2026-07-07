import XCTest
import PickleballCore
@testable import PickleballCapture

final class CapturePolicyEnforcementTests: XCTestCase {
    func testPolicyEnforcementPassesOnlyWhenLandscapeLockedAndStabilizationOff() {
        let policy = CapturePolicy.recommended(
            for: .standard60,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: .landscape
        )
        let achieved = CapturePolicyAchievedState(
            fps: 60,
            resolution: [1920, 1080],
            format: .hevc,
            orientation: .landscape,
            electronicStabilizationEnabled: false,
            exposureLocked: true,
            focusLocked: true,
            whiteBalanceLocked: true
        )

        let report = CapturePolicyEnforcer.evaluate(policy: policy, achieved: achieved)

        XCTAssertTrue(report.isCompliant)
        XCTAssertEqual(report.violations, [])
        XCTAssertEqual(report.requested.electronicStabilizationEnabled, false)
        XCTAssertEqual(report.achieved?.whiteBalanceLocked, true)
    }

    func testPolicyEnforcementReportsEveryCaptureViolation() {
        let policy = CapturePolicy.recommended(
            for: .standard60,
            deviceTier: .standard,
            capabilities: .hevcOnly,
            orientation: .landscape
        )
        let achieved = CapturePolicyAchievedState(
            fps: 30,
            resolution: [1280, 720],
            format: .hevc,
            orientation: .portrait,
            electronicStabilizationEnabled: true,
            exposureLocked: false,
            focusLocked: false,
            whiteBalanceLocked: false
        )

        let report = CapturePolicyEnforcer.evaluate(policy: policy, achieved: achieved)

        XCTAssertFalse(report.isCompliant)
        XCTAssertEqual(
            report.violations,
            [
                "fps_mismatch",
                "resolution_mismatch",
                "orientation_not_landscape",
                "electronic_stabilization_enabled",
                "exposure_not_locked",
                "focus_not_locked",
                "white_balance_not_locked",
            ]
        )
    }
}
