import XCTest
@testable import SwayGuidance

final class CaptureGuidanceEvaluatorTests: XCTestCase {
    func testGoodGuidanceSampleStaysSilent() {
        let state = CaptureGuidanceEvaluator.evaluate(
            GuidanceSample(
                visibleCourtCornerRatio: 1.0,
                isTrackingNormal: true,
                exposureTargetOffset: 0.0,
                clippedPixelRatio: 0.0,
                shutterSeconds: 1.0 / 1000.0,
                minimumSharpShutterSeconds: 1.0 / 500.0,
                rollDegrees: 0.5,
                pitchDegrees: 1.0,
                accelerationVarianceG: 0.01
            )
        )

        XCTAssertEqual(state.status, .good)
        XCTAssertEqual(state.flags, [])
        XCTAssertEqual(state.captureQuality.grade, .good)
        XCTAssertEqual(state.captureQuality.reasons, [])
    }

    func testBadGuidanceSampleFlagsEveryCaptureQualityRule() {
        let state = CaptureGuidanceEvaluator.evaluate(
            GuidanceSample(
                visibleCourtCornerRatio: 0.5,
                isTrackingNormal: false,
                exposureTargetOffset: 1.2,
                clippedPixelRatio: 0.08,
                shutterSeconds: 1.0 / 120.0,
                minimumSharpShutterSeconds: 1.0 / 500.0,
                rollDegrees: 6.0,
                pitchDegrees: 8.0,
                accelerationVarianceG: 0.12
            )
        )

        XCTAssertEqual(state.status, .needsCorrection)
        XCTAssertEqual(state.flags, [.framing, .tracking, .exposure, .blur, .level, .shake])
        XCTAssertEqual(state.captureQuality.grade, .poor)
        XCTAssertEqual(
            state.captureQuality.reasons,
            [
                "framing_court_corners_not_visible",
                "tracking_not_normal",
                "exposure_clipping_or_offset",
                "motion_blur_risk",
                "phone_not_level",
                "camera_shake",
            ]
        )
    }

    func testMissingGuidanceMeasurementsFailClosed() {
        let state = CaptureGuidanceEvaluator.evaluate(GuidanceSample())

        XCTAssertEqual(state.status, .failClosed)
        XCTAssertEqual(state.flags, [.framing, .tracking, .exposure, .blur, .level, .shake])
        XCTAssertEqual(state.captureQuality.grade, .poor)
        XCTAssertEqual(
            state.captureQuality.reasons,
            [
                "missing_framing_measurement",
                "missing_tracking_measurement",
                "missing_exposure_measurement",
                "missing_blur_measurement",
                "missing_level_measurement",
                "missing_shake_measurement",
            ]
        )
    }
}
