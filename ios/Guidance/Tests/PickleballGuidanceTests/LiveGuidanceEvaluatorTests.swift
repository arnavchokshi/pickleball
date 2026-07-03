import XCTest
@testable import PickleballGuidance

final class LiveGuidanceEvaluatorTests: XCTestCase {
    func testAllMeasurementsMissingRendersEveryCheckUnavailableAndGradeWarnNotGood() {
        let state = LiveGuidanceEvaluator.evaluate(LiveGuidanceSample())

        XCTAssertEqual(state.checks.map(\.status), Array(repeating: .unavailable, count: state.checks.count))
        XCTAssertEqual(state.grade, .warn)
        XCTAssertEqual(state.manualFramingTip, LiveGuidanceEvaluator.manualFramingTipText)
        XCTAssertEqual(state.setupTips, [])
    }

    func testGoodSampleAcrossAllMeasuredSignalsGradesGood() {
        let sample = LiveGuidanceSample(
            exposureTargetOffsetEV: 0.1,
            isExposureLocked: true,
            shutterSeconds: 1.0 / 1000.0,
            minimumSharpShutterSeconds: 1.0 / 500.0,
            tiltFromLevelDegrees: 1.2,
            requestedFPS: 60,
            configuredFPS: 60.0,
            expectedResolution: [1920, 1080],
            configuredResolution: [1920, 1080]
        )

        let state = LiveGuidanceEvaluator.evaluate(sample)

        XCTAssertEqual(state.grade, .good)
        XCTAssertTrue(state.checks.allSatisfy { $0.status == .good })
    }

    func testBadExposureAndLevelWarnWithoutPoisoningUnavailableChecks() {
        let sample = LiveGuidanceSample(
            exposureTargetOffsetEV: 2.0,
            tiltFromLevelDegrees: 12.0
            // shutter/fps/resolution left nil on purpose -- must stay unavailable, not warn.
        )

        let state = LiveGuidanceEvaluator.evaluate(sample)

        let byID = Dictionary(uniqueKeysWithValues: state.checks.map { ($0.id, $0.status) })
        XCTAssertEqual(byID["exposure"], .warn)
        XCTAssertEqual(byID["level"], .warn)
        XCTAssertEqual(byID["blur"], .unavailable)
        XCTAssertEqual(byID["frame_rate"], .unavailable)
        XCTAssertEqual(byID["resolution"], .unavailable)
        // Only 2 measured checks and both warn -> overall warn (not poor, which requires 3+ warns).
        XCTAssertEqual(state.grade, .warn)
    }

    func testThreeOrMoreMeasuredWarningsGradePoor() {
        let sample = LiveGuidanceSample(
            exposureTargetOffsetEV: 2.0,
            shutterSeconds: 1.0 / 60.0,
            minimumSharpShutterSeconds: 1.0 / 500.0,
            tiltFromLevelDegrees: 12.0,
            requestedFPS: 60,
            configuredFPS: 60.0,
            expectedResolution: [1920, 1080],
            configuredResolution: [1920, 1080]
        )

        let state = LiveGuidanceEvaluator.evaluate(sample)

        XCTAssertEqual(state.grade, .poor)
    }

    func testFrameRateAndResolutionShortfallsWarn() {
        let sample = LiveGuidanceSample(
            requestedFPS: 60,
            configuredFPS: 30.0,
            expectedResolution: [1920, 1080],
            configuredResolution: [1280, 720]
        )

        let state = LiveGuidanceEvaluator.evaluate(sample)

        let byID = Dictionary(uniqueKeysWithValues: state.checks.map { ($0.id, $0.status) })
        XCTAssertEqual(byID["frame_rate"], .warn)
        XCTAssertEqual(byID["resolution"], .warn)
    }

    func testSetupTipsSurfaceKnownReasonsAsHumanReadableTextAndPassThroughUnknownReasons() {
        let sample = LiveGuidanceSample(setupTipReasons: ["arkit_seed_missing", "court_plane_missing", "some_future_reason"])

        let state = LiveGuidanceEvaluator.evaluate(sample)

        XCTAssertEqual(state.setupTips.count, 3)
        XCTAssertTrue(state.setupTips[0].contains("ARKit"))
        XCTAssertTrue(state.setupTips[1].contains("court plane"))
        XCTAssertEqual(state.setupTips[2], "some_future_reason")
    }

    func testHumanReadableSetupTipFallsBackToRawReasonForUnknownCodes() {
        XCTAssertEqual(LiveGuidanceEvaluator.humanReadableSetupTip(for: "unmapped_code"), "unmapped_code")
        XCTAssertTrue(LiveGuidanceEvaluator.humanReadableSetupTip(for: "intrinsics_estimated_from_fov").contains("intrinsics"))
    }
}

final class LiveTiltEstimatorTests: XCTestCase {
    func testLevelLandscapeRightGravityYieldsNearZeroTilt() {
        let axis = LiveTiltEstimator.expectedLevelAxis(for: .landscapeRight)
        XCTAssertEqual(axis, [-1.0, 0.0, 0.0])

        let tilt = LiveTiltEstimator.tiltDegrees(gravity: [-1.0, 0.0, 0.0], expectedLevelAxis: axis)

        XCTAssertEqual(try XCTUnwrap(tilt), 0.0, accuracy: 0.001)
    }

    func test45DegreeTiltFromLevelMeasuresApproximately45Degrees() {
        let axis = LiveTiltEstimator.expectedLevelAxis(for: .landscapeRight)
        let tiltedGravity = [-cos(Double.pi / 4), sin(Double.pi / 4), 0.0]

        let tilt = LiveTiltEstimator.tiltDegrees(gravity: tiltedGravity, expectedLevelAxis: axis)

        XCTAssertEqual(try XCTUnwrap(tilt), 45.0, accuracy: 0.01)
    }

    func testFullyUpsideDownGravityMeasures180Degrees() {
        let axis = LiveTiltEstimator.expectedLevelAxis(for: .portrait)

        let tilt = LiveTiltEstimator.tiltDegrees(gravity: [0.0, 1.0, 0.0], expectedLevelAxis: axis)

        XCTAssertEqual(try XCTUnwrap(tilt), 180.0, accuracy: 0.001)
    }

    func testZeroMagnitudeGravityReturnsNil() {
        let axis = LiveTiltEstimator.expectedLevelAxis(for: .landscapeRight)

        XCTAssertNil(LiveTiltEstimator.tiltDegrees(gravity: [0.0, 0.0, 0.0], expectedLevelAxis: axis))
    }

    func testMalformedVectorLengthReturnsNil() {
        XCTAssertNil(LiveTiltEstimator.tiltDegrees(gravity: [1.0, 0.0], expectedLevelAxis: [1.0, 0.0, 0.0]))
    }
}
