import XCTest
@testable import PickleballFastTier

final class LiveBallIndicatorTests: XCTestCase {
    func testDefaultBuildFlagIsUntrained() {
        XCTAssertFalse(LiveBallIndicatorPolicy.modelIsTrainedInThisBuild)
    }

    func testUntrainedModelAlwaysReturnsComingSoonWithNoPosition() {
        let state = LiveBallIndicatorPolicy.evaluate(rawConfidence: 0.2, rawNormalizedX: 0.4, rawNormalizedY: 0.6)

        XCTAssertEqual(state.availability, .comingSoon)
        XCTAssertEqual(state.badgeText, "Ball tracking: coming soon")
        XCTAssertNil(state.normalizedX)
        XCTAssertNil(state.normalizedY)
        XCTAssertNil(state.confidence)
    }

    func testUntrainedModelIgnoresConfidenceEvenWhenHigh() {
        // The whole point of the gate: a suspiciously-confident untrained
        // model must never leak through as a real ball position.
        let state = LiveBallIndicatorPolicy.evaluate(rawConfidence: 0.99, rawNormalizedX: 0.5, rawNormalizedY: 0.5)

        XCTAssertEqual(state.availability, .comingSoon)
        XCTAssertNil(state.normalizedX)
        XCTAssertNil(state.normalizedY)
    }

    func testDefaultStaticStateMatchesEvaluateWithNilInputs() {
        XCTAssertEqual(LiveBallIndicatorState.comingSoon, LiveBallIndicatorPolicy.evaluate(rawConfidence: nil, rawNormalizedX: nil, rawNormalizedY: nil))
    }

    func testTrainedModelBelowThresholdReportsLowConfidenceWithoutAPosition() {
        let state = LiveBallIndicatorPolicy.evaluate(
            rawConfidence: 0.2,
            rawNormalizedX: 0.4,
            rawNormalizedY: 0.6,
            modelIsTrained: true
        )

        XCTAssertEqual(state.availability, .lowConfidence)
        XCTAssertNil(state.normalizedX)
        XCTAssertNil(state.normalizedY)
        XCTAssertEqual(state.confidence, 0.2)
    }

    func testTrainedModelAtOrAboveThresholdReportsTrackingWithAPosition() {
        let state = LiveBallIndicatorPolicy.evaluate(
            rawConfidence: 0.75,
            rawNormalizedX: 0.3,
            rawNormalizedY: 0.65,
            modelIsTrained: true
        )

        XCTAssertEqual(state.availability, .tracking)
        XCTAssertEqual(state.normalizedX, 0.3)
        XCTAssertEqual(state.normalizedY, 0.65)
        XCTAssertEqual(state.confidence, 0.75)
    }

    func testTrainedModelWithMissingPositionStillReportsLowConfidenceRatherThanCrashing() {
        let state = LiveBallIndicatorPolicy.evaluate(
            rawConfidence: 0.9,
            rawNormalizedX: nil,
            rawNormalizedY: 0.5,
            modelIsTrained: true
        )

        XCTAssertEqual(state.availability, .lowConfidence)
    }
}
