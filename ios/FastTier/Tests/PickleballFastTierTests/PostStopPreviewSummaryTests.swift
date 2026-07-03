import XCTest
import PickleballCore
@testable import PickleballFastTier

final class PostStopPreviewSummaryTests: XCTestCase {
    func testSummaryWithNoSampledFramesReportsNilPlayerCountRatherThanZero() {
        let summary = PostStopPreviewBuilder.summarize(
            durationSeconds: 12.4,
            requestedFPS: 60,
            captureQuality: CaptureQuality(grade: .warn, reasons: ["arkit_seed_missing"]),
            sampledFrameDetectionCounts: [],
            elapsedBuildSeconds: 2.1
        )

        XCTAssertNil(summary.estimatedPlayerCount)
        XCTAssertEqual(summary.playerCountSampleFrameCount, 0)
        XCTAssertEqual(summary.captureQualityGrade, .warn)
        XCTAssertEqual(summary.captureQualityReasons, ["arkit_seed_missing"])
        XCTAssertTrue(summary.isWithinPreviewBudget)
    }

    func testMedianPlayerCountFromOddNumberOfSamples() {
        let summary = PostStopPreviewBuilder.summarize(
            durationSeconds: 30,
            requestedFPS: 60,
            captureQuality: CaptureQuality(grade: .good),
            sampledFrameDetectionCounts: [2, 4, 2],
            elapsedBuildSeconds: 3
        )

        XCTAssertEqual(summary.estimatedPlayerCount, 2)
        XCTAssertEqual(summary.playerCountSampleFrameCount, 3)
    }

    func testMedianPlayerCountFromEvenNumberOfSamplesRoundsToNearestInt() {
        // sorted [2, 3] -> mean 2.5 -> rounds to 3 (Double.rounded() rounds half away from zero)
        let summary = PostStopPreviewBuilder.summarize(
            durationSeconds: 10,
            requestedFPS: 60,
            captureQuality: CaptureQuality(grade: .good),
            sampledFrameDetectionCounts: [3, 2],
            elapsedBuildSeconds: 1
        )

        XCTAssertEqual(summary.estimatedPlayerCount, 3)
    }

    func testElapsedBuildSecondsOverBudgetIsFlaggedFalse() {
        let summary = PostStopPreviewBuilder.summarize(
            durationSeconds: 60,
            requestedFPS: 60,
            captureQuality: CaptureQuality(grade: .poor, reasons: ["arkit_seed_missing"]),
            sampledFrameDetectionCounts: [4],
            elapsedBuildSeconds: 11.2
        )

        XCTAssertFalse(summary.isWithinPreviewBudget)
    }

    func testNegativeDurationAndElapsedSecondsAreClampedToZero() {
        let summary = PostStopPreviewBuilder.summarize(
            durationSeconds: -5,
            requestedFPS: 60,
            captureQuality: CaptureQuality(grade: .good),
            sampledFrameDetectionCounts: [1],
            elapsedBuildSeconds: -2
        )

        XCTAssertEqual(summary.durationSeconds, 0)
        XCTAssertEqual(summary.elapsedBuildSeconds, 0)
    }
}
