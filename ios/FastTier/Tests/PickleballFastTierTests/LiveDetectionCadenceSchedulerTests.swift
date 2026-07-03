import XCTest
@testable import PickleballFastTier

final class LiveDetectionCadenceSchedulerTests: XCTestCase {
    func testDefaultRunsEveryFourthFrame() {
        let scheduler = LiveDetectionCadenceScheduler()

        let runFrames = (0..<12).filter { scheduler.shouldRunDetection(forFrameIndex: $0) }

        XCTAssertEqual(runFrames, [0, 4, 8])
    }

    func testBudgetedDefaultIsWithinTheMilestonesThirdToFifthFrameRange() {
        XCTAssertGreaterThanOrEqual(LiveDetectionCadenceScheduler.budgeted.everyNFrames, 3)
        XCTAssertLessThanOrEqual(LiveDetectionCadenceScheduler.budgeted.everyNFrames, 5)
    }

    func testCustomCadenceOfThreeRunsEveryThirdFrame() {
        let scheduler = LiveDetectionCadenceScheduler(everyNFrames: 3)

        XCTAssertTrue(scheduler.shouldRunDetection(forFrameIndex: 0))
        XCTAssertFalse(scheduler.shouldRunDetection(forFrameIndex: 1))
        XCTAssertFalse(scheduler.shouldRunDetection(forFrameIndex: 2))
        XCTAssertTrue(scheduler.shouldRunDetection(forFrameIndex: 3))
    }

    func testNonPositiveIntervalClampsToOneSoEveryFrameRuns() {
        let scheduler = LiveDetectionCadenceScheduler(everyNFrames: 0)

        XCTAssertEqual(scheduler.everyNFrames, 1)
        XCTAssertTrue(scheduler.shouldRunDetection(forFrameIndex: 5))
    }
}
