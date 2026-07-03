import XCTest
@testable import PickleballReplay

final class ReplayScrubberLayoutTests: XCTestCase {
    func testBottomPaddingKeepsScrubberAboveHomeIndicatorArea() {
        XCTAssertGreaterThanOrEqual(ReplayScrubberLayout.bottomPadding, 44)
    }
}
