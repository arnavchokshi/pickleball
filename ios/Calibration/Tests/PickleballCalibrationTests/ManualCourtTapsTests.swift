import XCTest
@testable import PickleballCalibration

final class ManualCourtTapsTests: XCTestCase {
    func testFourCornerTapsAreValidatedAndOrderedForServer() throws {
        let taps = ManualCourtTaps(imagePoints: [
            [1840, 1010],
            [110, 95],
            [1810, 120],
            [95, 1000],
        ])

        let ordered = try taps.orderedFourCorners(imageSize: ImageSize(width: 1920, height: 1080))

        XCTAssertEqual(ordered.imagePoints, [
            [110, 95],
            [1810, 120],
            [1840, 1010],
            [95, 1000],
        ])
    }

    func testManualTapsRejectDuplicateOutOfBoundsAndCollinearInputs() {
        XCTAssertThrowsError(
            try ManualCourtTaps(imagePoints: [[10, 10], [10, 10], [100, 100], [20, 200]])
                .orderedFourCorners(imageSize: ImageSize(width: 1920, height: 1080))
        ) { error in
            XCTAssertEqual(error as? ManualCourtTaps.ValidationError, .duplicatePoint)
        }

        XCTAssertThrowsError(
            try ManualCourtTaps(imagePoints: [[-1, 10], [100, 10], [100, 100], [10, 100]])
                .orderedFourCorners(imageSize: ImageSize(width: 1920, height: 1080))
        ) { error in
            XCTAssertEqual(error as? ManualCourtTaps.ValidationError, .pointOutsideImage)
        }

        XCTAssertThrowsError(
            try ManualCourtTaps(imagePoints: [[10, 10], [100, 10], [200, 10], [300, 10]])
                .orderedFourCorners(imageSize: ImageSize(width: 1920, height: 1080))
        ) { error in
            XCTAssertEqual(error as? ManualCourtTaps.ValidationError, .degenerateQuadrilateral)
        }
    }
}
