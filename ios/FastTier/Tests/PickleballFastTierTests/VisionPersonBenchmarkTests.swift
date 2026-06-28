import XCTest
@testable import PickleballFastTier

#if canImport(ImageIO)
import ImageIO
#endif

final class VisionPersonBenchmarkTests: XCTestCase {
    func testNormalizedVisionRectConvertsToTopLeftPixelXYWH() {
        let bbox = normalizedVisionRectToPixelXYWH(
            x: 0.25,
            y: 0.10,
            width: 0.50,
            height: 0.40,
            imageWidth: 200,
            imageHeight: 100
        )

        XCTAssertEqual(bbox, [50.0, 50.0, 100.0, 40.0])
    }

    #if canImport(ImageIO)
    func testVisionProcessorAcceptsExplicitImageOrientation() {
        let processor = VisionHumanRectanglePersonProcessor(
            configuration: VisionPersonBenchmarkConfiguration(),
            orientation: .right
        )

        XCTAssertNotNil(processor)
    }
    #endif
}
