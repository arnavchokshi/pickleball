import XCTest
import CoreGraphics

#if canImport(Pickleball)
@testable import Pickleball
#endif

@MainActor
final class DinkVisionTabLayoutGeometryTests: XCTestCase {
    private let barThickness: CGFloat = 88

    func testPortraitPlacementPreservesExistingBottomBarGeometry() {
        let model = DinkVisionTabLayoutModel.brandV4
        let containerSize = CGSize(width: 393, height: 852)

        XCTAssertEqual(model.railEdge(for: containerSize), .bottom)

        let placement = model.recordButtonPlacement(
            in: containerSize,
            railEdge: .bottom,
            barThickness: barThickness,
            buttonDiameter: model.recordButtonDiameter
        )
        let previousCenterY = containerSize.height
            - model.totalOverlayHeight(tabBarHeight: barThickness)
            + model.recordButtonCenterY(tabBarHeight: barThickness)

        XCTAssertEqual(placement.center.x, 196.5, accuracy: 0.001)
        XCTAssertEqual(placement.center.y, 738, accuracy: 0.001)
        XCTAssertEqual(placement.center.y, previousCenterY, accuracy: 0.001)
        XCTAssertEqual(placement.exposedFraction, 62.0 / 72.0, accuracy: 0.001)
    }

    func testLandscapeLeadingRailPlacementFullyExposesButtonIntoContent() {
        let model = DinkVisionTabLayoutModel.brandV4
        let containerSize = CGSize(width: 852, height: 393)

        XCTAssertEqual(model.railEdge(for: containerSize), .leading)

        let placement = model.recordButtonPlacement(
            in: containerSize,
            railEdge: .leading,
            barThickness: barThickness,
            buttonDiameter: model.recordButtonDiameter
        )
        let buttonFrame = CGRect(
            x: placement.center.x - model.recordButtonDiameter / 2,
            y: placement.center.y - model.recordButtonDiameter / 2,
            width: model.recordButtonDiameter,
            height: model.recordButtonDiameter
        )

        XCTAssertEqual(placement.center, CGPoint(x: 124, y: 196.5))
        XCTAssertEqual(buttonFrame.minX, barThickness, accuracy: 0.001)
        XCTAssertGreaterThanOrEqual(placement.exposedFraction, 1)
    }

    func testLandscapeTrailingRailPlacementFullyExposesButtonIntoContent() {
        let model = DinkVisionTabLayoutModel.brandV4
        let containerSize = CGSize(width: 852, height: 393)
        let placement = model.recordButtonPlacement(
            in: containerSize,
            railEdge: .trailing,
            barThickness: barThickness,
            buttonDiameter: model.recordButtonDiameter
        )
        let buttonFrame = CGRect(
            x: placement.center.x - model.recordButtonDiameter / 2,
            y: placement.center.y - model.recordButtonDiameter / 2,
            width: model.recordButtonDiameter,
            height: model.recordButtonDiameter
        )

        XCTAssertEqual(placement.center, CGPoint(x: 728, y: 196.5))
        XCTAssertEqual(buttonFrame.maxX, containerSize.width - barThickness, accuracy: 0.001)
        XCTAssertGreaterThanOrEqual(placement.exposedFraction, 1)
    }
}
