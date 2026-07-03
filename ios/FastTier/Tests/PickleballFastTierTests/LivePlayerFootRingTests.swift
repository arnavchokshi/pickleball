import XCTest
@testable import PickleballFastTier

final class LivePlayerFootRingTests: XCTestCase {
    func testBuildsStableColoredFootRingsFromCourtDotPoints() {
        let points = [
            CourtDotMapPoint(trackID: 3, normalizedX: 0.25, normalizedY: 0.80, confidence: 0.9),
            CourtDotMapPoint(trackID: 7, normalizedX: 0.75, normalizedY: 0.45, confidence: 0.6),
        ]

        let rings = LivePlayerFootRingBuilder.build(
            points: points,
            frameIndex: 12,
            lastDetectionFrameIndex: 12
        )

        XCTAssertEqual(rings.map(\.trackID), [3, 7])
        XCTAssertEqual(rings[0].normalizedCenterX, 0.25, accuracy: 0.0001)
        XCTAssertEqual(rings[0].normalizedCenterY, 0.80, accuracy: 0.0001)
        XCTAssertEqual(rings[0].colorIndex, 3)
        XCTAssertEqual(rings[1].colorIndex, 3)
        XCTAssertEqual(rings[0].source, .screenSpaceProxy)
        XCTAssertFalse(rings[0].isStale)
        XCTAssertGreaterThan(rings[0].normalizedWidth, rings[0].normalizedHeight)
    }

    func testBuildsFootRingsFromDetectionBoxFootprints() {
        let detections = [
            OnDevicePersonDetection(trackID: 2, bboxXYWH: [100, 50, 40, 250], confidence: 0.7, source: "unit"),
            OnDevicePersonDetection(trackID: 5, bboxXYWH: [500, 50, 180, 250], confidence: 0.9, source: "unit"),
        ]

        let rings = LivePlayerFootRingBuilder.build(
            detections: detections,
            sourceWidth: 1000,
            sourceHeight: 500,
            rotationDegrees: 0,
            frameIndex: 44,
            lastDetectionFrameIndex: 44
        )

        XCTAssertEqual(rings.map(\.trackID), [2, 5])
        XCTAssertEqual(rings[0].normalizedCenterX, 0.12, accuracy: 0.0001)
        XCTAssertEqual(rings[0].normalizedCenterY, 0.60, accuracy: 0.0001)
        XCTAssertEqual(rings[1].normalizedCenterX, 0.59, accuracy: 0.0001)
        XCTAssertEqual(rings[1].normalizedCenterY, 0.60, accuracy: 0.0001)
        XCTAssertGreaterThan(rings[1].normalizedWidth, rings[0].normalizedWidth)
        XCTAssertEqual(rings[0].source, .screenSpaceProxy)
        XCTAssertEqual(rings[1].source, .screenSpaceProxy)
    }

    func testDetectionFootRingsRejectInvalidSourceOrBoxes() {
        let rings = LivePlayerFootRingBuilder.build(
            detections: [
                OnDevicePersonDetection(trackID: 1, bboxXYWH: [0, 0, 40], confidence: 0.5, source: "unit"),
                OnDevicePersonDetection(trackID: 2, bboxXYWH: [0, 0, 0, 80], confidence: 0.5, source: "unit"),
            ],
            sourceWidth: 1000,
            sourceHeight: 500,
            frameIndex: 1,
            lastDetectionFrameIndex: 1
        )
        let invalidSourceRings = LivePlayerFootRingBuilder.build(
            detections: [
                OnDevicePersonDetection(trackID: 3, bboxXYWH: [0, 0, 40, 80], confidence: 0.5, source: "unit"),
            ],
            sourceWidth: 0,
            sourceHeight: 500,
            frameIndex: 1,
            lastDetectionFrameIndex: 1
        )

        XCTAssertEqual(rings, [])
        XCTAssertEqual(invalidSourceRings, [])
    }

    func testRingOpacityFadesButRemainsVisibleAcrossSkippedCadenceFrames() {
        let points = [
            CourtDotMapPoint(trackID: 1, normalizedX: 0.4, normalizedY: 0.7, confidence: 0.9),
        ]

        let fresh = LivePlayerFootRingBuilder.build(points: points, frameIndex: 20, lastDetectionFrameIndex: 20)
        let propagated = LivePlayerFootRingBuilder.build(points: points, frameIndex: 23, lastDetectionFrameIndex: 20)

        XCTAssertEqual(propagated.first?.stalenessFrames, 3)
        XCTAssertEqual(propagated.first?.isStale, true)
        XCTAssertLessThan(propagated.first?.strokeOpacity ?? 1, fresh.first?.strokeOpacity ?? 0)
        XCTAssertGreaterThan(propagated.first?.strokeOpacity ?? 0, 0.45)
    }

    func testAspectFillLayoutKeepsRingsAlignedWithPreviewLayerCropping() {
        let ring = LivePlayerFootRing(
            trackID: 1,
            normalizedCenterX: 0.50,
            normalizedCenterY: 0.25,
            confidence: 0.8,
            stalenessFrames: 0,
            colorIndex: 1,
            normalizedWidth: 0.12,
            normalizedHeight: 0.04,
            strokeOpacity: 0.9,
            fillOpacity: 0.2,
            source: .screenSpaceProxy
        )

        let layout = LivePlayerFootRingLayout.layout(
            rings: [ring],
            viewportWidth: 390,
            viewportHeight: 844,
            videoAspectRatio: 16.0 / 9.0
        )

        let rendered = try! XCTUnwrap(layout.first)
        XCTAssertEqual(rendered.centerX, 195, accuracy: 0.001)
        XCTAssertEqual(rendered.centerY, 211, accuracy: 0.001)
        XCTAssertGreaterThan(rendered.width, rendered.height)
    }
}
