import XCTest
@testable import PickleballFastTier

final class LiveBallOverlayTests: XCTestCase {
    func testComingSoonAndLowConfidenceStatesDoNotCreateTrailOrContactMarkers() {
        var tracker = LiveBallOverlayTracker()

        let comingSoon = tracker.update(frameIndex: 1, ballState: .comingSoon)
        let lowConfidence = tracker.update(
            frameIndex: 2,
            ballState: LiveBallIndicatorPolicy.evaluate(
                rawConfidence: 0.2,
                rawNormalizedX: 0.4,
                rawNormalizedY: 0.4,
                modelIsTrained: true
            )
        )

        XCTAssertEqual(comingSoon.trailPoints, [])
        XCTAssertEqual(comingSoon.contactMarkers, [])
        XCTAssertEqual(lowConfidence.trailPoints, [])
        XCTAssertEqual(lowConfidence.contactMarkers, [])
    }

    func testTrackingStatesCreateAFadingTrailFromRealBallPositions() {
        var tracker = LiveBallOverlayTracker()

        _ = tracker.update(frameIndex: 10, ballState: trackingState(x: 0.20, y: 0.35, confidence: 0.75))
        let overlay = tracker.update(frameIndex: 11, ballState: trackingState(x: 0.26, y: 0.37, confidence: 0.80))

        XCTAssertEqual(overlay.trailPoints.map(\.frameIndex), [10, 11])
        XCTAssertEqual(overlay.trailPoints[0].normalizedX, 0.20, accuracy: 0.0001)
        XCTAssertEqual(overlay.trailPoints[1].normalizedY, 0.37, accuracy: 0.0001)
        XCTAssertLessThan(overlay.trailPoints[0].opacity, overlay.trailPoints[1].opacity)
        XCTAssertGreaterThan(overlay.trailPoints[1].radius, 0)
    }

    func testSharpTrajectoryKinkCreatesCandidateContactMarkerAtTheInflectionPoint() {
        var tracker = LiveBallOverlayTracker()

        _ = tracker.update(frameIndex: 20, ballState: trackingState(x: 0.20, y: 0.40, confidence: 0.90))
        _ = tracker.update(frameIndex: 21, ballState: trackingState(x: 0.32, y: 0.40, confidence: 0.88))
        let overlay = tracker.update(frameIndex: 22, ballState: trackingState(x: 0.22, y: 0.47, confidence: 0.86))

        let marker = try! XCTUnwrap(overlay.contactMarkers.first)
        XCTAssertEqual(marker.frameIndex, 21)
        XCTAssertEqual(marker.normalizedX, 0.32, accuracy: 0.0001)
        XCTAssertEqual(marker.normalizedY, 0.40, accuracy: 0.0001)
        XCTAssertEqual(marker.source, .kinematicInflectionCandidate)
        XCTAssertGreaterThan(marker.opacity, 0.85)
    }

    func testLowConfidenceGapBreaksContactInferenceAcrossSamples() {
        var tracker = LiveBallOverlayTracker()

        _ = tracker.update(frameIndex: 20, ballState: trackingState(x: 0.20, y: 0.40, confidence: 0.90))
        _ = tracker.update(
            frameIndex: 21,
            ballState: LiveBallIndicatorPolicy.evaluate(
                rawConfidence: 0.10,
                rawNormalizedX: 0.30,
                rawNormalizedY: 0.40,
                modelIsTrained: true
            )
        )
        _ = tracker.update(frameIndex: 22, ballState: trackingState(x: 0.32, y: 0.40, confidence: 0.88))
        let overlay = tracker.update(frameIndex: 23, ballState: trackingState(x: 0.22, y: 0.47, confidence: 0.86))

        XCTAssertEqual(overlay.trailPoints.map(\.frameIndex), [20, 22, 23])
        XCTAssertEqual(overlay.contactMarkers, [])
    }

    func testAspectFillLayoutMapsTrailAndContactMarkersIntoPreviewCoordinates() {
        let overlay = LiveBallOverlayState(
            trailPoints: [
                LiveBallTrailPoint(frameIndex: 1, normalizedX: 0.50, normalizedY: 0.25, confidence: 0.9, ageFrames: 0, opacity: 1, radius: 0.014),
            ],
            contactMarkers: [
                LiveBallContactMarker(frameIndex: 1, normalizedX: 0.50, normalizedY: 0.25, confidence: 0.9, ageFrames: 0, opacity: 1, radius: 0.055, source: .kinematicInflectionCandidate),
            ]
        )

        let rendered = LiveBallOverlayLayout.layout(
            overlay: overlay,
            viewportWidth: 390,
            viewportHeight: 844,
            videoAspectRatio: 16.0 / 9.0
        )

        let trailPoint = try! XCTUnwrap(rendered.trailPoints.first)
        let marker = try! XCTUnwrap(rendered.contactMarkers.first)
        XCTAssertEqual(trailPoint.centerX, 195, accuracy: 0.001)
        XCTAssertEqual(trailPoint.centerY, 211, accuracy: 0.001)
        XCTAssertEqual(marker.centerX, trailPoint.centerX, accuracy: 0.001)
        XCTAssertGreaterThan(marker.radius, trailPoint.radius)
    }

    func testAspectFillLayoutBreaksTrailSegmentsAcrossTrackingGaps() {
        let overlay = LiveBallOverlayState(
            trailPoints: [
                LiveBallTrailPoint(frameIndex: 20, normalizedX: 0.20, normalizedY: 0.40, confidence: 0.9, ageFrames: 3, opacity: 0.8, radius: 0.014),
                LiveBallTrailPoint(frameIndex: 22, normalizedX: 0.32, normalizedY: 0.40, confidence: 0.9, ageFrames: 1, opacity: 0.9, radius: 0.014),
                LiveBallTrailPoint(frameIndex: 23, normalizedX: 0.22, normalizedY: 0.47, confidence: 0.9, ageFrames: 0, opacity: 1.0, radius: 0.014),
            ]
        )

        let rendered = LiveBallOverlayLayout.layout(
            overlay: overlay,
            viewportWidth: 390,
            viewportHeight: 844,
            videoAspectRatio: 16.0 / 9.0
        )

        XCTAssertEqual(rendered.trailSegments.map { $0.points.map(\.point.frameIndex) }, [[20], [22, 23]])
    }

    private func trackingState(x: Double, y: Double, confidence: Double) -> LiveBallIndicatorState {
        LiveBallIndicatorPolicy.evaluate(
            rawConfidence: confidence,
            rawNormalizedX: x,
            rawNormalizedY: y,
            modelIsTrained: true
        )
    }
}
