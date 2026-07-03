import XCTest
@testable import PickleballFastTier

final class CourtDotMapTests: XCTestCase {
    func testFootPointIsBottomCenterOfBoundingBoxNormalizedWithNoRotation() {
        let detections = [
            OnDevicePersonDetection(trackID: 1, bboxXYWH: [100, 200, 40, 80], confidence: 0.9, source: "coreml_yolo26_end2end"),
        ]

        let points = CourtDotMapBuilder.build(detections: detections, sourceWidth: 1000, sourceHeight: 500)

        let point = try! XCTUnwrap(points.first)
        XCTAssertEqual(point.trackID, 1)
        XCTAssertEqual(point.normalizedX, 0.12, accuracy: 0.0001) // (100 + 40/2) / 1000
        XCTAssertEqual(point.normalizedY, 0.56, accuracy: 0.0001) // (200 + 80) / 500
        XCTAssertEqual(point.confidence, 0.9)
    }

    func testAtMostFourPlayersAreKeptEvenWithMoreDetections() {
        let detections = (0..<6).map {
            OnDevicePersonDetection(trackID: $0, bboxXYWH: [10, 10, 10, 10], confidence: 0.5, source: "coreml_yolo26_end2end")
        }

        let points = CourtDotMapBuilder.build(detections: detections, sourceWidth: 100, sourceHeight: 100)

        XCTAssertEqual(points.count, 4)
        XCTAssertEqual(points.map(\.trackID), [0, 1, 2, 3])
    }

    func testMalformedBoundingBoxIsSkippedRatherThanCrashing() {
        let detections = [
            OnDevicePersonDetection(trackID: 1, bboxXYWH: [1, 2, 3], confidence: 0.5, source: "bad"),
            OnDevicePersonDetection(trackID: 2, bboxXYWH: [10, 10, 10, 10], confidence: 0.5, source: "ok"),
        ]

        let points = CourtDotMapBuilder.build(detections: detections, sourceWidth: 100, sourceHeight: 100)

        XCTAssertEqual(points.map(\.trackID), [2])
    }

    func testZeroSourceDimensionsProduceNoPointsInsteadOfDividingByZero() {
        let detections = [OnDevicePersonDetection(trackID: 1, bboxXYWH: [1, 1, 1, 1], confidence: 0.5, source: "x")]

        XCTAssertEqual(CourtDotMapBuilder.build(detections: detections, sourceWidth: 0, sourceHeight: 100), [])
        XCTAssertEqual(CourtDotMapBuilder.build(detections: detections, sourceWidth: 100, sourceHeight: 0), [])
    }

    func testRotate90ClockwiseMatchesLandscapeRightConvention() {
        // A point at the top-left corner of a portrait-native 480x640 buffer
        // should land at the top-right of the resulting 640x480 landscape
        // frame after a 90-degree clockwise rotation (matching
        // CaptureOrientationPolicy.rotationAngleDegrees(.landscapeRight) == 90).
        let rotated = CourtDotMapBuilder.rotate(x: 0, y: 0, width: 480, height: 640, degrees: 90)

        XCTAssertEqual(rotated.x, 640, accuracy: 0.0001)
        XCTAssertEqual(rotated.y, 0, accuracy: 0.0001)
        XCTAssertEqual(rotated.width, 640)
        XCTAssertEqual(rotated.height, 480)
    }

    func testRotate180MirrorsBothAxes() {
        let rotated = CourtDotMapBuilder.rotate(x: 10, y: 20, width: 100, height: 200, degrees: 180)

        XCTAssertEqual(rotated.x, 90, accuracy: 0.0001)
        XCTAssertEqual(rotated.y, 180, accuracy: 0.0001)
        XCTAssertEqual(rotated.width, 100)
        XCTAssertEqual(rotated.height, 200)
    }

    func testRotate270IsInverseOfRotate90() {
        let width = 480.0
        let height = 640.0
        let original = (x: 37.0, y: 123.0)

        let forward = CourtDotMapBuilder.rotate(x: original.x, y: original.y, width: width, height: height, degrees: 90)
        let back = CourtDotMapBuilder.rotate(x: forward.x, y: forward.y, width: forward.width, height: forward.height, degrees: 270)

        XCTAssertEqual(back.x, original.x, accuracy: 0.0001)
        XCTAssertEqual(back.y, original.y, accuracy: 0.0001)
        XCTAssertEqual(back.width, width)
        XCTAssertEqual(back.height, height)
    }

    func testNormalizedCoordinatesAreClampedIntoZeroToOneRange() {
        let detections = [
            OnDevicePersonDetection(trackID: 1, bboxXYWH: [-50, -50, 10, 10], confidence: 0.5, source: "x"),
        ]

        let points = CourtDotMapBuilder.build(detections: detections, sourceWidth: 100, sourceHeight: 100)

        let point = try! XCTUnwrap(points.first)
        XCTAssertGreaterThanOrEqual(point.normalizedX, 0)
        XCTAssertGreaterThanOrEqual(point.normalizedY, 0)
    }

    /// "Linker integration" coverage: feed cadence-decimated frames through
    /// `PersonTrackLinker` (reused verbatim from FastTier, per the milestone
    /// spec) and confirm the resulting track IDs stay stable across frames
    /// the detector skipped, then map cleanly through the dot-map builder.
    func testCadenceDecimatedFramesStillProduceStableTrackIDsThroughTheLinkerAndDotMap() {
        let scheduler = LiveDetectionCadenceScheduler(everyNFrames: 4)
        var linker = PersonTrackLinker(maxTracks: 4)
        var lastDetections: [OnDevicePersonDetection] = []
        var dotMapHistory: [[CourtDotMapPoint]] = []

        // Simulate a moving player across 8 frames, but the detector (per
        // cadence) only actually runs on frames 0 and 4.
        // Frame-4's box is shifted by 15px (not 40px) so it still overlaps
        // frame-0's box enough (IoU > the linker's 0.3 default threshold) to
        // link onto the SAME track -- a believable one-player-walking delta
        // across 4 skipped frames, not a teleport that would legitimately
        // spawn a new track.
        let observationsPerFrame: [Int: [OnDevicePersonObservation]] = [
            0: [OnDevicePersonObservation(bboxXYWH: [100, 100, 40, 80], confidence: 0.8, source: "coreml_yolo26_end2end")],
            4: [OnDevicePersonObservation(bboxXYWH: [115, 100, 40, 80], confidence: 0.8, source: "coreml_yolo26_end2end")],
        ]

        for frameIndex in 0..<8 {
            if scheduler.shouldRunDetection(forFrameIndex: frameIndex) {
                let observations = observationsPerFrame[frameIndex] ?? []
                lastDetections = linker.update(frameIndex: frameIndex, observations: observations)
            }
            // Frames the detector skipped reuse `lastDetections` verbatim,
            // matching the replay pipeline's existing propagation pattern
            // (`CoreMLReplayPersonFrameProcessor`).
            dotMapHistory.append(
                CourtDotMapBuilder.build(detections: lastDetections, sourceWidth: 1000, sourceHeight: 500)
            )
        }

        // Same track ID (from linking) persists across all 8 frames, even
        // the ones the detector skipped.
        let trackIDsSeen = Set(dotMapHistory.flatMap { $0.map(\.trackID) })
        XCTAssertEqual(trackIDsSeen, [1])

        // Frames 0-3 use the frame-0 position; frames 4-7 use the frame-4 position.
        XCTAssertEqual(dotMapHistory[0].first?.normalizedX, dotMapHistory[3].first?.normalizedX)
        XCTAssertEqual(dotMapHistory[4].first?.normalizedX, dotMapHistory[7].first?.normalizedX)
        XCTAssertNotEqual(dotMapHistory[0].first?.normalizedX, dotMapHistory[4].first?.normalizedX)
    }
}
