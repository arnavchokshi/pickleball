import XCTest
@testable import PickleballFastTier

final class PersonTrackLinkerTests: XCTestCase {
    func testLinkerKeepsTrackIDsStableAcrossOverlappingBoxes() {
        var linker = PersonTrackLinker(iouThreshold: 0.3, maxTrackAgeFrames: 2, maxTracks: 4)

        let first = linker.update(
            frameIndex: 0,
            observations: [
                OnDevicePersonObservation(bboxXYWH: [10, 10, 20, 20], confidence: 0.9, source: "vision"),
                OnDevicePersonObservation(bboxXYWH: [100, 10, 20, 20], confidence: 0.8, source: "vision"),
            ]
        )
        let second = linker.update(
            frameIndex: 1,
            observations: [
                OnDevicePersonObservation(bboxXYWH: [12, 10, 20, 20], confidence: 0.9, source: "vision"),
                OnDevicePersonObservation(bboxXYWH: [102, 10, 20, 20], confidence: 0.8, source: "vision"),
            ]
        )

        XCTAssertEqual(first.map(\.trackID), [1, 2])
        XCTAssertEqual(second.map(\.trackID), [1, 2])
    }

    func testLinkerExpiresOldTracksAndCapsActiveOutput() {
        var linker = PersonTrackLinker(iouThreshold: 0.3, maxTrackAgeFrames: 0, maxTracks: 2)

        _ = linker.update(
            frameIndex: 0,
            observations: [OnDevicePersonObservation(bboxXYWH: [10, 10, 20, 20], confidence: 0.9, source: "vision")]
        )
        let next = linker.update(
            frameIndex: 2,
            observations: [
                OnDevicePersonObservation(bboxXYWH: [12, 10, 20, 20], confidence: 0.9, source: "vision"),
                OnDevicePersonObservation(bboxXYWH: [50, 10, 20, 20], confidence: 0.7, source: "vision"),
                OnDevicePersonObservation(bboxXYWH: [90, 10, 20, 20], confidence: 0.6, source: "vision"),
            ]
        )

        XCTAssertEqual(next.map(\.trackID), [2, 3])
        XCTAssertEqual(next.count, 2)
    }

    func testByteTrackStyleLowConfidenceObservationCanRescueExistingTrack() {
        var linker = PersonTrackLinker(
            iouThreshold: 0.3,
            maxTrackAgeFrames: 2,
            maxTracks: 4,
            highConfidenceThreshold: 0.6,
            lowConfidenceThreshold: 0.1
        )

        let first = linker.update(
            frameIndex: 0,
            observations: [OnDevicePersonObservation(bboxXYWH: [10, 10, 20, 20], confidence: 0.9, source: "yolo26n")]
        )
        let rescued = linker.update(
            frameIndex: 1,
            observations: [OnDevicePersonObservation(bboxXYWH: [12, 10, 20, 20], confidence: 0.2, source: "yolo26n")]
        )

        XCTAssertEqual(first.map(\.trackID), [1])
        XCTAssertEqual(rescued.map(\.trackID), [1])
        XCTAssertEqual(rescued.first?.confidence, 0.2)
    }
}
