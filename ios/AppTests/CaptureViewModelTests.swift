import XCTest
import PickleballCapture
@testable import Pickleball

final class CaptureViewModelTests: XCTestCase {
    @MainActor
    func testInitialCaptureStateIsIdleAndCameraFree() {
        let model = CaptureViewModel()

        XCTAssertEqual(model.status, .idle)
        XCTAssertFalse(model.isRecording)
        XCTAssertNil(model.descriptor)
        XCTAssertEqual(model.selectedMode, .standard60)
        XCTAssertEqual(CaptureViewModel.modes, [.standard60, .swing120, .ballPhysics240, .quality4K60])
        XCTAssertEqual(model.captureOrientationTitle, "Landscape")
        XCTAssertEqual(model.videoRotationTitle, "90°")
        XCTAssertEqual(model.previewRotationAngle, 90)
        XCTAssertEqual(model.replayBenchmarkTitle, "Replay")
        XCTAssertEqual(model.replayBenchmarkDetail, "Vision")
        XCTAssertNil(model.replayBenchmarkOutputPath)
        XCTAssertTrue(model.modeSummary.contains("60 fps"))
        XCTAssertTrue(model.modeSummary.contains("hevc"))
    }
}
