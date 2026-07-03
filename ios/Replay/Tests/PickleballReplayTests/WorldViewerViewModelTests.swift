import XCTest
@testable import PickleballReplay

@MainActor
final class WorldViewerViewModelTests: XCTestCase {
    private func makeViewModel() throws -> WorldViewerViewModel {
        WorldViewerViewModel(bundle: try WorldBundle.loadBundledSample())
    }

    func testInitialStateStartsAtEarliestPlayerFrameAndBuildsAScene() throws {
        let viewModel = try makeViewModel()
        XCTAssertGreaterThanOrEqual(viewModel.currentTime, 0)
        XCTAssertGreaterThan(viewModel.durationSeconds, viewModel.currentTime)
        XCTAssertTrue(viewModel.dimLowConfidence, "low-confidence entities must be dimmed by default")
        XCTAssertGreaterThan(viewModel.sceneBuilder.scene.rootNode.childNodes.count, 0)
    }

    func testSeekClampsToValidRangeAndUpdatesSnapshot() throws {
        let viewModel = try makeViewModel()
        viewModel.seek(to: -5)
        XCTAssertEqual(viewModel.currentTime, 0)

        viewModel.seek(to: 1_000_000)
        XCTAssertEqual(viewModel.currentTime, viewModel.durationSeconds)

        viewModel.seek(to: 2.6193)
        XCTAssertEqual(viewModel.currentTime, 2.6193, accuracy: 1e-9)
        XCTAssertEqual(viewModel.snapshot.timeSeconds, 2.6193, accuracy: 1e-9)
    }

    func testSeekingCompactVerifiedFixtureKeepsSkeletonTier() throws {
        let bundle = try WorldBundle.loadBundledSample()
        let player3World = try XCTUnwrap(bundle.world.players.first { $0.id == 3 })
        let jointTime = try XCTUnwrap(player3World.frames.first { !$0.jointsWorld.isEmpty }?.t)
        let viewModel = try makeViewModel()
        viewModel.seek(to: jointTime)
        let player3 = try XCTUnwrap(viewModel.snapshot.players.first { $0.id == 3 })
        guard case .joints = player3.tier else {
            return XCTFail("expected skeleton-level joints tier at t=\(jointTime), got \(player3.tier)")
        }
    }

    func testCameraPresetChangeRepositionsTheCameraNode() throws {
        let viewModel = try makeViewModel()
        let broadcastPosition = viewModel.sceneBuilder.cameraNode.position
        viewModel.selectCameraPreset(.topDown)
        let topDownPosition = viewModel.sceneBuilder.cameraNode.position
        XCTAssertNotEqual(broadcastPosition.z, topDownPosition.z)
        XCTAssertEqual(viewModel.cameraPreset, .topDown)
    }

    func testDimLowConfidenceToggleDoesNotChangeCurrentTimeOrTier() throws {
        let viewModel = try makeViewModel()
        viewModel.seek(to: 0.2)
        let before = viewModel.snapshot.players.first { $0.id == 1 }?.tier
        viewModel.dimLowConfidence.toggle()
        let after = viewModel.snapshot.players.first { $0.id == 1 }?.tier
        XCTAssertEqual(viewModel.currentTime, 0.2, accuracy: 1e-9)
        XCTAssertEqual(before, after, "dimming is a rendering style, not a tier change")
    }

    func testTogglePlaybackFlipsIsPlaying() throws {
        let viewModel = try makeViewModel()
        XCTAssertFalse(viewModel.isPlaying)
        viewModel.togglePlayback()
        XCTAssertTrue(viewModel.isPlaying)
        viewModel.togglePlayback()
        XCTAssertFalse(viewModel.isPlaying)
    }
}
