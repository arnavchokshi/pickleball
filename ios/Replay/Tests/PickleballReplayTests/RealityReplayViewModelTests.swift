import XCTest
@testable import PickleballReplay

@MainActor
final class RealityReplayViewModelTests: XCTestCase {
    func testBundledRealityFixtureResolvesUsdzAndKeepsSourceProvenance() throws {
        let asset = try RealityReplayAsset.loadBundledFixture()

        XCTAssertEqual(asset.clipID, "burlington_gold_0300_low_steep_corner")
        XCTAssertEqual(asset.assetURL.pathExtension, "usdz")
        XCTAssertEqual(asset.sourceAssetPath, "runs/usdz-compress_20260702T113045Z/body_mesh_animated_budget53.usdz")
        XCTAssertEqual(asset.sourceAssetByteCount, 11_823_255)
        XCTAssertEqual(asset.bundledAssetByteCount, asset.sourceAssetByteCount)
        XCTAssertEqual(asset.badgeTitle, "Budget-53 baked mesh preview")
        XCTAssertTrue(asset.badgeDetail.contains("Apple-strict-valid"))
    }

    func testRealityReplaySharesTheSceneKitWorldViewerTimeline() throws {
        let bundle = try WorldBundle.loadBundledSample()
        let worldViewModel = WorldViewerViewModel(bundle: bundle)
        let replayViewModel = try RealityReplayViewModel(
            asset: .loadBundledFixture(),
            timeline: worldViewModel.timeline
        )

        replayViewModel.seek(to: 2.6193)

        XCTAssertTrue(replayViewModel.timeline === worldViewModel.timeline)
        XCTAssertEqual(replayViewModel.currentTime, 2.6193, accuracy: 1e-9)
        XCTAssertEqual(worldViewModel.currentTime, 2.6193, accuracy: 1e-9)
        XCTAssertEqual(worldViewModel.snapshot.timeSeconds, 2.6193, accuracy: 1e-9)
    }

    func testRealityReplayTimelineMapsWorldSecondsIntoUsdAnimationSeconds() throws {
        let viewModel = try RealityReplayViewModel(asset: .loadBundledFixture())

        XCTAssertEqual(viewModel.animationTimeSeconds(for: 0), 0, accuracy: 1e-9)

        let twoSecondsIntoBake = viewModel.asset.usdTimelineStartSeconds + 2.0
        viewModel.seek(to: twoSecondsIntoBake)
        XCTAssertEqual(viewModel.animationTimeSeconds, 2.0, accuracy: 1e-9)

        viewModel.seek(to: 1_000)
        XCTAssertEqual(viewModel.currentTime, viewModel.durationSeconds, accuracy: 1e-9)
        XCTAssertEqual(viewModel.animationTimeSeconds, viewModel.asset.animationDurationSeconds, accuracy: 1e-9)
    }

    func testRealityReplayPlaybackUsesSharedTimelineState() throws {
        let timeline = ReplayTimelineModel(durationSeconds: 4, preferredFrameRate: 15)
        let viewModel = try RealityReplayViewModel(asset: .loadBundledFixture(), timeline: timeline)

        XCTAssertFalse(viewModel.isPlaying)
        viewModel.togglePlayback()
        XCTAssertTrue(timeline.isPlaying)
        XCTAssertTrue(viewModel.isPlaying)
        viewModel.togglePlayback()
        XCTAssertFalse(timeline.isPlaying)
        XCTAssertFalse(viewModel.isPlaying)
    }
}
