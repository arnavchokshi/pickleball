import XCTest
@preconcurrency import AVFoundation
import PickleballCapture
import PickleballCore
import PickleballFastTier
import PickleballGuidance
@testable import Pickleball

/// W3-LIVE-MLP wiring coverage: the pure evaluators/builders are unit
/// tested exhaustively in `PickleballGuidanceTests`/`PickleballFastTierTests`
/// (cadence scheduler, court dot map, ball indicator gate, post-stop
/// summary). These tests instead cover the App-target glue in
/// `CaptureViewModel` -- default state before any live signal has arrived,
/// and that the ball indicator never leaves `.comingSoon` in this build.
final class LiveOverlayCaptureViewModelTests: XCTestCase {
    @MainActor
    func testInitialLiveOverlayStateIsHonestlyEmptyBeforeAnySignalArrives() {
        let model = CaptureViewModel()

        XCTAssertNil(model.liveGuidanceState)
        XCTAssertEqual(model.courtDotMapPoints, [])
        XCTAssertEqual(model.playerFootRings, [])
        XCTAssertEqual(model.ballTrailPoints, [])
        XCTAssertEqual(model.ballContactMarkers, [])
        XCTAssertEqual(model.ballIndicatorState, .comingSoon)
        XCTAssertNil(model.postStopSummary)
    }

    @MainActor
    func testDefaultCameraControllerGuidanceSampleFailsClosedThroughTheRealEvaluator() async {
        // FakeCameraCaptureController (used elsewhere in
        // CaptureViewModelTests) does not override `currentLiveGuidanceSample()`,
        // so it uses `CameraCaptureControlling`'s default (all-nil) sample.
        // Feeding that through the real `LiveGuidanceEvaluator` must render
        // every check `.unavailable` and grade `.warn` -- never `.good` with
        // zero evidence.
        let controller = FakeCameraCaptureController()

        let sample = await controller.currentLiveGuidanceSample()
        let state = LiveGuidanceEvaluator.evaluate(sample)

        XCTAssertTrue(state.checks.allSatisfy { $0.status == .unavailable })
        XCTAssertEqual(state.grade, .warn)
    }

    @MainActor
    func testBallIndicatorPolicyDefaultAlwaysComingSoonRegardlessOfInput() {
        // Directly exercises the same gate CaptureViewModel.handleLiveOverlayFrame
        // calls on every live frame -- confirms the App layer cannot
        // accidentally bypass the untrained-model gate.
        let state = LiveBallIndicatorPolicy.evaluate(rawConfidence: 1.0, rawNormalizedX: 0.5, rawNormalizedY: 0.5)

        XCTAssertEqual(state, .comingSoon)
    }

    @MainActor
    func testLiveOverlayFrameBuildsPlayerFootRingsForCameraPreview() {
        let model = CaptureViewModel()
        let frame = LiveCourtOverlayFrame(
            points: [
                CourtDotMapPoint(trackID: 2, normalizedX: 0.35, normalizedY: 0.82, confidence: 0.8),
            ],
            frameIndex: 40,
            detectorInvoked: true,
            videoAspectRatio: 16.0 / 9.0
        )

        model.ingestLiveOverlayFrame(frame)

        XCTAssertEqual(model.courtDotMapPoints.count, 1)
        XCTAssertEqual(model.playerFootRings.count, 1)
        let ring = try! XCTUnwrap(model.playerFootRings.first)
        XCTAssertEqual(ring.trackID, 2)
        XCTAssertEqual(ring.normalizedCenterY, 0.82, accuracy: 0.0001)
        XCTAssertEqual(model.liveOverlayVideoAspectRatio, 16.0 / 9.0, accuracy: 0.0001)
        XCTAssertEqual(model.ballIndicatorState, .comingSoon)
        XCTAssertEqual(model.ballTrailPoints, [])
        XCTAssertEqual(model.ballContactMarkers, [])
    }

    @MainActor
    func testLiveOverlayFramePrefersEngineSizedFootRingsForCameraPreview() {
        let model = CaptureViewModel()
        let engineRing = LivePlayerFootRing(
            trackID: 4,
            normalizedCenterX: 0.42,
            normalizedCenterY: 0.74,
            confidence: 0.85,
            stalenessFrames: 0,
            colorIndex: 0,
            normalizedWidth: 0.21,
            normalizedHeight: 0.0714,
            strokeOpacity: 0.9,
            fillOpacity: 0.16,
            source: .screenSpaceProxy
        )
        let frame = LiveCourtOverlayFrame(
            points: [
                CourtDotMapPoint(trackID: 4, normalizedX: 0.42, normalizedY: 0.74, confidence: 0.85),
            ],
            frameIndex: 41,
            detectorInvoked: true,
            videoAspectRatio: 16.0 / 9.0,
            playerFootRings: [engineRing]
        )

        model.ingestLiveOverlayFrame(frame)

        XCTAssertEqual(model.playerFootRings, [engineRing])
    }

    @MainActor
    func testTrackingBallStateBuildsTrailAndCandidateContactOverlayForCameraPreview() {
        let model = CaptureViewModel()
        let frames = [
            LiveCourtOverlayFrame(points: [], frameIndex: 20, detectorInvoked: false, videoAspectRatio: 16.0 / 9.0, ballState: trackingState(x: 0.20, y: 0.40, confidence: 0.90)),
            LiveCourtOverlayFrame(points: [], frameIndex: 21, detectorInvoked: false, videoAspectRatio: 16.0 / 9.0, ballState: trackingState(x: 0.32, y: 0.40, confidence: 0.88)),
            LiveCourtOverlayFrame(points: [], frameIndex: 22, detectorInvoked: false, videoAspectRatio: 16.0 / 9.0, ballState: trackingState(x: 0.22, y: 0.47, confidence: 0.86)),
        ]

        frames.forEach { model.ingestLiveOverlayFrame($0) }

        XCTAssertEqual(model.ballIndicatorState.availability, .tracking)
        XCTAssertEqual(model.ballTrailPoints.map(\.frameIndex), [20, 21, 22])
        let marker = try! XCTUnwrap(model.ballContactMarkers.first)
        XCTAssertEqual(marker.frameIndex, 21)
        XCTAssertEqual(marker.source, .kinematicInflectionCandidate)
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

private final class FakeCameraCaptureController: CameraCaptureControlling, @unchecked Sendable {
    let session = AVCaptureSession()
    var onRecordingFinished: ((Result<CameraRecordingResult, Error>) -> Void)?

    func configure(
        mode _: CaptureMode,
        deviceTier _: DeviceTier,
        capabilities _: CaptureCodecCapabilities,
        captureDeviceOrientation _: CaptureDeviceOrientation,
        sessionID _: String,
        packageRootURL _: URL
    ) async throws -> CapturePackageDescriptor {
        try CapturePackageDescriptor(
            sessionID: "fake",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly, orientation: .landscape),
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .landscapeRight
        )
    }

    func startPreview() async {}
    func stopPreview() async {}

    func startRecording() async throws -> CapturePackageDescriptor {
        try CapturePackageDescriptor(
            sessionID: "fake-recording",
            policy: CapturePolicy.recommended(for: .standard60, deviceTier: .standard, capabilities: .hevcOnly, orientation: .landscape),
            startedAt: Date(timeIntervalSince1970: 0),
            captureDeviceOrientation: .landscapeRight
        )
    }

    func stopRecording() async throws {}
}
