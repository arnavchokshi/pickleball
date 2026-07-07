import XCTest
import Foundation
import CoreGraphics
import PickleballCapture
import PickleballCore
@testable import Pickleball

final class DinkVisionStateTests: XCTestCase {
    @MainActor
    func testSplashStateMachineRunsSettleBlinkOpenUpThenCompletes() {
        var machine = DinkVisionSplashStateMachine(reducedMotion: false)

        XCTAssertEqual(machine.phase, .settle)
        XCTAssertEqual(machine.animationPlan, .lidReveal)
        XCTAssertEqual(machine.advance(), .blink)
        XCTAssertEqual(machine.advance(), .openUp)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertEqual(DinkVisionSplashTiming.settleNanoseconds, 150_000_000)
        XCTAssertEqual(DinkVisionSplashTiming.blinkCloseNanoseconds, 260_000_000)
        XCTAssertEqual(DinkVisionSplashTiming.blinkHoldNanoseconds, 120_000_000)
        XCTAssertEqual(DinkVisionSplashTiming.blinkOpenNanoseconds, 340_000_000)
        XCTAssertEqual(DinkVisionSplashTiming.openUpNanoseconds, 380_000_000)
        XCTAssertEqual(DinkVisionSplashTiming.totalDurationNanoseconds, 1_250_000_000)
    }

    @MainActor
    func testSplashStateMachineUsesCrossfadeWhenReducedMotionIsEnabled() {
        var machine = DinkVisionSplashStateMachine(reducedMotion: true)

        XCTAssertEqual(machine.phase, .settle)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertEqual(machine.animationPlan, .crossfade)
        XCTAssertEqual(DinkVisionSplashTiming.reducedMotionCrossfadeNanoseconds, 180_000_000)
    }

    @MainActor
    func testSplashLidGeometryMatchesOwnerArtworkFrame() {
        let markFrame = CGRect(x: 20, y: 40, width: 322, height: 579)
        let eyeCenter = DinkVisionSplashLidGeometry.eyeCenter(in: markFrame)

        XCTAssertEqual(eyeCenter.x, 181, accuracy: 0.001)
        XCTAssertEqual(eyeCenter.y, 249.019, accuracy: 0.001)
        XCTAssertEqual(DinkVisionSplashLidGeometry.apertureHalfHeight(in: markFrame), 83.955, accuracy: 0.001)
        XCTAssertEqual(DinkVisionSplashLidGeometry.lidSpan(in: markFrame), 276.92, accuracy: 0.001)
        XCTAssertEqual(DinkVisionSplashLidGeometry.strokeWidth(in: markFrame), 24.15, accuracy: 0.001)

        let openUpper = DinkVisionSplashLidGeometry.curve(isUpper: true, closure: 0, markFrame: markFrame)
        let closedUpper = DinkVisionSplashLidGeometry.curve(isUpper: true, closure: 1, markFrame: markFrame)
        XCTAssertEqual(openUpper.left.x, 42.54, accuracy: 0.001)
        XCTAssertEqual(openUpper.right.x, 319.46, accuracy: 0.001)
        XCTAssertEqual(openUpper.edgeY, eyeCenter.y - 83.955, accuracy: 0.001)
        XCTAssertEqual(closedUpper.edgeY, eyeCenter.y, accuracy: 0.001)
    }

    @MainActor
    func testSplashLidsCoverOnlyTheEyeLensAndOpenToZeroCoverage() {
        let markFrame = CGRect(x: 20, y: 40, width: 322, height: 579)
        let eyeCenter = DinkVisionSplashLidGeometry.eyeCenter(in: markFrame)
        let halfHeight = DinkVisionSplashLidGeometry.apertureHalfHeight(in: markFrame)

        let openUpper = DinkVisionSplashLidGeometry.lidCover(isUpper: true, closure: 0, markFrame: markFrame)
        let openLower = DinkVisionSplashLidGeometry.lidCover(isUpper: false, closure: 0, markFrame: markFrame)
        XCTAssertEqual(openUpper.coverRect.height, 0, accuracy: 0.001)
        XCTAssertEqual(openLower.coverRect.height, 0, accuracy: 0.001)

        let closedUpper = DinkVisionSplashLidGeometry.lidCover(isUpper: true, closure: 1, markFrame: markFrame)
        let closedLower = DinkVisionSplashLidGeometry.lidCover(isUpper: false, closure: 1, markFrame: markFrame)

        XCTAssertEqual(closedUpper.outerCurve.edgeY, eyeCenter.y - halfHeight, accuracy: 0.001)
        XCTAssertEqual(closedUpper.innerCurve.edgeY, eyeCenter.y, accuracy: 0.001)
        XCTAssertEqual(closedUpper.coverRect.minY, eyeCenter.y - halfHeight, accuracy: 0.001)
        XCTAssertEqual(closedUpper.coverRect.maxY, eyeCenter.y, accuracy: 0.001)

        XCTAssertEqual(closedLower.outerCurve.edgeY, eyeCenter.y + halfHeight, accuracy: 0.001)
        XCTAssertEqual(closedLower.innerCurve.edgeY, eyeCenter.y, accuracy: 0.001)
        XCTAssertEqual(closedLower.coverRect.minY, eyeCenter.y, accuracy: 0.001)
        XCTAssertEqual(closedLower.coverRect.maxY, eyeCenter.y + halfHeight, accuracy: 0.001)
    }

    @MainActor
    func testReplayOpenTrailIsBriefAndDisabledForReducedMotion() {
        XCTAssertEqual(DinkVisionReplayOpenTransition.durationNanoseconds, 550_000_000)
        XCTAssertEqual(DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: true), 180_000_000)
        XCTAssertEqual(
            DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: false),
            DinkVisionReplayOpenTransition.durationNanoseconds
        )
        XCTAssertEqual(DinkVisionReplayOpenTransition.plan(reducedMotion: false), .diagonalSwoosh)
        XCTAssertEqual(DinkVisionReplayOpenTransition.plan(reducedMotion: true), .crossfade)
    }

    @MainActor
    func testBrandV4AccentSitesStayLimitedToOwnerApprovedScreens() {
        XCTAssertEqual(
            DinkVisionAccentSite.allCases,
            [.replaysEmptyState, .statsSampleWatermark, .profileCompletedStep, .permissionPrimer, .coachPlaceholder]
        )
    }

    @MainActor
    func testCenterRecordTabLayoutKeepsRecordAsRaisedMiddleItem() {
        let model = DinkVisionTabLayoutModel.brandV4

        XCTAssertEqual(model.tabs.map(\.title), ["Replays", "Stats", "Record", "Coach", "Profile"])
        XCTAssertEqual(model.centerTab, .record)
        XCTAssertEqual(model.recordButtonDiameter, 72)
        XCTAssertEqual(model.recordButtonCenterAboveBarTop, 26)
        XCTAssertEqual(model.totalOverlayHeight(tabBarHeight: DinkVisionMetric.tabBarHeight), 150)
        XCTAssertEqual(model.recordButtonFrame(tabBarHeight: DinkVisionMetric.tabBarHeight), CGRect(x: -36, y: 0, width: 72, height: 72))
        XCTAssertEqual(model.recordButtonCenterY(tabBarHeight: DinkVisionMetric.tabBarHeight), 36)
        XCTAssertGreaterThanOrEqual(model.minimumHitTarget, 44)
    }

    @MainActor
    func testTexturedRecordButtonStateMachineMatchesOwnerSpec() {
        var machine = DinkVisionRecordButtonStateMachine()

        XCTAssertEqual(machine.state, .idle)
        XCTAssertEqual(machine.visual.scale, 1.0, accuracy: 0.001)
        XCTAssertEqual(machine.visual.breathingScaleRange.upperBound, 1.02, accuracy: 0.001)
        XCTAssertEqual(machine.visual.breathingDurationSeconds, 3.0, accuracy: 0.001)
        XCTAssertEqual(machine.visual.centerShape, .circle)
        XCTAssertEqual(machine.visual.ring, .ink)
        XCTAssertEqual(machine.visual.holeCount, 8)
        XCTAssertEqual(machine.visual.holeDiameterRatio, 0.13, accuracy: 0.001)

        machine.press()
        XCTAssertEqual(machine.state, .pressed)
        XCTAssertEqual(machine.visual.scale, 0.94, accuracy: 0.001)
        XCTAssertGreaterThan(machine.visual.innerShadowStrength, DinkVisionRecordButtonVisual.idle.innerShadowStrength)

        machine.startRecording()
        XCTAssertEqual(machine.state, .recording)
        XCTAssertEqual(machine.visual.centerShape, .roundedSquareStop)
        XCTAssertEqual(machine.visual.ring, .trailRed)

        machine.stopRecording()
        XCTAssertEqual(machine.state, .idle)
    }

    @MainActor
    func testStrokeDrawOnAndStickerMotionRespectReducedMotion() {
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.default.durationSeconds, 0.35, accuracy: 0.001)
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.default.initialTrimEnd, 0)
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.default.finalTrimEnd, 1)
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.resolved(reducedMotion: true).durationSeconds, 0)
        XCTAssertEqual(DinkVisionStrokeDrawOnParameters.resolved(reducedMotion: true).initialTrimEnd, 1)

        XCTAssertEqual(DinkVisionScreenMotionParameters.default.slidePoints, 8)
        XCTAssertEqual(DinkVisionScreenMotionParameters.default.rotationDegrees, 1.2, accuracy: 0.001)
        XCTAssertEqual(DinkVisionScreenMotionParameters.resolved(reducedMotion: true).slidePoints, 0)
        XCTAssertEqual(DinkVisionScreenMotionParameters.resolved(reducedMotion: true).rotationDegrees, 0)
    }

    @MainActor
    func testCoachPlaceholderIsClearlyComingSoonWithoutFakeFeatures() {
        let model = DinkVisionCoachPlaceholderModel.brandV4

        XCTAssertEqual(model.title, "Your pocket coach is training...")
        XCTAssertEqual(model.roadmapID, "P6")
        XCTAssertTrue(model.isComingSoon)
        XCTAssertTrue(model.fakeFeatureBullets.isEmpty)
    }

    @MainActor
    func testPolicyChipMapperSeparatesEISLockAndLandscapeSignals() {
        let report = CapturePolicyEnforcementReport(
            requested: CapturePolicyRequestedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .landscape,
                electronicStabilizationEnabled: false,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            achieved: CapturePolicyAchievedState(
                fps: 60,
                resolution: [1920, 1080],
                format: .hevc,
                orientation: .portrait,
                electronicStabilizationEnabled: true,
                exposureLocked: true,
                focusLocked: true,
                whiteBalanceLocked: true
            ),
            violations: ["electronic_stabilization_enabled", "orientation_not_landscape"]
        )

        let chips = DinkVisionPolicyChipMapper.chips(for: report)

        XCTAssertEqual(chips.map(\.id), ["eis", "camera_locks", "landscape"])
        XCTAssertEqual(chips[0].status, .warning)
        XCTAssertEqual(chips[1].status, .pass)
        XCTAssertEqual(chips[2].status, .warning)
        XCTAssertEqual(chips[0].hint, "Turn Enhanced Stabilization off before recording.")
    }

    @MainActor
    func testReplayListDataSourceLoadsRealLocalCapturesNewestFirst() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dinkvision-replay-datasource-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCapture(root: root, sessionID: "older", startedAt: "2026-07-07T09:00:00Z", duration: 92)
        try Self.writeCapture(root: root, sessionID: "newer", startedAt: "2026-07-07T10:00:00Z", duration: 124)

        let dataSource = DinkVisionReplayListDataSource(packageRootURL: root)
        let rows = try dataSource.loadRows()

        XCTAssertEqual(rows.map(\.id), ["newer", "older"])
        XCTAssertEqual(rows[0].durationText, "02:04")
        XCTAssertEqual(rows[0].subtitle.contains("60 fps"), true)
    }

    @MainActor
    func testReplayListDataSourceReadsOptionalRealTrustSidecarFieldsWhenPresent() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dinkvision-replay-trust-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Self.writeCapture(
            root: root,
            sessionID: "trusted",
            startedAt: "2026-07-07T10:00:00Z",
            duration: 124,
            extraSidecarFields: ["trusted_3d": true, "ball_percent": 0.83]
        )

        let row = try XCTUnwrap(DinkVisionReplayListDataSource(packageRootURL: root).loadRows().first)

        XCTAssertEqual(row.trusted3DText, "Trusted 3D yes")
        XCTAssertEqual(row.ballTrustText, "Ball 83%")
    }

    @MainActor
    func testReplayListDataSourceReturnsEmptyRowsWhenCaptureDirectoryIsMissing() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dinkvision-replay-empty-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let rows = try DinkVisionReplayListDataSource(packageRootURL: root).loadRows()

        XCTAssertTrue(rows.isEmpty)
    }

    private static func writeCapture(
        root: URL,
        sessionID: String,
        startedAt: String,
        duration: Double,
        extraSidecarFields: [String: Any] = [:]
    ) throws {
        let packageURL = root.appendingPathComponent("captures", isDirectory: true)
            .appendingPathComponent(sessionID, isDirectory: true)
        try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)
        try Data().write(to: packageURL.appendingPathComponent("clip.mov"))
        let sidecar = CaptureSidecar(
            provenance: .liveRecording,
            deviceTier: .standard,
            deviceModel: "iPhone",
            fps: 60,
            format: .hevc,
            resolution: [1920, 1080],
            recordingStartedAt: startedAt,
            recordingDurationS: duration,
            locked: LockedCapture(exposureS: 1 / 120, iso: 160, focus: 1.0, wbLocked: true),
            intrinsics: nil,
            gravity: [0, -1, 0],
            captureQuality: CaptureQuality(grade: .good)
        )
        let data = try JSONEncoder().encode(sidecar)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        extraSidecarFields.forEach { key, value in
            object[key] = value
        }
        let merged = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try merged.write(to: packageURL.appendingPathComponent("capture_sidecar.json"))
    }
}
