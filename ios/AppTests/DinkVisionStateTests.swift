import XCTest
import Foundation
import PickleballCapture
import PickleballCore
@testable import Pickleball

final class DinkVisionStateTests: XCTestCase {
    @MainActor
    func testSplashStateMachineOpensLidsThenCompletes() {
        var machine = DinkVisionSplashStateMachine(reducedMotion: false)

        XCTAssertEqual(machine.phase, .zoomedClosed)
        XCTAssertEqual(machine.animationPlan, .lidReveal)
        XCTAssertEqual(machine.advance(), .lidsOpening)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertLessThanOrEqual(DinkVisionSplashTiming.totalDurationNanoseconds, 900_000_000)
    }

    @MainActor
    func testSplashStateMachineUsesCrossfadeWhenReducedMotionIsEnabled() {
        var machine = DinkVisionSplashStateMachine(reducedMotion: true)

        XCTAssertEqual(machine.phase, .zoomedClosed)
        XCTAssertEqual(machine.advance(), .done)
        XCTAssertEqual(machine.animationPlan, .crossfade)
    }

    @MainActor
    func testReplayOpenTrailIsBriefAndDisabledForReducedMotion() {
        XCTAssertLessThanOrEqual(DinkVisionReplayOpenTransition.durationNanoseconds, 450_000_000)
        XCTAssertEqual(DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: true), 0)
        XCTAssertEqual(
            DinkVisionReplayOpenTransition.durationNanoseconds(reducedMotion: false),
            DinkVisionReplayOpenTransition.durationNanoseconds
        )
    }

    @MainActor
    func testBrandV2AccentSitesStayLimitedToOwnerApprovedScreens() {
        XCTAssertEqual(
            DinkVisionAccentSite.allCases,
            [.replaysEmptyState, .statsSampleWatermark, .profileCompletedStep, .permissionPrimer]
        )
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
    func testReplayListDataSourceReturnsEmptyRowsWhenCaptureDirectoryIsMissing() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dinkvision-replay-empty-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let rows = try DinkVisionReplayListDataSource(packageRootURL: root).loadRows()

        XCTAssertTrue(rows.isEmpty)
    }

    private static func writeCapture(root: URL, sessionID: String, startedAt: String, duration: Double) throws {
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
        try data.write(to: packageURL.appendingPathComponent("capture_sidecar.json"))
    }
}
